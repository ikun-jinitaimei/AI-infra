#pragma once

#include "kernel_operator.h"

namespace RsmSolution {

constexpr uint32_t kScoreTile = 4096;
constexpr uint32_t kMaxD = 256;
constexpr uint32_t kAlignment = 8;
constexpr uint32_t kXTileElements = 8192;

class ComputeCore {
public:
    __aicore__ inline void Init(
        GM_ADDR score, GM_ADDR x, GM_ADDR offsets, GM_ADDR mean, GM_ADDR rstd,
        GM_ADDR logsumexp, uint32_t n, uint32_t d, float epsilon)
    {
        d_ = d;
        stride_ = d;
        epsilon_ = epsilon;
        score_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(score), n);
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), n * d);
        offsets_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
        mean_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));
        rstd_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rstd));
        logsumexp_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logsumexp));

        pipe_.InitBuffer(score_half_buffer_, kScoreTile * sizeof(half));
        pipe_.InitBuffer(score_float_buffer_, kScoreTile * sizeof(float));
        pipe_.InitBuffer(reduce_output_buffer_, 64 * sizeof(float));
        pipe_.InitBuffer(reduce_work_buffer_, kScoreTile * sizeof(float));
        pipe_.InitBuffer(x_half_buffer_, kXTileElements * sizeof(half));
        pipe_.InitBuffer(x_float_buffer_, kXTileElements * sizeof(float));
        pipe_.InitBuffer(weights_buffer_, kXTileElements * sizeof(float));
        pipe_.InitBuffer(broadcast_buffer_, kXTileElements * sizeof(float));
        pipe_.InitBuffer(centered_buffer_, kXTileElements * sizeof(float));
        pipe_.InitBuffer(mean_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(m2_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(mean_residual_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(variance_chunk_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(mean_correction_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(delta_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(temp_buffer_, kMaxD * sizeof(float));
    }

    // Cache several consecutive segments in the existing input buffers.
    // Bound every batch by real offsets and both buffer capacities.
    __aicore__ inline void ProcessRange(uint32_t first, uint32_t last)
    {
        uint32_t segment = first;
        const uint32_t capacity = Minimum(kScoreTile, kXTileElements / stride_);
        while (segment < last) {
            const uint32_t begin = static_cast<uint32_t>(offsets_gm_.GetValue(segment));
            uint32_t next = segment + 1;
            uint32_t end = static_cast<uint32_t>(offsets_gm_.GetValue(next));
            if (end - begin <= capacity) {
                while (next < last) {
                    const uint32_t candidate = static_cast<uint32_t>(offsets_gm_.GetValue(next + 1));
                    if (candidate - begin > capacity) break;
                    end = candidate;
                    ++next;
                }
            }
            batch_active_ = next > segment + 1;
            if (batch_active_) {
                batch_begin_ = begin;
                const uint32_t rows = end - begin;
                AscendC::DataCopyPad(x_half_buffer_.Get<half>(), x_gm_[begin * stride_],
                    {1, static_cast<uint16_t>(rows * stride_ * sizeof(half)), 0, 0}, {});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            while (segment < next) {
                const uint32_t start = static_cast<uint32_t>(offsets_gm_.GetValue(segment));
                uint32_t rows = static_cast<uint32_t>(offsets_gm_.GetValue(segment + 1)) - start;
                uint32_t groups = 1;
                if (batch_active_ && rows <= 16) {
                    const uint32_t limit = Minimum(512u / stride_, next - segment);
                    while (groups < limit) {
                        const uint32_t stop = static_cast<uint32_t>(offsets_gm_.GetValue(segment + groups + 1));
                        const uint32_t previous = static_cast<uint32_t>(offsets_gm_.GetValue(segment + groups));
                        const uint32_t length = stop - previous;
                        if (length > 16) break;
                        const uint32_t maximum = rows > length ? rows : length;
                        uint32_t padded = 1;
                        while (padded < maximum) padded *= 2;
                        // At least half the padded vector lanes must contain real rows.
                        if (padded * (groups + 1) > 2 * (stop - start)) break;
                        rows = maximum;
                        ++groups;
                    }
                }
                if (groups > 1) ProcessShortGroup(segment, start, rows, groups);
                else ProcessSegment(segment, 0, stride_);
                segment += groups;
            }
            batch_active_ = false;
        }
    }
    // Ragged neighboring segments occupy independent column lanes with zero padding.
    // Only the traversal is shared; each segment retains its own softmax Z.
    __aicore__ inline void ProcessShortGroup(uint32_t segment, uint32_t begin,
        uint32_t rows, uint32_t groups)
    {
        d_ = groups * stride_;
        rows_per_x_tile_ = 1;
        while (rows_per_x_tile_ < rows) rows_per_x_tile_ *= 2;
        const uint32_t elements = rows * d_;
        auto original = centered_buffer_.Get<float>();
        auto staging = broadcast_buffer_.Get<float>();
        auto matrix = x_float_buffer_.Get<float>();
        auto weights = weights_buffer_.Get<float>();
        // A group loads at most 256 half scores; upper float regions remain disjoint.
        auto zvec = score_half_buffer_.Get<float>()[512];
        auto inverse = score_half_buffer_.Get<float>()[1024];
        auto lses = variance_chunk_buffer_.Get<float>();
        auto scores = score_float_buffer_.Get<float>();
        auto scalar = reduce_output_buffer_.Get<float>();
        const uint32_t actual_rows = static_cast<uint32_t>(offsets_gm_.GetValue(segment + groups)) - begin;
        // Compact softmax lanes use eight floats per segment for UB alignment.
        // Perform one DMA/Cast/Exp and two row trees for the entire group.
        const uint32_t pitch = groups * kAlignment;
        const uint32_t score_elements = rows_per_x_tile_ * pitch;
        auto work = reduce_work_buffer_.Get<float>();
        auto maximum = mean_buffer_.Get<float>();
        AscendC::DataCopyPad(score_half_buffer_.Get<half>(), score_gm_[begin],
            {1, static_cast<uint16_t>(actual_rows * sizeof(half)), 0, 0}, {});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(staging, score_half_buffer_.Get<half>(),
            AscendC::RoundMode::CAST_NONE, actual_rows);
        AscendC::PipeBarrier<PIPE_ALL>();
        staging.SetValue(actual_rows, -3.402823466e38f);
        auto score_offsets = reduce_work_buffer_.Get<uint32_t>();
        AscendC::Duplicate(score_offsets, static_cast<uint32_t>(actual_rows * sizeof(float)), score_elements);
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t start = static_cast<uint32_t>(offsets_gm_.GetValue(segment + g));
            const uint32_t stop = static_cast<uint32_t>(offsets_gm_.GetValue(segment + g + 1));
            for (uint32_t r = 0; r < stop - start; ++r) {
                AscendC::Duplicate(score_offsets[r * pitch + g * kAlignment],
                    static_cast<uint32_t>((start - begin + r) * sizeof(float)), kAlignment);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Gather(scores, staging, score_offsets, 0u, score_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(work, scores, score_elements);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t half_rows = rows_per_x_tile_ / 2; half_rows; half_rows /= 2) {
            AscendC::Max(work, work, work[half_rows * pitch], half_rows * pitch);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopy(maximum, work, pitch);
        for (uint32_t r = 1; r < rows_per_x_tile_; r *= 2) {
            AscendC::DataCopy(work[r * pitch], work, r * pitch);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Sub(scores, scores, work, score_elements);
        AscendC::Exp(scores, scores, score_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(work, scores, score_elements);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t half_rows = rows_per_x_tile_ / 2; half_rows; half_rows /= 2) {
            AscendC::Add(work, work, work[half_rows * pitch], half_rows * pitch);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Ln(lses, work, pitch);
        AscendC::Add(lses, lses, maximum, pitch);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(original, 0.0f, elements);
        AscendC::Cast(staging,
            x_half_buffer_.Get<half>()[(begin - batch_begin_) * stride_],
            AscendC::RoundMode::CAST_NONE, actual_rows * stride_);
        AscendC::PipeBarrier<PIPE_V>();
        auto gather_offsets = score_half_buffer_.Get<uint32_t>()[1536];
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t start = static_cast<uint32_t>(offsets_gm_.GetValue(segment + g));
            const uint32_t stop = static_cast<uint32_t>(offsets_gm_.GetValue(segment + g + 1));
            const float z = work.GetValue(g * kAlignment);
            AscendC::Duplicate(zvec[g * stride_], z, stride_);
            AscendC::Duplicate(inverse[g * stride_], 1.0f / z, stride_);
            AscendC::Duplicate(gather_offsets[g * stride_],
                static_cast<uint32_t>(g * kAlignment * sizeof(float)), stride_);
            AscendC::DataCopy(original[g * stride_], staging[(start - begin) * stride_],
                {static_cast<uint16_t>(stop - start), static_cast<uint16_t>(stride_ / kAlignment),
                 0, static_cast<uint16_t>((d_ - stride_) / kAlignment)});
        }

        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::Gather(weights[r * d_], scores, gather_offsets,
                static_cast<uint32_t>(r * pitch * sizeof(float)), d_);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        auto mean = score_float_buffer_.Get<float>();
        auto residual = score_float_buffer_.Get<float>()[512];
        auto delta = score_float_buffer_.Get<float>()[1024];
        auto term = score_float_buffer_.Get<float>()[1536];
        auto m2 = score_float_buffer_.Get<float>()[2048];
        auto second = centered_buffer_.Get<float>();
        AscendC::Mul(matrix, original, weights, elements);
        ReduceRows(rows);
        AscendC::Mul(mean, matrix, inverse, d_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(staging, mean, d_);
        for (uint32_t r = 1; r < rows_per_x_tile_; r *= 2) {
            AscendC::DataCopy(staging[r * d_], staging, r * d_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Sub(matrix, original, staging, elements);
        AscendC::PipeBarrier<PIPE_V>();
        // Original input is dead after centering; reuse its 8192-float buffer.
        AscendC::Mul(second, matrix, matrix, elements);
        AscendC::Mul(second, second, weights, elements);
        AscendC::Mul(matrix, matrix, weights, elements);
        ReduceMomentPair(rows, second);
        AscendC::DataCopy(residual, matrix, d_);
        AscendC::DataCopy(m2, second, d_);
        AscendC::DataCopy(term, mean, d_);
        AscendC::Mul(delta, residual, inverse, d_);
        AscendC::Add(mean, mean, delta, d_);
        AscendC::Sub(delta, mean, term, d_);
        AscendC::Mul(term, delta, residual, d_);
        AscendC::Muls(term, term, 2.0f, d_);
        AscendC::Sub(m2, m2, term, d_);
        AscendC::Mul(term, delta, delta, d_);
        AscendC::Mul(term, term, zvec, d_);
        AscendC::Add(m2, m2, term, d_);
        AscendC::Mul(m2, m2, inverse, d_);
        AscendC::Maxs(m2, m2, 0.0f, d_);
        AscendC::Adds(m2, m2, epsilon_, d_);
        AscendC::Rsqrt(m2, m2, d_);
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint16_t bytes = static_cast<uint16_t>(d_ * sizeof(float));
        AscendC::DataCopyPad(mean_gm_[segment * stride_], mean, {1, bytes, 0, 0});
        AscendC::DataCopyPad(rstd_gm_[segment * stride_], m2, {1, bytes, 0, 0});
        for (uint32_t g = 0; g < groups; ++g) {
            AscendC::DataCopyPad(logsumexp_gm_[segment + g], lses[g * kAlignment],
                {1, sizeof(float), 0, 0});
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }


    __aicore__ inline void ProcessSegment(uint32_t segment, uint32_t column, uint32_t width)
    {
        column_ = column;
        d_ = Minimum(width, stride_ - column);
        rows_per_x_tile_ = d_ <= 64 ? 128 : (d_ <= 128 ? 64 : 32);
        loaded_rows_ = 0;
        weight_rows_ = 0;
        const uint32_t begin = static_cast<uint32_t>(offsets_gm_.GetValue(segment));
        const uint32_t end = static_cast<uint32_t>(offsets_gm_.GetValue(segment + 1));
        while (rows_per_x_tile_ > 1 && rows_per_x_tile_ / 2 >= end - begin)
            rows_per_x_tile_ /= 2;
        const float maximum = SegmentMaximum(begin, end);
        const float normalizer = AccumulateMoments(begin, end, maximum);
        FinalizeMoments(normalizer);
        StoreResults(segment, maximum, normalizer);
    }

private:
    // Pad row tiles to a power of two and reduce all columns concurrently.
    __aicore__ inline void ReduceRows(uint32_t rows)
    {
        auto matrix = x_float_buffer_.Get<float>();
        uint32_t padded_rows = 1;
        while (padded_rows < rows) padded_rows *= 2;
        if (rows < padded_rows) {
            AscendC::Duplicate(matrix[rows * d_], 0.0f, (padded_rows - rows) * d_);
        }
        for (uint32_t half_rows = padded_rows / 2; half_rows > 0; half_rows /= 2) {
            AscendC::Add(matrix, matrix, matrix[half_rows * d_], half_rows * d_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Reduce the two independent moments together, sharing loop/synchronization.
    __aicore__ inline void ReduceMomentPair(uint32_t rows, AscendC::LocalTensor<float> second)
    {
        auto first = x_float_buffer_.Get<float>();
        uint32_t padded = 1;
        while (padded < rows) padded *= 2;
        if (rows < padded) {
            AscendC::Duplicate(first[rows * d_], 0.0f, (padded - rows) * d_);
            AscendC::Duplicate(second[rows * d_], 0.0f, (padded - rows) * d_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t half_rows = padded / 2; half_rows > 0; half_rows /= 2) {
            AscendC::Add(first, first, first[half_rows * d_], half_rows * d_);
            AscendC::Add(second, second, second[half_rows * d_], half_rows * d_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void FillWeights(uint32_t row_base, uint32_t rows)
    {
        if (weight_rows_ == rows && weight_row_base_ == row_base) return;
        auto scores = score_float_buffer_.Get<float>();
        auto weights = weights_buffer_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Duplicate(weights[row * d_], scores.GetValue(row_base + row), d_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        weight_rows_ = rows;
        weight_row_base_ = row_base;
    }

    __aicore__ inline void BroadcastMean()
    {
        auto broadcast = broadcast_buffer_.Get<float>();
        AscendC::DataCopy(broadcast, mean_buffer_.Get<float>(), d_);
        for (uint32_t rows = 1; rows < rows_per_x_tile_; rows *= 2) {
            AscendC::DataCopy(broadcast[rows * d_], broadcast, rows * d_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline uint32_t Minimum(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void LoadScores(
        uint32_t begin, uint32_t count, float maximum, bool exponentiate)
    {
        weight_rows_ = 0;
        auto score_half = score_half_buffer_.Get<half>();
        const uint16_t bytes = static_cast<uint16_t>(count * sizeof(half));
        AscendC::DataCopyPad(score_half, score_gm_[begin], {1, bytes, 0, 0}, {});
        AscendC::PipeBarrier<PIPE_ALL>();
        auto score_float = score_float_buffer_.Get<float>();
        AscendC::Cast(score_float, score_half, AscendC::RoundMode::CAST_NONE, count);
        if (exponentiate) {
            AscendC::Adds(score_float, score_float, -maximum, count);
            AscendC::Exp(score_float, score_float, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadRows(uint32_t begin, uint32_t rows)
    {
        const uint32_t elements = rows * d_;
        const uint16_t bytes = static_cast<uint16_t>(elements * sizeof(half));
        auto x_half = x_half_buffer_.Get<half>();
        if (batch_active_) {
            x_half = x_half[(begin - batch_begin_) * stride_];
        } else if (begin != loaded_begin_ || rows != loaded_rows_) {
        if (d_ == stride_) {
            AscendC::DataCopyPad(x_half, x_gm_[begin * stride_], {1, bytes, 0, 0}, {});
        } else {
            const uint16_t row_bytes = static_cast<uint16_t>(d_ * sizeof(half));
            AscendC::DataCopyExtParams copy_params{
                static_cast<uint16_t>(rows), row_bytes,
                static_cast<uint32_t>((stride_ - d_) * sizeof(half)), 0, 0};
            AscendC::DataCopyPadExtParams<half> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(x_half, x_gm_[begin * stride_ + column_],
                copy_params, pad_params);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        loaded_begin_ = begin;
        loaded_rows_ = rows;
        }
        AscendC::Cast(
            x_float_buffer_.Get<float>(), x_half,
            AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float SegmentMaximum(uint32_t begin, uint32_t end)
    {
        auto score_float = score_float_buffer_.Get<float>();
        auto reduce_output = reduce_output_buffer_.Get<float>();
        auto reduce_work = reduce_work_buffer_.Get<float>();
        float maximum = -65504.0f;
        for (uint32_t base = begin; base < end; base += kScoreTile) {
            const uint32_t count = Minimum(kScoreTile, end - base);
            LoadScores(base, count, 0.0f, false);
            AscendC::ReduceMax(reduce_output, score_float, reduce_work, count, false);
            AscendC::PipeBarrier<PIPE_V>();
            const float tile_maximum = reduce_output.GetValue(0);
            maximum = maximum > tile_maximum ? maximum : tile_maximum;
        }
        return maximum;
    }

    // Sum exp(score-max) and exp(score-max)*x together, with scalar and
    // vector Kahan compensation. Normalize once after the segment. With
    // finite FP16 x and at most 2^20 rows, the unnormalized sum fits in FP32.
    // Refine the resulting mean before computing centered variance.
    __aicore__ inline float AccumulateMoments(
        uint32_t begin, uint32_t end, float maximum)
    {
        auto sum = mean_buffer_.Get<float>();
        auto correction = mean_correction_buffer_.Get<float>();
        auto term = temp_buffer_.Get<float>();
        auto delta = delta_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto score_float = score_float_buffer_.Get<float>();
        AscendC::Duplicate(correction, 0.0f, d_);
        AscendC::PipeBarrier<PIPE_V>();

        float weight_total = 0.0f;
        float weight_correction = 0.0f;
        for (uint32_t base = begin; base < end; base += kScoreTile) {
            const uint32_t count = Minimum(kScoreTile, end - base);
            if (end - begin <= kScoreTile) {
                // SegmentMaximum left the single FP32 score tile resident.
                AscendC::Adds(score_float, score_float, -maximum, count);
                AscendC::Exp(score_float, score_float, count);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                LoadScores(base, count, maximum, true);
            }
            auto reduced_weight = reduce_output_buffer_.Get<float>();
            AscendC::ReduceSum(reduced_weight, score_float,
                reduce_work_buffer_.Get<float>(), count);
            AscendC::PipeBarrier<PIPE_V>();
            const float weight_delta = reduced_weight.GetValue(0) - weight_correction;
            const float next_weight = weight_total + weight_delta;
            weight_correction = (next_weight - weight_total) - weight_delta;
            weight_total = next_weight;
            for (uint32_t row_base = 0; row_base < count; row_base += rows_per_x_tile_) {
                const uint32_t rows = Minimum(rows_per_x_tile_, count - row_base);
                LoadRows(base + row_base, rows);
                FillWeights(row_base, rows);
                AscendC::Mul(x_float, x_float, weights_buffer_.Get<float>(), rows * d_);
                ReduceRows(rows);
                if (base == begin && row_base == 0) {
                    AscendC::DataCopy(sum, x_float, d_);
                } else {
                AscendC::Sub(delta, x_float, correction, d_);
                AscendC::Add(term, sum, delta, d_);
                AscendC::Sub(correction, term, sum, d_);
                AscendC::Sub(correction, correction, delta, d_);
                AscendC::DataCopy(sum, term, d_);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(sum, sum, 1.0f / weight_total, d_);
        AscendC::PipeBarrier<PIPE_V>();
        if (end - begin <= rows_per_x_tile_ && (end - begin) * d_ <= kScoreTile) {
            RefineShortMoments(begin, end, weight_total);
        } else {
            RefineMean(begin, end, maximum, weight_total);
        }
        return weight_total;
    }

    // Fuse residual and variance traversal for a fully resident short segment.
    // Shift the centered second moment to the final rounded FP32 mean.
    __aicore__ inline void RefineShortMoments(uint32_t begin, uint32_t end, float z)
    {
        const uint32_t rows = end - begin;
        const uint32_t elements = rows * d_;
        auto matrix = x_float_buffer_.Get<float>();
        auto saved = centered_buffer_.Get<float>();
        auto mean = mean_buffer_.Get<float>();
        auto residual = mean_residual_buffer_.Get<float>();
        auto delta = delta_buffer_.Get<float>();
        auto term = temp_buffer_.Get<float>();
        auto m2 = m2_buffer_.Get<float>();
        BroadcastMean();
        LoadRows(begin, rows);
        FillWeights(0, rows);
        AscendC::Sub(matrix, matrix, broadcast_buffer_.Get<float>(), elements);
        AscendC::Mul(saved, matrix, matrix, elements);
        AscendC::Mul(saved, saved, weights_buffer_.Get<float>(), elements);
        AscendC::Mul(matrix, matrix, weights_buffer_.Get<float>(), elements);
        ReduceMomentPair(rows, saved);
        AscendC::DataCopy(residual, matrix, d_);
        AscendC::DataCopy(m2, saved, d_);
        AscendC::DataCopy(term, mean, d_);
        AscendC::Muls(delta, residual, 1.0f / z, d_);
        AscendC::Add(mean, mean, delta, d_);
        AscendC::Sub(delta, mean, term, d_);
        AscendC::Mul(term, delta, residual, d_);
        AscendC::Muls(term, term, 2.0f, d_);
        AscendC::Sub(m2, m2, term, d_);
        AscendC::Mul(term, delta, delta, d_);
        AscendC::Muls(term, term, z, d_);
        AscendC::Add(m2, m2, term, d_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Recenter around the preliminary mean. This keeps constant columns
    // exact and recovers sub-ULP updates before the final FP32 rounding,
    // without choosing an arbitrary input row as a potentially distant center.
    __aicore__ inline void RefineMean(
        uint32_t begin, uint32_t end, float maximum, float normalizer)
    {
        auto mean = mean_buffer_.Get<float>();
        auto residual = mean_residual_buffer_.Get<float>();
        auto correction = mean_correction_buffer_.Get<float>();
        auto term = temp_buffer_.Get<float>();
        auto delta = delta_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto saved = centered_buffer_.Get<float>();
        auto m2 = m2_buffer_.Get<float>();
        AscendC::Duplicate(correction, 0.0f, d_);
        BroadcastMean();
        for (uint32_t base = begin; base < end; base += kScoreTile) {
            const uint32_t count = Minimum(kScoreTile, end - base);
            // A single score tile remains resident from the preceding pass.
            if (end - begin > kScoreTile) LoadScores(base, count, maximum, true);
            for (uint32_t row_base = 0; row_base < count; row_base += rows_per_x_tile_) {
                const uint32_t rows = Minimum(rows_per_x_tile_, count - row_base);
                LoadRows(base + row_base, rows);
                FillWeights(row_base, rows);
                AscendC::Sub(x_float, x_float, broadcast_buffer_.Get<float>(), rows * d_);
                AscendC::Mul(saved, x_float, x_float, rows * d_);
                AscendC::Mul(saved, saved, weights_buffer_.Get<float>(), rows * d_);
                AscendC::Mul(x_float, x_float, weights_buffer_.Get<float>(), rows * d_);
                ReduceMomentPair(rows, saved);
                if (base == begin && row_base == 0) {
                    AscendC::DataCopy(residual, x_float, d_);
                } else {
                AscendC::Sub(delta, x_float, correction, d_);
                AscendC::Add(term, residual, delta, d_);
                AscendC::Sub(correction, term, residual, d_);
                AscendC::Sub(correction, correction, delta, d_);
                AscendC::DataCopy(residual, term, d_);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (base == begin && row_base == 0)
                    AscendC::DataCopy(m2, saved, d_);
                else
                    AscendC::Add(m2, m2, saved, d_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        // Shift to the actual rounded final mean, preserving the residual sum.
        AscendC::DataCopy(term, mean, d_);
        AscendC::Muls(delta, residual, 1.0f / normalizer, d_);
        AscendC::Add(mean, mean, delta, d_);
        AscendC::Sub(delta, mean, term, d_);
        AscendC::Mul(term, delta, residual, d_);
        AscendC::Muls(term, term, 2.0f, d_);
        AscendC::Sub(m2, m2, term, d_);
        AscendC::Mul(term, delta, delta, d_);
        AscendC::Muls(term, term, normalizer, d_);
        AscendC::Add(m2, m2, term, d_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateCenteredVariance(
        uint32_t begin, uint32_t end, float maximum)
    {
        auto mean = mean_buffer_.Get<float>();
        auto m2 = m2_buffer_.Get<float>();
        auto temp = temp_buffer_.Get<float>();
        auto chunk = variance_chunk_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto score_float = score_float_buffer_.Get<float>();
        BroadcastMean();
        for (uint32_t base = begin; base < end; base += kScoreTile) {
            const uint32_t count = Minimum(kScoreTile, end - base);
            if (end - begin > kScoreTile) LoadScores(base, count, maximum, true);
            for (uint32_t row_base = 0; row_base < count; row_base += rows_per_x_tile_) {
                const uint32_t rows = Minimum(rows_per_x_tile_, count - row_base);
                LoadRows(base + row_base, rows);
                FillWeights(row_base, rows);
                AscendC::Sub(x_float, x_float, broadcast_buffer_.Get<float>(), rows * d_);
                AscendC::Mul(x_float, x_float, x_float, rows * d_);
                AscendC::Mul(x_float, x_float, weights_buffer_.Get<float>(), rows * d_);
                ReduceRows(rows);
                if (base == begin && row_base == 0)
                    AscendC::DataCopy(m2, x_float, d_);
                else
                    AscendC::Add(m2, m2, x_float, d_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void FinalizeMoments(float normalizer)
    {
        auto m2 = m2_buffer_.Get<float>();
        AscendC::Muls(m2, m2, 1.0f / normalizer, d_);
        AscendC::Maxs(m2, m2, 0.0f, d_);
        AscendC::Adds(m2, m2, epsilon_, d_);
        AscendC::Rsqrt(m2, m2, d_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreResults(
        uint32_t segment, float maximum, float normalizer)
    {
        auto mean = mean_buffer_.Get<float>();
        auto m2 = m2_buffer_.Get<float>();
        auto temp = temp_buffer_.Get<float>();
        const uint16_t bytes = static_cast<uint16_t>(d_ * sizeof(float));
        AscendC::DataCopyPad(mean_gm_[segment * stride_ + column_], mean, {1, bytes, 0, 0});
        AscendC::DataCopyPad(rstd_gm_[segment * stride_ + column_], m2, {1, bytes, 0, 0});

        // One writer for the per-segment scalar, including for split segments.
        if (column_ != 0) {
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        auto scalar = reduce_output_buffer_.Get<float>();
        AscendC::Duplicate(scalar, normalizer, kAlignment);
        AscendC::Ln(scalar, scalar, kAlignment);
        AscendC::PipeBarrier<PIPE_V>();
        temp.SetValue(0, scalar.GetValue(0) + maximum);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(
            logsumexp_gm_[segment], temp, {1, sizeof(float), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> score_half_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> score_float_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_output_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_work_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_half_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_float_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weights_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadcast_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> centered_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mean_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> m2_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mean_residual_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> variance_chunk_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mean_correction_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> delta_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp_buffer_;
    AscendC::GlobalTensor<half> score_gm_;
    AscendC::GlobalTensor<half> x_gm_;
    AscendC::GlobalTensor<int32_t> offsets_gm_;
    AscendC::GlobalTensor<float> mean_gm_;
    AscendC::GlobalTensor<float> rstd_gm_;
    AscendC::GlobalTensor<float> logsumexp_gm_;
    uint32_t d_ = 0;
    uint32_t stride_ = 0;
    uint32_t column_ = 0;
    uint32_t rows_per_x_tile_ = 32;
    uint32_t loaded_begin_ = 0;
    uint32_t loaded_rows_ = 0;
    uint32_t weight_rows_ = 0;
    uint32_t weight_row_base_ = 0;
    bool batch_active_ = false;
    uint32_t batch_begin_ = 0;
    float epsilon_ = 0.0f;
};

}  // namespace RsmSolution
