#include "rsm_solution_host.h"

#include <algorithm>

namespace {
constexpr uint32_t kVectorCores = 40;
}

void ConfigureSolutionLaunch(uint32_t n, uint32_t d, uint32_t s, float epsilon,
    const int32_t *offsets, RsmSolutionTiling *tiling,
    uint32_t *block_dim, uint32_t *tiling_key)
{
    (void)offsets;
    // Expose column parallelism when there are too few segments to fill cores.
    // Tiles are multiples of 16 so FP16 row copies stay 32-byte aligned.
    uint32_t width = d;
    if (s < kVectorCores) {
        const uint32_t desired = (kVectorCores + s - 1) / s;
        width = std::max(16u, ((d + desired - 1) / desired + 15) / 16 * 16);
    }
    *tiling = {n, d, s, epsilon, width};
    *block_dim = std::min(s * ((d + width - 1) / width), kVectorCores);
    *tiling_key = 1;
}
