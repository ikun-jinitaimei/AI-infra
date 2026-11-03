#include "rsm_solution_tiling.h"
#include "rsm_solution_core.h"

extern "C" __global__ __aicore__ void rsm_solution(
    GM_ADDR score, GM_ADDR x, GM_ADDR offsets, GM_ADDR mean, GM_ADDR rstd,
    GM_ADDR logsumexp, GM_ADDR workspace, RsmSolutionTiling tiling)
{
    (void)workspace;
    RsmSolution::ComputeCore core;
    core.Init(score, x, offsets, mean, rstd, logsumexp,
        tiling.n, tiling.d, tiling.epsilon);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const uint32_t tiles = (tiling.d + tiling.column_tile - 1) / tiling.column_tile;
    if (tiles == 1) {
        // Partition by input work plus per-segment overhead, not segment count.
        // Adjacent cores use identical boundaries; every segment is owned once.
        AscendC::GlobalTensor<int32_t> boundaries;
        boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
        uint32_t cuts[2];
        const uint64_t total = static_cast<uint64_t>(tiling.n) + 16ull * tiling.s;
        for (uint32_t edge = 0; edge < 2; ++edge) {
            const uint32_t rank = block + edge;
            if (rank == 0) { cuts[edge] = 0; continue; }
            if (rank == blocks) { cuts[edge] = tiling.s; continue; }
            const uint64_t target = total * rank / blocks;
            uint32_t lo = 0, hi = tiling.s;
            while (lo < hi) {
                const uint32_t mid = lo + (hi - lo) / 2;
                const uint64_t work = static_cast<uint32_t>(boundaries.GetValue(mid))
                    + 16ull * mid;
                if (work < target) lo = mid + 1;
                else hi = mid;
            }
            cuts[edge] = lo;
        }
        const uint32_t first = cuts[0];
        const uint32_t last = cuts[1];
        core.ProcessRange(first, last);
        return;
    }
    for (uint32_t task = block; task < tiling.s * tiles; task += blocks) {
        core.ProcessSegment(task / tiles, (task % tiles) * tiling.column_tile,
            tiling.column_tile);
    }
}
