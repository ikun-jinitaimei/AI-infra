---
name: cpu-hpc-skill
description: CPU kernel optimization guide.
---

# Masked column popcount

N is a multiple of 64, 64..8192; buffers never overlap. Size storage for actual N/64 words (up to 128), not public examples.

1. Enumerate selected rows from mask words with ctz on nonzero words; never touch unselected rows.
2. Bit-slice counters: `plane[k][w]` = bit k of counts for the 64 columns of word w. Adding a row is a ripple-carry increment, 64 columns at once: `c=row[w]; for k: t=plane[k][w]; plane[k][w]=t^c; c&=t; if(!c) break;`
3. Fold 16 rows with carry-save adders first: the full adder `u=a^b; sum=u^c; carry=(a&b)|(u&c)` makes a sum and a double-weight carry from three equal-weight words. Ladder them so 16 rows become one weight-16 word, and ripple only that into `plane[4]`. Carry the ones/twos/fours/eights residue between blocks, flush it at weights 1,2,4,8 at the end, and add the sub-16 tail directly.
4. Size planes by selected-row count R: `K=ceil(log2(R+1))`; R=8192 needs 14 planes. R=0: zero every output and return. Bound ripple/extraction by K; never access plane[4] when K<5. Zero counters per call; overwrite all outputs.

Re-time with `tools/test_candidate.sh` after each edit.
