# Attribution and third-party notices

This project contains modifications and selected solution artifacts derived from the HelloHPC 2026 competition framework:

- Upstream: https://github.com/HPC-SJTU/hellohpc-2nd
- Reference commit: d7e85db2c403528479c47771c0ab65198a07c18e
- Upstream license text: CC BY-NC-SA 4.0, preserved in LICENSE.
- Changes: ARM SIMD and parsing optimizations, scheduling policies, Ascend kernel changes, inference launch configuration, and accompanying experiment documentation. Files or patches are selected from the locally recorded versions; not all candidates have been tested.

Miniclash incorporates Marc Stevens' MD5 collision code. Its original MIT license is retained at cpu/miniclash/source_code/LICENSE.txt and its source notices are preserved. Other notices embedded in source files remain applicable; the root license does not replace independently licensed third-party components.

Blackhole and Mahjx are represented as patches against the named upstream commit rather than entire redistributed frameworks. Obtain the upstream tree and preserve its existing source notices when applying these patches.

The competition framework, model architectures, hardware, vendor tools and evaluation infrastructure are not claimed as original work. Development used Codex assistance. Performance claims in this repository are limited to the evidence described in docs/results.md.
