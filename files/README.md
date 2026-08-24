# Bundled refusal direction

`direction_r1.pt` — a unit-norm 4096-dim refusal-direction tensor for
the optional runtime ablation path (`ABLATE=1`, see the main README).

Byte-identical (sha256 `6e4d8a8f3aa9e21795faab2c5b14d29b019acdf2ddbfbd8238430458a5837fe0`)
to `results/refusal_direction_r1.pt` from
[drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks](https://github.com/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks),
captured on the FP8 DSpark sibling of `deepseek-ai/DeepSeek-V4-Flash-0731`.
The first `ABLATE=1` boot copies it into `./data/files/` (runtime path,
container `/models/files/`); a pre-existing file there is never overwritten.

Redistributed under that repository's MIT license:

---

MIT License

Copyright (c) 2026 drowzeys / keys

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

NOTE: Model weights are derived from deepseek-ai/DeepSeek-V4-Flash-DSpark and
remain subject to the original model license terms.
