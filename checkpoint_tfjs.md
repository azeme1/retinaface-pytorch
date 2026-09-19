# TF.js checkpoints -- hashes and verified WIDER FACE metrics

Generated 2026-09-19 by `inference/generate_checkpoint_md.py` from the logs of `inference/validate_tfjs.sh` (`inference/export_check.py --format tfjs`).

- **Files:** `checkpoints/<backbone>/tfjs/<backbone>_<level>.zip` in the HF repo `azemel/retinaface-xs` (33 files).
- **Hash:** sha256 HF advertises (LFS) for the file. Every file was also re-downloaded fresh from HF and its sha256 recomputed: all matched (2026-09-19). No local copies are kept.
- **Metrics:** real full-val WIDER FACE AP (3226 images) of the artifact downloaded from HF, run through `@tensorflow/tfjs-node` under Node.js (CPU backend -- no usable GPU build), on the fixed-size 640x640 letterbox preprocessing; "vs PyTorch" is the difference in mean AP from the PyTorch `RetinaStaticExportWrapper` run in the same script. Not comparable to `results/<network>/full_eval_parallel/*.json` (variable-size pipeline).
- **Reference device:** `resnet50` rows ran against a CUDA PyTorch reference; every other backbone's rows ran against a CPU PyTorch reference (an earlier run pinned to CPU) -- PyTorch CPU vs CUDA differs by ~0.02% mean AP, so the "vs PyTorch" column is comparable to that precision.

| backbone | level | size (MB) | sha256 | easy | medium | hard | mean | mean vs PyTorch |
|---|---|---|---|---|---|---|---|---|
| mobilenetv1 | c2 | 2.12 | `f45b8bddbfe70ea69d5d2f96bf4ae45707528631887a2842a1d490f2e58557a8` | 80.81 | 68.64 | 32.95 | 60.80 | +0.02 |
| mobilenetv1 | c7 | 5.29 | `48b7563078c695dc1d7a05dfd9333efb5b039ffd929ff4c3270b76277c7a89be` | 87.84 | 80.25 | 46.24 | 71.44 | -0.01 |
| mobilenetv1 | c64 | 7.77 | `fe1e13b32372e2d8e5a749677d4b885aaa24f7ccde353c81b30126f4c0be1043` | 89.17 | 81.42 | 47.31 | 72.63 | +0.03 |
| mobilenetv1 | c256 | 7.82 | `19c15a25aad2ddfe83f3cc5c7fbee67afdd106bfd1b745872c43ca3874bea6fa` | 89.14 | 81.46 | 46.92 | 72.51 | +0.02 |
| mobilenetv1 | float32 | 7.82 | `423f073616b05c2846fc875f9d60c0d57a1015cca26fdfa2adeb9eda6ece9563` | n/a | n/a | n/a | n/a | n/a |
| mobilenetv1_0.25 | c2 | 0.33 | `934dc5664cd82cbc7d3adf5aebcc809faed44a620858478b149c935c6d35da03` | 60.87 | 42.87 | 18.11 | 40.62 | +0.06 |
| mobilenetv1_0.25 | c12 | 0.72 | `b40fd0ea26ab51ef2b2aadf00d668421dd5baa8902e82ddd1a96e48c0508fc8f` | 86.93 | 77.12 | 40.75 | 68.26 | +0.01 |
| mobilenetv1_0.25 | c128 | 0.94 | `1e12b78247adacdfd8436ab49403e85ec50a1d0a514629b3ab332256a208ea67` | 86.15 | 76.40 | 40.46 | 67.67 | +0.03 |
| mobilenetv1_0.25 | c256 | 0.94 | `10393c1175060419bc187b51f97222644f0d8eae1ee72efaf631cd412ed95765` | 86.23 | 76.36 | 40.56 | 67.71 | -0.01 |
| mobilenetv1_0.25 | float32 | 0.94 | `9ba68b5a4d4d9493af2994283060eb6d25ce17d91f9499becfd0de6873e9ee64` | n/a | n/a | n/a | n/a | n/a |
| mobilenetv1_0.50 | c2 | 0.81 | `6335e395eb4e793bb6e99d658731e48321b649a31925e3c813f6423f1ac9ceef` | 69.40 | 49.76 | 20.82 | 46.66 | -0.01 |
| mobilenetv1_0.50 | c8 | 1.99 | `9d1169f1a2019d7a6de88a14dd11e975db17d8b95deb3b2deacaf730843e8e05` | 86.56 | 77.95 | 42.35 | 68.96 | -0.01 |
| mobilenetv1_0.50 | c256 | 3.21 | `06e46fb64864d7413e3b48f5a777a22a4f6436d41dbb94d7cf9aaaf5d333f5e4` | 87.33 | 79.36 | 44.32 | 70.34 | +0.02 |
| mobilenetv1_0.50 | float32 | 3.21 | `4bd78ac9c623b9a49d77927203361ffee91b05223702552611ae2f2785c672af` | n/a | n/a | n/a | n/a | n/a |
| mobilenetv2 | c2 | 1.59 | `a4c6d3d089bb82bd3a1466a695e9d5ad52f7e3a39ab020b2fd3da398943556e0` | 79.56 | 65.82 | 29.83 | 58.40 | +0.06 |
| mobilenetv2 | c5 | 3.16 | `3a91b18b12a6a7af66fe59ca44d4ab01542452d2c7abd1ce6d0a4b09d6ac9ebe` | 91.16 | 84.25 | 51.88 | 75.76 | -0.03 |
| mobilenetv2 | c64 | 5.86 | `dc8c23e4c600a321b414e410ffcbd945522e10bc6208f27df2897efd4b411911` | 91.73 | 85.80 | 55.08 | 77.53 | +0.01 |
| mobilenetv2 | c256 | 5.92 | `4cd4eeb0c0eaad8c46c8772483c3b4846e0667fdee0c325de9717ca92d464e70` | 91.83 | 86.20 | 55.89 | 77.98 | -0.03 |
| mobilenetv2 | float32 | 5.92 | `b2b9bf0f5941ba77404ed9e47268fceb56e68585f02921db989084756bb6d134` | n/a | n/a | n/a | n/a | n/a |
| resnet18 | c2 | 5.06 | `0593b9bda466bcaefd9b6eef2c0a25beb2206826f2aed79545eee495b1622c29` | 85.90 | 74.43 | 36.23 | 65.52 | +0.04 |
| resnet18 | c5 | 10.92 | `862cdbc94759ec2352c0fecd1d4ca0b5c4f6c01fc26183a4eb4adf3b386dc309` | 91.22 | 84.41 | 51.12 | 75.58 | +0.01 |
| resnet18 | c32 | 21.75 | `c1247ebb56db6c4c850953065e146631576e2c1d57b245c3430af7fdda2354f0` | 91.96 | 85.68 | 52.05 | 76.56 | +0.02 |
| resnet18 | c256 | 22.39 | `ba013b09eaef5d4d69267e384cdf81f10d86e1b6cb17e2acfd91dd1cf645c16e` | 91.80 | 85.85 | 53.06 | 76.90 | -0.01 |
| resnet18 | float32 | 22.39 | `f37cb4cef9462aaf7b280005ec485b4553beb2baa7e64b6ae69eca2c9c3e8027` | n/a | n/a | n/a | n/a | n/a |
| resnet34 | c2 | 9.09 | `4ab45fdb0ac1e0455fd78b5e45b72ed3555cf4e05f1ce918804425d39f58dedc` | 89.09 | 81.21 | 43.95 | 71.42 | +0.02 |
| resnet34 | c4 | 17.11 | `38c6831655f89909899ef75885f987b7755e703e4eaeae8ce2245340209a2f1e` | 92.30 | 86.31 | 53.25 | 77.29 | -0.02 |
| resnet34 | c128 | 41.08 | `3122c862b28e4e9a5a48159a49f4ab50b9c9e5501c216fa6e892ed72ace24003` | 92.58 | 87.05 | 55.12 | 78.25 | -0.04 |
| resnet34 | c256 | 41.11 | `7b7f1ce1a6c430992f4f6da86ed75ebc6766b21be3649e14a970febc65ccbd3d` | 92.67 | 87.09 | 55.14 | 78.30 | +0.01 |
| resnet34 | float32 | 41.11 | `a30808ac628aaa209a4c9b36505336eef0e33a22abccf32db07529988687209f` | n/a | n/a | n/a | n/a | n/a |
| resnet50 | c2 | 13.05 | `17d5bcf4babe69608ac7afede6159cf1224d28671e803fa87e8e4ba9a59e5e2f` | 90.21 | 81.27 | 42.93 | 71.47 | -0.00 |
| resnet50 | c4 | 24.00 | `7b00bd5549b066a07aae8fe01ed13eff3853029fdf72bf1ca8242860162255ee` | 92.62 | 87.59 | 53.94 | 78.05 | -0.00 |
| resnet50 | c128 | 50.58 | `1b10f1fb4ebb8013df323207ac3ebab2c08f46cd9db714bb3e5298b7b4cfb1cd` | 92.82 | 87.09 | 52.49 | 77.47 | -0.01 |
| resnet50 | c256 | 50.62 | `5032e3ab53a202ecc46c705e9bec2a580664dc3968e304601e15fdb0f0385701` | 92.78 | 87.11 | 52.28 | 77.39 | -0.00 |

AP values are percentages; `n/a` = not verified (6 file(s)) -- the validate scripts skip `float32` levels.
