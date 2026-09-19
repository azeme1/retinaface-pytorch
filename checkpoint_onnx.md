# ONNX checkpoints -- hashes and verified WIDER FACE metrics

Generated 2026-09-19 by `inference/generate_checkpoint_md.py` from the logs of `inference/validate_onnx.sh` (`inference/export_check.py --format onnx`).

- **Files:** `checkpoints/<backbone>/onnx/<backbone>_<level>.zip` in the HF repo `azemel/retinaface-xs` (33 files).
- **Hash:** sha256 HF advertises (LFS) for the file. Every file was also re-downloaded fresh from HF and its sha256 recomputed: all matched (2026-09-19). No local copies are kept.
- **Metrics:** real full-val WIDER FACE AP (3226 images) of the artifact downloaded from HF, run through ONNX Runtime CUDAExecutionProvider vs. a CUDA PyTorch reference, on the fixed-size 640x640 letterbox preprocessing; "vs PyTorch" is the difference in mean AP from the PyTorch `RetinaStaticExportWrapper` run in the same script. Not comparable to `results/<network>/full_eval_parallel/*.json` (variable-size pipeline).

| backbone | level | size (MB) | sha256 | easy | medium | hard | mean | mean vs PyTorch |
|---|---|---|---|---|---|---|---|---|
| mobilenetv1 | c2 | 1.00 | `07795bc075619a6391fdcfd71dc58ddb15d4c4dc6f053e1b538b9d5da0df9a9e` | 80.84 | 68.68 | 32.99 | 60.84 | +0.04 |
| mobilenetv1 | c7 | 2.03 | `355698539f5ac23874143d733612965b141a1b709ceaa6efacaaf706a3dde7e5` | 87.82 | 80.25 | 46.23 | 71.43 | -0.03 |
| mobilenetv1 | c64 | 5.27 | `8fa54412c7c553601c5e03092d8a7713b64369881c2222f627d30b2d635698b6` | 89.14 | 81.40 | 47.30 | 72.61 | +0.00 |
| mobilenetv1 | c256 | 10.44 | `ce23fb0512301195f5f6fcbb4c6baa05bbf1badf7f5c4d326d317a357ea66240` | 89.15 | 81.44 | 46.88 | 72.49 | +0.01 |
| mobilenetv1 | float32 | 15.58 | `7653e8403ebaa3d92d0a39410f451109c5e855777f2f63db5ea32ea222e7ffb5` | n/a | n/a | n/a | n/a | n/a |
| mobilenetv1_0.25 | c2 | 0.31 | `d95b93e5275fe92707bf19e01ecbd32c0561b8faa15e82ad6242f4ff1ec36b93` | 60.85 | 42.86 | 18.11 | 40.61 | +0.07 |
| mobilenetv1_0.25 | c12 | 0.54 | `6cdd3714190a520f107e924c7031a26dfc5825bead3fbe2ed27a419a52fb7de8` | 86.94 | 77.08 | 40.71 | 68.24 | -0.02 |
| mobilenetv1_0.25 | c128 | 1.36 | `529c9b90469f1ab7e6e186a5e89920f80bb3e3c01c8fc4572a03e1caa3afdaa1` | 86.15 | 76.38 | 40.45 | 67.66 | +0.00 |
| mobilenetv1_0.25 | c256 | 1.56 | `7f3a3f1ea0881faa8043b42fc0ad623d6d887b3700dcf7f4a7d8cf4bd481515f` | 86.24 | 76.37 | 40.57 | 67.73 | -0.00 |
| mobilenetv1_0.25 | float32 | 1.77 | `3416c103477e161e3c5335bcf743cde66e58ae80f2c5724c4e34b8adc8c8108b` | n/a | n/a | n/a | n/a | n/a |
| mobilenetv1_0.50 | c2 | 0.54 | `ef66fb396635225376c3507e9316c7d7dacdf749eacf50cba5c8053921d31e4c` | 69.41 | 49.76 | 20.82 | 46.66 | +0.01 |
| mobilenetv1_0.50 | c8 | 1.05 | `d2f2f9bc79cc738d3e613f9e7ca7836a92ec7a365584bdd825971d2ac064e0e1` | 86.59 | 77.97 | 42.38 | 68.98 | +0.01 |
| mobilenetv1_0.50 | c256 | 4.75 | `0e0c874fa671a5cfed3299198496ae51aa2b6c9c12bbd7605da560190605e114` | 87.33 | 79.33 | 44.29 | 70.32 | -0.00 |
| mobilenetv1_0.50 | float32 | 6.32 | `832372391dccddabf2798694d322957c548fc14ff95ac12a981d7e5e5748d206` | n/a | n/a | n/a | n/a | n/a |
| mobilenetv2 | c2 | 0.93 | `95a2530b1fceca7c0080700c8a5506f62c285b927cf01a669dd911600cd07424` | 79.53 | 65.78 | 29.81 | 58.37 | +0.00 |
| mobilenetv2 | c5 | 1.65 | `60064edffd68a8310959b21dd3380b436cf1bc92b1b4d5a2bbc3dd6d1602a735` | 91.18 | 84.27 | 51.86 | 75.77 | -0.01 |
| mobilenetv2 | c64 | 5.24 | `52bd7d98cbf0578d27c5209324632107bb06d8ac18fc4c24261a703d9153c28a` | 91.73 | 85.79 | 55.05 | 77.52 | -0.01 |
| mobilenetv2 | c256 | 8.51 | `3a05f5fb9ba7eb0fc65b6a59ca447eee1a0cec2c63ca518189ff7a320de87d3a` | 91.88 | 86.23 | 55.88 | 78.00 | -0.02 |
| mobilenetv2 | float32 | 11.75 | `9c24808bc360710df947bac609db54968b901ffa1f0999733042ba4dc4323982` | n/a | n/a | n/a | n/a | n/a |
| resnet18 | c2 | 1.90 | `9b6ea48344453ff32ac45711bc67c06e116c2e9a1d48796c4915340bff50139b` | 85.88 | 74.41 | 36.22 | 65.50 | +0.02 |
| resnet18 | c5 | 3.53 | `39e3d2a2a7b60f5965c54bb4538682f9f46b656a72885d3dd9c89f4e43c83144` | 91.22 | 84.41 | 51.11 | 75.58 | +0.00 |
| resnet18 | c32 | 8.71 | `1a01422385b4f4ddd71be97fe6e6c9cc5a98f609b57725434bc3321ddbabac3a` | 91.94 | 85.64 | 52.05 | 76.54 | -0.01 |
| resnet18 | c256 | 17.36 | `002fa8502905ab95bfd5ae15fb93a4c1dfcc8039bda2fb3f25b1250de75e434d` | 91.83 | 85.86 | 53.07 | 76.92 | +0.01 |
| resnet18 | float32 | 44.80 | `4ddcc8542ae0381483e07bdc60a27c2c0d6493b8294edadb9ea5b632f9d76107` | n/a | n/a | n/a | n/a | n/a |
| resnet34 | c2 | 3.24 | `fa882c33b582f9297295890bce64d0dab392634d53debafbf4286ea0821d3696` | 89.06 | 81.19 | 43.94 | 71.40 | -0.04 |
| resnet34 | c4 | 5.42 | `9fdab1a90576f506bc3365b514aa6251b932f7911dc49a18df2a08f461d917f1` | 92.31 | 86.30 | 53.26 | 77.29 | +0.01 |
| resnet34 | c128 | 24.17 | `e48250d31d6eec17a0b2ec0c90b2210afebd5806fe880909cff103dc6e178e42` | 92.63 | 87.09 | 55.15 | 78.29 | -0.00 |
| resnet34 | c256 | 31.02 | `27c033302b87cc8a7b268dbe056967fab31765c354fb3b15b793a14026a14449` | 92.65 | 87.07 | 55.12 | 78.28 | -0.01 |
| resnet34 | float32 | 82.38 | `50f46ed931ab672b1101b68717526ec383c03eaa8e730da25ca4e34106d70cf1` | n/a | n/a | n/a | n/a | n/a |
| resnet50 | c2 | 4.50 | `47c51a3e1e64cc3db9d7f986e64ed697bd5731fe20d9c53dd892ca2754b4545b` | 90.19 | 81.25 | 42.93 | 71.46 | -0.01 |
| resnet50 | c4 | 7.74 | `23691b05a3128740266175132d1544861c8f308dd184be6e120861f748b17db9` | 92.62 | 87.58 | 53.93 | 78.05 | -0.01 |
| resnet50 | c128 | 37.51 | `b03c1236ea04f5caa93d51f855236d036f251aef7c8196d679108d529123af01` | 92.82 | 87.10 | 52.49 | 77.47 | -0.01 |
| resnet50 | c256 | 50.64 | `de31d179cb992a60344950049261847e0fca0843b4c23699197c7f64d580355b` | 92.79 | 87.12 | 52.26 | 77.39 | -0.00 |

AP values are percentages; `n/a` = not verified (6 file(s)) -- the validate scripts skip `float32` levels.
