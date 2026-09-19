# TFLite checkpoints -- hashes and verified WIDER FACE metrics

Generated 2026-09-19 by `inference/generate_checkpoint_md.py` from the logs of `inference/validate_tflite.sh` (`inference/export_check.py --format tflite`).

- **Files:** `checkpoints/<backbone>/tflite/<backbone>_<level>.zip` in the HF repo `azemel/retinaface-xs` (35 files).
- **Hash:** sha256 HF advertises (LFS) for the file. Every file was also re-downloaded fresh from HF and its sha256 recomputed: all matched (2026-09-19). No local copies are kept.
- **Metrics:** real full-val WIDER FACE AP (3226 images) of the artifact downloaded from HF, run through `tf.lite.Interpreter` (CPU -- no GPU delegate exists for it here), on the fixed-size 640x640 letterbox preprocessing; "Δ" is the difference in AP (percentage points) from the PyTorch `RetinaStaticExportWrapper` run in the same script, per subset -- easy, medium and hard are three separate metrics (they score 7,211 / 13,319 / 31,958 faces) and are never averaged. Not comparable to `results/<network>/full_eval_parallel/*.json` (variable-size pipeline).
- **Reference device:** `resnet50` rows ran against a CUDA PyTorch reference; every other backbone's rows ran against a CPU PyTorch reference (an earlier run pinned to CPU) -- PyTorch CPU vs CUDA differs by up to 0.05 / 0.05 / 0.07 points (easy / medium / hard, measured over the 30 overlapping runs), so the Δ columns are only meaningful to about that precision.

| backbone | level | size (MB) | sha256 | easy | medium | hard | Δ easy | Δ medium | Δ hard |
|---|---|---|---|---|---|---|---|---|---|
| mobilenetv1 | c2 | 1.16 | `b606927ccb2e0e5bdac2e66897e6bd223719e131f2e422bf010bff9b37bec2e0` | 80.74 | 68.42 | 32.70 | -0.08 | -0.19 | -0.23 |
| mobilenetv1 | c7 | 2.54 | `f19c3c98cdfbdc4aa817c33a40d681517c2ada33b1486f931583706bf01b0c95` | 88.02 | 80.37 | 46.07 | +0.17 | +0.10 | -0.17 |
| mobilenetv1 | c64 | 4.08 | `b8aa48a5e12f377ddee1436a51fea2ec534ebd0bd3e920030913ccf3b04d73ff` | 89.15 | 81.34 | 47.17 | +0.02 | -0.05 | -0.13 |
| mobilenetv1 | c256 | 4.08 | `04178feecae442db3954d5e295a0f653aad4aa8870154818d325c8bc64819df4` | 89.22 | 81.56 | 46.95 | +0.08 | +0.12 | +0.06 |
| mobilenetv1 | float32 | 4.08 | `bb8e73de48284c11c6cb5da2996cab31c407162f7307a48a88410d743ed7a078` | 88.90 | 81.16 | 47.08 | +0.12 | +0.04 | -0.05 |
| mobilenetv1_0.25 | c2 | 0.31 | `eaa12858c647b77dd43d87c8ce08faa27c1fd6259234a2c469c9b779e94b0f78` | 60.01 | 42.18 | 17.83 | -0.77 | -0.63 | -0.26 |
| mobilenetv1_0.25 | c12 | 0.56 | `3a4e74babb9415ede38c217c0bd8b4ca128bb9917661b5e5ee309b287d9feb69` | 87.03 | 77.31 | 41.06 | +0.09 | +0.20 | +0.34 |
| mobilenetv1_0.25 | c128 | 0.60 | `97576d9b6951c22f68f3b7ae0df2148c598bcdd3f44983c7fae159b7a57a8db5` | 86.05 | 76.31 | 40.38 | -0.08 | -0.06 | -0.06 |
| mobilenetv1_0.25 | c256 | 0.60 | `17c97b442943dd290efdfb3166ac9c6aaab51f9171119c8c551cd813f19a9adb` | 86.20 | 76.55 | 40.89 | -0.03 | +0.18 | +0.32 |
| mobilenetv1_0.25 | float32 | 0.60 | `3f4ff05bc53d5692f0fa6b6c1ca314192322f0e3d2140e7e852f3cb8db4203e5` | 86.47 | 77.02 | 41.11 | -0.09 | +0.07 | +0.17 |
| mobilenetv1_0.50 | c2 | 0.60 | `8dc265e4efe9a3b4b6a3a10a50939a97b3e54b8867e75279d1812f4576e7a2e5` | 69.20 | 49.54 | 20.73 | -0.23 | -0.22 | -0.09 |
| mobilenetv1_0.50 | c8 | 1.26 | `f0a7a85abb31e2b96eaec05551c9170e229a9b39c3b30cdd280b237710f197aa` | 86.77 | 77.82 | 41.91 | +0.19 | -0.14 | -0.45 |
| mobilenetv1_0.50 | c12 | 1.51 | `b39dc6b46ba8210fde4139135b34fad2bc3049f839da6ca4d5a0de631ce93b1a` | 86.70 | 78.42 | 43.56 | +0.16 | +0.21 | +0.46 |
| mobilenetv1_0.50 | c256 | 1.75 | `81c65f30559d04c7e856045be560593f238f22a609fd04b94050dbc46fe85aed` | 87.25 | 79.28 | 44.33 | -0.08 | -0.04 | +0.04 |
| mobilenetv1_0.50 | float32 | 1.75 | `b781c49e7f360c8b648af5d4478c313dbbe03cd151c3ddb67e692a78900a5999` | 87.16 | 78.80 | 43.65 | -0.09 | -0.04 | -0.04 |
| mobilenetv2 | c2 | 1.04 | `4f25d5088533037406b4c9ef988b921fda05b17d23c43dec3f2be548b3e89650` | 79.62 | 66.00 | 29.97 | +0.09 | +0.24 | +0.21 |
| mobilenetv2 | c5 | 1.84 | `14cfc3547033c4a5ad7b40fe7c114e4b55050e67c46ddb6c758b3dc0598c1eb5` | 91.08 | 84.35 | 51.56 | -0.12 | +0.05 | -0.33 |
| mobilenetv2 | c64 | 3.18 | `0069a33766496a32804a339c7afda894bf894e9e1d6edd1074cb631890e73909` | 91.57 | 85.71 | 55.22 | -0.17 | -0.09 | +0.18 |
| mobilenetv2 | c256 | 3.19 | `c912a29dace400b53a6e033a18ebb4119c81d8c9d6c7839583d2d8d49169ae1a` | 91.90 | 86.15 | 55.89 | +0.02 | -0.09 | +0.00 |
| mobilenetv2 | float32 | 3.19 | `b3b2b7be6bb812e324c60e4236578b449b90deef4b422c4e6488aeccca1128de` | 91.90 | 86.30 | 55.30 | -0.08 | -0.12 | -0.09 |
| resnet18 | c2 | 2.32 | `6746afeafc611f81b98786bd1110f4d8465b5d90e34ca9520705ddd9d4d534cb` | 84.18 | 72.74 | 35.38 | -1.64 | -1.67 | -0.84 |
| resnet18 | c5 | 4.69 | `e0bec4335d82ec48135d959a6e2d73aec7985537a1eab098682b388cd7ad77f2` | 90.63 | 83.66 | 49.68 | -0.59 | -0.74 | -1.42 |
| resnet18 | c32 | 9.85 | `393a192ef399a204de58153050682bebfc6e6167d012fb448f4168a3461c4d65` | 91.76 | 85.64 | 52.22 | -0.18 | +0.00 | +0.17 |
| resnet18 | c256 | 10.45 | `15e74ecb43d747ae7a5869738ac5a2e67849b05ca48de764a0b2cc180c679a3c` | 91.79 | 85.88 | 53.25 | -0.04 | +0.03 | +0.19 |
| resnet18 | float32 | 10.38 | `09443e780f015a21732497f6d68ff84a34623df9342800c3372928a87e640989` | 91.75 | 85.60 | 52.54 | +0.01 | +0.07 | +0.10 |
| resnet34 | c2 | 4.07 | `0dc4bb063d2a162e73e173a9bb982e2265f6e128abfdffbd45c88931e99da675` | 88.47 | 80.66 | 43.36 | -0.58 | -0.53 | -0.59 |
| resnet34 | c4 | 7.37 | `285b0b318de4e80c34b50c8d1316708775fa4811d371e304bade17eac320d638` | 92.46 | 86.57 | 53.36 | +0.10 | +0.25 | +0.11 |
| resnet34 | c128 | 19.12 | `2020f74ff9e1aa8ac8a5e25e560ed62dc16647c32b3d348a06c6357285e79537` | 92.63 | 87.00 | 54.80 | -0.00 | -0.09 | -0.34 |
| resnet34 | c256 | 19.15 | `dc09171c7bba1821513a043e5204dcd05f04ff1c710e23d0256877d01dd9b866` | 92.49 | 86.93 | 55.08 | -0.16 | -0.15 | -0.05 |
| resnet34 | float32 | 19.01 | `16839356888f4f177f54e1c3f54172c023a56f7774fab92840aac0874d4222af` | 92.65 | 87.28 | 55.17 | -0.03 | +0.01 | +0.01 |
| resnet50 | c2 | 5.51 | `d7c50283e69434dfe200a9e568458d706e382be1bdc39889fa35ab97d3341998` | 89.95 | 80.92 | 42.26 | -0.24 | -0.36 | -0.69 |
| resnet50 | c4 | 9.91 | `62588f2c4bbe9d5e9a1a72fd963f3c44c90ef3eae323c10e68ac1ed3e9bb2fbe` | 92.40 | 87.38 | 53.96 | -0.22 | -0.21 | +0.01 |
| resnet50 | c128 | 24.17 | `f9ceeb5269f7b6d51b955c7a1a32f2a436165bf1546a66c77a8ce32b8643cbd4` | 92.78 | 86.96 | 52.31 | -0.06 | -0.14 | -0.19 |
| resnet50 | c256 | 24.17 | `e87c986c585e11ea17a92656b8c6efb683ed39fbecfbf4839a6a78f61d896808` | 92.78 | 87.13 | 52.23 | -0.01 | +0.01 | -0.03 |
| resnet50 | float32 | 23.42 | `99ba5e2eb838963a7795053a2cd1763d96c5d5fe1ca94f28ee098396ce158f32` | 93.37 | 88.78 | 58.00 | -0.14 | -0.09 | -0.04 |

AP values are percentages.
