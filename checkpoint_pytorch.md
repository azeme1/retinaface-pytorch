# resnet50 PyTorch checkpoints -- hashes and verified WIDER FACE metrics

Generated 2026-09-19 by `inference/validate_pytorch.sh resnet50 ...` (`inference/export_check.py --format pytorch`).

- **Files:** `checkpoints/resnet50/pytorch/resnet50_<level>.zip` in `retinaface-xs/` (local) and in the HF repo `azemel/retinaface-xs`.
- **Hash:** sha256 of the local zip, compared to the sha256 HF advertises for the same file (LFS). All 14 match.
- **Metrics:** real full-val WIDER FACE AP (3226 images) of the checkpoint downloaded from HF, run through the fixed-size `RetinaStaticExportWrapper` (640x640 letterbox) on CUDA. This is the same reference AP the ONNX/TFLite/TFJS checks compare against. It is *not* comparable to `results/resnet50/full_eval_parallel/*.json` (that uses the variable-size pipeline).

| level | size (MB) | sha256 | HF match | easy | medium | hard | mean |
|---|---|---|---|---|---|---|---|
| c2 | 14.7 | `7d3ddfc537f8999ffa2dd55954d62b54692d014d81d3868bf3f242ea743bb013` | yes | 90.20 | 81.28 | 42.95 | 71.47 |
| c3 | 20.0 | `e76eb2c0689658029136b19a2b7bbcb4ac7c0df0e7d2d17c31301ec77885ce54` | yes | 92.46 | 86.48 | 52.64 | 77.19 |
| c4 | 24.7 | `d30efad84ab432007b4cfae00a8b38f45408f4ad5bd6e40e72fe333ac515c9d3` | yes | 92.62 | 87.59 | 53.95 | 78.05 |
| c5 | 27.0 | `8f1060a4f2fa127f5791470a3d958024f447bde1565fdf3fc5868593c64b6c33` | yes | 93.30 | 88.02 | 55.17 | 78.83 |
| c6 | 29.6 | `42758d8a80041187f96b7bfed4f8d458c2eaf1aacc959ca525a3eaf5dad97d9c` | yes | 93.18 | 88.11 | 55.79 | 79.03 |
| c7 | 32.1 | `bfa7fa7308da55c4360e709a2ac518b8e7f01d1b5e6ea5589e916a0efafe9a43` | yes | 93.26 | 88.50 | 56.30 | 79.35 |
| c8 | 34.5 | `21aca85ed954ca0fd1c986231c786ffe25d8f5bd3be5d42935eba68aaa1d21f0` | yes | 93.45 | 88.64 | 56.63 | 79.57 |
| c12 | 42.1 | `419abda4449d8810a745f895cead70654bfd48d129025bdd6eef50693dce3396` | yes | 93.40 | 88.65 | 56.55 | 79.53 |
| c16 | 47.7 | `29e85bb92f9061c0c017c1b10d30d0abca6aa5395a30d4a05788409f1dbf7e5a` | yes | 93.76 | 89.01 | 57.50 | 80.09 |
| c32 | 61.8 | `15be507db17b2430448129ce255cc3d6072ffb9086ba449eaeb5deaf1ac87ade` | yes | 93.54 | 88.88 | 57.67 | 80.03 |
| c64 | 80.0 | `b8ffe62afc6952785b6edda94c394017428931638540776997a03fd4a529105a` | yes | 93.60 | 89.05 | 57.91 | 80.19 |
| c128 | 102.7 | `17157a5208f9a88a7c2433f654659e5bfb52f41d158c5ebbd5488cfb7dbc0315` | yes | 92.84 | 87.10 | 52.50 | 77.48 |
| c256 | 132.0 | `19ffdbbffa27742dd5b180e60ab9b69b42e65c3459f221e43faefed750f87be1` | yes | 92.79 | 87.12 | 52.26 | 77.39 |
| float32 | 101.7 | `01273cde7b1fa72d280e114ea6fe864f885a07c25a2654e76fc6518935970cf2` | yes | 93.51 | 88.86 | 58.03 | 80.14 |

AP values are percentages.

## Provenance notes

Local-file comparison (batchnorm/bias tensors vs. local training checkpoints, done 2026-09-19): `c2`, `c4`, `c256` match `results/resnet50/` (last run); `c3`, `c5`, `c6`, `c7`, `c8`, `c12`, `c16`, `c32`, `c64` match `results/resnet50_fromscratch_baseline_experiment/` (2026-09-06); `c128` matches no local checkpoint. The README's statement that levels 2-6 all completed the current QAT schedule is unverified for `c3`, `c5`, `c6`.
