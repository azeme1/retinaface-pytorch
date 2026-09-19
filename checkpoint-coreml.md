# CoreML verification (all published CoreML files, GPU / MPS)

Verification of the CoreML files published in [`azemel/retinaface-xs`](https://huggingface.co/azemel/retinaface-xs), each bound to the SHA-256 of the exact file that was tested.

**Status: 27 of 27 files verified; every SHA-256 identical to the checksum HF reports: yes; largest |mean-AP difference| PyTorch vs CoreML: 0.00 points.**

| | |
|---|---|
| started / last update | 2026-09-19 16:35 CEST / 2026-09-19 21:31 CEST |
| HF repo at start | `a7639a47ae`; now `aa9c10797e` (2026-09-19 16:19 UTC) - **the repo changed during the run** |
| machine | Apple M1 Pro, macOS 26.6.2 |
| software | coremltools 9.0, torch 2.12.0, Python 3.12.13 |
| code | `inference/export_check.py` at `c68361ce09` (working tree has uncommitted changes) |
| dataset | WIDER FACE val, all 3226 images, fixed 640x640 letterboxed canvas |
| compute units | `CPU_AND_GPU` (Apple GPU, "MPS") for CoreML; PyTorch reference on MPS |

## Method

1. Each file is downloaded fresh from HF (no cache) and hashed with SHA-256.
2. The hash is compared with the checksum HF itself reports for the file (LFS `sha256` from the repo API).
3. `inference/export_check.py --format coreml` runs the full val set on exactly those downloaded bytes, comparing CoreML against the PyTorch checkpoint it was exported from (also downloaded from HF and hashed).
4. After the run the file is hashed again and HF is queried again: it must be unchanged.

## Results (Easy / Medium / Hard / mean AP, PyTorch / CoreML)

| backbone | level | Easy | Medium | Hard | mean | diff | CoreML file SHA-256 (as published on HF) |
|---|---|---|---|---|---|---|---|
| mobilenetv1 | 2 | 80.80 / 80.80 | 68.59 / 68.59 | 32.91 / 32.91 | 60.77 / 60.77 | -0.00% | `4f4fbfd8bd55e89fc28f54eb89f1df75d13292798e57433ae2ecfdf5dcb5bae1` |
| mobilenetv1 | 7 | 87.86 / 87.86 | 80.28 / 80.28 | 46.26 / 46.26 | 71.47 / 71.47 | -0.00% | `bb61f46a05755e21a88558bf26c0b980a255f313b748185783665151f93c4442` |
| mobilenetv1 | 64 | 89.16 / 89.16 | 81.41 / 81.41 | 47.31 / 47.31 | 72.63 / 72.63 | +0.00% | `a661ec425b3bc999d19243e15aeba6efbefbc760d8d5cd2c5d7f1fd1ed99097c` |
| mobilenetv1 | 256 | 89.16 / 89.16 | 81.46 / 81.46 | 46.91 / 46.91 | 72.51 / 72.51 | -0.00% | `bfe9dbe853e4f90425d144c3f76c6aed3ada72f1e447a1bfb0857f9b617df639` |
| mobilenetv1_0.25 | 2 | 60.81 / 60.81 | 42.80 / 42.80 | 18.09 / 18.09 | 40.57 / 40.57 | -0.00% | `0926c0d250a3dd5558d70771fa625bfad772e894072398ff3990010bba7cf60c` |
| mobilenetv1_0.25 | 12 | 86.92 / 86.92 | 77.11 / 77.11 | 40.75 / 40.75 | 68.26 / 68.26 | -0.00% | `1ea5adc2d6d185f16ba964fd130c5eb2022682624c4fbc7b191825e01e5b867d` |
| mobilenetv1_0.25 | 128 | 86.15 / 86.15 | 76.37 / 76.37 | 40.43 / 40.43 | 67.65 / 67.65 | -0.00% | `876fea32c7cb3453ff030538335cfff6ab0a9961940b1999a6676d222f675f63` |
| mobilenetv1_0.25 | 256 | 86.24 / 86.24 | 76.38 / 76.38 | 40.57 / 40.57 | 67.73 / 67.73 | -0.00% | `809b9d88a9c3f9ea79a428091c9f7d449be0146cec56e68825cc7b998f55ee28` |
| mobilenetv1_0.50 | 2 | 69.43 / 69.43 | 49.76 / 49.76 | 20.82 / 20.82 | 46.67 / 46.67 | -0.00% | `12183db2f61d7b65427eee8e31e69374da324946b436d3cc23131bbb2949dc5d` |
| mobilenetv1_0.50 | 8 | 86.59 / 86.59 | 77.98 / 77.98 | 42.36 / 42.36 | 68.97 / 68.97 | +0.00% | `67d686a5daff03947e8b23a823105bfba1ca03a2a7aa9fe9c0dc86bd62e4755b` |
| mobilenetv1_0.50 | 256 | 87.35 / 87.35 | 79.35 / 79.35 | 44.32 / 44.32 | 70.34 / 70.34 | +0.00% | `34d1008deee26cca31f32db4f07eb2ee26eafc3a1226007fb0c51ada5478ca92` |
| mobilenetv2 | 2 | 79.53 / 79.53 | 65.77 / 65.77 | 29.76 / 29.76 | 58.35 / 58.35 | +0.00% | `09d9b747670ab6c6bedf4125d4348b848c563b77b58f9a337705033dfd0f885d` |
| mobilenetv2 | 5 | 91.20 / 91.20 | 84.31 / 84.31 | 51.91 / 51.91 | 75.81 / 75.81 | -0.00% | `2c3b7b8db3126d18194201c0731163770213e361c1c198a16e487e3575e9ddd3` |
| mobilenetv2 | 64 | 91.73 / 91.73 | 85.80 / 85.80 | 55.07 / 55.07 | 77.53 / 77.53 | -0.00% | `84c1519a90226ef01228d55996422575e8f8046ffa0befa74b4c8519f03cbd23` |
| mobilenetv2 | 256 | 91.83 / 91.83 | 86.20 / 86.20 | 55.88 / 55.88 | 77.97 / 77.97 | -0.00% | `ee45e124c1de59344d5a8050ba0aca229dfae64e62fd54bbf5339d532df77a8d` |
| resnet18 | 2 | 85.83 / 85.83 | 74.43 / 74.43 | 36.24 / 36.24 | 65.50 / 65.50 | +0.00% | `9399854179d3e2cd902125ae8281496c3bd48375e184094bc59d72a2fb97da1c` |
| resnet18 | 5 | 91.20 / 91.20 | 84.39 / 84.39 | 51.10 / 51.10 | 75.57 / 75.57 | -0.00% | `6b101cfb5c31a837460d16b96a08f22246ed01a61e89a5cb9e4f98fb377db129` |
| resnet18 | 32 | 91.96 / 91.96 | 85.68 / 85.68 | 52.06 / 52.06 | 76.57 / 76.57 | -0.00% | `f537b7072e9445687aa0d48eb1591e869ea860066ee70a3af68dfbf42b5038dd` |
| resnet18 | 256 | 91.80 / 91.80 | 85.85 / 85.85 | 53.05 / 53.05 | 76.90 / 76.90 | -0.00% | `8c8dfd32125b88b53e7bfbb3f461582b98cb60f1b6d29eb4921b5c6b6637d6db` |
| resnet34 | 2 | 89.10 / 89.10 | 81.19 / 81.19 | 43.95 / 43.95 | 71.41 / 71.41 | +0.00% | `7627959075f705fdc0103780c5dce66532c8822d9c1dc670f93eb89be8b1d771` |
| resnet34 | 4 | 92.34 / 92.34 | 86.33 / 86.33 | 53.26 / 53.26 | 77.31 / 77.31 | +0.00% | `bc7470bec7f9300e3e6adacf5c5cf6a4307015cf93ae19d18709216a0028ae87` |
| resnet34 | 128 | 92.59 / 92.59 | 87.06 / 87.06 | 55.10 / 55.10 | 78.25 / 78.25 | -0.00% | `5067139a0ee9996c1c93c92e4ea73d141c9d9820fb6858cf8f2d0a585d47019f` |
| resnet34 | 256 | 92.67 / 92.67 | 87.11 / 87.11 | 55.14 / 55.14 | 78.30 / 78.30 | -0.00% | `b033441a00e465e4f39c5f9e89c4870a6b5d5756ab21d20e9b6b8cafddc90bbd` |
| resnet50 | 2 | 90.22 / 90.22 | 81.30 / 81.30 | 42.97 / 42.97 | 71.50 / 71.50 | +0.00% | `c164c8257fc8d4010540c3664a49e357797eb170794e8b7d7b38ffa32d4943f3` |
| resnet50 | 4 | 92.61 / 92.61 | 87.59 / 87.59 | 53.92 / 53.92 | 78.04 / 78.04 | +0.00% | `85a398ec8366c8a05aac31f818384b3a7384e25849b7d63c91ae66f8d4da8716` |
| resnet50 | 128 | 92.82 / 92.82 | 87.10 / 87.10 | 52.48 / 52.48 | 77.47 / 77.47 | +0.00% | `29801f0d2c2ba6b13a3eb3336a78a0fbf34818fd750ff9816d124672ca0e9afe` |
| resnet50 | 256 | 92.78 / 92.78 | 87.11 / 87.11 | 52.27 / 52.27 | 77.38 / 77.38 | +0.00% | `8f40a8e2ce4dabac9d15cf3fe364a3fcb540fef94ac495118e66e4b8b9cb25ac` |

## Checksums

| backbone | level | file on HF | size (bytes) | SHA-256 == HF-reported | unchanged after run | `model.mlpackage` content fingerprint | PyTorch source SHA-256 (== HF) |
|---|---|---|---|---|---|---|---|
| mobilenetv1 | 2 | `checkpoints/mobilenetv1/coreml/mobilenetv1_c2.zip` | 725129 | yes | yes | `ba28e789e396a098bcc2c5ddad12e64b41c2615f3f6229502896f52113a437c4` | `a0b49961e5ffd2c74f1c8bedc266161b08a7c4196f90be402690ec73378a31ea` (yes) |
| mobilenetv1 | 7 | `checkpoints/mobilenetv1/coreml/mobilenetv1_c7.zip` | 1943562 | yes | yes | `f40070de6ee5d0d95a59242bb51c5facd2de6144ca5189c113065b8cd8c22537` | `d379971bca6c21f6496de8ef255c705dcd0a50ffee0f3d4a5f96ef684f5b91f3` (yes) |
| mobilenetv1 | 64 | `checkpoints/mobilenetv1/coreml/mobilenetv1_c64.zip` | 5081354 | yes | yes | `fa8dd1be43560482422e67b2b363466329d5666406a7e2a45bc76b7a1018430f` | `5ff43dc3d7107fcac156f3ec5d6c5d9364c71b2a13759f8d33efd5d2a10b749b` (yes) |
| mobilenetv1 | 256 | `checkpoints/mobilenetv1/coreml/mobilenetv1_c256.zip` | 10493125 | yes | yes | `bfb896953b20cbbabf4f22705a3126ea78d9173b27bf41795c51089345522bdf` | `6401d65ab21b698c2533d5fc98efde295bf87c4dcd63555d76345d24b4f46cdc` (yes) |
| mobilenetv1_0.25 | 2 | `checkpoints/mobilenetv1_0.25/coreml/mobilenetv1_0.25_c2.zip` | 160658 | yes | yes | `727780cd22be5ee9a8c63a65219f7756204729fb0caa1c52b3501ad29e455f9a` | `34563577fead1da4523351c438ef29df1160e0dabe79b7e1ff2266eccc92a836` (yes) |
| mobilenetv1_0.25 | 12 | `checkpoints/mobilenetv1_0.25/coreml/mobilenetv1_0.25_c12.zip` | 450289 | yes | yes | `94131636465c77900cd1beb1e3acc0d200250c4d9bbe327d5112a0cc871ec26c` | `0d2904febc4fc5c84030bdde2dbd559907215aec131a1b3e70b1b1e9e3505762` (yes) |
| mobilenetv1_0.25 | 128 | `checkpoints/mobilenetv1_0.25/coreml/mobilenetv1_0.25_c128.zip` | 1289180 | yes | yes | `3e5a72ad78a87ba40e91eca641f0d4bef5fad5351f1bd27f73c0d7c7ed1d993c` | `6917c7efa02af790bf627d11e66c1d1d5bd5d61d0c2ce0a4459fa830a1d67f69` (yes) |
| mobilenetv1_0.25 | 256 | `checkpoints/mobilenetv1_0.25/coreml/mobilenetv1_0.25_c256.zip` | 1438343 | yes | yes | `ff71fab0a1aa974dd020b40913f8a846c054e977cb76c187ca81e931f88c7101` | `1c6a9b2b7ca3669d1c3dd7bcaaedcfa543b08102ba207b7035890b7eaa7f84cd` (yes) |
| mobilenetv1_0.50 | 2 | `checkpoints/mobilenetv1_0.50/coreml/mobilenetv1_0.50_c2.zip` | 351726 | yes | yes | `1d52a42d8d21bc0f12cd4fd80cc4158e407557c6e16b5f460a76682a157a1232` | `6a5c8c0bd9ece3ec8a0805ac2c4b8e490805b2d3ce3fb15d0d188a945502ad7b` (yes) |
| mobilenetv1_0.50 | 8 | `checkpoints/mobilenetv1_0.50/coreml/mobilenetv1_0.50_c8.zip` | 907266 | yes | yes | `347162aa1c9cab83e8f03583e63e366d9929d83db6500512f19199f92eebffcf` | `996922fcb644ed293e8435764739c2474a72784750c553d8a230076fdf7f52df` (yes) |
| mobilenetv1_0.50 | 256 | `checkpoints/mobilenetv1_0.50/coreml/mobilenetv1_0.50_c256.zip` | 4811687 | yes | yes | `7b4e1a1ba3dbcf2fc393c0382081b54c4bb89c54ec8b5e0e13120ca495b1f0bd` | `613e5ac94a0e31581f3b9e29007bdfe1c43f8be0af0c248076933e3114ca564b` (yes) |
| mobilenetv2 | 2 | `checkpoints/mobilenetv2/coreml/mobilenetv2_c2.zip` | 669000 | yes | yes | `c0eb3ca5455bf536a67ded57da3ab801439326986a7bcd803b270c2b27be5af0` | `9200decff4fec1a6d5b4e76a0f10278e99372c5fc2395f4f45e7c5c84d470393` (yes) |
| mobilenetv2 | 5 | `checkpoints/mobilenetv2/coreml/mobilenetv2_c5.zip` | 1558603 | yes | yes | `3da7ed1751686c09ab7db2e0b227bc2e46b955ea162671eb1046774476371fcc` | `ce439c049ced3c04282dafca5065501476f7f3cc291a5f4cf76d4d6d01002afc` (yes) |
| mobilenetv2 | 64 | `checkpoints/mobilenetv2/coreml/mobilenetv2_c64.zip` | 5060304 | yes | yes | `0082e8f99fcbcd7aac78f8d59d7f5fb35e5d32a4c0b255c0f0b85cad2c5344d7` | `7c21c93bfccf06607d47cdb5945b8d9a51bdb28f4369f2c03d18eeefebeb0027` (yes) |
| mobilenetv2 | 256 | `checkpoints/mobilenetv2/coreml/mobilenetv2_c256.zip` | 8572278 | yes | yes | `ab82c78c0cb170278da011004c2ef0736afce65bf2db36ad8d869568fc417bc4` | `40ad79ffbba9606a0e71706a596572e2ea429c6a1d223885ee070d7ee3db48b1` (yes) |
| resnet18 | 2 | `checkpoints/resnet18/coreml/resnet18_c2.zip` | 1459809 | yes | yes | `6292d3bf51f2c7ea8a1a5c317b48c1f03b6189d1023323d483564b1c30b513f2` | `2876b5ab1a7a4cc09cf6cb14912e45c76d64883704f256e27b57f74046e65747` (yes) |
| resnet18 | 5 | `checkpoints/resnet18/coreml/resnet18_c5.zip` | 3759424 | yes | yes | `5a53d86c09254c9c11719607e3807bce40b493172a8e6024926065f63a82e382` | `84f6905080b14cc7207691eb0651ba218f79294f8488666ba5e6f1278ca725d7` (yes) |
| resnet18 | 32 | `checkpoints/resnet18/coreml/resnet18_c32.zip` | 9229904 | yes | yes | `e3e8c15925acc267e1483aa8c248a13c7be484bd831159b2c70471e6e82e085d` | `9a12970c27e43bfedd4a71bc5cb1b9bb603a76a471049732e7e65c8ccaaf9385` (yes) |
| resnet18 | 256 | `checkpoints/resnet18/coreml/resnet18_c256.zip` | 22534458 | yes | yes | `23e565b62dc2885c9861031664b9c4a37de83cfba1653ea9bd367d72ed34870f` | `a6403919332fc88a7885090c94ac9eaf28185cdd9e97b445dc48a159d561f61a` (yes) |
| resnet34 | 2 | `checkpoints/resnet34/coreml/resnet34_c2.zip` | 2564857 | yes | yes | `3961fd2f75b5e775e6ab2db65e739fc45877cdccde28ff41cbf305ec2bea1f68` | `032880f33c68861d6c29bd8568d7440b457512460fcdb1010ec9af6764eed3fa` (yes) |
| resnet34 | 4 | `checkpoints/resnet34/coreml/resnet34_c4.zip` | 5841223 | yes | yes | `99c6ae490365001dcf09a090f6df78aa7e795410b4a764226103c414854fa121` | `6be0ea33adea21d921fe2c3a2e97210b27d77eebcc099466317cb64fd5def858` (yes) |
| resnet34 | 128 | `checkpoints/resnet34/coreml/resnet34_c128.zip` | 34489596 | yes | yes | `ac86fb4d7196200c424602215cf57394f346e9440beee051b815cabee5bdc2be` | `a207a001037681e905263963e688f86fa9f467ed12f1cca0cb02cdff6f8df308` (yes) |
| resnet34 | 256 | `checkpoints/resnet34/coreml/resnet34_c256.zip` | 40705803 | yes | yes | `17c8ac5648b6b3ae1c2e76b84f682de512e3c6e9a39e5167aef0e2017262e22f` | `23111fc9f52ba566c127d243236a406ebe217a2f9e53c7b7109748bbcf22e912` (yes) |
| resnet50 | 2 | `checkpoints/resnet50/coreml/resnet50_c2.zip` | 3613302 | yes | yes | `2392a1ea16e312dd78b2c9d37c5ebea7a39aad58ba17e9ba93b496bb6ad1735d` | `7d3ddfc537f8999ffa2dd55954d62b54692d014d81d3868bf3f242ea743bb013` (yes) |
| resnet50 | 4 | `checkpoints/resnet50/coreml/resnet50_c4.zip` | 7483226 | yes | yes | `88adb46acf940795a334a283453e145b9992787dfaf4a9085c7633b4947aabb9` | `d30efad84ab432007b4cfae00a8b38f45408f4ad5bd6e40e72fe333ac515c9d3` (yes) |
| resnet50 | 128 | `checkpoints/resnet50/coreml/resnet50_c128.zip` | 44500161 | yes | yes | `f3aaaac92f382511c9ea258ab7067b14844a1acf42d136d12ae5bf3a2a9fa32e` | `17157a5208f9a88a7c2433f654659e5bfb52f41d158c5ebbd5488cfb7dbc0315` (yes) |
| resnet50 | 256 | `checkpoints/resnet50/coreml/resnet50_c256.zip` | 56936575 | yes | yes | `e291fe1a80de1025c8ecbb00f45b34105f3fb1c563f59074e10f4f8368a84928` | `19ffdbbffa27742dd5b180e60ab9b69b42e65c3459f221e43faefed750f87be1` (yes) |

The content fingerprint is SHA-256 over the sorted (relative path, SHA-256 of bytes) of every file inside the unpacked `model.mlpackage`, so it identifies the model independently of how it was zipped.

## Attestation

Digest of the machine-readable result set so far (27 files, all fields above, canonical JSON): `sha256:e0aeb87604d6dd6718282b367f8f3d9053a9dd2dec88c6b790e4bedede3b744f`

This is a hash binding the results to the files, not a cryptographic signature: no signing key was used.

## Check it yourself

```
huggingface-cli download azemel/retinaface-xs checkpoints/<backbone>/coreml/<backbone>_c2.zip --local-dir .
shasum -a 256 checkpoints/<backbone>/coreml/<backbone>_c2.zip     # must equal the SHA-256 in the tables above
python inference/export_check.py --format coreml --network <backbone> --hf-repo azemel/retinaface-xs --hf-level c2 \
  --input-color-order rgb --compute-units CPU_AND_GPU --image-size 640
```

## Scope

- Covers the CoreML files listed above exactly as they are on HF, on the GPU compute unit, full val set.
- CPU (`CPU_ONLY`) is not part of this run.
- The comparison is against the PyTorch checkpoints on HF at the time of the run; if those are replaced, the exports and this verification must be redone.
- ONNX, TFLite and TFJS exports are not verified here.
