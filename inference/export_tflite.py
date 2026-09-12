"""ONNX -> TensorFlow SavedModel -> TFLite, via onnx2tf. Takes the .onnx
produced by export_onnx.py and converts it to both a SavedModel (needed by
export_tfjs.py) and a set of .tflite files.

Two onnx2tf gotchas this works around, confirmed by direct testing against
this repo's own mobilenetv1_0.25 export -- worth knowing before touching
this script:

1. tflite_backend="tf_converter" (NOT onnx2tf's newer, faster default
   "flatbuffer_direct") is required to get a real SavedModel out at all --
   the default backend's own from-flatbuffer SavedModel reconstruction
   (flatbuffer_direct_output_saved_model) crashed converting this
   architecture's depthwise convolutions ("input.shape.rank must be at
   least 5"). tf_converter needs the `tf_keras` package installed
   alongside `tensorflow` (`pip install tf_keras`) -- pure tflite-only
   conversion doesn't need it, but this script always asks for a
   SavedModel too.
2. The resulting model takes NHWC input ([1, H, W, 3]), not the NCHW
   ([1, 3, H, W]) every other export in this directory uses -- onnx2tf's
   keep_ncw_or_nchw_or_ncdhw_input_names has no effect under this backend.
   And its output tensor NAMES land on the wrong tensors (confirmed:
   "landmarks" bound to the (4200, 1) tensor and "scores" to the
   (4200, 10) one -- swapped, an alphabetical-vs-declared-order mismatch
   inside onnx2tf's own name-copying) -- every example under examples/
   identifies boxes/scores/landmarks by their distinctive last-dim size
   (4 / 1 / 10) instead of trusting the name, and every example feeds
   this model NHWC. Do this too if you write new inference code against
   these files.

IMPORTANT: inference/onnx_palettize.py's Gather-based palettization (the
mechanism that makes export_onnx.py's quantized output smaller on disk)
does NOT survive this conversion -- confirmed empirically: a palettized and
a plain dense .onnx of the same architecture produce byte-for-byte
identically sized plain .tflite files. onnx2tf's conversion constant-folds
an all-initializer Cast/Add/Reshape/Gather chain (nothing in it depends on
the model's input, so from the optimizer's point of view it's just a
constant to precompute) back into one dense float32 tensor before the
.tflite is ever written.

The real compression path for TFLite is its OWN native
output_dynamic_range_quantized_tflite (int8 weights, chosen post-training
per-tensor/per-channel by the TFLite converter itself) -- confirmed to
shrink this repo's mobilenetv1_0.25 export from 1.94 MB (plain float32
.tflite) to 0.74 MB, smaller than even the palettized .onnx. Because the
weights it's quantizing already sit on a tight, small-cardinality grid,
TFLite's int8 range-quantization loses very little extra precision on top
of what this checkpoint's own training already accepted -- so this is
enabled by default here, not left as an opt-in extra.

Usage:
    python inference/export_onnx.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_c16.zip --image-size 640 --out-dir onnx_export
    python inference/export_tflite.py --onnx onnx_export/mobilenetv1_0.25_c16.onnx \\
        --out-dir tflite_export
"""

import argparse
from pathlib import Path

import onnx2tf


def export_tflite(onnx_path: str, out_dir: str, dynamic_range_quant: bool = True) -> Path:
    out_dir_p = Path(out_dir)
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir_p),
        output_signaturedefs=True,           # writes a SavedModel with real signatures -- needed by export_tfjs.py
        copy_onnx_input_output_names_to_tflite=True,
        tflite_backend="tf_converter",        # see module docstring -- the default backend can't SavedModel-export this architecture
        output_dynamic_range_quantized_tflite=dynamic_range_quant,
        non_verbose=True,
    )

    tflite_files = sorted(out_dir_p.glob("*.tflite"))
    for f in tflite_files:
        tag = " (recommended -- real int8 compression, see module docstring)" if "dynamic_range" in f.name else ""
        print(f"TFLite model: {f.stat().st_size / 1e6:.2f} MB -> {f}{tag}")
    if (out_dir_p / "saved_model.pb").exists():
        print(f"SavedModel (for export_tfjs.py) -> {out_dir_p}")
    return out_dir_p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="path to a .onnx produced by export_onnx.py")
    p.add_argument("--out-dir", default="tflite_export")
    p.add_argument("--no-dynamic-range-quant", action="store_true",
                    help="skip the int8 dynamic-range-quantized .tflite -- keeps only the plain "
                         "float32/float16 outputs onnx2tf always writes")
    args = p.parse_args()
    export_tflite(args.onnx, args.out_dir, dynamic_range_quant=not args.no_dynamic_range_quant)


if __name__ == "__main__":
    main()
