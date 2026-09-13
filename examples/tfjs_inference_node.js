/*
 * Run face detection with a TF.js graph model (from this repo's
 * scripts/export_tfjs.py, or downloaded from Hugging Face) under Node.js.
 * Same NHWC input and shape-based output identification as the TFLite
 * example in this directory -- see scripts/export_tflite.py's module
 * docstring for why (both trace back through the same onnx2tf SavedModel).
 * Prints detections rather than drawing them -- see the Python examples
 * in this directory (which use cv2) for a version that saves an annotated
 * image instead.
 *
 *   npm install @tensorflow/tfjs-node jimp
 *   node examples/tfjs_inference_node.js tfjs/model.json photo.jpg 640
 */

const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');

function nms(boxes, scores, threshold) {
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const areas = boxes.map(b => (b[2] - b[0] + 1) * (b[3] - b[1] + 1));
  const keep = [];
  const suppressed = new Set();
  for (const i of order) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (const j of order) {
      if (j === i || suppressed.has(j)) continue;
      const xx1 = Math.max(boxes[i][0], boxes[j][0]);
      const yy1 = Math.max(boxes[i][1], boxes[j][1]);
      const xx2 = Math.min(boxes[i][2], boxes[j][2]);
      const yy2 = Math.min(boxes[i][3], boxes[j][3]);
      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const inter = w * h;
      const overlap = inter / (areas[i] + areas[j] - inter);
      if (overlap > threshold) suppressed.add(j);
    }
  }
  return keep;
}

async function main() {
  const [modelPath, imagePath, imageSizeArg] = process.argv.slice(2);
  const imageSize = parseInt(imageSizeArg || '640', 10);
  const confThreshold = 0.5;
  const nmsThreshold = 0.4;

  const image = await Jimp.read(imagePath);
  const origWidth = image.bitmap.width;
  const origHeight = image.bitmap.height;
  const resized = image.clone().resize(imageSize, imageSize);

  // RGB (Jimp's native order) -> BGR, HWC, float32, raw 0-255 pixel values, NHWC batch dim
  const data = new Float32Array(imageSize * imageSize * 3);
  let idx = 0;
  resized.scan(0, 0, imageSize, imageSize, function (x, y, offset) {
    const r = this.bitmap.data[offset + 0];
    const g = this.bitmap.data[offset + 1];
    const b = this.bitmap.data[offset + 2];
    data[idx++] = b;
    data[idx++] = g;
    data[idx++] = r;
  });
  const input = tf.tensor4d(data, [1, imageSize, imageSize, 3], 'float32');

  const model = await tf.loadGraphModel(`file://${modelPath}`);
  const outputs = model.execute(input);
  // match by last-dim size, not name -- same onnx2tf naming bug as TFLite (see export_tflite.py)
  const byWidth = {};
  for (const t of outputs) byWidth[t.shape[t.shape.length - 1]] = await t.array();
  const boxesRaw = byWidth[4][0];
  const scoresRaw = byWidth[1][0].map(s => s[0]);
  const landmarksRaw = byWidth[10][0];

  const kept = [];
  for (let i = 0; i < scoresRaw.length; i++) {
    if (scoresRaw[i] > confThreshold) kept.push(i);
  }
  kept.sort((a, b) => scoresRaw[b] - scoresRaw[a]);

  const boxes = kept.map(i => boxesRaw[i]);
  const scores = kept.map(i => scoresRaw[i]);
  const landmarks = kept.map(i => landmarksRaw[i]);
  const keepIdx = nms(boxes, scores, nmsThreshold);

  const sx = origWidth / imageSize;
  const sy = origHeight / imageSize;
  const detections = keepIdx.map(i => ({
    box: [boxes[i][0] * sx, boxes[i][1] * sy, boxes[i][2] * sx, boxes[i][3] * sy],
    score: scores[i],
    landmarks: landmarks[i].map((v, j) => (j % 2 === 0 ? v * sx : v * sy)),
  }));

  console.log(`${detections.length} face(s) detected`);
  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box.map(v => Math.round(v));
    console.log(`  box=[${x1}, ${y1}, ${x2}, ${y2}] score=${det.score.toFixed(3)}`);
  }
}

main();
