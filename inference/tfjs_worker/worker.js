/*
 * Long-lived TF.js inference worker for export_check.py --format tfjs.
 *
 *   node worker.js <model.json> <image_size>
 *
 * Loads the graph model once, then serves requests over stdin/stdout:
 *   request : image_size*image_size*3 float32 (little-endian), NHWC BGR, raw 0-255
 *   response: for each of boxes / scores / landmarks (in that order):
 *             uint32 LE element count, then that many float32 (flat, [1,N,d] order)
 * Outputs are identified by last-dim size (4=boxes, 1=scores, 10=landmarks),
 * not by name -- same onnx2tf naming bug as TFLite (see export_tflite.py).
 * Nothing but responses is ever written to stdout; TF's own logging goes to stderr.
 */
// tfjs-node 4.22 still calls util.isNullOrUndefined & co., removed in Node >= 23.
const util = require('util');
const legacy = {
  isNullOrUndefined: v => v === null || v === undefined,
  isNull: v => v === null,
  isUndefined: v => v === undefined,
  isArray: Array.isArray,
  isNumber: v => typeof v === 'number',
  isString: v => typeof v === 'string',
  isBoolean: v => typeof v === 'boolean',
  isFunction: v => typeof v === 'function',
  isObject: v => v !== null && typeof v === 'object',
};
for (const [k, f] of Object.entries(legacy)) if (!util[k]) util[k] = f;

const tf = require('@tensorflow/tfjs-node');

async function main() {
  const [modelPath, sizeArg] = process.argv.slice(2);
  const size = parseInt(sizeArg, 10);
  const reqBytes = size * size * 3 * 4;
  const model = await tf.loadGraphModel(`file://${modelPath}`);

  let buf = Buffer.alloc(0);
  let busy = false;
  let ended = false;

  async function pump() {
    if (busy) return;
    busy = true;
    while (buf.length >= reqBytes) {
      const chunk = buf.subarray(0, reqBytes);
      buf = buf.subarray(reqBytes);
      const data = new Float32Array(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + reqBytes));
      const outs = tf.tidy(() => model.execute(tf.tensor4d(data, [1, size, size, 3], 'float32')));
      const byWidth = {};
      for (const t of outs) byWidth[t.shape[t.shape.length - 1]] = t;
      const parts = [];
      for (const w of [4, 1, 10]) {
        const arr = await byWidth[w].data();
        const head = Buffer.alloc(4);
        head.writeUInt32LE(arr.length, 0);
        parts.push(head, Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
      }
      for (const t of outs) t.dispose();
      const out = Buffer.concat(parts);
      if (!process.stdout.write(out)) await new Promise(r => process.stdout.once('drain', r));
    }
    busy = false;
    if (ended) process.stdout.write('', () => process.exit(0));
  }

  process.stdin.on('data', d => { buf = Buffer.concat([buf, d]); pump(); });
  process.stdin.on('end', () => { ended = true; if (!busy) process.stdout.write('', () => process.exit(0)); });
  process.stdout.write(Buffer.from('READY'));
}

main().catch(e => { console.error(e); process.exit(1); });
