// ----------------------------
// 1. LOAD PaddleOCR Lite
// ----------------------------
let ocrReady = false;

async function initOCR() {
  await PaddleOCR.init({
    wasmFolder: "paddleocr",
    chunk: true,
    short: true,
    splitPatch: true,
  });
  ocrReady = true;
  console.log("PaddleOCR WASM loaded.");
}
initOCR();

// ----------------------------
// 2. OCR IMAGE
// ----------------------------
async function runOCR(imageBlob) {
  if (!ocrReady) {
    throw new Error("OCR not loaded yet");
  }

  const img = new Image();
  img.src = URL.createObjectURL(imageBlob);

  await new Promise((res) => (img.onload = res));

  const result = await PaddleOCR.recognize(img);
  return result.map((r) => r.text).join("\n");
}

// ----------------------------
// 3. ONNX Runtime for Tiny NLP Model
// ----------------------------

// Load ONNX model (T5-small, MiniLM, etc.)
let session;

async function initONNX() {
  session = await ort.InferenceSession.create("./onnx/model.onnx", {
    executionProviders: ["wasm"], // mobile-friendly
    graphOptimizationLevel: "all",
  });
  console.log("ONNX model ready.");
}
initONNX();

// Prepare text → tensor
function tokenize(text) {
  // Replace this with your tokenizer (WordPiece/BPE/etc.)
  // This is a simple placeholder example

  const maxLength = 256;
  const tokens = new Array(maxLength).fill(0);

  // Fake tokenization: map char codes
  for (let i = 0; i < Math.min(text.length, maxLength); i++) {
    tokens[i] = text.charCodeAt(i) % 255;
  }

  return new ort.Tensor(
    "int64",
    BigInt64Array.from(tokens.map((x) => BigInt(x))),
    [1, maxLength]
  );
}

// Run ONNX model (text extraction)
async function runONNXExtraction(text) {
  if (!session) throw new Error("ONNX model not loaded.");

  const input = tokenize(text);
  const results = await session.run({ input_ids: input });

  // Assuming output is a simple text field
  const outputData = results.output.data; // adjust for your model
  const decoded = new TextDecoder().decode(outputData);

  return decoded;
}

// ----------------------------
// 4. Combined Pipeline:
//     IMAGE → OCR → TEXT → ONNX
// ----------------------------
document.getElementById("imageInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  document.getElementById("ocrOutput").innerText = "Running OCR...";

  const ocrText = await runOCR(file);
  document.getElementById("ocrOutput").innerText = "OCR Text:\n" + ocrText;

  document.getElementById("llmOutput").innerText = "Extracting fields...";

  const extraction = await runONNXExtraction(ocrText);
  document.getElementById("llmOutput").innerText = extraction;
});
