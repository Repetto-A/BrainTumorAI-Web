import * as ort from "onnxruntime-web"

// Model configuration
const MODEL_PATH = "/model/brain_tumor_model.onnx"
const INPUT_SIZE = 128
const NUM_CHANNELS = 3
const CLASSES: TumorClass[] = ["glioma", "meningioma", "notumor", "pituitary"]

// Types
export type TumorClass = "glioma" | "meningioma" | "notumor" | "pituitary"

export interface PredictionResult {
  className: TumorClass
  confidence: number
}

export interface ModelPrediction {
  predictions: PredictionResult[]
  topPrediction: PredictionResult
  inferenceTime: number
}

// Global session instance (singleton pattern)
let session: ort.InferenceSession | null = null
let isLoading = false

/**
 * Load the ONNX model
 * Uses WebGL for acceleration with WASM as fallback
 */
export async function loadModel(): Promise<void> {
  if (session) {
    console.log("[v0] Model already loaded")
    return
  }

  if (isLoading) {
    console.log("[v0] Model is currently loading")
    return
  }

  try {
    isLoading = true
    console.log("[v0] Loading ONNX model from:", MODEL_PATH)

    // Configure ONNX Runtime with WebGL (preferred) and WASM (fallback)
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgl", "wasm"],
      graphOptimizationLevel: "all",
    })

    console.log("[v0] Model loaded successfully")
    console.log("[v0] Input names:", session.inputNames)
    console.log("[v0] Output names:", session.outputNames)
  } catch (error) {
    session = null
    console.error("[v0] Failed to load model:", error)
    throw new Error(`Failed to load ONNX model: ${error instanceof Error ? error.message : "Unknown error"}`)
  } finally {
    isLoading = false
  }
}

/**
 * Check if model is loaded and ready
 */
export function isModelLoaded(): boolean {
  return session !== null
}

/**
 * Preprocess image to match PyTorch training format
 * - Resize to 128x128
 * - Normalize: (pixel/255 - 0.5) / 0.5
 * - Convert to channels-first format [1, 3, 128, 128]
 */
function preprocessImage(image: HTMLImageElement): Float32Array {
  // Create canvas for image processing
  const canvas = document.createElement("canvas")
  canvas.width = INPUT_SIZE
  canvas.height = INPUT_SIZE
  const ctx = canvas.getContext("2d")

  if (!ctx) {
    throw new Error("Failed to get canvas context")
  }

  // Draw and resize image
  ctx.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE)

  // Get image data
  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
  const pixels = imageData.data // RGBA format

  // Create tensor with shape [1, 3, 128, 128]
  const tensorSize = 1 * NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE
  const tensor = new Float32Array(tensorSize)

  // Normalize and convert to channels-first format
  // Formula: (pixel/255 - 0.5) / 0.5
  const pixelCount = INPUT_SIZE * INPUT_SIZE

  for (let i = 0; i < pixelCount; i++) {
    const pixelIndex = i * 4 // RGBA = 4 bytes per pixel

    // Red channel (positions 0-16383)
    tensor[i] = (pixels[pixelIndex] / 255 - 0.5) / 0.5

    // Green channel (positions 16384-32767)
    tensor[pixelCount + i] = (pixels[pixelIndex + 1] / 255 - 0.5) / 0.5

    // Blue channel (positions 32768-49151)
    tensor[pixelCount * 2 + i] = (pixels[pixelIndex + 2] / 255 - 0.5) / 0.5
  }

  return tensor
}

/**
 * Apply softmax to convert logits to probabilities
 */
function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits)
  const expScores = logits.map((logit) => Math.exp(logit - maxLogit))
  const sumExpScores = expScores.reduce((a, b) => a + b, 0)
  return expScores.map((score) => score / sumExpScores)
}

/**
 * Run inference on an image
 * @param image - HTMLImageElement to classify
 * @returns ModelPrediction with class probabilities and inference time
 */
export async function predict(image: HTMLImageElement): Promise<ModelPrediction> {
  if (!session) {
    throw new Error("Model not loaded. Call loadModel() first.")
  }

  const startTime = performance.now()

  try {
    // Preprocess image
    const inputTensor = preprocessImage(image)

    // Create ONNX tensor
    const tensor = new ort.Tensor("float32", inputTensor, [1, NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE])

    // Run inference
    const feeds: Record<string, ort.Tensor> = {}
    feeds[session.inputNames[0]] = tensor

    const results = await session.run(feeds)

    // Get output tensor
    const outputTensor = results[session.outputNames[0]]
    const logits = Array.from(outputTensor.data as Float32Array)

    // Apply softmax to get probabilities
    const probabilities = softmax(logits)

    // Create prediction results
    const predictions: PredictionResult[] = CLASSES.map((className, index) => ({
      className,
      confidence: probabilities[index],
    }))

    // Sort by confidence
    predictions.sort((a, b) => b.confidence - a.confidence)

    const inferenceTime = performance.now() - startTime

    console.log("[v0] Inference completed in", inferenceTime.toFixed(2), "ms")
    console.log("[v0] Top prediction:", predictions[0].className, predictions[0].confidence.toFixed(4))

    return {
      predictions,
      topPrediction: predictions[0],
      inferenceTime,
    }
  } catch (error) {
    console.error("[v0] Prediction failed:", error)
    throw new Error(`Prediction failed: ${error instanceof Error ? error.message : "Unknown error"}`)
  }
}
