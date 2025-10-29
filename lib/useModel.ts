"use client"

import { useState, useEffect, useCallback } from "react"
import { loadModel, predict, isModelLoaded, type ModelPrediction } from "./modelService"

export type ModelStatus = "idle" | "loading" | "ready" | "error"

export interface UseModelReturn {
  status: ModelStatus
  error: string | null
  isReady: boolean
  predictImage: (image: HTMLImageElement) => Promise<ModelPrediction | null>
  loadModelManually: () => Promise<void>
}

/**
 * React hook for managing ONNX model state and predictions
 * Automatically loads the model on mount
 */
export function useModel(): UseModelReturn {
  const [status, setStatus] = useState<ModelStatus>("idle")
  const [error, setError] = useState<string | null>(null)

  // Auto-load model on mount
  useEffect(() => {
    const initModel = async () => {
      // Check if already loaded
      if (isModelLoaded()) {
        setStatus("ready")
        return
      }

      setStatus("loading")
      setError(null)

      try {
        await loadModel()
        setStatus("ready")
        console.log("[v0] Model ready for inference")
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to load model"
        setError(errorMessage)
        setStatus("error")
        console.error("[v0] Model loading error:", errorMessage)
      }
    }

    initModel()
  }, [])

  // Manual model loading function
  const loadModelManually = useCallback(async () => {
    if (isModelLoaded()) {
      console.log("[v0] Model already loaded")
      return
    }

    setStatus("loading")
    setError(null)

    try {
      await loadModel()
      setStatus("ready")
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load model"
      setError(errorMessage)
      setStatus("error")
      throw err
    }
  }, [])

  // Predict function
  const predictImage = useCallback(async (image: HTMLImageElement): Promise<ModelPrediction | null> => {
    if (!isModelLoaded()) {
      setError("Model not loaded")
      return null
    }

    try {
      const prediction = await predict(image)
      return prediction
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Prediction failed"
      setError(errorMessage)
      console.error("[v0] Prediction error:", errorMessage)
      return null
    }
  }, [])

  return {
    status,
    error,
    isReady: status === "ready",
    predictImage,
    loadModelManually,
  }
}
