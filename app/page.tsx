"use client"

import type React from "react"

import { useState } from "react"
import { Upload, AlertTriangle, Brain, Github, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { useModel } from "@/lib/useModel"
import type { TumorClass } from "@/lib/modelService"

interface ClassificationResult {
  class: TumorClass
  confidence: number
  probabilities: Record<TumorClass, number>
  inferenceTime?: number
}

interface SampleImage {
  id: number
  src: string
  label: string
  class: TumorClass
}

const samples: SampleImage[] = [
  { id: 1, src: "/glioma1.jpg", label: "Glioma", class: "glioma" },
  { id: 2, src: "/glioma2.jpg", label: "Glioma", class: "glioma" },
  { id: 3, src: "/meningioma1.jpg", label: "Meningioma", class: "meningioma" },
  { id: 4, src: "/meningioma2.jpg", label: "Meningioma", class: "meningioma" },
  { id: 5, src: "/notumor1.jpg", label: "No Tumor", class: "notumor" },
  { id: 6, src: "/notumor2.jpg", label: "No Tumor", class: "notumor" },
  { id: 7, src: "/pituitary1.jpg", label: "Pituitary", class: "pituitary" },
  { id: 8, src: "/pituitary2.jpg", label: "Pituitary", class: "pituitary" },
]

const classColors: Record<TumorClass, string> = {
  glioma: "bg-red-500",
  meningioma: "bg-orange-500",
  notumor: "bg-green-500",
  pituitary: "bg-blue-500",
}

const classLabels: Record<TumorClass, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No Tumor",
  pituitary: "Pituitary",
}

export default function BrainTumorClassifier() {
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { status: modelStatus, error: modelError, isReady, predictImage } = useModel()

  const handleImageUpload = async (file: File) => {
    // Validate file type
    if (!file.type.startsWith("image/")) {
      setError("Please upload a JPG or PNG image")
      return
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("Image too large. Please use a smaller file.")
      return
    }

    setError(null)

    // Display preview
    const url = URL.createObjectURL(file)
    setImagePreview(url)

    if (!isReady) {
      setError("Model is not ready yet. Please wait...")
      return
    }

    setLoading(true)

    try {
      // Load image as HTMLImageElement
      const img = new Image()
      img.crossOrigin = "anonymous"

      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = reject
        img.src = url
      })

      // Run prediction
      const prediction = await predictImage(img)

      if (!prediction) {
        setError("Prediction failed. Please try again.")
        setLoading(false)
        return
      }

      // Convert to ClassificationResult format
      const probabilities: Record<TumorClass, number> = {
        glioma: 0,
        meningioma: 0,
        notumor: 0,
        pituitary: 0,
      }

      prediction.predictions.forEach((pred) => {
        probabilities[pred.className] = pred.confidence
      })

      const classificationResult: ClassificationResult = {
        class: prediction.topPrediction.className,
        confidence: prediction.topPrediction.confidence,
        probabilities,
        inferenceTime: prediction.inferenceTime,
      }

      setResult(classificationResult)
    } catch (err) {
      console.error("[v0] Classification error:", err)
      setError("Failed to classify image. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const handleSampleClick = async (sample: SampleImage) => {
    setImagePreview(sample.src)
    setError(null)

    if (!isReady) {
      setError("Model is not ready yet. Please wait...")
      return
    }

    setLoading(true)

    try {
      // Load sample image
      const img = new Image()
      img.crossOrigin = "anonymous"

      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = reject
        img.src = sample.src
      })

      // Run prediction
      const prediction = await predictImage(img)

      if (!prediction) {
        setError("Prediction failed. Please try again.")
        setLoading(false)
        return
      }

      // Convert to ClassificationResult format
      const probabilities: Record<TumorClass, number> = {
        glioma: 0,
        meningioma: 0,
        notumor: 0,
        pituitary: 0,
      }

      prediction.predictions.forEach((pred) => {
        probabilities[pred.className] = pred.confidence
      })

      setResult({
        class: prediction.topPrediction.className,
        confidence: prediction.topPrediction.confidence,
        probabilities,
        inferenceTime: prediction.inferenceTime,
      })
    } catch (err) {
      console.error("[v0] Classification error:", err)
      setError("Failed to classify image. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0])
    }
  }

  const clearImage = () => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview)
    }
    setImagePreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 py-8">
        <div className="container mx-auto px-4 max-w-6xl">
          <div className="flex items-center gap-3 mb-2">
            <Brain className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-semibold text-gray-900">Brain Tumor Classifier</h1>
          </div>
          <p className="text-gray-600 mb-4">AI-powered MRI analysis using deep learning</p>
          {modelStatus === "loading" && (
            <Alert className="bg-blue-50 border-blue-200 mb-4">
              <AlertDescription className="text-blue-800">Loading ONNX model... Please wait.</AlertDescription>
            </Alert>
          )}
          {modelStatus === "error" && (
            <Alert className="bg-red-50 border-red-200 mb-4">
              <AlertTriangle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">
                <strong>Model Error:</strong> {modelError}
              </AlertDescription>
            </Alert>
          )}
          {modelStatus === "ready" && (
            <Alert className="bg-green-50 border-green-200 mb-4">
              <AlertDescription className="text-green-800">
                <strong>Model Ready</strong> - ONNX Runtime Web loaded successfully
              </AlertDescription>
            </Alert>
          )}
          <Alert className="bg-yellow-50 border-yellow-200">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <AlertDescription className="text-yellow-800">
              <strong>Research Demo</strong> - Not for Medical Diagnosis
            </AlertDescription>
          </Alert>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Error Alert */}
        {error && (
          <Alert className="mb-6 bg-red-50 border-red-200">
            <AlertTriangle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}

        {/* Main Interface */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          {/* Left Column: Image Input */}
          <Card>
            <CardHeader>
              <CardTitle>Upload MRI Scan</CardTitle>
              <CardDescription>Accepted formats: JPG, PNG</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Drag & Drop Zone */}
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={cn(
                  "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
                  dragActive ? "border-blue-600 bg-blue-50" : "border-gray-300 bg-gray-50",
                  !imagePreview && "cursor-pointer hover:border-blue-400",
                )}
              >
                {!imagePreview ? (
                  <div className="space-y-4">
                    <Upload className="w-12 h-12 mx-auto text-gray-400" />
                    <div>
                      <p className="text-gray-600 mb-2">Drag & drop your MRI scan here</p>
                      <p className="text-sm text-gray-500">or</p>
                    </div>
                    <label htmlFor="file-upload">
                      <Button asChild>
                        <span className="cursor-pointer">
                          <Upload className="w-4 h-4 mr-2" />
                          Upload MRI Scan
                        </span>
                      </Button>
                    </label>
                    <input
                      id="file-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleFileInput}
                      className="hidden"
                    />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <img
                      src={imagePreview || "/placeholder.svg"}
                      alt="MRI Preview"
                      className="w-full max-w-[300px] h-[300px] object-cover rounded-lg mx-auto"
                    />
                    <Button variant="outline" onClick={clearImage}>
                      Clear Image
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Right Column: Results Panel */}
          <Card>
            <CardHeader>
              <CardTitle>Classification Results</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex flex-col items-center justify-center py-12 space-y-4">
                  <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  <p className="text-gray-600">Analyzing MRI scan...</p>
                </div>
              ) : result ? (
                <div className="space-y-6">
                  {/* Confidence & Class */}
                  <div className="text-center space-y-2">
                    <div className="text-5xl font-semibold text-gray-900">{(result.confidence * 100).toFixed(1)}%</div>
                    <div className="flex items-center justify-center gap-2">
                      <Badge className={cn("text-white", classColors[result.class])}>{classLabels[result.class]}</Badge>
                    </div>
                    {result.inferenceTime && (
                      <p className="text-sm text-gray-500">Inference: {result.inferenceTime.toFixed(0)}ms</p>
                    )}
                  </div>

                  {/* Probability Bars */}
                  <div className="space-y-3">
                    <h3 className="font-medium text-sm text-gray-700">All Probabilities</h3>
                    {(Object.entries(result.probabilities) as [TumorClass, number][])
                      .sort(([, a], [, b]) => b - a)
                      .map(([cls, prob]) => (
                        <div key={cls} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-700">{classLabels[cls]}</span>
                            <span className="text-gray-600">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <Progress value={prob * 100} className="h-2" />
                        </div>
                      ))}
                  </div>

                  <Button onClick={clearImage} className="w-full">
                    Classify Another
                  </Button>
                </div>
              ) : (
                <div className="flex items-center justify-center py-12 text-gray-500">
                  Upload an image to get started
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sample Images Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">Try Sample Images</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {samples.map((sample) => (
              <Card
                key={sample.id}
                className="cursor-pointer transition-all hover:shadow-lg hover:scale-105"
                onClick={() => handleSampleClick(sample)}
              >
                <CardContent className="p-4">
                  <img
                    src={sample.src || "/placeholder.svg"}
                    alt={sample.label}
                    className="w-full h-[150px] object-cover rounded-md mb-2"
                  />
                  <p className="text-sm font-medium text-center text-gray-700">{sample.label}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Info Accordion */}
        <section className="mb-12">
          <Accordion type="single" collapsible className="bg-white rounded-lg border border-gray-200">
            <AccordionItem value="architecture">
              <AccordionTrigger className="px-6">Model Architecture</AccordionTrigger>
              <AccordionContent className="px-6 pb-4">
                <ul className="space-y-2 text-gray-700">
                  <li>• ONNX Runtime Web with WebGL acceleration</li>
                  <li>• 3-layer Convolutional Neural Network (CNN)</li>
                  <li>• Input: 128×128 RGB images (channels-first format)</li>
                  <li>• Output: 4 classes (Glioma, Meningioma, No Tumor, Pituitary)</li>
                  <li>• Trained using PyTorch, exported to ONNX</li>
                  <li>• Preprocessing: (pixel/255 - 0.5) / 0.5 normalization</li>
                </ul>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="performance">
              <AccordionTrigger className="px-6">Performance Metrics</AccordionTrigger>
              <AccordionContent className="px-6 pb-4">
                <ul className="space-y-2 text-gray-700">
                  <li>• Test Accuracy: ~95%</li>
                  <li>• Dataset: 3000+ MRI scans</li>
                  <li>• Training: 50 epochs with data augmentation</li>
                  <li>• Validation split: 80/20</li>
                </ul>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="links">
              <AccordionTrigger className="px-6">Resources & Links</AccordionTrigger>
              <AccordionContent className="px-6 pb-4">
                <div className="space-y-3">
                  <a
                    href="https://github.com/Repetto-A/BrainTumorAI-Web"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-blue-600 hover:text-blue-700"
                  >
                    <Github className="w-4 h-4" />
                    Web Application Repository
                    <ExternalLink className="w-3 h-3" />
                  </a>
                  <a
                    href="https://github.com/Repetto-A/Brain-Tumor-Classifier"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-blue-600 hover:text-blue-700"
                  >
                    <Github className="w-4 h-4" />
                    Model Repository
                    <ExternalLink className="w-3 h-3" />
                  </a>
                  <a
                    href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-blue-600 hover:text-blue-700"
                  >
                    <ExternalLink className="w-4 h-4" />
                    Kaggle Dataset
                    <ExternalLink className="w-3 h-3" />
                  </a>
                  </a>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-8 mt-12">
        <div className="container mx-auto px-4 max-w-6xl">
          <Alert className="mb-4 bg-red-50 border-red-200">
            <AlertTriangle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-800">
              <strong>Disclaimer:</strong> This is a research prototype. Never use for actual medical decisions. Always
              consult healthcare professionals.
            </AlertDescription>
          </Alert>
          <p className="text-center text-gray-600 text-sm">Made by Alejandro Repetto | {new Date().getFullYear()}</p>
        </div>
      </footer>
    </div>
  )
}
