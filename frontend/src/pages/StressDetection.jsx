// File: src/pages/StressDetection.jsx

import { useState, useEffect, useRef } from "react"
/* eslint-disable-next-line no-unused-vars */
import { motion } from "framer-motion"
import Webcam from "react-webcam"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, Camera, Music } from "lucide-react"
import Navbar from "@/components/Navbar"
import Footer from "@/components/Footer"
import { useNavigate } from "react-router-dom"

const API_BASE = "http://127.0.0.1:5000" // <-- change if your backend is elsewhere
const CAPTURE_INTERVAL_MS = 700

function StressDetection() {
  const navigate = useNavigate()

  const [isWebcamActive, setIsWebcamActive] = useState(false)
  const [detectedEmotion, setDetectedEmotion] = useState(null) // "Anxiety" / "No Anxiety" / "No face detected"
  const [stressLevel, setStressLevel] = useState(null)          // Low / Medium / High
  const [confidence, setConfidence] = useState(null)            // 0..1
  const [faces, setFaces] = useState(0)
  const [error, setError] = useState("")
  const [elapsedMs, setElapsedMs] = useState(null)

  const webcamRef = useRef(null)
  const overlayRef = useRef(null)
  const timerRef = useRef(null)
  const inFlightRef = useRef(false)
  const bboxRef = useRef(null) // last bbox for drawing

  // --- Animations (unchanged) ---
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.3 } },
  }
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { type: "spring", stiffness: 100 } },
  }

  // --- Helpers ---
  const inferStressLevel = (label, conf) => {
    if (label === "Anxiety") {
      if (conf >= 0.85) return "High"
      if (conf >= 0.6) return "Medium"
      return "Low"
    } else if (label === "No Anxiety") {
      if (conf !== null && conf < 0.55) return "Medium" // low confidence, be cautious
      return "Low"
    }
    return null
  }

  const getSuggestion = () => {
    switch (stressLevel) {
      case "Low":
        return "You're doing okay. Keep up supportive habits: hydration, short walks, and steady rest."
      case "Medium":
        return "Try a quick reset: 4–7–8 breathing or a 5-minute stretch. Want me to open the Panic SOS chatbot?"
      case "High":
        return "Let’s take care of you right now. Consider a guided breathing exercise or reach out to someone you trust."
      default:
        return detectedEmotion === "No face detected"
          ? "I can't see a face. Adjust lighting, face the camera, or move a bit closer."
          : "Waiting for stress level detection…"
    }
  }

  const drawOverlay = () => {
    const canvas = overlayRef.current
    const video = webcamRef.current?.video
    if (!canvas || !video) return
    const ctx = canvas.getContext("2d")

    // Match overlay canvas to displayed video size
    const vw = video.videoWidth || 0
    const vh = video.videoHeight || 0
    if (vw === 0 || vh === 0) return
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw
      canvas.height = vh
    }

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw bbox if present
    const box = bboxRef.current
    if (box && Array.isArray(box) && box.length === 4) {
      const [x, y, w, h] = box.map(v => Math.max(0, Math.floor(v)))
      ctx.lineWidth = 3
      ctx.strokeStyle = "#22c55e" // Tailwind emerald-500-ish
      ctx.strokeRect(x, y, w, h)

      // Label badge
      const label = detectedEmotion ? `${detectedEmotion} ${(confidence != null ? `(${Math.round(confidence*100)}%)` : "")}` : ""
      if (label) {
        const padX = 8, padY = 6
        ctx.font = "14px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto"
        const textW = ctx.measureText(label).width
        const boxW = textW + padX * 2
        const boxH = 22 + padY

        // background box
        ctx.fillStyle = "rgba(34,197,94,0.85)" // green-ish
        ctx.fillRect(x, Math.max(0, y - boxH - 4), boxW, boxH)
        // text
        ctx.fillStyle = "#fff"
        ctx.fillText(label, x + padX, Math.max(14, y - 10))
      }
    }
  }

  // Keep overlay drawing smooth even between network responses
  useEffect(() => {
    let raf
    const loop = () => {
      drawOverlay()
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isWebcamActive])

  // Capture & send frames loop
  const startCaptureLoop = () => {
    if (timerRef.current) return
    timerRef.current = setInterval(async () => {
      if (inFlightRef.current) return
      const shot = webcamRef.current?.getScreenshot()
      if (!shot) return
      inFlightRef.current = true
      try {
        const { data } = await axios.post(`${API_BASE}/api/stress/face/predict`, { image: shot })
        setFaces(data?.faces ?? 0)
        setElapsedMs(data?.elapsed_ms ?? null)

        if ((data?.faces ?? 0) > 0) {
          const label = data?.label || null
          const conf = typeof data?.confidence === "number" ? data.confidence : null
          bboxRef.current = data?.bbox || null
          setDetectedEmotion(label)
          setConfidence(conf)
          setStressLevel(inferStressLevel(label, conf))
        } else {
          bboxRef.current = null
          setDetectedEmotion("No face detected")
          setConfidence(null)
          setStressLevel(null)
        }
        setError("")
      } catch (e) {
        setError("Could not reach the stress detector. Is the backend running?")
      } finally {
        inFlightRef.current = false
      }
    }, CAPTURE_INTERVAL_MS)
  }

  const stopCaptureLoop = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const activateWebcam = async () => {
    setError("")
    setIsWebcamActive(true)
    // give the camera a moment to initialize before first screenshot
    setTimeout(() => startCaptureLoop(), 500)
  }

  // Cleanup on unmount or deactivation
  useEffect(() => {
    return () => {
      stopCaptureLoop()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handlePanicSOSClick = () => navigate("/panic-chatbot")
  const handleMusicTherapyClick = () => navigate("/music-therapy")

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <motion.div
        className="container mx-auto px-4 py-8 max-w-4xl"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.header variants={itemVariants} className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-4">AI Stress Detection</h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Your webcam frame is analyzed in real time to detect facial stress signals. We infer a simple stress level
            and suggest quick, supportive next steps.
          </p>
        </motion.header>

        {!isWebcamActive ? (
          <motion.div variants={itemVariants} className="flex justify-center">
            <Card>
              <CardHeader>
                <CardTitle>Live Webcam Feed</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col items-center gap-4">
                <Button
                  size="lg"
                  onClick={activateWebcam}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Camera className="mr-2 h-5 w-5" /> Activate Webcam
                </Button>
                <div className="text-sm text-muted-foreground">
                  Tip: Ensure good lighting and face the camera directly.
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <div className="flex flex-col md:flex-row gap-6">
            {/* Left: Webcam with overlay */}
            <motion.div className="w-full md:w-1/2" variants={itemVariants}>
              <Card>
                <CardHeader>
                  <CardTitle>Live Webcam Feed</CardTitle>
                </CardHeader>
                <CardContent className="flex justify-center">
                  <div className="w-full max-w-md mx-auto">
                    <div className="relative rounded-lg overflow-hidden border">
                      <Webcam
                        ref={webcamRef}
                        audio={false}
                        mirrored={true}
                        className="w-full h-auto"
                        screenshotFormat="image/jpeg"
                        // Optional: constrain the capture size for bandwidth/CPU
                        screenshotQuality={0.8}
                        videoConstraints={{
                          width: { ideal: 640 },
                          height: { ideal: 480 },
                          facingMode: "user",
                        }}
                      />
                      {/* Canvas overlay for bbox */}
                      <canvas
                        ref={overlayRef}
                        className="absolute inset-0 pointer-events-none"
                      />
                    </div>
                    {error && (
                      <div className="mt-2 text-sm text-red-600">
                        {error}
                      </div>
                    )}
                    {elapsedMs != null && (
                      <div className="mt-2 text-xs text-muted-foreground">
                        Inference time: {elapsedMs} ms {faces !== null ? `• Faces: ${faces}` : ""}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Right: Results */}
            <motion.div className="w-full md:w-1/2 flex flex-col gap-6" variants={itemVariants}>
              <Card>
                <CardHeader>
                  <CardTitle>Detected Emotion</CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <p className="text-2xl font-semibold">
                    {detectedEmotion || "Analyzing…"}
                  </p>
                  {confidence != null && detectedEmotion !== "No face detected" && (
                    <p className="text-sm text-muted-foreground mt-1">
                      Confidence: {Math.round(confidence * 100)}%
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Stress Level</CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <p className="text-2xl font-semibold">
                    {stressLevel || (detectedEmotion === "No face detected" ? "—" : "Analyzing…")}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Personalized Suggestion</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-lg">{getSuggestion()}</p>    
                  {stressLevel === "High" && (
                    <motion.div
                      className="mt-4 flex justify-center"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                      <Button
                        onClick={() => navigate("/panic-chatbot")}
                        size="lg"
                        className="bg-red-600 hover:bg-red-700 text-white mr-4"
                      >
                        <AlertCircle className="mr-2 h-5 w-5" /> Panic SOS
                      </Button>
                    </motion.div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </div>
        )}
      </motion.div>
      <Footer />
    </div>
  )
}

export default StressDetection
