import { useEffect, useRef, useState } from "react"
import axios from "axios"
import Webcam from "react-webcam"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import Navbar from "@/components/Navbar"
import Footer from "@/components/Footer"
import { Wind, Mic, MicOff, Play, Pause, Activity, Sparkles } from "lucide-react"
import { useNavigate } from "react-router-dom"

const API_BASE = "http://127.0.0.1:5000"
const CAPTURE_INTERVAL_MS = 1200 // webcam snapshot to backend
const STATUS_POLL_MS = 5000 // refresh rate for live stress
const WINDOW_SEC = 60

const PHASES = ["Inhale", "Hold", "Exhale", "Hold"]
const fmt = (n) => (typeof n === "number" ? n.toFixed(2) : "—")

export default function BreathingCoach() {
  const navigate = useNavigate()

  // State
  const webcamRef = useRef(null)
  const captureTimer = useRef(null)
  const phaseTimer = useRef(null)
  const nextPlanTimer = useRef(null)

  const [running, setRunning] = useState(false)
  const [withVoice, setWithVoice] = useState(false)
  const [phase, setPhase] = useState("—")
  const [secondsLeft, setSecondsLeft] = useState(0)
  const [cycle, setCycle] = useState(0)
  const [plan, setPlan] = useState(null)
  const [status, setStatus] = useState(null)
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState("")

  const phaseIdxRef = useRef(0)
  const cycleRef = useRef(0)
  const stopSignal = useRef(false)
  const stressEMA = useRef(0.5)

  const speak = (text) => {
    if (!withVoice || !window?.speechSynthesis) return
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 0.95
    u.pitch = 0.95
    u.volume = 0.8
    const v = window.speechSynthesis.getVoices()
    const en = v.find(x => /en-?US|en-?IN|en-?GB/i.test(x.lang)) || v[0]
    if (en) u.voice = en
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(u)
  }

  const stopSpeech = () => { try { window.speechSynthesis.cancel() } catch {} }

  // -------------- Camera + Face Loop --------------
  const startCameraLoop = () => {
    if (captureTimer.current) return
    captureTimer.current = setInterval(async () => {
      const img = webcamRef.current?.getScreenshot()
      if (!img) return
      try {
        await axios.post(`${API_BASE}/api/stress/face/predict`, { image: img })
      } catch {}
    }, CAPTURE_INTERVAL_MS)
  }

  const stopCameraLoop = () => {
    if (captureTimer.current) {
      clearInterval(captureTimer.current)
      captureTimer.current = null
    }
  }

  // -------------- Backend fetches --------------
  const fetchStatus = async () => {
    try {
      const { data } = await axios.get(`${API_BASE}/api/breath/status`)
      setStatus(data)
      if (data?.stress_smoothed !== undefined) {
        const α = 0.2
        stressEMA.current = α * data.stress_smoothed + (1 - α) * stressEMA.current
      }
    } catch {}
  }

  const fetchPlan = async () => {
    const { data } = await axios.get(`${API_BASE}/api/breath/plan?window=${WINDOW_SEC}`)
    return data
  }

  const startSession = async () => {
    try { await axios.post(`${API_BASE}/api/breath/session`, { action: "start" }) } catch {}
  }

  const stopSession = async () => {
    try { await axios.post(`${API_BASE}/api/breath/session`, { action: "stop" }) } catch {}
  }

  // -------------- Breathing Engine --------------
  const startPlan = (newPlan) => {
    if (!newPlan?.pattern?.length) return
    stopPlan()

    phaseIdxRef.current = 0
    cycleRef.current = 1
    setCycle(1)
    setPlan(newPlan)
    setPhase("Inhale")
    runPhase(newPlan, 0)
  }

  const runPhase = (p, idx) => {
    if (stopSignal.current) return
    if (!p?.pattern?.length) return

    const thisPhase = p.pattern[idx]
    const secs = Math.max(0.5, Number(thisPhase.seconds || 0))
    setPhase(PHASES[idx] || "—")
    setSecondsLeft(secs)
    setProgress(0)
    if (withVoice) speak(PHASES[idx])

    const tickRate = 100
    let elapsed = 0
    clearInterval(phaseTimer.current)
    phaseTimer.current = setInterval(() => {
      if (stopSignal.current) return
      elapsed += tickRate / 1000
      setSecondsLeft(Math.max(0, secs - elapsed))
      setProgress(elapsed / secs)
      if (elapsed >= secs) {
        clearInterval(phaseTimer.current)
        const nextIdx = (idx + 1) % 4
        if (nextIdx === 0) {
          cycleRef.current += 1
          setCycle(cycleRef.current)
          if (cycleRef.current > (p.cycles || 6)) {
            handleWindowComplete()
            return
          }
        }
        runPhase(p, nextIdx)
      }
    }, tickRate)
  }

  const stopPlan = () => {
    clearInterval(phaseTimer.current)
    clearTimeout(nextPlanTimer.current)
  }

  const handleWindowComplete = async () => {
    setMessage("Great job! Adjusting your pace...")
    speak("Great job! Adjusting your pace.")
    stopPlan()
    setTimeout(async () => {
      const np = await fetchPlan()
      setMessage("")
      startPlan(np)
    }, 3000)
  }

  const handleStart = async () => {
    stopSignal.current = false
    await startSession()
    await fetchStatus()
    const p = await fetchPlan()
    setRunning(true)
    startCameraLoop()
    startPlan(p)
  }

  const handleStop = async () => {
    stopSignal.current = true
    stopCameraLoop()
    stopPlan()
    stopSpeech()
    await stopSession()
    setRunning(false)
    setMessage("Session ended. Take a moment to relax.")
  }

  useEffect(() => {
    fetchStatus()
    const s = setInterval(fetchStatus, STATUS_POLL_MS)
    return () => clearInterval(s)
  }, [])

  useEffect(() => {
    return () => {
      stopCameraLoop()
      stopPlan()
      stopSpeech()
    }
  }, [])

  // -------------- UI --------------
  const signal = plan?.signal_quality || status?.signal_quality || "ok"
  const mode = plan?.mode || "—"
  const stress = status?.stress_smoothed ?? plan?.stress_smoothed ?? 0.5

  const colorMap = {
    Calm: "from-green-300 via-emerald-300 to-teal-300",
    Focus: "from-blue-300 via-cyan-300 to-sky-300",
    "Wind-down": "from-indigo-300 via-purple-300 to-violet-300",
    Relief: "from-rose-300 via-pink-300 to-orange-300",
  }
  const gradient = colorMap[mode] || "from-emerald-300 via-cyan-300 to-blue-300"

  const scale = phase === "Inhale"
    ? 0.6 + 0.4 * progress
    : phase === "Exhale"
      ? 1.0 - 0.4 * progress
      : 1.0

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <motion.div className="container mx-auto px-4 py-8 max-w-6xl" initial={{opacity:0}} animate={{opacity:1}}>
        <header className="text-center mb-6">
          <h1 className="text-3xl font-bold mb-2">AI Breathing Coach</h1>
          <p className="text-gray-600 dark:text-gray-300">
            Adaptive breathing guided by your facial stress signal. Long exhale, soft jaw, lowered shoulders.
          </p>
        </header>

        {/* Hidden webcam */}
        <div className="hidden">
          <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg" videoConstraints={{ facingMode: "user" }}/>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left panel */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Wind className="h-5 w-5" /> Guided Session
              </CardTitle>
              <div className="flex gap-2">
                <Button
                  variant={withVoice ? "default" : "outline"}
                  onClick={() => setWithVoice(v => !v)}
                  className={withVoice ? "bg-emerald-600 text-white hover:bg-emerald-700" : ""}
                >
                  {withVoice ? <Mic className="h-4 w-4 mr-1"/> : <MicOff className="h-4 w-4 mr-1"/>}
                  {withVoice ? "Voice On" : "Voice Off"}
                </Button>
                {!running ? (
                  <Button onClick={handleStart} className="bg-blue-600 text-white">
                    <Play className="h-4 w-4 mr-1"/> Start
                  </Button>
                ) : (
                  <Button variant="destructive" onClick={handleStop}>
                    <Pause className="h-4 w-4 mr-1"/> Stop
                  </Button>
                )}
              </div>
            </CardHeader>

            <CardContent>
              <div className="flex flex-col items-center">
                {/* Animated Circle */}
                <motion.div
                  className={`relative h-64 w-64 rounded-full bg-gradient-to-br ${gradient} border shadow-xl`}
                  animate={{ scale }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                >
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="text-lg font-semibold">{phase}</div>
                    <div className="text-sm text-muted-foreground">{Math.ceil(secondsLeft)}s</div>
                  </div>
                </motion.div>

                {/* Stats */}
                <div className="mt-4 grid grid-cols-3 gap-3 w-full max-w-sm text-center">
                  <Stat label="Mode" value={mode}/>
                  <Stat label="Cycle" value={cycle <= (plan?.cycles || 6) ? `${cycle}/${plan?.cycles || 6}` : "Done"}/>
                  <Stat label="Signal" value={signal}/>
                </div>

                <div className="mt-3 text-center text-sm text-gray-700 dark:text-gray-300">
                  <em>{message || plan?.affirmation || "Breathe gently. Let the jaw unclench."}</em>
                </div>

                <div className="mt-5 flex gap-3">
                  <Button variant="outline" onClick={() => navigate("/stress-detection")}>
                    <Activity className="h-4 w-4 mr-1"/> Back to Face View
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Right panel */}
          <div className="flex flex-col gap-6">
            <Card>
              <CardHeader><CardTitle>Live Stress & Plan</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <Row k="Smoothed Stress" v={fmt(stress)}/>
                <Row k="Mode" v={mode}/>
                <Row k="Signal Quality" v={signal}/>
                <Row k="Window" v={`${plan?.window_sec || WINDOW_SEC}s`}/>
                <Row k="Notes" v={plan?.notes || "—"}/>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Relaxation Tools</CardTitle></CardHeader>
              <CardContent className="text-sm space-y-3">
                <p className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-emerald-500"/> 
                  Try <strong>5–4–3–2–1 grounding:</strong> name 5 things you see, 4 feel, 3 hear, 2 smell, 1 taste.
                </p>
                <p>
                  Tap <strong>Voice On</strong> to hear gentle cues for each phase, or close your eyes and follow the rhythm.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </motion.div>
      <Footer />
    </div>
  )
}

function Row({k,v}) {
  return (
    <div className="flex items-center justify-between border rounded-md px-3 py-2">
      <div className="text-xs text-muted-foreground">{k}</div>
      <div className="text-sm font-medium">{v}</div>
    </div>
  )
}

function Stat({label, value}) {
  return (
    <div className="p-3 rounded-lg border">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-base font-medium">{value}</div>
    </div>
  )
}
