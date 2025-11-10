// File: src/pages/FocusCompanion.jsx
import { useState, useMemo } from "react"
import axios from "axios"
import { motion } from "framer-motion"
import Navbar from "@/components/Navbar"
import Footer from "@/components/Footer"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Calendar, Send, Loader2, AlarmClock, Clock } from "lucide-react"

const API_BASE = "http://127.0.0.1:5000" // keep consistent with your other pages

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.2 } },
}
const itemVariants = {
  hidden: { y: 12, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { type: "spring", stiffness: 110 } },
}

function toLocal(dt) {
  try {
    const d = new Date(dt)
    const date = d.toLocaleDateString([], { year: "numeric", month: "short", day: "2-digit" })
    const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    return `${date} • ${time}`
  } catch {
    return dt
  }
}

function minutesBetween(aIso, bIso) {
  try {
    const a = new Date(aIso).getTime()
    const b = new Date(bIso).getTime()
    return Math.max(1, Math.round((b - a) / 60000))
  } catch {
    return null
  }
}

export default function FocusCompanion() {
  // --- form state
  const [userEmail, setUserEmail] = useState("")
  const [title, setTitle] = useState("")
  const [description, setDescription] = useState("")
  const [timebox, setTimebox] = useState(45) // minutes
  const [deadline, setDeadline] = useState("") // HTML datetime-local string

  // --- results / UI
  const [loading, setLoading] = useState(false)
  const [createdTaskId, setCreatedTaskId] = useState(null)
  const [subtasks, setSubtasks] = useState([])
  const [error, setError] = useState("")

  const totalPlanned = useMemo(() => {
    if (!subtasks?.length) return 0
    return subtasks.reduce((acc, s) => acc + (minutesBetween(s.planned_start_ts, s.planned_end_ts) || 0), 0)
  }, [subtasks])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError("")
    if (!userEmail || !title) {
      setError("Please enter your email and a task title.")
      return
    }
    setLoading(true)
    try {
      const payload = {
        user_email: userEmail.trim(),
        title: title.trim(),
        description: description.trim() || undefined,
        timebox_min: Number(timebox) || 45,
        deadline_ts: deadline ? new Date(deadline).toISOString() : undefined,
        constraints: {},
      }
      const { data } = await axios.post(`${API_BASE}/api/focus/task/create`, payload)
      setCreatedTaskId(data.task_id)
      setSubtasks(data.subtasks || [])
    } catch (err) {
      setError(err?.response?.data?.error || "Could not create/schedule the task. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <motion.main
        className="container mx-auto px-4 py-8 max-w-5xl"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.header variants={itemVariants} className="mb-8 text-center">
          <h1 className="text-3xl font-bold">Focus Companion — Planner · Scheduler · Notifier</h1>
          <p className="text-muted-foreground mt-2">
            Turn a big goal into 3–5 atomic sub-tasks, schedule them within your work hours,
            and receive email magic-links to <b>Start / Done / Snooze / Blocked</b> from anywhere.
          </p>
        </motion.header>

        {/* Creator */}
        <motion.section variants={itemVariants} className="mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Create & Schedule a Focus Task</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="grid gap-4 md:grid-cols-2">
                <div className="md:col-span-1">
                  <label className="text-sm font-medium">Your Email</label>
                  <Input
                    type="email"
                    placeholder="you@example.com"
                    value={userEmail}
                    onChange={e => setUserEmail(e.target.value)}
                    required
                  />
                </div>

                <div className="md:col-span-1">
                  <label className="text-sm font-medium">Timebox (minutes)</label>
                  <Input
                    type="number"
                    min={15}
                    step={5}
                    value={timebox}
                    onChange={e => setTimebox(e.target.value)}
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="text-sm font-medium">Task Title / Goal</label>
                  <Input
                    placeholder="e.g., Prepare 2-slide demo deck"
                    value={title}
                    onChange={e => setTitle(e.target.value)}
                    required
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="text-sm font-medium">Description (optional)</label>
                  <Textarea
                    placeholder="Any context or constraints to consider…"
                    value={description}
                    onChange={e => setDescription(e.target.value)}
                    rows={3}
                  />
                </div>

                <div className="md:col-span-1">
                  <label className="text-sm font-medium">Deadline (optional)</label>
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-muted-foreground" />
                    <Input
                      type="datetime-local"
                      value={deadline}
                      onChange={e => setDeadline(e.target.value)}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    If omitted, subtasks start from “now” within your work hours.
                  </p>
                </div>

                <div className="md:col-span-1 flex items-end">
                  <Button
                    type="submit"
                    disabled={loading}
                    className="w-full md:w-auto"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Planning & Scheduling…
                      </>
                    ) : (
                      <>
                        <Send className="mr-2 h-4 w-4" /> Create Plan
                      </>
                    )}
                  </Button>
                </div>

                {error && (
                  <div className="md:col-span-2 text-red-600 text-sm">{error}</div>
                )}
              </form>
            </CardContent>
          </Card>
        </motion.section>

        {/* Results */}
        {subtasks?.length > 0 && (
          <motion.section variants={itemVariants} className="mb-8">
            <Card>
              <CardHeader className="flex flex-col md:flex-row md:items-center md:justify-between">
                <CardTitle>Today’s Scheduled Sub-tasks</CardTitle>
                <div className="text-sm text-muted-foreground mt-2 md:mt-0">
                  <span className="inline-flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    Planned total:&nbsp;<b>{totalPlanned}</b>&nbsp;min
                  </span>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {subtasks.map((s) => (
                  <div
                    key={s.id}
                    className="flex flex-col md:flex-row md:items-center md:justify-between rounded-lg border p-4"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">Step {s.idx}</Badge>
                        <span className="font-medium">{s.title}</span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        <span className="inline-flex items-center gap-1 mr-3">
                          <AlarmClock className="w-3.5 h-3.5" />
                          {toLocal(s.planned_start_ts)} → {toLocal(s.planned_end_ts)}
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5" />
                          {minutesBetween(s.planned_start_ts, s.planned_end_ts)} min
                        </span>
                      </div>
                    </div>
                    <div className="mt-3 md:mt-0 text-xs text-muted-foreground">
                      You’ll receive an email with one-tap links:
                      <span className="ml-1 font-medium">Start</span> · <span className="font-medium">Done</span> · <span className="font-medium">Snooze</span> · <span className="font-medium">Blocked</span>
                    </div>
                  </div>
                ))}

                <p className="text-xs text-muted-foreground pt-2">
                  Tip: if you’re running locally, ensure your backend emails can reach you (SMTP/App Password). For public callbacks in demo, set <code>PUBLIC_BASE_URL</code> to an <b>ngrok</b> URL.
                </p>
              </CardContent>
            </Card>
          </motion.section>
        )}

        {/* How it works */}
        <motion.section variants={itemVariants} className="mb-4">
          <Card>
            <CardHeader>
              <CardTitle>How this works</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-2">
              <p>
                When you create a task, the backend uses <b>Gemini 2.5 Flash</b> to produce 3–5 atomic steps (with
                a “definition of done”) and <b>schedules</b> them within your work hours. At each planned start
                time, you’ll get an email with magic links to change status from anywhere.
              </p>
              <ul className="list-disc ml-5">
                <li><b>Blocked</b> splits the step into 2–3 micro-steps and <b>reschedules</b> instantly.</li>
                <li><b>Snooze</b> shifts the step forward (downstream steps adjust to avoid overlap).</li>
                <li><b>Done</b> marks completion—future steps may pull forward depending on policy.</li>
              </ul>
            </CardContent>
          </Card>
        </motion.section>
      </motion.main>
      <Footer />
    </div>
  )
}
