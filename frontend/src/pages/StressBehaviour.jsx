import { useEffect, useMemo, useRef, useState, useCallback } from "react"
import Navbar from "@/components/Navbar"
import Footer from "@/components/Footer"
import { api } from "@/lib/api" // your axios wrapper

// shadcn/ui
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Progress } from "@/components/ui/progress"

// charts
import { ResponsiveContainer, LineChart, Line, YAxis, XAxis, Tooltip as RTooltip } from "recharts"

// icons
import { Activity, RefreshCw, AlertTriangle, Play, Pause, Keyboard } from "lucide-react"

const WINDOW_MS = 30_000
const LS_WINDOWS = "stress-behavior:windows"
const LS_AUTO = "stress-behavior:auto"
const LS_TRACK = "stress-behavior:track"

function pct(n) { return Math.round(Number(n || 0) * 100) }
function statusFor(p) { const v = Number(p || 0); if (v > 0.6) return { label: "High", variant: "destructive" }; if (v >= 0.4) return { label: "Elevated", variant: "secondary" }; return { label: "Calm", variant: "default" } }
function quantile(arr, q) { if (!arr?.length) return 0; const a = [...arr].sort((x,y)=>x-y); const pos=(a.length-1)*q; const b=Math.floor(pos); const r=pos-b; return a[b+1]!==undefined ? a[b] + r*(a[b+1]-a[b]) : a[b] }

// =============== TRACKER HOOK (17 features) + DEBUG ===============
function useBehaviorTracker(enabled, setDebug) {
  const ref = useRef({
    // keyboard
    keydowns: 0,
    keyups: 0,
    eventCount: 0,
    uniqueKeys: new Set(),
    downTimes: new Map(),     // code -> t
    dwellMs: [],
    keydownTimes: [],
    ikgMs: [],
    lastKey: "",

    // mouse
    moveCount: 0,
    clickCount: 0,
    scrollCount: 0,
    lastMouse: null,          // {x,y,t}
    totalDist: 0,
    speeds: [],
    maxSpeed: 0,

    // activity
    activeSeconds: new Set(),

    windowStart: performance.now(),
  })

  const markActive = () => { ref.current.activeSeconds.add(Math.floor(Date.now() / 1000)) }

  const resetWindow = useCallback(() => {
    const now = performance.now()
    Object.assign(ref.current, {
      keydowns: 0, keyups: 0, eventCount: 0,
      uniqueKeys: new Set(),
      downTimes: new Map(),
      dwellMs: [],
      keydownTimes: [],
      ikgMs: [],
      lastKey: "",
      moveCount: 0, clickCount: 0, scrollCount: 0,
      lastMouse: null,
      totalDist: 0,
      speeds: [],
      maxSpeed: 0,
      activeSeconds: new Set(),
      windowStart: now,
    })
    setDebug?.((d)=>({ ...d, lastKey: "", kd:0, ku:0, moves:0, clicks:0, scrolls:0 }))
  }, [setDebug])

  const updateDebug = () => {
    const s = ref.current
    setDebug?.((d)=>({
      ...d,
      kd: s.keydowns,
      ku: s.keyups,
      moves: s.moveCount,
      clicks: s.clickCount,
      scrolls: s.scrollCount,
      lastKey: s.lastKey
    }))
  }

  // KEYBOARD — listen on BOTH document (capture) and window (bubble)
  const onKeyDownDoc = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    s.eventCount += 1
    s.keydowns += 1
    const code = e.code || e.key || "Unknown"
    s.lastKey = `↓ ${code}`
    s.uniqueKeys.add(code)
    const now = performance.now()
    const lastKd = s.keydownTimes.length ? s.keydownTimes[s.keydownTimes.length - 1] : null
    if (lastKd != null) {
      const gap = now - lastKd
      if (gap >= 0 && isFinite(gap)) s.ikgMs.push(gap)
    }
    s.keydownTimes.push(now)
    if (!s.downTimes.has(code)) s.downTimes.set(code, now)
    markActive()
    updateDebug()
  }, [enabled])

  const onKeyUpDoc = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    s.eventCount += 1
    s.keyups += 1
    const code = e.code || e.key || "Unknown"
    s.lastKey = `↑ ${code}`
    const t0 = s.downTimes.get(code)
    if (typeof t0 === "number") {
      const dwell = performance.now() - t0
      if (dwell >= 0 && isFinite(dwell)) s.dwellMs.push(dwell)
      s.downTimes.delete(code)
    }
    markActive()
    updateDebug()
  }, [enabled])

  const onKeyDownWin = useCallback((e)=>onKeyDownDoc(e), [onKeyDownDoc])
  const onKeyUpWin   = useCallback((e)=>onKeyUpDoc(e), [onKeyUpDoc])

  // MOUSE
  const onMouseMove = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    const now = performance.now()
    const last = s.lastMouse
    if (last) {
      const dx = e.clientX - last.x
      const dy = e.clientY - last.y
      const dist = Math.hypot(dx, dy)
      if (dist < 2) return // jitter guard
      s.moveCount += 1
      const dt = (now - last.t) / 1000
      if (isFinite(dist)) s.totalDist += dist
      if (dt > 0 && isFinite(dist)) {
        const speed = dist / dt
        if (isFinite(speed)) {
          s.speeds.push(speed)
          if (speed > s.maxSpeed) s.maxSpeed = speed
        }
      }
    } else {
      s.moveCount += 1
    }
    s.lastMouse = { x: e.clientX, y: e.clientY, t: now }
    markActive()
    updateDebug()
  }, [enabled])

  const onMouseDown = useCallback(() => {
    if (!enabled) return
    ref.current.clickCount += 1
    markActive()
    updateDebug()
  }, [enabled])

  const onWheel = useCallback(() => {
    if (!enabled) return
    ref.current.scrollCount += 1
    markActive()
    updateDebug()
  }, [enabled])

  // Attach listeners (document capture + window bubble)
  useEffect(() => {
    if (!enabled) return
    resetWindow()

    // keyboard
    document.addEventListener("keydown", onKeyDownDoc, { capture: true })
    document.addEventListener("keyup", onKeyUpDoc, { capture: true })
    window.addEventListener("keydown", onKeyDownWin)
    window.addEventListener("keyup", onKeyUpWin)

    // mouse
    window.addEventListener("mousemove", onMouseMove, { passive: true })
    document.addEventListener("mousedown", onMouseDown, { capture: true })
    document.addEventListener("wheel", onWheel, { capture: true })

    return () => {
      document.removeEventListener("keydown", onKeyDownDoc, { capture: true })
      document.removeEventListener("keyup", onKeyUpDoc, { capture: true })
      window.removeEventListener("keydown", onKeyDownWin)
      window.removeEventListener("keyup", onKeyUpWin)
      window.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mousedown", onMouseDown, { capture: true })
      document.removeEventListener("wheel", onWheel, { capture: true })
    }
  }, [enabled, onKeyDownDoc, onKeyUpDoc, onKeyDownWin, onKeyUpWin, onMouseMove, onMouseDown, onWheel, resetWindow])

  const snapshotFeatures = useCallback(() => {
    const s = ref.current
    const now = performance.now()
    const elapsedS = Math.max(0.001, (now - s.windowStart) / 1000)

    const ks_mean_dwell_ms = s.dwellMs.length ? s.dwellMs.reduce((a,b)=>a+b,0) / s.dwellMs.length : 0
    const ks_median_dwell_ms = quantile(s.dwellMs, 0.5)
    const ks_p95_dwell_ms = quantile(s.dwellMs, 0.95)

    const ks_mean_ikg_ms = s.ikgMs.length ? s.ikgMs.reduce((a,b)=>a+b,0) / s.ikgMs.length : 0
    const ks_median_ikg_ms = quantile(s.ikgMs, 0.5)
    const ks_p95_ikg_ms = quantile(s.ikgMs, 0.95)

    const mouse_mean_speed_px_s = s.speeds.length ? s.speeds.reduce((a,b)=>a+b,0) / s.speeds.length : 0
    const mouse_max_speed_px_s = s.maxSpeed

    const active_seconds_fraction = Math.min(1, s.activeSeconds.size / 30)

    return {
      ks_event_count: s.eventCount,
      ks_keydowns: s.keydowns,
      ks_keyups: s.keyups,
      ks_unique_keys: s.uniqueKeys.size,
      ks_mean_dwell_ms,
      ks_median_dwell_ms,
      ks_p95_dwell_ms,
      ks_mean_ikg_ms,
      ks_median_ikg_ms,
      ks_p95_ikg_ms,
      mouse_move_count: s.moveCount,
      mouse_click_count: s.clickCount,
      mouse_scroll_count: s.scrollCount,
      mouse_total_distance_px: s.totalDist,
      mouse_mean_speed_px_s,
      mouse_max_speed_px_s,
      active_seconds_fraction: Number(active_seconds_fraction.toFixed(3)),
    }
  }, [])

  return { snapshotFeatures, resetWindow }
}

// =================== PAGE ===================
export default function StressBehaviour() {
  const [health, setHealth] = useState(null)
  const [windows, setWindows] = useState(() => { try { return JSON.parse(localStorage.getItem(LS_WINDOWS) || "[]") } catch { return [] } })
  const [auto, setAuto] = useState(() => { try { return JSON.parse(localStorage.getItem(LS_AUTO) || "true") } catch { return true } })
  const [tracking, setTracking] = useState(() => { try { return JSON.parse(localStorage.getItem(LS_TRACK) || "true") } catch { return true } })
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState("")

  // focus/debug states
  const [hasFocus, setHasFocus] = useState(() => document.hasFocus())
  const [debug, setDebug] = useState({ kd:0, ku:0, moves:0, clicks:0, scrolls:0, lastKey:"" })

  const sendTickRef = useRef(null)

  const { snapshotFeatures, resetWindow } = useBehaviorTracker(tracking, setDebug)

  useEffect(() => { try { localStorage.setItem(LS_WINDOWS, JSON.stringify(windows.slice(0, 24))) } catch {} }, [windows])
  useEffect(() => { localStorage.setItem(LS_AUTO, JSON.stringify(auto)) }, [auto])
  useEffect(() => { localStorage.setItem(LS_TRACK, JSON.stringify(tracking)) }, [tracking])

  // focus listeners
  useEffect(() => {
    const onFocus = () => setHasFocus(true)
    const onBlur = () => setHasFocus(false)
    window.addEventListener("focus", onFocus)
    window.addEventListener("blur", onBlur)
    document.addEventListener("visibilitychange", () => setHasFocus(document.visibilityState === "visible"))
    return () => {
      window.removeEventListener("focus", onFocus)
      window.removeEventListener("blur", onBlur)
    }
  }, [])

  const loadHealth = useCallback(async () => {
    setErr("")
    try {
      const { data } = await api.get("/api/stress/behavior/health")
      setHealth(data)
    } catch (e) {
      setHealth({ ok: false })
      setErr(e?.response?.data?.error || e?.message || "Failed to reach /health")
    }
  }, [])
  useEffect(() => { loadHealth() }, [loadHealth])

  const sendWindowForPrediction = useCallback(async () => {
    setErr("")
    setBusy(true)
    try {
      const feats = snapshotFeatures()
      const { data } = await api.post("/api/stress/behavior/predict", feats)
      if (!data?.ok) throw new Error(data?.error || "Prediction failed")
      const res = data.result || {}
      const row = {
        time: new Date().toISOString(),
        raw: Number(res.raw_prob || 0),
        cal: Number(res.calibrated_prob || 0),
        smo: Number(res.smoothed_prob || 0),
        on: !!res.is_stressed,
        thresh: Number(res.threshold_used ?? 0.5),
        hasCal: !!res.has_calibrator,
        feat: feats,
      }
      setWindows((prev) => [row, ...prev].slice(0, 24))
    } catch (e) {
      setErr(e?.response?.data?.error || e?.message || "Prediction error")
    } finally {
      setBusy(false)
    }
  }, [snapshotFeatures])

  // Single 30s sender
  useEffect(() => {
    if (sendTickRef.current) clearInterval(sendTickRef.current)
    if (!tracking) return
    sendTickRef.current = setInterval(async () => {
      await sendWindowForPrediction()
      resetWindow()
    }, WINDOW_MS)
    return () => { if (sendTickRef.current) clearInterval(sendTickRef.current) }
  }, [tracking, sendWindowForPrediction, resetWindow])

  const manualSend = async () => { await sendWindowForPrediction() }
  const clearHistory = () => { setWindows([]); localStorage.removeItem(LS_WINDOWS) }

  const latest = windows[0]
  const status = statusFor(latest?.smo ?? 0)
  const chartData = useMemo(() => {
    const arr = [...windows].reverse().map((w, i) => ({ idx: i + 1, smoothed: Math.round((w.smo || 0) * 1000) / 10 }))
    return arr
  }, [windows])

  return (
    <>
      <Navbar />

      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Focus notice */}
        <div className="mb-4 flex items-center gap-2">
          <Keyboard className={`h-4 w-4 ${hasFocus ? "text-green-600" : "text-yellow-600"}`} />
          <span className="text-sm">
            Focus: <b className={hasFocus ? "text-green-600" : "text-yellow-600"}>{hasFocus ? "This tab is focused" : "Click anywhere on the page to focus this tab"}</b>
          </span>
          {!hasFocus && (
            <Button size="sm" variant="outline" className="ml-2" onClick={() => window.focus()}>
              Click to focus
            </Button>
          )}
        </div>

        {/* Header */}
        <div className="flex flex-col gap-2 mb-6">
          <div className="flex items-center gap-2">
            <Activity className="h-6 w-6 text-primary" />
            <h1 className="text-2xl md:text-3xl font-bold">Behavior Stress Monitor</h1>
            {health?.ok ? <Badge className="ml-2">Service OK</Badge> : <Badge variant="destructive" className="ml-2">Service Issue</Badge>}
          </div>
          <p className="text-sm text-muted-foreground">
            Tracks keyboard & mouse dynamics in your browser and sends <b>exactly the 17 features</b> your model expects every 30 seconds.
          </p>
        </div>

        {/* Live status */}
        <Card className="mb-6">
          <CardHeader className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="flex items-center gap-3">
              <CardTitle className="text-xl md:text-2xl">Live Status</CardTitle>
              <Badge variant={status.variant}>{status.label}</Badge>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <Label htmlFor="track" className="text-sm">Tracking</Label>
                <Switch id="track" checked={tracking} onCheckedChange={setTracking} />
                <Button size="sm" variant={tracking ? "outline" : "default"} onClick={() => setTracking(t => !t)}>
                  {tracking ? <><Pause className="h-4 w-4 mr-1" /> Stop</> : <><Play className="h-4 w-4 mr-1" /> Start</>}
                </Button>
              </div>
              <Separator orientation="vertical" className="hidden md:block h-6" />
              <Button variant="outline" size="sm" onClick={manualSend} disabled={busy}>
                <RefreshCw className="h-4 w-4 mr-1" /> Send current window
              </Button>
            </div>
          </CardHeader>

          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-[1fr_1.2fr] gap-6">
              {/* Left: big numbers */}
              <div className="space-y-4">
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl font-extrabold tracking-tight">{pct(latest?.smo ?? 0)}%</span>
                  <span className="text-muted-foreground">smoothed</span>
                </div>
                <Progress value={pct(latest?.smo ?? 0)} className="h-2" />
                <div className="grid grid-cols-2 gap-3">
                  <Card className="p-3"><div className="text-xs text-muted-foreground">Raw</div><div className="text-lg font-semibold">{pct(latest?.raw ?? 0)}%</div></Card>
                  <Card className="p-3"><div className="text-xs text-muted-foreground">Calibrated</div><div className="text-lg font-semibold">{pct(latest?.cal ?? 0)}%</div></Card>
                  <Card className="p-3"><div className="text-xs text-muted-foreground">Threshold</div><div className="text-lg font-semibold">{Math.round((latest?.thresh ?? 0.5) * 100)}%</div></Card>
                  <Card className="p-3"><div className="text-xs text-muted-foreground">Calibrator</div><div className="text-lg font-semibold">{latest?.hasCal ? "Active" : "None"}</div></Card>
                </div>

                {/* Debug row */}
                <div className="rounded-md border p-3">
                  <div className="text-xs font-semibold mb-2">Live Counters (debug)</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                    <div>Keydowns: <b>{debug.kd}</b></div>
                    <div>Keyups: <b>{debug.ku}</b></div>
                    <div>Moves: <b>{debug.moves}</b></div>
                    <div>Clicks: <b>{debug.clicks}</b></div>
                    <div>Scrolls: <b>{debug.scrolls}</b></div>
                    <div>Last key: <b>{debug.lastKey || "—"}</b></div>
                  </div>
                  {!hasFocus && (
                    <div className="mt-2 text-xs text-yellow-700">
                      Tip: This tab is not focused. Click anywhere on the page, then type again.
                    </div>
                  )}
                  {err && (
                    <div className="mt-2 flex items-center gap-2 text-sm text-red-600">
                      <AlertTriangle className="h-4 w-4" />
                      <span>{err}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Right: sparkline */}
              <div className="h-48 md:h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <YAxis hide domain={[0, 100]} />
                    <XAxis hide dataKey="idx" />
                    <RTooltip formatter={(value) => [`${value}%`, "Smoothed"]} />
                    <Line type="monotone" dataKey="smoothed" strokeWidth={2} dot={false} isAnimationActive />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent windows */}
        <Card>
          <CardHeader><CardTitle className="text-xl">Recent Windows</CardTitle></CardHeader>
          <CardContent>
            <Table>
              <TableCaption>Last {windows.length} predictions (most recent first)</TableCaption>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[160px]">Time</TableHead>
                  <TableHead className="text-right">Raw</TableHead>
                  <TableHead className="text-right">Calibrated</TableHead>
                  <TableHead className="text-right">Smoothed</TableHead>
                  <TableHead className="text-center">State</TableHead>
                  <TableHead className="text-center">Payload</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {windows.map((w, i) => (
                  <TableRow key={`${w.time}-${i}`}>
                    <TableCell>{new Date(w.time).toLocaleString()}</TableCell>
                    <TableCell className="text-right">{pct(w.raw)}%</TableCell>
                    <TableCell className="text-right">{pct(w.cal)}%</TableCell>
                    <TableCell className="text-right">{pct(w.smo)}%</TableCell>
                    <TableCell className="text-center">
                      <Badge variant={w.on ? "destructive" : "default"}>{w.on ? "Stressed" : "Calm"}</Badge>
                    </TableCell>
                    <TableCell className="text-center">
                      <Button variant="outline" size="sm" onClick={() => { try { navigator.clipboard.writeText(JSON.stringify(w.feat, null, 2)) } catch {} }}>
                        Copy JSON
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
                {!windows.length && (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center text-muted-foreground">
                      No predictions yet. Turn on <b>Tracking</b>, ensure this tab is focused, and type/move/scroll.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <div className="flex justify-end mt-4">
          <Button variant="outline" onClick={clearHistory}>Clear history</Button>
        </div>
      </div>

      <Footer />
    </>
  )
}
