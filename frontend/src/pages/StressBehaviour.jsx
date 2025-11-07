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

// ==== NEW: 10s window with 5s stride ====
const WINDOW_MS = 10_000
const STEP_MS = 5_000

const LS_WINDOWS = "stress-behavior:windows"
const LS_AUTO = "stress-behavior:auto"
const LS_TRACK = "stress-behavior:track"

function pct(n) { return Math.round(Number(n || 0) * 100) }
function statusFor(p) { const v = Number(p || 0); if (v > 0.6) return { label: "High", variant: "destructive" }; if (v >= 0.4) return { label: "Elevated", variant: "secondary" }; return { label: "Calm", variant: "default" } }
function quantile(arr, q) { if (!arr?.length) return 0; const a = [...arr].sort((x,y)=>x-y); const pos=(a.length-1)*q; const b=Math.floor(pos); const r=pos-b; return a[b+1]!==undefined ? a[b] + r*(a[b+1]-a[b]) : a[b] }

// ================= TRACKER =================
function useBehaviorTracker(enabled, setDebug) {
  const ref = useRef({
    // keyboard raw presses (timestamps are performance.now() ms)
    keyPresses: [], // { down_ts, up_ts?, code }
    keydowns_times: [], // for ikg in-window computation

    // mouse raw events
    mouseEvents: [], // [t_ms, x, y, type] type: 0 move, 1 click, 2 scroll
    lastMouse: null,

    // “activity seconds” set (epoch seconds)
    activeSeconds: new Set(),

    // last debug snapshot
    lastKey: "",
    // jitter guard
    jitterPx: 2,
  })

  const nowPerf = () => performance.now()
  const nowEpochSec = () => Math.floor(Date.now() / 1000)

  const markActive = () => {
    ref.current.activeSeconds.add(nowEpochSec())
  }

  const updateDebug = () => {
    const { keyPresses, mouseEvents, lastKey } = ref.current
    const kd = keyPresses.length
    const ku = keyPresses.filter(p => p.up_ts != null).length
    const moves = mouseEvents.filter(e => e[3] === 0).length
    const clicks = mouseEvents.filter(e => e[3] === 1).length
    const scrolls = mouseEvents.filter(e => e[3] === 2).length
    setDebug?.((d)=>({ ...d, kd, ku, moves, clicks, scrolls, lastKey }))
  }

  // prune everything older than WINDOW_MS + small buffer
  const pruneOld = useCallback((now) => {
    const cutoff = now - WINDOW_MS - 500 // 0.5s buffer
    const s = ref.current
    s.keyPresses = s.keyPresses.filter(p => (p.down_ts >= cutoff) || (p.up_ts != null && p.up_ts >= cutoff))
    s.keydowns_times = s.keydowns_times.filter(t => t >= cutoff)
    s.mouseEvents = s.mouseEvents.filter(ev => ev[0] >= cutoff)

    // keep only seconds in the last window
    const epochCut = nowEpochSec() - Math.ceil(WINDOW_MS / 1000)
    s.activeSeconds = new Set([...s.activeSeconds].filter(sec => sec >= epochCut))
  }, [])

  // ----- KEYBOARD
  const onKeyDownDoc = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    const code = e.code || e.key || "Unknown"
    s.lastKey = `↓ ${code}`
    const t = nowPerf()

    // record press
    s.keyPresses.push({ down_ts: t, up_ts: null, code })
    s.keydowns_times.push(t)

    markActive()
    updateDebug()
    pruneOld(t)
  }, [enabled, pruneOld])

  const onKeyUpDoc = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    const code = e.code || e.key || "Unknown"
    s.lastKey = `↑ ${code}`
    const t1 = nowPerf()

    // close latest open press of same code
    for (let i = s.keyPresses.length - 1; i >= 0; i--) {
      const p = s.keyPresses[i]
      if (p.up_ts == null && p.code === code) { p.up_ts = t1; break }
    }

    markActive()
    updateDebug()
    pruneOld(t1)
  }, [enabled, pruneOld])

  const onKeyDownWin = useCallback((e)=>onKeyDownDoc(e), [onKeyDownDoc])
  const onKeyUpWin   = useCallback((e)=>onKeyUpDoc(e), [onKeyUpDoc])

  // ----- MOUSE
  const onMouseMove = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    const t = nowPerf()

    // raw event
    s.mouseEvents.push([t, e.clientX, e.clientY, 0])

    // jitter guard state (just for next move dist calc)
    s.lastMouse = { x: e.clientX, y: e.clientY, t }

    markActive()
    updateDebug()
    pruneOld(t)
  }, [enabled, pruneOld])

  const onMouseDown = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    const t = nowPerf()
    s.mouseEvents.push([t, e.clientX, e.clientY, 1])
    markActive(); updateDebug(); pruneOld(t)
  }, [enabled, pruneOld])

  const onWheel = useCallback((e) => {
    if (!enabled) return
    const s = ref.current
    const t = nowPerf()
    s.mouseEvents.push([t, e.clientX ?? 0, e.clientY ?? 0, 2])
    markActive(); updateDebug(); pruneOld(t)
  }, [enabled, pruneOld])

  // Attach/detach listeners
  useEffect(() => {
    if (!enabled) return

    document.addEventListener("keydown", onKeyDownDoc, { capture: true })
    document.addEventListener("keyup", onKeyUpDoc, { capture: true })
    window.addEventListener("keydown", onKeyDownWin)
    window.addEventListener("keyup", onKeyUpWin)

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
  }, [enabled, onKeyDownDoc, onKeyUpDoc, onKeyDownWin, onKeyUpWin, onMouseMove, onMouseDown, onWheel])

  // ---------- build payload for the last WINDOW_MS ----------
  const snapshotFeatures = useCallback(() => {
    const s = ref.current
    const now = nowPerf()
    const cut = now - WINDOW_MS

    // --- keyboard window ---
    const keyPressesW = s.keyPresses.filter(p => (p.down_ts >= cut) || (p.up_ts != null && p.up_ts >= cut))
    const downsW = keyPressesW.map(p => p.down_ts).sort((a,b)=>a-b)
    const uniqueKeys = new Set(keyPressesW.map(p=>p.code))
    const keydowns = keyPressesW.filter(p => p.down_ts >= cut).length
    const keyups   = keyPressesW.filter(p => p.up_ts != null && p.up_ts >= cut).length
    const eventCount = keydowns + keyups

    // dwell list in-window
    const dwellMs = keyPressesW.map(p => Math.max((p.up_ts ?? p.down_ts) - p.down_ts, 0))
    const ks_mean_dwell_ms = dwellMs.length ? dwellMs.reduce((a,b)=>a+b,0) / dwellMs.length : 0
    const ks_median_dwell_ms = quantile(dwellMs, 0.5)
    const ks_p95_dwell_ms = quantile(dwellMs, 0.95)

    // ikg from downsW
    let ikgMs = []
    for (let i = 1; i < downsW.length; i++) {
      const g = Math.max(downsW[i] - downsW[i-1], 0)
      if (isFinite(g)) ikgMs.push(g)
    }
    const ks_mean_ikg_ms = ikgMs.length ? ikgMs.reduce((a,b)=>a+b,0) / ikgMs.length : 0
    const ks_median_ikg_ms = quantile(ikgMs, 0.5)
    const ks_p95_ikg_ms = quantile(ikgMs, 0.95)

    // --- mouse window ---
    const mev = s.mouseEvents.filter(ev => ev[0] >= cut).sort((a,b)=>a[0]-b[0])
    const moveCount = mev.filter(ev => ev[3] === 0).length
    const clickCount = mev.filter(ev => ev[3] === 1).length
    const scrollCount = mev.filter(ev => ev[3] === 2).length

    let totalDist = 0
    let speeds = []
    let maxSpeed = 0
    for (let i = 1; i < mev.length; i++) {
      const [t1, x1, y1, typ1] = mev[i]
      const [t0, x0, y0, typ0] = mev[i-1]
      const dt = (t1 - t0) / 1000
      if (typ1 === 0 || typ0 === 0) {
        const dist = Math.hypot((x1 - x0), (y1 - y0))
        if (isFinite(dist)) totalDist += dist
        if (dt > 0 && isFinite(dist)) {
          const v = dist / dt
          if (isFinite(v)) {
            speeds.push(v)
            if (v > maxSpeed) maxSpeed = v
          }
        }
      }
    }
    const mouse_mean_speed_px_s = speeds.length ? speeds.reduce((a,b)=>a+b,0)/speeds.length : 0
    const mouse_max_speed_px_s = maxSpeed

    // activity seconds in-window
    const secWindow = Math.ceil(WINDOW_MS / 1000)
    const epochCut = nowEpochSec() - secWindow
    const activeSecondsCount = [...s.activeSeconds].filter(sec => sec >= epochCut).length
    const active_seconds_fraction = Math.min(1, activeSecondsCount / secWindow)

    // fill next_down_ts for encoders
    const key_events = keyPressesW
      .filter(p => p.down_ts != null)
      .map(p => ({ down_ts: p.down_ts, up_ts: p.up_ts ?? p.down_ts }))

    key_events.sort((a,b)=>a.down_ts - b.down_ts)
    for (let i = 0; i < key_events.length; i++) {
      const nd = (i + 1 < key_events.length) ? key_events[i + 1].down_ts : key_events[i].up_ts
      key_events[i].next_down_ts = nd
    }

    const mouse_events = mev // already formatted [t,x,y,type]

    return {
      // 17 MVP features
      ks_event_count: eventCount,
      ks_keydowns: keydowns,
      ks_keyups: keyups,
      ks_unique_keys: uniqueKeys.size,
      ks_mean_dwell_ms,
      ks_median_dwell_ms,
      ks_p95_dwell_ms,
      ks_mean_ikg_ms,
      ks_median_ikg_ms,
      ks_p95_ikg_ms,
      mouse_move_count: moveCount,
      mouse_click_count: clickCount,
      mouse_scroll_count: scrollCount,
      mouse_total_distance_px: totalDist,
      mouse_mean_speed_px_s,
      mouse_max_speed_px_s,
      active_seconds_fraction: Number(active_seconds_fraction.toFixed(3)),

      // sequences for encoders
      key_events,
      mouse_events,

      // optional: force personal head
      head: "hybrid",
    }
  }, [])

  return { snapshotFeatures }
}

// =================== PAGE ===================
export default function StressBehaviour() {
  const [health, setHealth] = useState(null)
  const [windows, setWindows] = useState(() => { try { return JSON.parse(localStorage.getItem(LS_WINDOWS) || "[]") } catch { return [] } })
  const [auto, setAuto] = useState(() => { try { return JSON.parse(localStorage.getItem(LS_AUTO) || "true") } catch { return true } })
  const [tracking, setTracking] = useState(() => { try { return JSON.parse(localStorage.getItem(LS_TRACK) || "true") } catch { return true } })
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState("")
  const [hasFocus, setHasFocus] = useState(() => document.hasFocus())
  const [debug, setDebug] = useState({ kd:0, ku:0, moves:0, clicks:0, scrolls:0, lastKey:"" })
  const sendTickRef = useRef(null)

  const { snapshotFeatures } = useBehaviorTracker(tracking, setDebug)

  useEffect(() => { try { localStorage.setItem(LS_WINDOWS, JSON.stringify(windows.slice(0, 24))) } catch {} }, [windows])
  useEffect(() => { localStorage.setItem(LS_AUTO, JSON.stringify(auto)) }, [auto])
  useEffect(() => { localStorage.setItem(LS_TRACK, JSON.stringify(tracking)) }, [tracking])

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

  useEffect(() => {
    if (sendTickRef.current) clearInterval(sendTickRef.current)
    if (!tracking) return
    // 5-second stride, overlapping 10s windows
    sendTickRef.current = setInterval(async () => {
      await sendWindowForPrediction()
    }, STEP_MS)
    return () => { if (sendTickRef.current) clearInterval(sendTickRef.current) }
  }, [tracking, sendWindowForPrediction])

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
            Tracks keyboard &amp; mouse dynamics in your browser and sends <b>10-second windows</b> (with 5-second overlap) including sequences + 17 features to the hybrid model.
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
                <Switch
                  id="track"
                  checked={tracking}
                  onCheckedChange={setTracking}
                  tabIndex={-1}
                  onKeyDown={(e) => {
                    if (e.code === "Space" || e.key === " ") { e.preventDefault(); e.stopPropagation(); }
                  }}
                  onKeyUp={(e) => {
                    if (e.code === "Space" || e.key === " ") { e.preventDefault(); e.stopPropagation(); }
                  }}
                />
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
                  <div className="text-xs font-semibold mb-2">Live Counters (debug, last 10s)</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                    <div>Keydowns: <b>{debug.kd}</b></div>
                    <div>Keyups: <b>{debug.ku}</b></div>
                    <div>Moves: <b>{debug.moves}</b></div>
                    <div>Clicks: <b>{debug.clicks}</b></div>
                    <div>Scrolls: <b>{debug.scrolls}</b></div>
                    <div>Last key: <b>{debug.lastKey || "—"}</b></div>
                  </div>
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
                      No predictions yet. Turn on <b>Tracking</b>, keep this tab focused, and type/move/scroll.
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
