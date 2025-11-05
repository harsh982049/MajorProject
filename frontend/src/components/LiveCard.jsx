import { Area, AreaChart, XAxis, YAxis, ResponsiveContainer } from "recharts"

function LiveCard({ prediction, recentWindows, autoUpdate, onAutoUpdateChange }) {
  const getStressLevel = (smoothedProb) => {
    if (!smoothedProb) return { level: "Unknown", className: "unknown" }

    const percent = smoothedProb * 100
    if (percent < 40) return { level: "Calm", className: "calm" }
    if (percent <= 60) return { level: "Elevated", className: "elevated" }
    return { level: "High", className: "high" }
  }

  const formatPercent = (value) => {
    return Math.round((value || 0) * 100)
  }

  const stressScore = formatPercent(prediction?.smoothed_prob)
  const stressLevel = getStressLevel(prediction?.smoothed_prob)

  // Prepare sparkline data (last 20 windows)
  const sparklineData = recentWindows
    .slice(0, 20)
    .reverse()
    .map((window, index) => ({
      index,
      value: window.smoothed_prob * 100,
      timestamp: window.timestamp,
    }))

  return (
    <div className="live-card">
      <h2 style={{ fontSize: "24px", fontWeight: "600", marginBottom: "8px" }}>Current Stress Level</h2>

      <div className="stress-score">{stressScore}%</div>

      <div className={`stress-status ${stressLevel.className}`}>{stressLevel.level}</div>

      {prediction && (
        <div className="stress-details">
          Raw: {formatPercent(prediction.raw_prob)}% • Calibrated: {formatPercent(prediction.calibrated_prob)}% •
          Threshold: {formatPercent(prediction.threshold_used)}% • Calibrator:{" "}
          {prediction.has_calibrator ? "Active" : "None"}
        </div>
      )}

      {sparklineData.length > 0 && (
        <div className="sparkline-container">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={sparklineData}>
              <defs>
                <linearGradient id="stressGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="rgba(255,255,255,0.8)" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="rgba(255,255,255,0.2)" stopOpacity={0.2} />
                </linearGradient>
              </defs>
              <XAxis hide />
              <YAxis hide domain={[0, 100]} />
              <Area
                type="monotone"
                dataKey="value"
                stroke="rgba(255,255,255,0.9)"
                strokeWidth={2}
                fill="url(#stressGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="auto-update-toggle">
        <span style={{ fontSize: "14px", opacity: 0.9 }}>Auto Update (30s)</span>
        <div className={`toggle-switch ${autoUpdate ? "active" : ""}`} onClick={() => onAutoUpdateChange(!autoUpdate)}>
          <div className="toggle-slider"></div>
        </div>
      </div>
    </div>
  )
}

export default LiveCard
