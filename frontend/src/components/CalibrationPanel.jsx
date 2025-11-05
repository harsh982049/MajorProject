import { useState } from "react"

function CalibrationPanel({ hasCalibrator, onCalibrate }) {
  const [minRows, setMinRows] = useState(200)
  const [isCalibrating, setIsCalibrating] = useState(false)
  const [calibrationHistory, setCalibrationHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("stress-monitor-calibration-history")
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  const handleCalibrate = async () => {
    setIsCalibrating(true)
    try {
      await onCalibrate(minRows)
      const newEntry = {
        timestamp: new Date().toISOString(),
        minRows,
        userId: localStorage.getItem("stress-monitor-user-id") || "unknown",
      }
      const updatedHistory = [newEntry, ...calibrationHistory.slice(0, 9)]
      setCalibrationHistory(updatedHistory)
      localStorage.setItem("stress-monitor-calibration-history", JSON.stringify(updatedHistory))
    } finally {
      setIsCalibrating(false)
    }
  }

  const clearHistory = () => {
    setCalibrationHistory([])
    localStorage.removeItem("stress-monitor-calibration-history")
  }

  return (
    <div className="card">
      <h3 style={{ marginBottom: "16px", color: "var(--gray-800)" }}>Calibration</h3>

      <div className="calibration-status">
        <div className={`calibration-indicator ${hasCalibrator ? "active" : "inactive"}`}></div>
        <span style={{ fontSize: "14px", color: "var(--gray-700)" }}>
          Calibrator: {hasCalibrator ? "Active" : "Not Set"}
        </span>
      </div>

      <div style={{ marginBottom: "16px" }}>
        <label
          htmlFor="minRows"
          style={{
            display: "block",
            fontSize: "12px",
            fontWeight: "600",
            color: "var(--gray-700)",
            marginBottom: "4px",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
          }}
        >
          Minimum Rows
        </label>
        <input
          id="minRows"
          type="number"
          value={minRows}
          onChange={(e) => setMinRows(Number.parseInt(e.target.value) || 200)}
          min="1"
          max="10000"
          style={{
            width: "100%",
            padding: "10px 12px",
            border: "1px solid var(--gray-300)",
            borderRadius: "var(--border-radius-sm)",
            fontSize: "14px",
          }}
        />
      </div>

      <button
        className="btn btn-success"
        onClick={handleCalibrate}
        disabled={isCalibrating}
        style={{ width: "100%", marginBottom: "16px" }}
      >
        {isCalibrating && <div className="loading-spinner"></div>}
        Train Calibrator
      </button>

      {calibrationHistory.length > 0 && (
        <div style={{ marginTop: "16px", paddingTop: "16px", borderTop: "1px solid var(--gray-200)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <h4 style={{ fontSize: "12px", fontWeight: "600", color: "var(--gray-700)", textTransform: "uppercase" }}>
              Recent Calibrations
            </h4>
            <button
              onClick={clearHistory}
              style={{
                background: "none",
                border: "none",
                color: "var(--gray-500)",
                fontSize: "12px",
                cursor: "pointer",
                textDecoration: "underline",
              }}
            >
              Clear
            </button>
          </div>
          <div style={{ maxHeight: "120px", overflowY: "auto" }}>
            {calibrationHistory.map((entry, index) => (
              <div
                key={index}
                style={{
                  fontSize: "12px",
                  color: "var(--gray-600)",
                  padding: "4px 0",
                  borderBottom: index < calibrationHistory.length - 1 ? "1px solid var(--gray-100)" : "none",
                }}
              >
                <div>{new Date(entry.timestamp).toLocaleString()}</div>
                <div style={{ color: "var(--gray-500)" }}>
                  {entry.minRows} rows • {entry.userId}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <p
        style={{
          fontSize: "12px",
          color: "var(--gray-500)",
          marginTop: "12px",
          lineHeight: "1.4",
        }}
      >
        Calibration improves prediction accuracy by learning user-specific patterns from historical data.
        {hasCalibrator && " Current model is using personalized calibration."}
      </p>
    </div>
  )
}

export default CalibrationPanel
