function RecentTable({ windows }) {
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const formatPercent = (value) => {
    return `${Math.round(value * 100)}%`
  }

  const copyFeatures = (features) => {
    navigator.clipboard.writeText(JSON.stringify(features, null, 2))
  }

  if (windows.length === 0) {
    return (
      <div>
        <h3 style={{ marginBottom: "16px", color: "var(--gray-800)" }}>Recent Windows</h3>
        <div className="empty-state">
          <p>No prediction windows yet. Submit some feature data to get started.</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <h3 style={{ marginBottom: "16px", color: "var(--gray-800)" }}>Recent Windows</h3>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Raw</th>
              <th>Calibrated</th>
              <th>Smoothed</th>
              <th>Stressed</th>
              <th>Features</th>
            </tr>
          </thead>
          <tbody>
            {windows.map((window, index) => (
              <tr key={index}>
                <td>{formatTime(window.timestamp)}</td>
                <td>{formatPercent(window.raw_prob)}</td>
                <td>{formatPercent(window.calibrated_prob)}</td>
                <td>{formatPercent(window.smoothed_prob)}</td>
                <td>
                  <span
                    style={{
                      color: window.is_stressed ? "var(--danger-color)" : "var(--success-color)",
                      fontWeight: "600",
                    }}
                  >
                    {window.is_stressed ? "✓" : "✗"}
                  </span>
                </td>
                <td>
                  <button
                    className="btn btn-secondary"
                    onClick={() => copyFeatures(window.features)}
                    style={{
                      padding: "4px 8px",
                      fontSize: "12px",
                      minHeight: "auto",
                    }}
                    title="Copy feature JSON to clipboard"
                  >
                    Copy JSON
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default RecentTable
