function Header({ serviceHealth, userId, onUserIdChange }) {
  const getHealthStatus = () => {
    if (!serviceHealth) return { className: "error", text: "Unknown" }
    if (serviceHealth.ok) return { className: "healthy", text: "Healthy" }
    return { className: "error", text: "Error" }
  }

  const healthStatus = getHealthStatus()

  return (
    <header className="header">
      <div className="header-content">
        <h1>Behavior Stress Monitor</h1>
        <p>Real-time stress inference from keyboard and mouse behavior</p>
      </div>

      <div className="header-controls">
        <div>
          <label htmlFor="userId" style={{ fontSize: "12px", color: "var(--gray-600)", marginRight: "8px" }}>
            User ID:
          </label>
          <input
            id="userId"
            type="text"
            value={userId}
            onChange={(e) => onUserIdChange(e.target.value)}
            className="user-input"
            placeholder="Enter user ID"
          />
        </div>

        <div className={`status-pill ${healthStatus.className}`}>Service: {healthStatus.text}</div>
      </div>
    </header>
  )
}

export default Header
