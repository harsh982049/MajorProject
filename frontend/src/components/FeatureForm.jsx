import { useState } from "react"

function FeatureForm({ features, onSubmit }) {
  const [formData, setFormData] = useState({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleInputChange = (featureName, value) => {
    // Coerce to number, default to 0 for invalid values
    const numValue = Number.parseFloat(value) || 0
    // Ensure non-negative values
    const finalValue = Math.max(0, numValue)

    setFormData((prev) => ({
      ...prev,
      [featureName]: finalValue,
    }))
  }

  const populateTypicalActivity = () => {
    const typicalValues = {
      ks_keydowns: 45,
      ks_mean_dwell_ms: 95,
      ks_std_dwell_ms: 25,
      mouse_move_count: 320,
      mouse_total_distance_px: 8500,
      mouse_click_count: 12,
      mouse_scroll_count: 8,
      active_seconds_fraction: 0.85,
      window_switches: 3,
      typing_speed_wpm: 65,
      pause_count: 8,
      long_pause_count: 2,
      backspace_ratio: 0.08,
      mouse_velocity_mean: 180,
      mouse_acceleration_mean: 45,
      click_pressure_mean: 0.7,
      dwell_rhythm_variance: 0.15,
      typing_burst_count: 6,
      idle_time_fraction: 0.12,
      multitask_switches: 2,
      error_correction_rate: 0.05,
      focus_duration_mean: 25,
      interaction_density: 0.75,
      cognitive_load_proxy: 0.6,
      stress_indicator_composite: 0.35,
      fatigue_proxy: 0.25,
      attention_stability: 0.8,
    }

    const newFormData = {}
    features.forEach((feature) => {
      newFormData[feature] = typicalValues[feature] || Math.random() * 100
    })

    setFormData(newFormData)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsSubmitting(true)

    try {
      // Create payload with all features, defaulting missing ones to 0
      const payload = {}
      features.forEach((feature) => {
        payload[feature] = formData[feature] || 0
      })

      await onSubmit(payload)
    } finally {
      setIsSubmitting(false)
    }
  }

  const formatFeatureName = (name) => {
    return name
      .replace(/_/g, " ")
      .replace(/\b\w/g, (l) => l.toUpperCase())
      .replace(/Ms$/, "(ms)")
      .replace(/Px$/, "(px)")
      .replace(/Wpm$/, "(wpm)")
  }

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <h3 style={{ color: "var(--gray-800)" }}>Feature Input</h3>
        <div
          style={{
            fontSize: "12px",
            color: "var(--gray-500)",
            cursor: "help",
            padding: "4px 8px",
            background: "var(--gray-100)",
            borderRadius: "var(--border-radius-sm)",
          }}
          title="All inputs represent 30-second aggregated behavior data"
        >
          ℹ️ 30s aggregates
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {features.map((feature) => (
            <div key={feature} className="form-group">
              <label htmlFor={feature}>{formatFeatureName(feature)}</label>
              <input
                id={feature}
                type="number"
                step="any"
                min="0"
                value={formData[feature] || ""}
                onChange={(e) => handleInputChange(feature, e.target.value)}
                placeholder="0"
              />
            </div>
          ))}
        </div>

        <div className="button-group">
          <button type="button" className="btn btn-secondary" onClick={populateTypicalActivity} disabled={isSubmitting}>
            Populate with Typical Activity
          </button>

          <button type="submit" className="btn btn-primary" disabled={isSubmitting || features.length === 0}>
            {isSubmitting && <div className="loading-spinner"></div>}
            Send Window
          </button>
        </div>
      </form>

      {features.length === 0 && (
        <div className="empty-state">
          <p>No features available. Check service connection.</p>
        </div>
      )}
    </div>
  )
}

export default FeatureForm
