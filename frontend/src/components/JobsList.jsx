/**
 * JobsList.jsx
 * Sidebar list of transcription jobs with status badges.
 * Clicking a job selects it and displays its transcript in the detail pane.
 */

const STATUS_BADGE = {
  queued:     { label: 'Queued',     color: '#f59e0b', icon: '⏳' },
  processing: { label: 'Processing', color: '#6366f1', icon: '⚙️' },
  completed:  { label: 'Done',       color: '#10b981', icon: '✓' },
  failed:     { label: 'Failed',     color: '#ef4444', icon: '✗' },
};

function formatTime(iso) {
  if (!iso) return '';
  // Show just HH:MM:SS from "YYYY-MM-DD HH:MM:SS"
  const parts = iso.split(' ');
  return parts[1] || iso;
}

export default function JobsList({ jobs, selectedJobId, onSelect, onDelete, onResubmit, onDeleteAll }) {
  const jobEntries = Object.entries(jobs).sort((a, b) => {
    // Sort by created_at descending (newest first)
    const ta = parseFloat(a[1].created_at || '0');
    const tb = parseFloat(b[1].created_at || '0');
    return tb - ta;
  });

  if (jobEntries.length === 0) {
    return (
      <div className="jobs-list jobs-list--empty">
        <p className="jobs-empty-hint">No jobs yet. Upload audio to get started.</p>
      </div>
    );
  }

  return (
    <div className="jobs-list">
      <div className="jobs-list-header">
        <h3 className="jobs-list-title">Jobs</h3>
        {onDeleteAll && (
          <button
            className="jobs-action-btn jobs-action-btn--delete"
            title="Clear all jobs"
            onClick={onDeleteAll}
          >
            Clear All
          </button>
        )}
      </div>
      {(() => {
        const total = jobEntries.length;
        const done = jobEntries.filter(([, j]) => j.status === 'completed' || j.status === 'failed').length;
        const pct = Math.round((done / total) * 100);
        return (
          <div className="jobs-progress">
            <div className="jobs-progress-bar">
              <div className="jobs-progress-fill" style={{ width: `${pct}%` }} />
            </div>
            <span className="jobs-progress-label">{done} / {total} completed ({pct}%)</span>
          </div>
        );
      })()}
      <ul className="jobs-list-items">
        {jobEntries.map(([id, job]) => {
          const badge = STATUS_BADGE[job.status] || STATUS_BADGE.queued;
          const isSelected = id === selectedJobId;
          return (
            <li
              key={id}
              className={`jobs-list-item ${isSelected ? 'jobs-list-item--selected' : ''}`}
              onClick={() => onSelect(id)}
            >
              <div className="jobs-item-top">
                <span className="jobs-item-filename" title={job.original_filename || job.filename}>
                  {job.original_filename || job.filename || id.slice(0, 8)}
                </span>
                <span
                  className="jobs-item-badge"
                  style={{ backgroundColor: badge.color }}
                  title={badge.label}
                >
                  {badge.icon} {badge.label}
                </span>
              </div>
              <div className="jobs-item-meta">
                <span className="jobs-item-time">{formatTime(job.created_at_iso)}</span>
                {job.processing_time_s && (
                  <span className="jobs-item-duration">{job.processing_time_s}s</span>
                )}
              </div>
              <div className="jobs-item-actions">
                {(job.status === 'failed' || job.status === 'completed') && (
                  <button
                    className="jobs-action-btn"
                    title="Re-submit"
                    onClick={(e) => { e.stopPropagation(); onResubmit?.(id); }}
                  >
                    🔄
                  </button>
                )}
                <button
                  className="jobs-action-btn jobs-action-btn--delete"
                  title="Delete"
                  onClick={(e) => { e.stopPropagation(); onDelete?.(id); }}
                >
                  🗑
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
