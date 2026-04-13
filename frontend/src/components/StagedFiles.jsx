/**
 * StagedFiles.jsx
 * Staging area for files awaiting batch submission.
 * Users add files here, then click "Submit All" to create batch jobs.
 * Clicking a file selects it for audio preview.
 */

export default function StagedFiles({ files, selectedIndex, onSelect, onRemove, onSubmit, onClear, disabled, previewUrl, uploadProgress }) {
  if (files.length === 0 && !uploadProgress) {
    return (
      <div className="staged-files staged-files--empty">
        <p className="staged-empty-hint">Upload or record audio to stage files for batch transcription.</p>
      </div>
    );
  }

  return (
    <div className="staged-files">
      <div className="staged-header">
        <h3 className="staged-title">Staged Files ({files.length})</h3>
        <button
          className="staged-clear-btn"
          onClick={onClear}
          disabled={disabled}
          title="Clear all staged files"
        >
          Clear
        </button>
      </div>
      <ul className="staged-list">
        {files.map((file, idx) => (
          <li
            key={`${file.name}-${idx}`}
            className={`staged-item${idx === selectedIndex ? ' staged-item--selected' : ''}`}
            onClick={() => onSelect?.(idx)}
          >
            <span className="staged-item-name" title={file.name}>
              📎 {file.name}
            </span>
            <span className="staged-item-size">
              {file.size < 1024 * 1024
                ? `${(file.size / 1024).toFixed(0)} KB`
                : `${(file.size / (1024 * 1024)).toFixed(1)} MB`}
            </span>
            <button
              className="staged-remove-btn"
              onClick={(e) => { e.stopPropagation(); onRemove(idx); }}
              disabled={disabled}
              title="Remove"
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
      {previewUrl && (
        <audio
          className="staged-preview-player"
          controls
          src={previewUrl}
          key={previewUrl}
        />
      )}
      {uploadProgress && (
        <div className="staged-progress">
          <div className="staged-progress-bar">
            <div
              className="staged-progress-fill"
              style={{ width: `${Math.round(((uploadProgress.queued + uploadProgress.failed) / uploadProgress.total) * 100)}%` }}
            />
          </div>
          <span className="staged-progress-label">
            {uploadProgress.queued} / {uploadProgress.total} queued
            {uploadProgress.failed > 0 && <span className="staged-progress-failed"> · {uploadProgress.failed} failed</span>}
          </span>
        </div>
      )}
      <button
        className="btn btn-primary staged-submit-btn"
        onClick={onSubmit}
        disabled={disabled || files.length === 0 || !!uploadProgress}
      >
        {uploadProgress
          ? `⏳ Uploading… ${Math.round(((uploadProgress.queued + uploadProgress.failed) / uploadProgress.total) * 100)}%`
          : `📤 Submit ${files.length === 1 ? '1 File' : `${files.length} Files`}`}
      </button>
    </div>
  );
}
