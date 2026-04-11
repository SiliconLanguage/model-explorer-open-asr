/**
 * TranscriptDisplay.jsx
 * Shows the live and final transcription output.
 *
 * During streaming:
 *   - `stable`   text is rendered normally
 *   - `unstable` text is rendered dimmed/italic (LocalAgreement-2 unconfirmed prefix)
 *
 * Props:
 *   stable   {string}  – confirmed transcript text
 *   unstable {string}  – pending / unconfirmed suffix
 *   loading  {bool}    – show a spinner while waiting for the first token
 *   mode     {string}  – 'webgpu' | 'server'
 */

export default function TranscriptDisplay({ stable, unstable, loading, mode }) {
  const isEmpty = !stable && !unstable && !loading;

  return (
    <div className="transcript-display">
      <div className="transcript-header">
        <span className="transcript-title">Transcript</span>
      </div>

      <div className="transcript-body transcript-output" aria-live="polite" aria-label="Transcription output">
        {loading && !stable && !unstable && (
          <span className="transcript-loading">
            <span className="spinner" aria-hidden="true" />
            Waiting for first token…
          </span>
        )}

        {isEmpty && !loading && (
          <span className="transcript-placeholder">
            Transcription will appear here after you submit audio.
          </span>
        )}

        {/* Stable (confirmed) text */}
        {stable && <span className="transcript-stable">{stable}</span>}

        {/* Unstable (unconfirmed) text – LocalAgreement-2 pending prefix */}
        {unstable && (
          <span
            className="transcript-unstable"
            title="Partial hypothesis – not yet confirmed by LocalAgreement-2"
          >
            {unstable}
          </span>
        )}
      </div>
    </div>
  );
}
