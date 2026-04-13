/**
 * AudioRecorder.jsx
 * Provides two audio input modes:
 *   1. File upload  – user selects a local audio file
 *   2. Live record  – uses MediaRecorder to capture microphone audio
 *
 * Calls onAudioReady(File) when audio is available.
 */

import { useRef, useState, useEffect } from 'react';

const SUPPORTED_TYPES = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/webm'];

export default function AudioRecorder({ onAudioReady, onFilesStaged, disabled, externalAudioUrl }) {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [fileName, setFileName] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const isInternalChange = useRef(false);

  // Clear internal state when external audio (sample clip) is selected,
  // but not when the change was triggered by our own upload.
  useEffect(() => {
    if (externalAudioUrl) {
      if (isInternalChange.current) {
        isInternalChange.current = false;
        return;
      }
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
      setFileName(null);
    }
  }, [externalAudioUrl]);

  // ── File upload ────────────────────────────────────────────────────────────
  function handleFileChange(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    // Stage all selected files (no immediate submission)
    if (onFilesStaged) {
      onFilesStaged(Array.from(files));
    }
    setFileName(files.length === 1 ? files[0].name : `${files.length} files staged`);

    // Set the first file as active audio for playback / single Transcribe
    const first = files[0];
    setAudioUrl(URL.createObjectURL(first));
    isInternalChange.current = true;
    onAudioReady(first);
  }

  // ── Microphone recording ───────────────────────────────────────────────────
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      recorder.addEventListener('dataavailable', (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      });

      recorder.addEventListener('stop', () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const recName = `recording_${ts}.webm`;
        const file = new File([blob], recName, { type: mimeType });
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        setFileName(recName);
        onAudioReady(file);
        // Stage the recording (user clicks "Submit" in the staged files panel)
        if (onFilesStaged) {
          onFilesStaged([file]);
        }
      });

      recorder.start(100); // collect data every 100 ms
      mediaRecorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      console.error('Microphone access denied:', err);
      alert('Could not access microphone. Please allow microphone permissions.');
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  return (
    <div className="audio-recorder">
      <div className="recorder-row">
        {/* File upload */}
        <label className="btn btn-secondary" title="Upload an audio file">
          📁 Upload Audio
          <input
            type="file"
            accept={SUPPORTED_TYPES.join(',')}
            multiple
            onChange={handleFileChange}
            disabled={disabled || recording}
            style={{ display: 'none' }}
          />
        </label>

        {/* Record button */}
        {recording ? (
          <button className="btn btn-danger" onClick={stopRecording} disabled={disabled}>
            ⏹ Stop Recording
          </button>
        ) : (
          <button className="btn btn-primary" onClick={startRecording} disabled={disabled}>
            🎙 Record
          </button>
        )}
      </div>

      {/* Filename display */}
      {fileName && (
        <p className="file-name" title={fileName}>
          📎 {fileName}
        </p>
      )}

      {/* Playback — show player for internal blob URLs or external (sample) URLs */}
      {(externalAudioUrl || (audioUrl && audioUrl.startsWith('blob:'))) && (
        <audio
          className="audio-player"
          controls
          src={externalAudioUrl || audioUrl}
          key={externalAudioUrl || audioUrl}
        />
      )}
    </div>
  );
}
