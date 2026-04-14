//! Scribe Audio Normalizer — Rust Sidecar
//!
//! A lightweight HTTP service that accepts raw audio (WAV, MP3, FLAC, OGG, AAC)
//! and returns 16 kHz mono 16-bit PCM suitable for Whisper inference.
//!
//! POST /normalize  → 16kHz mono PCM (raw bytes, no WAV header)
//! GET  /health     → {"status": "ok"}

use axum::{
    body::Bytes,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::{info, warn};

const TARGET_SAMPLE_RATE: u32 = 16_000;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/health", get(health))
        .route("/normalize", post(normalize));

    let addr = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:9090".to_string());
    info!("Scribe Normalizer listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "{\"status\":\"ok\"}")
}

async fn normalize(body: Bytes) -> impl IntoResponse {
    match decode_and_resample(&body) {
        Ok(pcm) => {
            // Convert f32 samples to i16 PCM bytes
            let mut output = Vec::with_capacity(pcm.len() * 2);
            for sample in &pcm {
                let clamped = sample.clamp(-1.0, 1.0);
                let i16_val = (clamped * 32767.0) as i16;
                output.extend_from_slice(&i16_val.to_le_bytes());
            }
            info!("Normalized {} samples → {} bytes PCM16", pcm.len(), output.len());
            (StatusCode::OK, Bytes::from(output))
        }
        Err(e) => {
            warn!("Normalization failed: {}", e);
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Bytes::from(format!("{{\"error\":\"{}\"}}", e)),
            )
        }
    }
}

fn decode_and_resample(data: &[u8]) -> Result<Vec<f32>, String> {
    // ── Decode with Symphonia ───────────────────────────────────────────────
    let cursor = Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let meta_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &meta_opts)
        .map_err(|e| format!("probe failed: {}", e))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or("no audio track found")?;

    let source_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| format!("codec init failed: {}", e))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let num_frames = decoded.capacity();
        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        let samples = sample_buf.samples();

        // Mix to mono if multi-channel
        if channels > 1 {
            for frame in samples.chunks(channels) {
                let mono: f32 = frame.iter().sum::<f32>() / channels as f32;
                all_samples.push(mono);
            }
        } else {
            all_samples.extend_from_slice(samples);
        }
    }

    if all_samples.is_empty() {
        return Err("no audio samples decoded".to_string());
    }

    // ── Resample to 16 kHz if needed ────────────────────────────────────────
    if source_rate == TARGET_SAMPLE_RATE {
        return Ok(all_samples);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        TARGET_SAMPLE_RATE as f64 / source_rate as f64,
        2.0,
        params,
        all_samples.len(),
        1, // mono
    )
    .map_err(|e| format!("resampler init: {}", e))?;

    let waves_in = vec![all_samples];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| format!("resample failed: {}", e))?;

    Ok(waves_out.into_iter().next().unwrap_or_default())
}
