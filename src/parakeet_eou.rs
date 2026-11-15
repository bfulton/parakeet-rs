use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_eou::{EncoderCache, ParakeetEOUModel};
use ndarray::{s, Array2, Array3};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

const SAMPLE_RATE: usize = 16000;

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

/// Shared handle to a loaded ParakeetEOU model.
/// The ONNX session is loaded once and reference-counted.
///
/// Use [`ParakeetEOUHandle::from_pretrained`] to load from disk, then
/// [`ParakeetEOU::from_shared`] to spawn each stream with its own decoder state.
#[derive(Clone)]
pub struct ParakeetEOUHandle {
    model: Arc<Mutex<ParakeetEOUModel>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    mel_basis: Arc<Array2<f32>>,
    blank_id: i32,
    eou_id: i32,
}

/// Parakeet RealTime EOU model for streaming ASR with end-of-utterance detection.
/// Uses cache-aware streaming with audio buffering for pre-encode context.
///
/// For a single stream use [`ParakeetEOU::from_pretrained`]. For multiple
/// concurrent streams sharing one loaded model, use
/// [`ParakeetEOUHandle::from_pretrained`] followed by [`ParakeetEOU::from_shared`].
pub struct ParakeetEOU {
    model: Arc<Mutex<ParakeetEOUModel>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    mel_basis: Arc<Array2<f32>>,
    blank_id: i32,
    eou_id: i32,
    blank_penalty: f32,
    encoder_cache: EncoderCache,
    state_h: Array3<f32>,
    state_c: Array3<f32>,
    last_token: Array2<i32>,
    audio_buffer: VecDeque<f32>,
    buffer_size_samples: usize,
}

impl ParakeetEOUHandle {
    /// Load the ParakeetEOU model, tokenizer, and mel filterbank from a directory.
    ///
    /// Required files:
    /// - `encoder.onnx`, `decoder_joint.onnx`
    /// - `tokenizer.json`
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Config(format!("Failed to load tokenizer: {e}")))?;

        let vocab_size = tokenizer.get_vocab_size(true);
        // blank is appended after vocab
        let blank_id = vocab_size as i32;
        let eou_id = tokenizer
            .token_to_id("<EOU>")
            .map(|id| id as i32)
            .unwrap_or(1024);

        let exec_config = config.unwrap_or_default();
        let model = ParakeetEOUModel::from_pretrained(path, exec_config)?;
        let mel_basis = mel_filterbank();

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            mel_basis: Arc::new(mel_basis),
            blank_id,
            eou_id,
        })
    }

    /// Backwards-compatible alias for [`ParakeetEOUHandle::from_pretrained`].
    pub fn load<P: AsRef<Path>>(path: P, config: Option<ExecutionConfig>) -> Result<Self> {
        Self::from_pretrained(path, config)
    }
}

impl ParakeetEOU {
    /// Load ParakeetEOU from a directory and return a ready-to-use instance.
    /// Convenience wrapper for the single-stream case.
    ///
    /// For multiple concurrent streams sharing one loaded model, use
    /// [`ParakeetEOUHandle::from_pretrained`] + [`ParakeetEOU::from_shared`] instead.
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Ok(Self::from_shared(&ParakeetEOUHandle::from_pretrained(
            path, config,
        )?))
    }

    /// Spawn a new ParakeetEOU instance bound to a shared model.
    ///
    /// Each instance owns independent encoder cache / decoder state while the
    /// expensive ONNX session is shared through the handle. The model lock is
    /// held only during encoder/decoder inference.
    pub fn from_shared(handle: &ParakeetEOUHandle) -> Self {
        // Buffer size: 4 seconds of audio
        // Provides long history for feature extraction context
        // Note that, I pick those "magic numbers" by looking NeMo's ring buffer approach.
        let buffer_size_samples = SAMPLE_RATE * 4;
        Self {
            model: Arc::clone(&handle.model),
            tokenizer: Arc::clone(&handle.tokenizer),
            mel_basis: Arc::clone(&handle.mel_basis),
            blank_id: handle.blank_id,
            eou_id: handle.eou_id,
            blank_penalty: 0.0,
            encoder_cache: EncoderCache::new(),
            state_h: Array3::zeros((1, 1, 640)),
            state_c: Array3::zeros((1, 1, 640)),
            last_token: Array2::from_elem((1, 1), handle.blank_id),
            audio_buffer: VecDeque::with_capacity(buffer_size_samples),
            buffer_size_samples,
        }
    }

    /// Blank-logit penalty: positive recovers borderline words, negative suppresses spurious emissions. Default 0.0 = off.
    pub fn set_blank_penalty(&mut self, penalty: f32) {
        self.blank_penalty = penalty;
    }

    /// Transcribe a chunk of audio samples.
    ///
    /// # Arguments
    /// * `chunk` - Audio chunk (typically 160ms / 2560 samples at 16kHz)
    /// * `reset_on_eou` - If true, reset decoder state when end-of-utterance is detected
    ///
    /// # Streaming Behavior
    /// Cache-aware streaming
    /// - Maintains 4-second ring buffer for feature extraction context
    /// - Extracts features from full buffer
    /// - Slices last (pre_encode_cache + new_frames) for encoder input
    /// - pre_encode_cache=9 frames, new_frames=~16, total=~25 frames to encoder
    pub fn transcribe(&mut self, chunk: &[f32], reset_on_eou: bool) -> Result<String> {
        // Add new chunk to rolling buffer
        self.audio_buffer.extend(chunk.iter().copied());

        // Trim buffer to keep only the most recent samples
        while self.audio_buffer.len() > self.buffer_size_samples {
            self.audio_buffer.pop_front();
        }

        // Wait until buffer has minimum samples (at least 1 second for stable features)
        const MIN_BUFFER_SAMPLES: usize = SAMPLE_RATE; // 1 second
        if self.audio_buffer.len() < MIN_BUFFER_SAMPLES {
            return Ok(String::new());
        }

        // Extract features from FULL buffer (provides context for feature extraction)
        let buffer_slice: Vec<f32> = self.audio_buffer.iter().copied().collect();
        let full_features = self.extract_mel_features(&buffer_slice)?;
        let total_frames = full_features.shape()[2];

        // Slice to take only (pre_encode_cache + new_frames) for encoder
        // pre_encode_cache = 9 frames, new_frames = ~16 for 160ms chunk
        const PRE_ENCODE_CACHE: usize = 9;
        const FRAMES_PER_CHUNK: usize = 16;
        const SLICE_LEN: usize = PRE_ENCODE_CACHE + FRAMES_PER_CHUNK;

        let start_frame = total_frames.saturating_sub(SLICE_LEN);

        let features = full_features.slice(s![.., .., start_frame..]).to_owned();
        let time_steps = features.shape()[2];

        // Encode with cache - encoder sees full buffer context
        let (encoder_out, new_cache) = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            model.run_encoder(&features, time_steps as i64, &self.encoder_cache)?
        };
        self.encoder_cache = new_cache;

        let total_frames = encoder_out.shape()[2];
        if total_frames == 0 {
            return Ok(String::new());
        }

        // Process all output frames (typically 1 frame per chunk)
        let new_frames = encoder_out;

        let mut text_output = String::new();

        // Hold the lock once across the decoder loop to avoid per-step acquire/release.
        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;

        for t in 0..new_frames.shape()[2] {
            let current_frame = new_frames.slice(s![.., .., t..t + 1]).to_owned();
            let mut syms_added = 0;

            while syms_added < 5 {
                let (mut logits, new_h, new_c) = model.run_decoder(
                    &current_frame,
                    &self.last_token,
                    &self.state_h,
                    &self.state_c,
                )?;

                apply_blank_penalty(logits.slice_mut(s![0, 0, ..]), self.blank_penalty);
                let vocab = logits.slice(s![0, 0, ..]);

                let (max_idx, _) = crate::tensor_utils::argmax_f32(vocab.iter().copied());
                let max_idx = max_idx as i32;

                if max_idx == self.blank_id || max_idx == 0 {
                    break;
                }

                if max_idx == self.eou_id {
                    if reset_on_eou {
                        drop(model);
                        self.reset_states();
                        return Ok(text_output + " [EOU]");
                    }
                    break;
                }

                if max_idx as usize >= self.tokenizer.get_vocab_size(true) {
                    break;
                }

                self.state_h = new_h;
                self.state_c = new_c;
                self.last_token.fill(max_idx);

                if let Ok(decoded) = self.tokenizer.decode(&[max_idx as u32], true) {
                    text_output.push_str(&decoded);
                }
                syms_added += 1;
            }
        }
        Ok(text_output)
    }

    fn reset_states(&mut self) {
        // Soft reset: Only reset decoder states
        // at this state, we need to keep encoder cache and audio buffer flowing for continuous context
        // self.encoder_cache = EncoderCache::new();  // DON'T reset!!!
        self.state_h.fill(0.0);
        self.state_c.fill(0.0);
        self.last_token.fill(self.blank_id);
        // self.audio_buffer.clear();  // DON'T clear!!
    }

    fn extract_mel_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        let audio_pre = crate::audio::apply_preemphasis(audio, PREEMPH);
        let spec = crate::audio::stft(&audio_pre, N_FFT, HOP_LENGTH, WIN_LENGTH)?;
        let mel = self.mel_basis.dot(&spec);
        let mel_log = mel.mapv(|x| (x.max(0.0) + LOG_ZERO_GUARD).ln());
        Ok(mel_log.insert_axis(ndarray::Axis(0)))
    }
}

/// Subtract `penalty` from the blank logit (index 1026 in the 1027-wide
/// joint output). No-op at 0.0.
fn apply_blank_penalty(mut vocab: ndarray::ArrayViewMut1<f32>, penalty: f32) {
    if penalty == 0.0 {
        return;
    }
    if let Some(blank_logit) = vocab.last_mut() {
        *blank_logit -= penalty;
    }
}

/// The mel filterbank NeMo builds for the Parakeet EOU model, which the
/// encoder was trained on: `AudioToMelSpectrogramPreprocessor` with the
/// model's preprocessor config (128 features, n_fft 512, 16 kHz, nothing else
/// about the filterbank overridden), i.e.
/// `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney")` on
/// librosa's default Slaney mel scale. That is
/// [`crate::audio::create_mel_filterbank`]; an HTK-scale filterbank is a
/// different set of bands (see the test below).
fn mel_filterbank() -> Array2<f32> {
    crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::aview_mut1;

    // NeMo's mel filterbank for this model at a spread of (band, FFT bin)
    // cells: each band's peak and its nonzero neighbours. From
    // `FilterbankFeatures.fb` of NeMo 2.6.0 (the version in the model's
    // config) built from the model's preprocessor config, which is exactly
    // `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney")`
    // (librosa 0.11.0).
    const NEMO_FILTERBANK: &[(usize, usize, f32)] = &[
        (0, 1, 2.83775423e-02),
        (1, 1, 1.43890083e-02),
        (1, 2, 1.39885349e-02),
        (10, 8, 2.95789633e-02),
        (20, 15, 2.00236728e-03),
        (20, 16, 2.63751745e-02),
        (40, 30, 4.00473457e-03),
        (40, 31, 2.43728068e-02),
        (64, 54, 1.12167057e-02),
        (64, 55, 1.88182835e-02),
        (64, 56, 6.49144524e-04),
        (90, 101, 5.54661918e-03),
        (90, 102, 1.08582862e-02),
        (90, 103, 9.82366409e-03),
        (100, 129, 5.85763063e-03),
        (100, 130, 9.13691055e-03),
        (100, 131, 8.00091028e-03),
        (127, 249, 4.50653443e-03),
        (127, 250, 5.22318855e-03),
        (127, 251, 4.35265712e-03),
    ];
    // The FFT bin each of those bands peaks at in NeMo's filterbank.
    const NEMO_PEAK_BINS: &[(usize, usize)] = &[
        (0, 1),
        (10, 8),
        (20, 16),
        (40, 31),
        (64, 55),
        (90, 102),
        (100, 130),
        (127, 250),
    ];
    // The sum of all 128 x 257 weights of NeMo's filterbank.
    const NEMO_FILTERBANK_SUM: f64 = 4.090487653877062;

    // The model must be fed the mel bands it was trained on: NeMo's, on the
    // Slaney mel scale. An HTK-scale filterbank is a different set of bands
    // (123 of the 128 peak at a different FFT bin; band 100 at 4344 Hz where
    // NeMo's peaks at 4063 Hz), and most of the cells above are zero in it.
    #[test]
    fn mel_filterbank_is_the_one_nemo_builds_for_this_model() {
        let fb = mel_filterbank();
        assert_eq!(fb.dim(), (N_MELS, N_FFT / 2 + 1));

        for &(band, bin, want) in NEMO_FILTERBANK {
            let got = fb[[band, bin]];
            assert!(
                (got - want).abs() < 1e-7,
                "band {band}, bin {bin}: weight {got:e}, NeMo's is {want:e}"
            );
        }

        for &(band, want) in NEMO_PEAK_BINS {
            let (got, _) = crate::tensor_utils::argmax_f32(fb.row(band).iter().copied());
            assert_eq!(
                got, want,
                "band {band} peaks at bin {got}, NeMo's at {want}"
            );
        }

        // The whole matrix, not only the sampled cells.
        let sum: f64 = fb.iter().map(|&w| f64::from(w)).sum();
        assert!(
            (sum - NEMO_FILTERBANK_SUM).abs() < 1e-5,
            "weights sum to {sum}, NeMo's to {NEMO_FILTERBANK_SUM}"
        );
    }

    // Joint output: 1026 tokenizer tokens (ids 0..=1025) + blank at 1026.
    const LOGIT_WIDTH: usize = 1027;
    const BLANK: usize = LOGIT_WIDTH - 1;
    const EOB_ID: usize = 1025;

    // The penalty must hit blank (1026), not <EOB> (1025), the last tokenizer token.
    #[test]
    fn penalty_targets_blank_not_last_tokenizer_token() {
        let mut v = vec![-20.0; LOGIT_WIDTH];
        v[EOB_ID] = 5.0;
        v[BLANK] = 5.0;
        apply_blank_penalty(aview_mut1(&mut v), 3.0);
        assert_eq!(v[EOB_ID], 5.0);
        assert_eq!(v[BLANK], 2.0);
    }

    // The 0.0 default must leave the logits bit-identical (feature off = stock decode).
    #[test]
    fn zero_penalty_is_a_no_op() {
        let mut v = vec![-20.0; LOGIT_WIDTH];
        v[300] = 6.5;
        v[BLANK] = 5.0;
        let original = v.clone();
        apply_blank_penalty(aview_mut1(&mut v), 0.0);
        assert_eq!(v, original);
    }
}
