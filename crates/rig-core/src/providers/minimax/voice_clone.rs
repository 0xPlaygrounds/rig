//! MiniMax voice cloning: upload a sample, clone it under a voice id, then
//! speak with that id through the client's `audio_generation`. MiniMax
//! deletes a cloned voice that goes unused for 7 days.
//!
//! ```no_run
//! use rig_core::audio_generation::AudioGenerationRequestBuilder;
//! use rig_core::providers::minimax::{self, voice_clone::{CloneAudio, VoiceClone}};
//!
//! # async fn run(sample: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
//! let client = minimax::from_env()?;
//! let cloning = minimax::voice_cloning(&client);
//! let file = cloning.upload(CloneAudio::sample("sample.mp3", sample)).await?;
//! let voice = cloning.clone_voice(VoiceClone::new(file.file_id, "MyVoice001")).await?;
//! let request = AudioGenerationRequestBuilder::new("Hello", voice.voice_id).build();
//! let audio = client.audio_generation(minimax::SPEECH_2_8_HD).call(request).await?;
//! # let _ = audio;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::driver::Model;
use crate::error::{EncodeError, ProviderError};
use crate::operation::Whole;
use crate::providers::openai::OpenAI;
use crate::providers::openai::wire::OpenAIConfig;
use crate::wire::{
    Body, Call, Decoder, Descriptor, Encoded, Flow, Free, Mode, Operation, Out, Wire, WireEvent,
    WireFrame,
};

const FILES_UPLOAD_PATH: &str = "/files/upload";
const VOICE_CLONE_PATH: &str = "/voice_clone";

/// The voice-cloning model of a MiniMax chat-completions `client`. Calls
/// through a client of another provider fail before they are sent.
pub fn voice_cloning(client: &OpenAI) -> Model<VoiceClones> {
    client.model(VoiceClones::new(client.config().clone()))
}

/// What an uploaded audio file is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioPurpose {
    /// The voice to clone (`voice_clone`).
    VoiceClone,
    /// A short prompt clip that improves similarity (`prompt_audio`).
    PromptAudio,
}

impl AudioPurpose {
    /// The `purpose` form field.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VoiceClone => "voice_clone",
            Self::PromptAudio => "prompt_audio",
        }
    }
}

/// An mp3, m4a or wav file to upload, at most 20 MB. Other extensions are
/// refused before the request is sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CloneAudio {
    /// What the file is for.
    pub purpose: AudioPurpose,
    /// The file name, whose extension names the format.
    pub filename: String,
    /// The file's bytes.
    pub data: Vec<u8>,
}

impl CloneAudio {
    /// The voice to clone: 10 seconds to 5 minutes of audio.
    pub fn sample(filename: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Self {
            purpose: AudioPurpose::VoiceClone,
            filename: filename.into(),
            data: data.into(),
        }
    }

    /// A prompt clip under 8 seconds, for [`VoiceClone::prompt`].
    pub fn prompt(filename: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Self {
            purpose: AudioPurpose::PromptAudio,
            filename: filename.into(),
            data: data.into(),
        }
    }
}

/// A file MiniMax stored.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct UploadedFile {
    /// The id a clone request names the file by.
    pub file_id: i64,
    /// Size in bytes.
    #[serde(default)]
    pub bytes: Option<u64>,
    /// Unix time of the upload, in seconds.
    #[serde(default)]
    pub created_at: Option<u64>,
    /// The uploaded file name.
    #[serde(default)]
    pub filename: Option<String>,
    /// The `purpose` it was uploaded for.
    #[serde(default)]
    pub purpose: Option<String>,
}

/// A prompt clip and its transcript, which make the cloned voice closer and
/// steadier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ClonePrompt {
    /// The `file_id` of a [`CloneAudio::prompt`] upload.
    pub prompt_audio: i64,
    /// What the clip says, ending with punctuation.
    pub prompt_text: String,
}

/// A request to clone an uploaded sample as `voice_id`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VoiceClone {
    /// The `file_id` of a [`CloneAudio::sample`] upload.
    pub file_id: i64,
    /// The new voice's id: 8 to 256 letters, digits, `-` and `_`, starting
    /// with a letter, not ending with `-` or `_`, and not already in use.
    pub voice_id: String,
    /// An optional prompt clip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clone_prompt: Option<ClonePrompt>,
    /// Text the cloned voice reads for a preview, billed as speech.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// The speech model that reads the preview.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Other request fields, such as `need_noise_reduction` or
    /// `language_boost`, merged over the ones above.
    #[serde(skip)]
    pub additional_params: Option<Value>,
}

impl VoiceClone {
    /// Clone the sample `file_id` as `voice_id`.
    pub fn new(file_id: i64, voice_id: impl Into<String>) -> Self {
        Self {
            file_id,
            voice_id: voice_id.into(),
            clone_prompt: None,
            text: None,
            model: None,
            additional_params: None,
        }
    }

    /// Guide the clone with the uploaded prompt clip `prompt_audio`, which
    /// says `prompt_text`.
    pub fn prompt(mut self, prompt_audio: i64, prompt_text: impl Into<String>) -> Self {
        self.clone_prompt = Some(ClonePrompt {
            prompt_audio,
            prompt_text: prompt_text.into(),
        });
        self
    }

    /// Have `model` read `text` in the new voice, returning a preview URL.
    pub fn preview(mut self, text: impl Into<String>, model: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self.model = Some(model.into());
        self
    }

    /// The voice id the request creates: a `voice_id` in
    /// [`Self::additional_params`] replaces [`Self::voice_id`].
    pub fn created_voice_id(&self) -> &str {
        self.additional_params
            .as_ref()
            .and_then(|params| params.get("voice_id"))
            .and_then(Value::as_str)
            .unwrap_or(&self.voice_id)
    }

    /// Merge `params` over earlier ones, key by key for JSON objects; `None`
    /// clears them.
    pub fn additional_params(mut self, params: impl Into<Option<Value>>) -> Self {
        self.additional_params =
            crate::json_utils::merge_params(self.additional_params.take(), params.into());
        self
    }
}

/// A cloned voice.
#[derive(Clone, Debug, PartialEq)]
pub struct ClonedVoice {
    /// The id to speak with, as [`VoiceClone::created_voice_id`] names it.
    pub voice_id: String,
    /// A URL of the preview, when one was asked for.
    pub demo_audio: Option<String>,
    /// The preview's length, size and billed characters, verbatim.
    pub extra_info: Option<Value>,
}

/// One voice-cloning request.
#[derive(Clone, Debug, PartialEq)]
pub enum VoiceCloneRequest {
    /// `POST /v1/files/upload`.
    Upload(CloneAudio),
    /// `POST /v1/voice_clone`.
    Clone(VoiceClone),
}

/// The reply to either request. An upload carries `file`; a clone carries
/// the preview fields.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct VoiceCloneReply {
    /// The stored file.
    #[serde(default)]
    pub file: Option<UploadedFile>,
    /// A URL of the preview audio, empty when none was asked for.
    #[serde(default)]
    pub demo_audio: Option<String>,
    /// The preview's length, size and billed characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_info: Option<Value>,
}

/// The voice-cloning operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceCloning;

impl Operation for VoiceCloning {
    type Request = VoiceCloneRequest;
    type Event = std::convert::Infallible;
    type End = VoiceCloneReply;
    type Response = VoiceCloneReply;
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Whole::new()
    }
}

/// The voice-cloning wire: file upload and voice clone on one MiniMax
/// configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoiceClones {
    /// Which MiniMax endpoint, and the credential.
    pub provider: OpenAIConfig,
}

impl VoiceClones {
    /// The voice-cloning wire for `provider`.
    pub fn new(provider: OpenAIConfig) -> Self {
        Self { provider }
    }
}

impl Wire for VoiceClones {
    type Op = VoiceCloning;
    type Event = Vec<u8>;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = VoiceClonesDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
    }

    fn encode(&self, request: VoiceCloneRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        if self.provider.dialect.name != super::PROVIDER_NAME {
            return Err(EncodeError::request(format!(
                "voice cloning is a MiniMax endpoint, not one {} serves",
                self.provider.dialect.name
            )));
        }
        let provider = &self.provider;
        match request {
            VoiceCloneRequest::Upload(audio) => {
                use crate::http_client::MultipartForm;
                use crate::http_client::multipart::Part;

                let format = std::path::Path::new(&audio.filename)
                    .extension()
                    .and_then(std::ffi::OsStr::to_str)
                    .map(str::to_ascii_lowercase);
                if !matches!(format.as_deref(), Some("mp3" | "m4a" | "wav")) {
                    return Err(EncodeError::request(format!(
                        "MiniMax clones voices from mp3, m4a or wav files, not `{}`",
                        audio.filename
                    )));
                }

                let form = MultipartForm::new()
                    .text("purpose", audio.purpose.as_str())
                    .part(Part::bytes("file", audio.data).filename(audio.filename));
                let builder = http::Request::post(provider.uri(FILES_UPLOAD_PATH, None));
                crate::providers::openai::wire::encoded(provider, builder, Body::Multipart(form))
            }
            VoiceCloneRequest::Clone(mut clone) => {
                let additional_params = clone.additional_params.take();
                let mut body = serde_json::to_value(clone)?;
                if let Some(additional_params) = additional_params {
                    crate::json_utils::merge_inplace(&mut body, additional_params);
                }
                crate::providers::openai::wire::json_post(provider, VOICE_CLONE_PATH, None, &body)
            }
        }
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        VoiceClonesDecoder
    }
}

/// Decodes either voice-cloning reply after checking its status envelope.
pub struct VoiceClonesDecoder;

impl<'id> Decoder<'id, VoiceCloning> for VoiceClonesDecoder {
    type Event = Vec<u8>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(match frame {
            WireFrame::Text(text) => text.into_bytes(),
            WireFrame::Bytes(bytes) => bytes,
        })
    }

    fn decode(
        &mut self,
        body: Vec<u8>,
        out: Out<'id, VoiceCloning>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(super::decode_reply(&body)?))
    }
}
