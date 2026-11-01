//! Text-to-speech through the Interactions API. A request speaks its text
//! in its voice; a [`Script`] speaks turns, each with an optional speaker
//! and style, in one voice or one voice per speaker. [`format_of`] reads the
//! format of the audio a reply carries.
//!
//! ```no_run
//! use rig_core::audio_generation::AudioGenerationRequestBuilder;
//! use rig_core::providers::gemini::Gemini;
//! use rig_core::providers::gemini::audio_generation::{
//!     AudioFormat, GEMINI_3_8_FLASH_TTS, Script, Turn,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.audio_generation(GEMINI_3_8_FLASH_TTS);
//! let hello = AudioGenerationRequestBuilder::new("Have a wonderful day!", "Kore").build();
//! let wav = model.call(hello).await?.audio;
//!
//! let podcast = Script::speakers([("Joe", "Puck"), ("Jane", "Kore")])
//!     .turn(Turn::new("How's it going, Jane?").speaker("Joe").style("cheerful"))
//!     .turn(Turn::new("Not bad at all.").speaker("Jane"))
//!     .format(AudioFormat::pcm(24_000))
//!     .build();
//! let pcm = model.call(podcast).await?.audio;
//! # let _ = (wav, pcm);
//! # Ok(())
//! # }
//! ```

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Serialize;
use serde_json::{Value, json};

use super::interactions_api::streaming::usage_of;
use super::interactions_api::{InteractionSpeech, InteractionSpeechConfig};
use crate::audio_generation::{AudioGenerationRequest, AudioGenerationResponse};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::operation::AudioGeneration;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::wire::{
    Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

/// `gemini-3.8-flash-tts`: the most expressive speech model.
pub const GEMINI_3_8_FLASH_TTS: &str = "gemini-3.8-flash-tts";
/// `gemini-3.8-flash-lite-tts`: the fast, low-cost speech model.
pub const GEMINI_3_8_FLASH_LITE_TTS: &str = "gemini-3.8-flash-lite-tts";

/// One turn of a [`Script`]: its text, read verbatim, and how to read it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Turn {
    /// The words to speak. Inline tags such as `<laugh>` mark momentary
    /// sounds.
    pub text: String,
    /// The speaker, one of the script's speakers. Every turn of a
    /// multi-speaker script names one.
    pub speaker: Option<String>,
    /// How the whole turn sounds, such as `"whispered urgently"`.
    pub style: Option<String>,
}

impl Turn {
    /// A turn that speaks `text`.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// The speaker.
    pub fn speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    /// The style.
    pub fn style(mut self, style: impl Into<String>) -> Self {
        self.style = Some(style.into());
        self
    }
}

impl Serialize for Turn {
    /// A text content item, with a `speech_metadata` annotation when the
    /// turn names a speaker or a style.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Metadata<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            #[serde(skip_serializing_if = "Option::is_none")]
            speaker: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            style: Option<&'a str>,
        }
        #[derive(Serialize)]
        struct Text<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            text: &'a str,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            annotations: Vec<Metadata<'a>>,
        }
        let annotations = match (&self.speaker, &self.style) {
            (None, None) => Vec::new(),
            (speaker, style) => vec![Metadata {
                kind: "speech_metadata",
                speaker: speaker.as_deref(),
                style: style.as_deref(),
            }],
        };
        Text {
            kind: "text",
            text: &self.text,
            annotations,
        }
        .serialize(serializer)
    }
}

/// The encoding of the audio a reply carries.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum AudioEncoding {
    /// `audio/wav`, the API's default: 16-bit PCM with a RIFF header.
    #[serde(rename = "audio/wav")]
    Wav,
    /// `audio/l16`: headerless 16-bit signed little-endian PCM, which
    /// joins by concatenation.
    #[serde(rename = "audio/l16")]
    L16,
    /// `audio/mulaw`: 8-bit G.711 mu-law.
    #[serde(rename = "audio/mulaw")]
    Mulaw,
    /// `audio/alaw`: 8-bit G.711 A-law.
    #[serde(rename = "audio/alaw")]
    Alaw,
    /// `audio/mp3`.
    #[serde(rename = "audio/mp3")]
    Mp3,
    /// `audio/ogg_opus`.
    #[serde(rename = "audio/ogg_opus")]
    OggOpus,
}

/// The audio format to answer in: the request's `response_format`. A field
/// left unset takes the API's default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct AudioFormat {
    /// `mime_type`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<AudioEncoding>,
    /// `sample_rate`, in hertz.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
}

impl AudioFormat {
    /// Audio in `encoding`.
    pub fn new(encoding: AudioEncoding) -> Self {
        Self {
            mime_type: Some(encoding),
            sample_rate: None,
        }
    }

    /// Headerless PCM ([`AudioEncoding::L16`]) at `sample_rate` hertz.
    pub fn pcm(sample_rate: u32) -> Self {
        Self::new(AudioEncoding::L16).sample_rate(sample_rate)
    }

    /// `sample_rate`, in hertz.
    pub fn sample_rate(mut self, hertz: u32) -> Self {
        self.sample_rate = Some(hertz);
        self
    }
}

/// `response_format` for audio.
#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(flatten)]
    format: AudioFormat,
}

impl From<AudioFormat> for ResponseFormat {
    fn from(format: AudioFormat) -> Self {
        Self {
            kind: "audio",
            format,
        }
    }
}

/// One `user_input` step of `turns`.
fn input(turns: &[Turn]) -> Value {
    json!([{ "type": "user_input", "content": turns }])
}

/// Turns spoken in one voice or one voice per speaker, in a format: a
/// speech request [`build`](Self::build) turns into an
/// [`AudioGenerationRequest`] for the [`Speech`] wire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Script {
    speech: InteractionSpeechConfig,
    turns: Vec<Turn>,
    format: Option<AudioFormat>,
}

impl Script {
    /// A script spoken in `voice`, such as `"Kore"` or a custom voice id.
    pub fn voice(voice: impl Into<String>) -> Self {
        Self::with(InteractionSpeechConfig::voice(InteractionSpeech::voice(
            voice,
        )))
    }

    /// A script whose turns each speaker of the `(speaker, voice)` pairs
    /// speaks in its voice. The API takes up to two speakers.
    pub fn speakers<S: Into<String>, V: Into<String>>(
        speakers: impl IntoIterator<Item = (S, V)>,
    ) -> Self {
        Self::with(InteractionSpeechConfig::speakers(speakers))
    }

    /// A script spoken as `speech` configures.
    pub fn with(speech: InteractionSpeechConfig) -> Self {
        Self {
            speech,
            turns: Vec::new(),
            format: None,
        }
    }

    /// Add a turn.
    pub fn turn(mut self, turn: Turn) -> Self {
        self.turns.push(turn);
        self
    }

    /// The audio format.
    pub fn format(mut self, format: AudioFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// The request that speaks this script. Its text is the turns' text,
    /// one per line, and its `additional_params` carry the turns, the
    /// voices and the format.
    pub fn build(self) -> AudioGenerationRequest {
        let text = self
            .turns
            .iter()
            .map(|turn| turn.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let voice = match &self.speech {
            InteractionSpeechConfig::Voice([voice]) => voice.voice.clone().unwrap_or_default(),
            InteractionSpeechConfig::Speakers { .. } => String::new(),
        };
        let mut params = json!({
            "input": input(&self.turns),
            "generation_config": { "speech_config": self.speech },
        });
        if let (Some(format), Some(fields)) = (self.format, params.as_object_mut()) {
            fields.insert(
                "response_format".to_owned(),
                json!(ResponseFormat::from(format)),
            );
        }
        crate::audio_generation::AudioGenerationRequestBuilder::new(text, voice)
            .additional_params(params)
            .build()
    }
}

/// The audio content item a reply ends with: the last `audio` item of its
/// model output steps.
fn last_audio(reply: &Value) -> Option<&Value> {
    reply
        .arr("steps")
        .iter()
        .filter(|step| step.str("type") == Some("model_output"))
        .flat_map(|step| step.arr("content"))
        .rfind(|content| content.str("type") == Some("audio"))
}

/// The format of the audio a reply carries, as the API states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechFormat {
    /// The media type, with its parameters, such as
    /// `audio/l16; rate=24000; channels=1`.
    pub mime_type: Option<String>,
    /// The sample rate in hertz, which the API states for headerless
    /// audio.
    pub sample_rate: Option<u64>,
    /// The number of channels, which the API states for headerless audio.
    pub channels: Option<u64>,
}

/// The format of the audio in `response`, read from its reply document.
pub fn format_of(response: &AudioGenerationResponse) -> Option<SpeechFormat> {
    let audio = last_audio(&response.raw)?;
    Some(SpeechFormat {
        mime_type: audio.str("mime_type").map(str::to_owned),
        sample_rate: audio.u64("sample_rate"),
        channels: audio.u64("channels"),
    })
}

/// The text-to-speech wire: `POST /v1beta/interactions` with audio output.
///
/// Both [`Mode`]s send a unary request and decode the whole interaction.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Speech {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
    /// Name of the model, for example [`GEMINI_3_8_FLASH_TTS`].
    pub model: String,
}

impl Speech {
    /// The text-to-speech wire for `model`.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

/// The interaction `request` creates: its text as one turn in its voice,
/// with `additional_params` merged over it key by key. An `input` there,
/// such as a [`Script`]'s, takes the place of the text.
fn create_request_body(model: &str, request: AudioGenerationRequest) -> Value {
    let mut body = json!({
        "model": model,
        "input": input(&[Turn::new(request.text)]),
        "response_format": ResponseFormat::from(AudioFormat::default()),
    });
    if let (false, Some(fields)) = (request.voice.is_empty(), body.as_object_mut()) {
        let speech = InteractionSpeechConfig::voice(InteractionSpeech::voice(request.voice));
        fields.insert(
            "generation_config".to_owned(),
            json!({ "speech_config": speech }),
        );
    }
    match (&mut body, request.additional_params) {
        (Value::Object(body), Some(Value::Object(params))) => {
            crate::completion::options::deep_merge(body, params);
        }
        (body, Some(params)) => *body = params,
        (_, None) => {}
    }
    body
}

impl Wire for Speech {
    type Op = AudioGeneration;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Event = Value;
    type Decoder<'id> = SpeechDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME).model(self.model.as_str())
    }

    /// # Errors
    ///
    /// When the request asks for a speed other than 1.0: the API has none,
    /// and a turn's style sets the pace instead.
    fn encode(&self, request: AudioGenerationRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        if request.speed != 1.0 {
            return Err(EncodeError::request(
                "Gemini speech takes no speed; ask for the pace in a turn's style",
            ));
        }
        let body = serde_json::to_vec(&create_request_body(&self.model, request))?;
        let request = http::Request::post(self.provider.interactions_uri("/v1beta/interactions"))
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(
                super::GeminiConfig::INTERACTIONS_KEY_HEADER,
                self.provider.api_key.expose(),
            )
            .body(crate::wire::Body::Bytes(body))?;
        // Gemini reports no transport request-id header.
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        SpeechDecoder
    }
}

/// Decode the last audio item of an interaction, with its usage and
/// identity. An interaction with no inline audio is a response error.
#[derive(Default)]
pub struct SpeechDecoder;

impl<'id> Decoder<'id, AudioGeneration> for SpeechDecoder {
    type Event = Value;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame(&frame.as_str(), &["steps", "status", "usage", "id"])
    }

    fn decode(
        &mut self,
        event: Self::Event,
        out: Out<'id, AudioGeneration>,
    ) -> Result<Flow, ProviderError> {
        let data = last_audio(&event)
            .and_then(|audio| audio.str("data"))
            .ok_or_else(|| {
                ProviderError::Response(format!(
                    "the Gemini speech interaction ({}) carried no inline audio",
                    event.str("status").unwrap_or("no status")
                ))
            })?;
        let audio = BASE64_STANDARD.decode(data).map_err(|err| {
            ProviderError::Response(format!("Gemini audio data was not valid base64: {err}"))
        })?;
        Ok(out.end(AudioGenerationResponse {
            model: event.str("model").map(str::to_owned),
            response_id: event.str("id").map(str::to_owned),
            usage: event
                .get("usage")
                .map(usage_of)
                .transpose()?
                .unwrap_or_default(),
            ..AudioGenerationResponse::new(audio)
        }))
    }
}

impl super::GeminiConfig {
    /// The text-to-speech wire.
    pub(crate) fn audio_generation(&self, model: impl Into<String>) -> Speech {
        Speech::new(self.clone(), model)
    }
}

#[cfg(test)]
mod tests;
