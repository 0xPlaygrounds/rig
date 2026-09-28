//! The [Gemini GenerateContent API](https://ai.google.dev/api/generate-content)
//! completion wire: unary `generateContent` and SSE `streamGenerateContent`.
//! Requests go through the [edge](super::edge); Gemini options rig does not
//! own are the model's [`RequestSettings`](super::api::RequestSettings).
//!
//! ```no_run
//! use rig_core::providers::gemini::{self, Gemini, api};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?
//!     .completion(gemini::GEMINI_3_8_FLASH)
//!     .settings(api::RequestSettings {
//!         service_tier: Some(api::ServiceTier::Flex),
//!         ..Default::default()
//!     });
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// `gemini-3.8-flash` completion model.
pub const GEMINI_3_8_FLASH: &str = "gemini-3.8-flash";
/// `gemini-3.1-flash-lite-preview` completion model
pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
/// `gemini-3-flash-preview` completion model
pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use crate::completion::CompletionRequest;
use crate::error::EncodeError;
use crate::operation::Completion;
use crate::telemetry::GenAiOperation;
use crate::wire::{Body, Descriptor, Encoded, Framing, Mode, Wire};

use super::api;
use super::generate_content::GenerateContentDecoder;
use super::prefix::CachedPrefix;

/// Provider name used in normalized responses, streams, and telemetry.
pub const PROVIDER_NAME: &str = "gcp.gemini";

/// Completion wire for unary `generateContent` and SSE
/// `streamGenerateContent`. Both modes use [`GenerateContentDecoder`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GenerateContent {
    /// The key and the API root.
    pub provider: super::GeminiConfig,
    /// The model to address, e.g. [`GEMINI_3_8_FLASH`].
    pub model: String,
    /// Every Gemini request option rig does not own, as Google names it.
    #[serde(default)]
    pub settings: api::RequestSettings,
    /// The explicit cache holding this model's system instruction, tools and
    /// tool config. Requests leave those out and read them from the cache.
    #[serde(default)]
    pub cached_content: Option<CachedPrefix>,
}

impl GenerateContent {
    /// The wire for `model`.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            settings: api::RequestSettings::default(),
            cached_content: None,
        }
    }

    /// Send `settings` with every request.
    pub fn with_settings(mut self, settings: api::RequestSettings) -> Self {
        self.settings = settings;
        self
    }

    /// The body this wire sends for `request`, as Google's schema reads it.
    pub fn to_api(
        &self,
        request: CompletionRequest,
    ) -> Result<api::GenerateContentRequest, EncodeError> {
        Ok(self.body(request)?.to_api()?)
    }

    fn body(
        &self,
        request: CompletionRequest,
    ) -> Result<super::generate_content::Body, EncodeError> {
        let request = request.replayable_to(&[super::ISSUER])?;
        super::generate_content::body(request, &self.settings, self.cached_content.as_ref())
    }
}

impl<T> crate::driver::Model<GenerateContent, T> {
    /// Send `settings` with every request: thinking, media resolution,
    /// safety, service tier, hosted tools and anything else Google offers
    /// that rig does not own.
    pub fn settings(mut self, settings: api::RequestSettings) -> Self {
        self.wire.settings = settings;
        self
    }

    /// Read the system instruction, tools and tool config from `prefix`
    /// instead of sending them. A request whose own differ fails to encode.
    pub fn cached_content(mut self, prefix: CachedPrefix) -> Self {
        self.wire.cached_content = Some(prefix);
        self
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = GenerateContentDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .telemetry(|mode| match mode {
                Mode::Unary => GenAiOperation::GenerateContent,
                Mode::Streaming => GenAiOperation::ChatStreaming,
            })
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        // The request may name a model of its own; the wire's is the default.
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let body = self.body(request)?;
        let (path, framing, target) = match mode {
            Mode::Unary => (
                completion_endpoint(&model),
                Framing::Whole,
                crate::providers::internal::LogTarget::Completions,
            ),
            // `alt=sse` makes the streamed reply an event stream rather than
            // a JSON array of the same chunks.
            Mode::Streaming => (
                format!("{}?alt=sse", streaming_endpoint(&model)),
                Framing::Sse,
                crate::providers::internal::LogTarget::Streaming,
            ),
        };
        crate::providers::internal::trace_json(target, "Gemini completion request", &body);
        let request = http::Request::post(self.provider.uri(&path))
            .header("Content-Type", "application/json")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        // Gemini supplies no transport request-id response header.
        Ok(Encoded::new(request, framing)
            .with_projection(super::streaming::project)
            .with_analysis_only(super::streaming::is_analysis_only))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        GenerateContentDecoder::new()
    }
}

/// The unary endpoint for `model`.
pub(crate) fn completion_endpoint(model: &str) -> String {
    format!("/v1beta/models/{model}:generateContent")
}

/// The streaming endpoint for `model`.
pub(crate) fn streaming_endpoint(model: &str) -> String {
    format!("/v1beta/models/{model}:streamGenerateContent")
}
