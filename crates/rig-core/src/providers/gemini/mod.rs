//! Gemini provider configuration and endpoint wires.
//!
//! ```no_run
//! use rig_core::providers::gemini;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = gemini::Gemini::from_env()?;
//!
//! let embeddings = provider.embeddings(gemini::EMBEDDING_001, None);
//! # Ok(())
//! # }
//! ```
//!
//! Bind a wire with `.bind(transport)` to obtain a [`Bound`](crate::driver::Bound)
//! implementing the consumer-facing model traits.

pub mod cached_content;
pub mod completion;
pub mod embedding;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod interactions_api;
pub mod model_listing;
pub mod streaming;
pub mod transcription;

pub use cached_content::{CacheExpiry, CachedContent, CachedContents, NewCachedContent};
pub use embedding::{EMBEDDING_001, EMBEDDING_004};
#[cfg(feature = "image")]
pub use image_generation::GEMINI_2_5_FLASH_IMAGE;
pub use model_listing::*;

use crate::client::env::{self, EnvError};
use crate::driver::{HasCompletion, HasEmbedding, HasModelListing, HasTranscription, HasVerify};
use crate::wire::Secret;

/// Stable descriptor name for both Gemini surfaces, as records and
/// telemetry spell it.
pub use completion::PROVIDER_NAME;

/// Where both Gemini surfaces live.
pub const BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// The environment variable holding the API key.
pub const API_KEY_ENV: &str = "GEMINI_API_KEY";

/// The text-block `AdditionalParams` key containing a verbatim Gemini part.
/// This preserves image data that the normalized block vocabulary cannot represent.
pub const GEMINI_RAW_CONTENT_KEY: &str = "gemini_content";

/// The text-block `AdditionalParams` key holding Gemini API extras for that
/// text, today its `thoughtSignature`. Gemini may sign an answer text part,
/// and a signature returns on the part that carried it, so it rides on the
/// text rather than on reasoning. Only the Gemini API codecs (REST and gRPC)
/// read it, which keeps the signature with its issuer.
pub const GEMINI_TEXT_EXTRAS_KEY: &str = "gemini";

const THOUGHT_SIGNATURE: &str = "thoughtSignature";

/// The signature Gemini put on this answer text part, if any.
pub fn text_thought_signature(text: &crate::message::Text) -> Option<&str> {
    text_signature_at(text, GEMINI_TEXT_EXTRAS_KEY)
}

/// Extras recording `signature` for an answer text part under `extras_key`.
pub fn text_signature_extras(
    extras_key: &str,
    signature: String,
) -> Option<crate::message::AdditionalParams> {
    crate::message::AdditionalParams::from_entries(Some((
        extras_key,
        serde_json::json!({ THOUGHT_SIGNATURE: signature }),
    )))
}

/// The signature recorded on `text` under `extras_key`, for codecs of other
/// Gemini-model services that keep their own key.
pub fn text_signature_at<'a>(text: &'a crate::message::Text, extras_key: &str) -> Option<&'a str> {
    text.additional_params
        .as_ref()
        .and_then(|params| params.wire_extras(extras_key))
        .and_then(|extras| extras.get(THOUGHT_SIGNATURE))
        .and_then(serde_json::Value::as_str)
}

/// Shared configuration for Gemini's GenerateContent and Interactions APIs.
/// Endpoint wires place the key in a query parameter or header as required.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gemini {
    /// The API key, redacted in `Debug` and in serialized form.
    pub api_key: Secret,
    /// The API root every wire builds its URI from.
    pub base_url: String,
}

impl Gemini {
    /// The provider configured with `api_key` and the public base URL.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: BASE_URL.to_owned(),
        }
    }

    /// Load `GEMINI_API_KEY`, returning an error if it is missing or invalid.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(env::required(API_KEY_ENV)?))
    }

    /// Send to a different API root (a proxy, a regional endpoint).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Build a GenerateContent URI, appending the API key after any path query.
    /// The returned URI contains credentials and must not be logged.
    pub(crate) fn uri(&self, path: &str) -> String {
        let trimmed = path.trim_start_matches('/');
        let separator = if trimmed.contains('?') { "&" } else { "?" };
        let base = self.base_url.trim_end_matches('/');
        format!("{base}/{trimmed}{separator}key={}", self.api_key.expose())
    }

    /// The URI for an Interactions-family `path`. No key: that family
    /// authenticates by the `x-goog-api-key` header instead.
    pub(crate) fn interactions_uri(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{base}/{}", path.trim_start_matches('/'))
    }

    /// The header that authenticates an Interactions-family request.
    pub(crate) const INTERACTIONS_KEY_HEADER: &'static str = "x-goog-api-key";

    /// The `generateContent` / `streamGenerateContent` completion wire.
    pub fn generate_content(&self, model: impl Into<String>) -> completion::GenerateContent {
        completion::GenerateContent::new(self.clone(), model)
    }

    /// The Interactions API completion wire.
    pub fn interactions(&self, model: impl Into<String>) -> interactions_api::Interactions {
        interactions_api::Interactions::new(self.clone(), model)
    }

    /// The `batchEmbedContents` embedding wire. `ndims` defaults from the
    /// model identifier.
    pub fn embeddings(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> embedding::Embeddings {
        embedding::Embeddings::new(self.clone(), model, ndims)
    }

    /// The audio transcription wire.
    pub fn transcriptions(&self, model: impl Into<String>) -> transcription::Transcriptions {
        transcription::Transcriptions::new(self.clone(), model)
    }

    /// The image generation wire.
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub fn images(&self, model: impl Into<String>) -> image_generation::Images {
        image_generation::Images::new(self.clone(), model)
    }

    /// Build the GenerateContent model-listing wire used by `Bound::models()`.
    pub fn models(&self) -> model_listing::Models {
        model_listing::Models::new(self.clone())
    }

    /// The model listing wire for the Interactions family: the same
    /// endpoint, authenticated the way Interactions authenticates.
    pub fn interactions_models(&self) -> model_listing::InteractionsModels {
        model_listing::InteractionsModels::new(self.clone())
    }

    /// Build a wire to retrieve an existing interaction or resume its stream.
    /// Unary reads fetch the current resource; callers control repeated polling.
    pub fn interaction(
        &self,
        interaction_id: impl Into<String>,
    ) -> interactions_api::InteractionResume {
        interactions_api::InteractionResume::new(self.clone(), interaction_id)
    }

    /// [`Self::interaction`], resuming a streamed read after the last event
    /// the consumer saw.
    pub fn interaction_resumed(
        &self,
        interaction_id: impl Into<String>,
        last_event_id: Option<&str>,
    ) -> interactions_api::InteractionResume {
        let wire = self.interaction(interaction_id);
        match last_event_id {
            Some(last_event_id) => wire.after_event(last_event_id),
            None => wire,
        }
    }
}

impl HasCompletion for Gemini {
    type Wire = completion::GenerateContent;

    fn completion(&self, model: impl Into<String>) -> Self::Wire {
        self.generate_content(model)
    }
}

impl HasEmbedding for Gemini {
    type Wire = embedding::Embeddings;

    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Self::Wire {
        self.embeddings(model, ndims)
    }
}

impl HasTranscription for Gemini {
    type Wire = transcription::Transcriptions;

    fn transcription(&self, model: impl Into<String>) -> Self::Wire {
        self.transcriptions(model)
    }
}

#[cfg(feature = "image")]
impl crate::driver::HasImageGeneration for Gemini {
    type Wire = image_generation::Images;

    fn image_generation(&self, model: impl Into<String>) -> Self::Wire {
        self.images(model)
    }
}

impl HasModelListing for Gemini {
    type Wire = model_listing::Models;

    fn model_listing(&self) -> Self::Wire {
        self.models()
    }
}

impl HasVerify for Gemini {
    type Wire = model_listing::VerifyKey;

    fn verify(&self) -> Self::Wire {
        model_listing::VerifyKey::new(self.clone())
    }
}

#[cfg(test)]
mod tests;

pub mod gemini_api_types {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum ExecutionLanguage {
        /// Unspecified language. This value should not be used.
        LanguageUnspecified,
        /// Python >= 3.10, with numpy and simply available.
        Python,
    }

    /// Code generated and executed by the hosted CodeExecution tool.
    /// Execution produces a corresponding [`CodeExecutionResult`].
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct ExecutableCode {
        /// Programming language of the code.
        pub language: ExecutionLanguage,
        /// The code to be executed.
        pub code: String,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct CodeExecutionResult {
        /// Outcome of the code execution.
        pub outcome: CodeExecutionOutcome,
        /// Contains stdout when code execution is successful, stderr or other description otherwise.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub output: Option<String>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub enum CodeExecutionOutcome {
        /// Unspecified status. This value should not be used.
        #[serde(rename = "OUTCOME_UNSPECIFIED")]
        Unspecified,
        /// Code execution completed successfully.
        #[serde(rename = "OUTCOME_OK")]
        Ok,
        /// Code execution finished but with a failure. stderr should contain the reason.
        #[serde(rename = "OUTCOME_FAILED")]
        Failed,
        /// Execution exceeded its deadline and was cancelled. Partial output may be present.
        #[serde(rename = "OUTCOME_DEADLINE_EXCEEDED")]
        DeadlineExceeded,
    }
}
