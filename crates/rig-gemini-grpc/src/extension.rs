//! Typed request options and reply extras for the Gemini gRPC API, keyed by
//! `"gemini-grpc"`. [`GeminiGrpcOptions`] carries the Gemini API's
//! [`GeminiOptions`], of which the gRPC request sends the GenerateContent
//! fields its proto declares; [`GeminiGrpcExtras`] reads its reply.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::gemini::extension::{
//!     GeminiOptions, GenerateContentOptions, GenerationConfig,
//! };
//! use rig_gemini_grpc::extension::{GeminiGrpcExt, GeminiGrpcOptions};
//!
//! # fn main() -> Result<(), rig_core::completion::OptionsError> {
//! let options = GeminiOptions::new()
//!     .generate_content(GenerateContentOptions::new().generation_config(GenerationConfig::new().top_k(40)));
//! let request = CompletionRequest::new("hi").provider_options(
//!     ProviderOptions::new().with::<GeminiGrpcExt>(&GeminiGrpcOptions::from(options))?,
//! );
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use rig_core::completion::{
    CompletionRequest, ExtensionOptions, ProviderExtension, ReplayTarget, ReplyExtras,
};
use rig_core::message::Api;
use rig_core::providers::gemini::extension::{
    CitationMetadata, GeminiExtras, GeminiOptions, HarmCategory, LogprobsResult,
    ModalityTokenCount, PromptFeedback, SafetyRating,
};
use rig_core::serde::de::Error as _;
use rig_core::serde::{Serialize, Serializer};
use serde_json::Value;

/// The API name of the gRPC route, the GenerateContent one.
const API: &str = "gemini.generate_content";

/// The Gemini gRPC API's extension: [`GeminiGrpcOptions`] and
/// [`GeminiGrpcExtras`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeminiGrpcExt;

impl ProviderExtension for GeminiGrpcExt {
    const PROVIDER: &'static str = crate::completion::PROVIDER_NAME;
    type Options = GeminiGrpcOptions;
    type Extras = GeminiGrpcExtras;
}

/// The Gemini API's options, sent over gRPC: its `"*"` and
/// `gemini.generate_content` sections, serialized as they are. The proto
/// declares no `store` or `labels` and no `HARM_CATEGORY_JAILBREAK`, so
/// those go through the request's unsupported-option policy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeminiGrpcOptions(pub GeminiOptions);

impl From<GeminiOptions> for GeminiGrpcOptions {
    fn from(options: GeminiOptions) -> Self {
        Self(options)
    }
}

impl Serialize for GeminiGrpcOptions {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl ExtensionOptions for GeminiGrpcOptions {
    fn unsupported(
        &self,
        _target: &dyn ReplayTarget,
        _request: &CompletionRequest,
    ) -> Vec<(&'static str, String)> {
        let mut refused = vec![
            (
                "store",
                "the Gemini gRPC request declares no `store`".to_owned(),
            ),
            (
                "labels",
                "the Gemini gRPC request declares no `labels`".to_owned(),
            ),
        ];
        let settings = &self.0.generate_content.safety_settings;
        if settings
            .iter()
            .any(|setting| setting.category == HarmCategory::Jailbreak)
        {
            refused.push((
                "safetySettings",
                "the Gemini gRPC proto declares no HARM_CATEGORY_JAILBREAK".to_owned(),
            ));
        }
        refused
    }
}

/// The Gemini gRPC API's reply fields rig does not normalize. A candidate
/// field is the first candidate's, the one rig reads. A field the reply
/// left out reads `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeminiGrpcExtras {
    /// `modelVersion`.
    pub model_version: Option<String>,
    /// `responseId`.
    pub response_id: Option<String>,
    /// `usageMetadata.promptTokensDetails`.
    pub prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// `usageMetadata.cacheTokensDetails`.
    pub cache_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// `usageMetadata.candidatesTokensDetails`.
    pub candidates_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// `usageMetadata.toolUsePromptTokensDetails`.
    pub tool_use_prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// `promptFeedback`.
    pub prompt_feedback: Option<PromptFeedback>,
    /// `safetyRatings` of the candidate.
    pub safety_ratings: Option<Vec<SafetyRating>>,
    /// `finishMessage` of the candidate.
    pub finish_message: Option<String>,
    /// `citationMetadata` of the candidate.
    pub citation_metadata: Option<CitationMetadata>,
    /// `avgLogprobs` of the candidate.
    pub avg_logprobs: Option<f64>,
    /// `logprobsResult` of the candidate.
    pub logprobs_result: Option<LogprobsResult>,
}

impl ReplyExtras for GeminiGrpcExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() != API {
            return Err(serde_json::Error::custom(format!(
                "the Gemini gRPC API returns no `{}` reply",
                api.as_str()
            )));
        }
        // The reply is kept as its REST JSON, a GenerateContent document.
        let shared = GeminiExtras::from_reply(api, raw)?;
        Ok(Self {
            model_version: shared.model_version,
            response_id: shared.response_id,
            prompt_tokens_details: shared.prompt_tokens_details,
            cache_tokens_details: shared.cache_tokens_details,
            candidates_tokens_details: shared.candidates_tokens_details,
            tool_use_prompt_tokens_details: shared.tool_use_prompt_tokens_details,
            prompt_feedback: shared.prompt_feedback,
            safety_ratings: shared.safety_ratings,
            finish_message: shared.finish_message,
            citation_metadata: shared.citation_metadata,
            avg_logprobs: shared.avg_logprobs,
            logprobs_result: shared.logprobs_result,
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
