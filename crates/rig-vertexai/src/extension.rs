//! Typed request options and reply extras for Vertex AI, keyed by
//! `"vertexai"`. [`VertexOptions`] holds the GenerateContent fields Vertex
//! AI's request declares, and [`VertexExtras`] reads its reply.
//!
//! ```
//! use rig_core::completion::CompletionRequest;
//! use rig_vertexai::extension::{VertexOptions};
//!
//! let options = VertexOptions::new()
//!     .label("team", "rig")
//!     .top_k(40);
//! let request = CompletionRequest::new("hi").provider_option(options);
//! # let _ = request;
//! ```

use std::collections::BTreeMap;

use rig_core::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use rig_core::message::Api;
use rig_core::providers::gemini::extension::{
    CandidateCount, GeminiExtras, GenerationConfig, GroundingMetadata, HarmBlockThreshold,
    HarmCategory, LogprobsResult, MediaResolution, ModalityTokenCount, PromptFeedback,
    ResponseModality, SafetyRating, UrlContextMetadata,
};
use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

/// The API name of the Vertex AI route.
const API: &str = "vertexai.generate_content";

/// Vertex AI's extension: [`VertexOptions`] and [`VertexExtras`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexExt;

impl ProviderExtension for VertexExt {
    const PROVIDER: &'static str = crate::types::completion_response::PROVIDER_NAME;
    type Options = VertexOptions;
    type Extras = VertexExtras;
}

/// Whether `value` is its type's default, so a config that sets nothing
/// writes no key.
fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// Vertex AI's request options. Vertex AI has one route, so every field is
/// in its section.
///
/// The `generationConfig` setters here (`top_k`, `include_thoughts`, ...)
/// write the same entry as [`generation_config`](Self::generation_config),
/// which replaces every such entry set before it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct VertexOptions {
    /// The GenerateContent fields.
    #[serde(rename = "vertexai.generate_content")]
    pub generate_content: VertexGenerateContent,
}

impl VertexOptions {
    /// No field set.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `generationConfig` entries every GenerateContent route takes.
    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.generate_content.generation_config.common = config;
        self
    }

    /// `generationConfig.routingConfig`: how Vertex AI picks the model.
    pub fn routing(mut self, routing: RoutingConfig) -> Self {
        self.generate_content.generation_config.routing_config = Some(routing);
        self
    }

    /// `generationConfig.audioTimestamp`: read timestamps in audio inputs.
    pub fn audio_timestamp(mut self, enable: bool) -> Self {
        self.generate_content.generation_config.audio_timestamp = Some(enable);
        self
    }

    /// Add a safety setting (`safetySettings`).
    pub fn safety_setting(mut self, setting: VertexSafetySetting) -> Self {
        self.generate_content.safety_settings.push(setting);
        self
    }

    /// Add the label `key` = `value` (`labels`), user metadata Vertex AI
    /// reports billing by.
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.generate_content
            .labels
            .insert(key.into(), value.into());
        self
    }

    /// `modelArmorConfig`: the Model Armor templates that screen the prompt
    /// and the reply.
    pub fn model_armor(mut self, config: ModelArmorConfig) -> Self {
        self.generate_content.model_armor_config = Some(config);
        self
    }

    /// Apply `set` to the `generationConfig` entries every GenerateContent
    /// route takes.
    fn generation_config_entry(
        mut self,
        set: impl FnOnce(GenerationConfig) -> GenerationConfig,
    ) -> Self {
        let common = std::mem::take(&mut self.generate_content.generation_config.common);
        self.generate_content.generation_config.common = set(common);
        self
    }

    /// `generationConfig.thinkingConfig.includeThoughts`, as
    /// [`GenerationConfig::include_thoughts`] through
    /// [`generation_config`](Self::generation_config).
    pub fn include_thoughts(self, include: bool) -> Self {
        self.generation_config_entry(|config| config.include_thoughts(include))
    }

    /// `generationConfig.topK`, as [`GenerationConfig::top_k`].
    pub fn top_k(self, top_k: u32) -> Self {
        self.generation_config_entry(|config| config.top_k(top_k))
    }

    /// `generationConfig.presencePenalty`, as
    /// [`GenerationConfig::presence_penalty`].
    pub fn presence_penalty(self, penalty: f64) -> Self {
        self.generation_config_entry(|config| config.presence_penalty(penalty))
    }

    /// `generationConfig.frequencyPenalty`, as
    /// [`GenerationConfig::frequency_penalty`].
    pub fn frequency_penalty(self, penalty: f64) -> Self {
        self.generation_config_entry(|config| config.frequency_penalty(penalty))
    }

    /// `generationConfig.responseLogprobs`, as
    /// [`GenerationConfig::response_logprobs`].
    pub fn response_logprobs(self, enable: bool) -> Self {
        self.generation_config_entry(|config| config.response_logprobs(enable))
    }

    /// `generationConfig.logprobs`, as [`GenerationConfig::logprobs`].
    pub fn logprobs(self, top: u32) -> Self {
        self.generation_config_entry(|config| config.logprobs(top))
    }

    /// `generationConfig.candidateCount`, as
    /// [`GenerationConfig::candidate_count`].
    pub fn candidate_count(self, count: CandidateCount) -> Self {
        self.generation_config_entry(|config| config.candidate_count(count))
    }

    /// `generationConfig.responseModalities`, as
    /// [`GenerationConfig::response_modalities`].
    pub fn response_modalities(
        self,
        modalities: impl IntoIterator<Item = ResponseModality>,
    ) -> Self {
        self.generation_config_entry(|config| config.response_modalities(modalities))
    }

    /// `generationConfig.mediaResolution`, as
    /// [`GenerationConfig::media_resolution`].
    pub fn media_resolution(self, resolution: MediaResolution) -> Self {
        self.generation_config_entry(|config| config.media_resolution(resolution))
    }
}

impl ExtensionOptions for VertexOptions {
    type Ext = VertexExt;
}

/// The fields of Vertex AI's `GenerateContentRequest` rig does not set.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexGenerateContent {
    /// `generationConfig` entries, merged key by key with the ones the
    /// request and its generation options write.
    #[serde(skip_serializing_if = "is_default")]
    pub generation_config: VertexGenerationConfig,
    /// `safetySettings`, which take the place of the `null` sent otherwise.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<VertexSafetySetting>,
    /// `labels`.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
    /// `modelArmorConfig`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_armor_config: Option<ModelArmorConfig>,
}

/// Vertex AI's `generationConfig` entries: the ones every GenerateContent
/// route takes and Vertex AI's own.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexGenerationConfig {
    /// The entries every GenerateContent route takes.
    #[serde(flatten)]
    pub common: GenerationConfig,
    /// `routingConfig`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_config: Option<RoutingConfig>,
    /// `audioTimestamp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_timestamp: Option<bool>,
}

/// How Vertex AI picks the model that answers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingConfig {
    /// `autoMode`: Vertex AI picks by the preference.
    Auto(ModelRoutingPreference),
    /// `manualMode`: the named model answers.
    Manual(String),
}

impl Serialize for RoutingConfig {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(1))?;
        match self {
            Self::Auto(preference) => map.serialize_entry(
                "autoMode",
                &BTreeMap::from([("modelRoutingPreference", preference)]),
            )?,
            Self::Manual(model) => {
                map.serialize_entry("manualMode", &BTreeMap::from([("modelName", model)]))?;
            }
        }
        map.end()
    }
}

/// What automatic routing favours.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelRoutingPreference {
    /// `PRIORITIZE_QUALITY`.
    PrioritizeQuality,
    /// `BALANCED`.
    Balanced,
    /// `PRIORITIZE_COST`.
    PrioritizeCost,
}

/// How a safety threshold is measured.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarmBlockMethod {
    /// `SEVERITY`: by probability and severity.
    Severity,
    /// `PROBABILITY`: by probability alone.
    Probability,
}

impl HarmBlockMethod {
    /// Vertex AI's spelling.
    fn as_str(self) -> &'static str {
        match self {
            Self::Severity => "SEVERITY",
            Self::Probability => "PROBABILITY",
        }
    }
}

/// Block `category` at `threshold`, measured by `method` when set.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexSafetySetting {
    /// The category.
    pub category: HarmCategory,
    /// The threshold.
    pub threshold: HarmBlockThreshold,
    /// The method, Vertex AI's default when `None`.
    pub method: Option<HarmBlockMethod>,
}

impl VertexSafetySetting {
    /// Block `category` at `threshold`.
    pub fn new(category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        Self {
            category,
            threshold,
            method: None,
        }
    }

    /// Measure the threshold by `method`.
    pub fn method(mut self, method: HarmBlockMethod) -> Self {
        self.method = Some(method);
        self
    }
}

impl Serialize for VertexSafetySetting {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("category", self.category.as_str())?;
        map.serialize_entry("threshold", self.threshold.as_str())?;
        if let Some(method) = self.method {
            map.serialize_entry("method", method.as_str())?;
        }
        map.end()
    }
}

/// The Model Armor templates that screen a request.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelArmorConfig {
    /// `promptTemplateName`: the template that screens the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template_name: Option<String>,
    /// `responseTemplateName`: the template that screens the reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_template_name: Option<String>,
}

impl ModelArmorConfig {
    /// No template.
    pub fn new() -> Self {
        Self::default()
    }

    /// Screen the prompt with the template `name`
    /// (`projects/{p}/locations/{l}/templates/{t}`).
    pub fn prompt_template(mut self, name: impl Into<String>) -> Self {
        self.prompt_template_name = Some(name.into());
        self
    }

    /// Screen the reply with the template `name`.
    pub fn response_template(mut self, name: impl Into<String>) -> Self {
        self.response_template_name = Some(name.into());
        self
    }
}

/// Vertex AI's reply fields rig does not normalize. A candidate field is
/// the first candidate's, the one rig reads. A field the reply left out
/// reads `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VertexExtras {
    /// `modelVersion`.
    pub model_version: Option<String>,
    /// `responseId`.
    pub response_id: Option<String>,
    /// `createTime`, an RFC 3339 timestamp.
    pub create_time: Option<String>,
    /// `usageMetadata.trafficType`, such as `ON_DEMAND`.
    pub traffic_type: Option<String>,
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
    /// `citationMetadata.citations` of the candidate.
    pub citations: Option<Vec<VertexCitation>>,
    /// `groundingMetadata` of the candidate.
    pub grounding_metadata: Option<GroundingMetadata>,
    /// `urlContextMetadata` of the candidate.
    pub url_context_metadata: Option<UrlContextMetadata>,
    /// `avgLogprobs` of the candidate.
    pub avg_logprobs: Option<f64>,
    /// `logprobsResult` of the candidate.
    pub logprobs_result: Option<LogprobsResult>,
}

impl ReplyExtras for VertexExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() != API {
            return Err(serde_json::Error::custom(format!(
                "Vertex AI returns no `{}` reply",
                api.as_str()
            )));
        }
        // The reply is a GenerateContent document, as rig keeps it.
        let shared = GeminiExtras::from_reply(&Api::from_static("gemini.generate_content"), raw)?;
        let own = VertexReply::deserialize(raw)?;
        let candidate = own.candidates.into_iter().next();
        Ok(Self {
            model_version: shared.model_version,
            response_id: shared.response_id,
            create_time: own.create_time,
            traffic_type: own.usage_metadata.and_then(|usage| usage.traffic_type),
            prompt_tokens_details: shared.prompt_tokens_details,
            cache_tokens_details: shared.cache_tokens_details,
            candidates_tokens_details: shared.candidates_tokens_details,
            tool_use_prompt_tokens_details: shared.tool_use_prompt_tokens_details,
            prompt_feedback: shared.prompt_feedback,
            safety_ratings: shared.safety_ratings,
            finish_message: shared.finish_message,
            citations: candidate
                .and_then(|candidate| candidate.citation_metadata)
                .and_then(|metadata| metadata.citations),
            grounding_metadata: shared.grounding_metadata,
            url_context_metadata: shared.url_context_metadata,
            avg_logprobs: shared.avg_logprobs,
            logprobs_result: shared.logprobs_result,
        })
    }
}

/// The parts of a Vertex AI reply only Vertex AI returns.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexReply {
    create_time: Option<String>,
    usage_metadata: Option<VertexUsage>,
    #[serde(default)]
    candidates: Vec<VertexCandidate>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexUsage {
    traffic_type: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexCandidate {
    citation_metadata: Option<VertexCitationMetadata>,
}

#[derive(Deserialize)]
struct VertexCitationMetadata {
    citations: Option<Vec<VertexCitation>>,
}

/// A source a Vertex AI candidate recites.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexCitation {
    /// `startIndex` of the reciting text.
    pub start_index: Option<u64>,
    /// `endIndex` of the reciting text.
    pub end_index: Option<u64>,
    /// `uri` of the source.
    pub uri: Option<String>,
    /// `title` of the source.
    pub title: Option<String>,
    /// `license` of the source.
    pub license: Option<String>,
    /// `publicationDate` of the source.
    pub publication_date: Option<PublicationDate>,
}

/// A calendar date; a part the source left out reads `None`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct PublicationDate {
    /// `year`.
    pub year: Option<i32>,
    /// `month`, from 1.
    pub month: Option<u32>,
    /// `day`, from 1.
    pub day: Option<u32>,
}

#[cfg(test)]
mod tests;
