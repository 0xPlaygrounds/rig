//! Typed request options and reply extras for the Gemini API, keyed by
//! [`PROVIDER_NAME`](super::PROVIDER_NAME). One [`GeminiOptions`] serves both
//! routes: its `"*"` section goes to GenerateContent and Interactions alike,
//! and each route section only to its own route.
//!
//! [`GeminiExtras`] reads the reply document of either route. A field the
//! route taken does not return reads `None`.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::gemini::extension::{
//!     GeminiExt, GeminiOptions, GenerateContentOptions, GenerationConfig,
//! };
//!
//! # fn main() -> Result<(), rig_core::completion::OptionsError> {
//! let options = GeminiOptions::new()
//!     .generate_content(GenerateContentOptions::new().generation_config(GenerationConfig::new().top_k(40)));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<GeminiExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;

use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use crate::completion::provider_options::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// The API name of the GenerateContent route.
const GENERATE_CONTENT: &str = "gemini.generate_content";

/// The API name of the Interactions route.
const INTERACTIONS: &str = "gemini.interactions";

/// The Gemini API's extension: [`GeminiOptions`] and [`GeminiExtras`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeminiExt;

impl ProviderExtension for GeminiExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = GeminiOptions;
    type Extras = GeminiExtras;
}

/// Whether `value` is its type's default, so a section or config that sets
/// nothing writes no key.
fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// The Gemini API's request options: a section both routes read, and one
/// per route.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GeminiOptions {
    /// Fields both routes spell the same.
    #[serde(rename = "*")]
    pub shared: GeminiShared,
    /// Fields only GenerateContent reads.
    #[serde(rename = "gemini.generate_content")]
    pub generate_content: GenerateContentOptions,
    /// Fields only Interactions reads.
    #[serde(rename = "gemini.interactions")]
    pub interactions: InteractionsOptions,
}

impl GeminiOptions {
    /// No field set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the API stores the request and its reply (`store`).
    pub fn store(mut self, store: bool) -> Self {
        self.shared.store = Some(store);
        self
    }

    /// Add the label `key` = `value` (`labels`), user metadata the API
    /// reports billing by.
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.shared.labels.insert(key.into(), value.into());
        self
    }

    /// The GenerateContent section.
    pub fn generate_content(mut self, section: GenerateContentOptions) -> Self {
        self.generate_content = section;
        self
    }

    /// The Interactions section.
    pub fn interactions(mut self, section: InteractionsOptions) -> Self {
        self.interactions = section;
        self
    }
}

impl ExtensionOptions for GeminiOptions {}

/// The fields both Gemini routes read, at the top level of the body.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GeminiShared {
    /// `store`: whether the API stores the request and its reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// `labels`: user metadata, each key and value at most 63 lowercase
    /// letters, digits, underscores and dashes.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
}

/// The GenerateContent-only fields: `generationConfig` entries and
/// `safetySettings`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GenerateContentOptions {
    /// `generationConfig` entries, merged key by key with the ones the
    /// request and its generation options write.
    #[serde(rename = "generationConfig", skip_serializing_if = "is_default")]
    pub generation_config: GeminiGenerationConfig,
    /// `safetySettings`, which take the place of the `null` sent otherwise.
    #[serde(rename = "safetySettings", skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<SafetySetting>,
}

impl GenerateContentOptions {
    /// No field set.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `generationConfig` entries every GenerateContent route takes.
    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.generation_config.common = config;
        self
    }

    /// `generationConfig.enableEnhancedCivicAnswers`.
    pub fn enable_enhanced_civic_answers(mut self, enable: bool) -> Self {
        self.generation_config.enable_enhanced_civic_answers = Some(enable);
        self
    }

    /// Add a safety setting: block `category` at `threshold`.
    pub fn safety_setting(mut self, category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        self.safety_settings
            .push(SafetySetting::new(category, threshold));
        self
    }
}

/// The Gemini API's `generationConfig` entries: the ones every
/// GenerateContent route takes and the API's own.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GeminiGenerationConfig {
    /// The entries every GenerateContent route takes.
    #[serde(flatten)]
    pub common: GenerationConfig,
    /// `enableEnhancedCivicAnswers`.
    #[serde(
        rename = "enableEnhancedCivicAnswers",
        skip_serializing_if = "Option::is_none"
    )]
    pub enable_enhanced_civic_answers: Option<bool>,
}

/// The `generationConfig` entries the Gemini API, Vertex AI and the gRPC
/// API all take. None of them is one a request or its generation options
/// write: `temperature`, `maxOutputTokens`, `topP`, `seed`, `stopSequences`
/// and the thinking level or budget are set there.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// `thinkingConfig.includeThoughts`, next to the thinking level or
    /// budget the request's reasoning sets.
    #[serde(skip_serializing_if = "is_default")]
    pub thinking_config: ThinkingConfig,
    /// `topK`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// `presencePenalty`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// `frequencyPenalty`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// `responseLogprobs`: return the chosen tokens' log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_logprobs: Option<bool>,
    /// `logprobs`: how many top candidates to return per token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    /// `candidateCount`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<CandidateCount>,
    /// `responseModalities`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub response_modalities: Vec<ResponseModality>,
    /// `imageConfig`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config: Option<ImageConfig>,
    /// `speechConfig`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<SpeechConfig>,
    /// `mediaResolution`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<MediaResolution>,
}

impl GenerationConfig {
    /// No entry set.
    pub fn new() -> Self {
        Self::default()
    }

    /// `thinkingConfig.includeThoughts`: return thought summaries.
    pub fn include_thoughts(mut self, include: bool) -> Self {
        self.thinking_config.include_thoughts = Some(include);
        self
    }

    /// `topK`.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// `presencePenalty`.
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// `frequencyPenalty`.
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// `responseLogprobs`.
    pub fn response_logprobs(mut self, enable: bool) -> Self {
        self.response_logprobs = Some(enable);
        self
    }

    /// `logprobs`.
    pub fn logprobs(mut self, top: u32) -> Self {
        self.logprobs = Some(top);
        self
    }

    /// `candidateCount`.
    pub fn candidate_count(mut self, count: CandidateCount) -> Self {
        self.candidate_count = Some(count);
        self
    }

    /// `responseModalities`.
    pub fn response_modalities(
        mut self,
        modalities: impl IntoIterator<Item = ResponseModality>,
    ) -> Self {
        self.response_modalities = modalities.into_iter().collect();
        self
    }

    /// `imageConfig`.
    pub fn image_config(mut self, config: ImageConfig) -> Self {
        self.image_config = Some(config);
        self
    }

    /// `speechConfig`.
    pub fn speech_config(mut self, config: SpeechConfig) -> Self {
        self.speech_config = Some(config);
        self
    }

    /// `mediaResolution`.
    pub fn media_resolution(mut self, resolution: MediaResolution) -> Self {
        self.media_resolution = Some(resolution);
        self
    }
}

/// The provider half of `thinkingConfig`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// `includeThoughts`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
}

/// How many candidates to generate. Rig reads only the first candidate of
/// a reply, so one is the only count it can ask for.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateCount {
    /// One candidate, sent as `1`.
    One,
}

impl Serialize for CandidateCount {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::One => serializer.serialize_u32(1),
        }
    }
}

/// A modality of the reply.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResponseModality {
    /// Text.
    Text,
    /// Images.
    Image,
    /// Audio.
    Audio,
}

/// The resolution media inputs are read at.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MediaResolution {
    /// `MEDIA_RESOLUTION_LOW`.
    #[serde(rename = "MEDIA_RESOLUTION_LOW")]
    Low,
    /// `MEDIA_RESOLUTION_MEDIUM`.
    #[serde(rename = "MEDIA_RESOLUTION_MEDIUM")]
    Medium,
    /// `MEDIA_RESOLUTION_HIGH`.
    #[serde(rename = "MEDIA_RESOLUTION_HIGH")]
    High,
}

/// How generated images are shaped.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageConfig {
    /// `aspectRatio`, such as `"16:9"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    /// `imageSize`, such as `"2K"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,
}

impl ImageConfig {
    /// No entry set.
    pub fn new() -> Self {
        Self::default()
    }

    /// `aspectRatio`.
    pub fn aspect_ratio(mut self, ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(ratio.into());
        self
    }

    /// `imageSize`.
    pub fn image_size(mut self, size: impl Into<String>) -> Self {
        self.image_size = Some(size.into());
        self
    }
}

/// How generated speech sounds: one voice, or one per speaker.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechConfig {
    /// `voiceConfig`: the one voice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_config: Option<VoiceConfig>,
    /// `multiSpeakerVoiceConfig`: a voice per speaker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_speaker_voice_config: Option<MultiSpeakerVoiceConfig>,
    /// `languageCode`, such as `"en-US"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
}

impl SpeechConfig {
    /// Speak with the prebuilt voice `voice_name`.
    pub fn voice(voice_name: impl Into<String>) -> Self {
        Self {
            voice_config: Some(VoiceConfig::prebuilt(voice_name)),
            multi_speaker_voice_config: None,
            language_code: None,
        }
    }

    /// Speak each `(speaker, voice_name)` pair's lines with its prebuilt
    /// voice.
    pub fn speakers<S: Into<String>, V: Into<String>>(
        speakers: impl IntoIterator<Item = (S, V)>,
    ) -> Self {
        let speaker_voice_configs = speakers
            .into_iter()
            .map(|(speaker, voice)| SpeakerVoiceConfig {
                speaker: speaker.into(),
                voice_config: VoiceConfig::prebuilt(voice),
            })
            .collect();
        Self {
            voice_config: None,
            multi_speaker_voice_config: Some(MultiSpeakerVoiceConfig {
                speaker_voice_configs,
            }),
            language_code: None,
        }
    }

    /// `languageCode`.
    pub fn language_code(mut self, code: impl Into<String>) -> Self {
        self.language_code = Some(code.into());
        self
    }
}

/// A voice.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceConfig {
    /// `prebuiltVoiceConfig`.
    pub prebuilt_voice_config: PrebuiltVoiceConfig,
}

impl VoiceConfig {
    /// The prebuilt voice `voice_name`, such as `"Kore"`.
    pub fn prebuilt(voice_name: impl Into<String>) -> Self {
        Self {
            prebuilt_voice_config: PrebuiltVoiceConfig {
                voice_name: voice_name.into(),
            },
        }
    }
}

/// A prebuilt voice, by name.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrebuiltVoiceConfig {
    /// `voiceName`.
    pub voice_name: String,
}

/// A voice per speaker.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MultiSpeakerVoiceConfig {
    /// `speakerVoiceConfigs`.
    pub speaker_voice_configs: Vec<SpeakerVoiceConfig>,
}

/// One speaker's voice.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerVoiceConfig {
    /// `speaker`: the name the prompt gives the speaker.
    pub speaker: String,
    /// `voiceConfig`.
    pub voice_config: VoiceConfig,
}

/// A harm category a safety setting applies to.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarmCategory {
    /// Hate speech.
    HateSpeech,
    /// Dangerous content.
    DangerousContent,
    /// Harassment.
    Harassment,
    /// Sexually explicit content.
    SexuallyExplicit,
    /// Content that may be used to harm civic integrity.
    CivicIntegrity,
    /// Jailbreak attempts.
    Jailbreak,
}

impl HarmCategory {
    /// The GenerateContent spelling, such as `HARM_CATEGORY_HATE_SPEECH`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HateSpeech => "HARM_CATEGORY_HATE_SPEECH",
            Self::DangerousContent => "HARM_CATEGORY_DANGEROUS_CONTENT",
            Self::Harassment => "HARM_CATEGORY_HARASSMENT",
            Self::SexuallyExplicit => "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            Self::CivicIntegrity => "HARM_CATEGORY_CIVIC_INTEGRITY",
            Self::Jailbreak => "HARM_CATEGORY_JAILBREAK",
        }
    }

    /// The Interactions spelling, such as `hate_speech`.
    fn interactions(self) -> &'static str {
        match self {
            Self::HateSpeech => "hate_speech",
            Self::DangerousContent => "dangerous_content",
            Self::Harassment => "harassment",
            Self::SexuallyExplicit => "sexually_explicit",
            Self::CivicIntegrity => "civic_integrity",
            Self::Jailbreak => "jailbreak",
        }
    }
}

/// The probability at and above which content is blocked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarmBlockThreshold {
    /// Block low probability and above.
    BlockLowAndAbove,
    /// Block medium probability and above.
    BlockMediumAndAbove,
    /// Block only high probability.
    BlockOnlyHigh,
    /// Block nothing.
    BlockNone,
    /// Turn the safety filter off.
    Off,
}

impl HarmBlockThreshold {
    /// The GenerateContent spelling, such as `BLOCK_ONLY_HIGH`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BlockLowAndAbove => "BLOCK_LOW_AND_ABOVE",
            Self::BlockMediumAndAbove => "BLOCK_MEDIUM_AND_ABOVE",
            Self::BlockOnlyHigh => "BLOCK_ONLY_HIGH",
            Self::BlockNone => "BLOCK_NONE",
            Self::Off => "OFF",
        }
    }

    /// The Interactions spelling, such as `block_only_high`.
    fn interactions(self) -> &'static str {
        match self {
            Self::BlockLowAndAbove => "block_low_and_above",
            Self::BlockMediumAndAbove => "block_medium_and_above",
            Self::BlockOnlyHigh => "block_only_high",
            Self::BlockNone => "block_none",
            Self::Off => "off",
        }
    }
}

/// Block `category` at `threshold`. Sent as
/// `{"category": "HARM_CATEGORY_…", "threshold": "BLOCK_…"}` on
/// GenerateContent and `{"type": "…", "threshold": "block_…"}` on
/// Interactions.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SafetySetting {
    /// The category.
    pub category: HarmCategory,
    /// The threshold.
    pub threshold: HarmBlockThreshold,
}

impl SafetySetting {
    /// Block `category` at `threshold`.
    pub fn new(category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        Self {
            category,
            threshold,
        }
    }
}

impl Serialize for SafetySetting {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("category", self.category.as_str())?;
        map.serialize_entry("threshold", self.threshold.as_str())?;
        map.end()
    }
}

/// `settings` in the Interactions spelling.
fn interactions_safety<S: Serializer>(
    settings: &[SafetySetting],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(settings.iter().map(|setting| {
        BTreeMap::from([
            ("type", setting.category.interactions()),
            ("threshold", setting.threshold.interactions()),
        ])
    }))
}

/// The Interactions-only fields, at the top level of the create body but
/// for `generation_config`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct InteractionsOptions {
    /// `agent`: the agent to run in place of the model, which the body then
    /// leaves out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// `agent_config`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_config: Option<AgentConfig>,
    /// `background`: run the interaction in the background.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// `previous_interaction_id`: continue a stored interaction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_interaction_id: Option<String>,
    /// `safety_settings`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "interactions_safety"
    )]
    pub safety_settings: Vec<SafetySetting>,
    /// `generation_config` entries, merged key by key with the ones the
    /// request and its generation options write.
    #[serde(skip_serializing_if = "is_default")]
    pub generation_config: InteractionsGenerationConfig,
}

impl InteractionsOptions {
    /// No field set.
    pub fn new() -> Self {
        Self::default()
    }

    /// `agent`.
    pub fn agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    /// `agent_config`.
    pub fn agent_config(mut self, config: AgentConfig) -> Self {
        self.agent_config = Some(config);
        self
    }

    /// `background`.
    pub fn background(mut self, background: bool) -> Self {
        self.background = Some(background);
        self
    }

    /// `previous_interaction_id`.
    pub fn previous_interaction_id(mut self, id: impl Into<String>) -> Self {
        self.previous_interaction_id = Some(id.into());
        self
    }

    /// Add a safety setting: block `category` at `threshold`.
    pub fn safety_setting(mut self, category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        self.safety_settings
            .push(SafetySetting::new(category, threshold));
        self
    }

    /// `generation_config.thinking_summaries`.
    pub fn thinking_summaries(mut self, summaries: ThinkingSummaries) -> Self {
        self.generation_config.thinking_summaries = Some(summaries);
        self
    }

    /// Add a voice to `generation_config.speech_config`.
    pub fn speech(mut self, speech: InteractionSpeech) -> Self {
        self.generation_config.speech_config.push(speech);
        self
    }
}

/// The provider entries of the Interactions `generation_config`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct InteractionsGenerationConfig {
    /// `thinking_summaries`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_summaries: Option<ThinkingSummaries>,
    /// `speech_config`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub speech_config: Vec<InteractionSpeech>,
}

/// Whether the reply carries summaries of the model's thoughts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingSummaries {
    /// `auto`.
    Auto,
    /// `none`.
    None,
}

/// An agent's configuration, tagged by `type`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AgentConfig {
    /// `dynamic`.
    Dynamic,
    /// `deep-research`.
    DeepResearch {
        /// `thinking_summaries`.
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking_summaries: Option<ThinkingSummaries>,
    },
}

/// One voice of the Interactions `speech_config`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct InteractionSpeech {
    /// `voice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// `language`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// `speaker`: the name the prompt gives the speaker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

impl InteractionSpeech {
    /// Speak with `voice`.
    pub fn voice(voice: impl Into<String>) -> Self {
        Self {
            voice: Some(voice.into()),
            ..Self::default()
        }
    }

    /// `language`.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// `speaker`.
    pub fn speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }
}

/// The Gemini API's reply fields rig does not normalize, from the reply
/// document of either route. Each field says which route returns it; on the
/// other route it reads `None`, as does a field the reply left out. A
/// candidate field is the first candidate's, the one rig reads.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeminiExtras {
    /// GenerateContent `modelVersion`.
    pub model_version: Option<String>,
    /// GenerateContent `responseId`.
    pub response_id: Option<String>,
    /// The tier that served the request: GenerateContent
    /// `usageMetadata.serviceTier`, Interactions `service_tier`.
    pub service_tier: Option<String>,
    /// GenerateContent `usageMetadata.promptTokensDetails`.
    pub prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// GenerateContent `usageMetadata.cacheTokensDetails`.
    pub cache_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// GenerateContent `usageMetadata.candidatesTokensDetails`.
    pub candidates_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// GenerateContent `usageMetadata.toolUsePromptTokensDetails`.
    pub tool_use_prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// GenerateContent `promptFeedback`.
    pub prompt_feedback: Option<PromptFeedback>,
    /// GenerateContent `safetyRatings` of the candidate.
    pub safety_ratings: Option<Vec<SafetyRating>>,
    /// GenerateContent `finishMessage` of the candidate.
    pub finish_message: Option<String>,
    /// GenerateContent `citationMetadata` of the candidate.
    pub citation_metadata: Option<CitationMetadata>,
    /// GenerateContent `groundingMetadata` of the candidate.
    pub grounding_metadata: Option<GroundingMetadata>,
    /// GenerateContent `urlContextMetadata` of the candidate.
    pub url_context_metadata: Option<UrlContextMetadata>,
    /// GenerateContent `avgLogprobs` of the candidate.
    pub avg_logprobs: Option<f64>,
    /// GenerateContent `logprobsResult` of the candidate.
    pub logprobs_result: Option<LogprobsResult>,
    /// Interactions `id`.
    pub id: Option<String>,
    /// Interactions `status`.
    pub status: Option<InteractionStatus>,
    /// Interactions `created`, an RFC 3339 timestamp.
    pub created: Option<String>,
    /// Interactions `updated`, an RFC 3339 timestamp.
    pub updated: Option<String>,
    /// Interactions `usage.input_tokens_by_modality`.
    pub input_tokens_by_modality: Option<Vec<ModalityTokens>>,
    /// Interactions `usage.output_tokens_by_modality`.
    pub output_tokens_by_modality: Option<Vec<ModalityTokens>>,
    /// Interactions `usage.cached_tokens_by_modality`.
    pub cached_tokens_by_modality: Option<Vec<ModalityTokens>>,
    /// Interactions `usage.grounding_tool_count`.
    pub grounding_tool_count: Option<Vec<GroundingToolCount>>,
}

impl ReplyExtras for GeminiExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        match api.as_str() {
            GENERATE_CONTENT => Ok(GenerateContentReply::deserialize(raw)?.into()),
            INTERACTIONS => Ok(InteractionReply::deserialize(raw)?.into()),
            other => Err(serde_json::Error::custom(format!(
                "the Gemini API returns no `{other}` reply"
            ))),
        }
    }
}

/// The parts of a `generateContent` reply [`GeminiExtras`] reads.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentReply {
    model_version: Option<String>,
    response_id: Option<String>,
    usage_metadata: Option<UsageDetails>,
    prompt_feedback: Option<PromptFeedback>,
    #[serde(default)]
    candidates: Vec<CandidateDetails>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageDetails {
    service_tier: Option<String>,
    prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
    cache_tokens_details: Option<Vec<ModalityTokenCount>>,
    candidates_tokens_details: Option<Vec<ModalityTokenCount>>,
    tool_use_prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateDetails {
    safety_ratings: Option<Vec<SafetyRating>>,
    finish_message: Option<String>,
    citation_metadata: Option<CitationMetadata>,
    grounding_metadata: Option<GroundingMetadata>,
    url_context_metadata: Option<UrlContextMetadata>,
    avg_logprobs: Option<f64>,
    logprobs_result: Option<LogprobsResult>,
}

impl From<GenerateContentReply> for GeminiExtras {
    fn from(reply: GenerateContentReply) -> Self {
        let mut extras = Self {
            model_version: reply.model_version,
            response_id: reply.response_id,
            prompt_feedback: reply.prompt_feedback,
            ..Self::default()
        };
        if let Some(usage) = reply.usage_metadata {
            extras.service_tier = usage.service_tier;
            extras.prompt_tokens_details = usage.prompt_tokens_details;
            extras.cache_tokens_details = usage.cache_tokens_details;
            extras.candidates_tokens_details = usage.candidates_tokens_details;
            extras.tool_use_prompt_tokens_details = usage.tool_use_prompt_tokens_details;
        }
        if let Some(candidate) = reply.candidates.into_iter().next() {
            extras.safety_ratings = candidate.safety_ratings;
            extras.finish_message = candidate.finish_message;
            extras.citation_metadata = candidate.citation_metadata;
            extras.grounding_metadata = candidate.grounding_metadata;
            extras.url_context_metadata = candidate.url_context_metadata;
            extras.avg_logprobs = candidate.avg_logprobs;
            extras.logprobs_result = candidate.logprobs_result;
        }
        extras
    }
}

/// The parts of an interaction resource [`GeminiExtras`] reads.
#[derive(Deserialize)]
struct InteractionReply {
    id: Option<String>,
    status: Option<InteractionStatus>,
    service_tier: Option<String>,
    created: Option<String>,
    updated: Option<String>,
    usage: Option<InteractionUsage>,
}

#[derive(Deserialize)]
struct InteractionUsage {
    input_tokens_by_modality: Option<Vec<ModalityTokens>>,
    output_tokens_by_modality: Option<Vec<ModalityTokens>>,
    cached_tokens_by_modality: Option<Vec<ModalityTokens>>,
    grounding_tool_count: Option<Vec<GroundingToolCount>>,
}

impl From<InteractionReply> for GeminiExtras {
    fn from(reply: InteractionReply) -> Self {
        let mut extras = Self {
            id: reply.id,
            status: reply.status,
            service_tier: reply.service_tier,
            created: reply.created,
            updated: reply.updated,
            ..Self::default()
        };
        if let Some(usage) = reply.usage {
            extras.input_tokens_by_modality = usage.input_tokens_by_modality;
            extras.output_tokens_by_modality = usage.output_tokens_by_modality;
            extras.cached_tokens_by_modality = usage.cached_tokens_by_modality;
            extras.grounding_tool_count = usage.grounding_tool_count;
        }
        extras
    }
}

/// A token count of one modality, as GenerateContent reports it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModalityTokenCount {
    /// `modality`, such as `TEXT`.
    pub modality: Option<String>,
    /// `tokenCount`.
    pub token_count: Option<u64>,
}

/// Why the prompt was blocked, and how it rated.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    /// `blockReason`, such as `SAFETY`, when the prompt was blocked.
    pub block_reason: Option<String>,
    /// `safetyRatings`.
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

/// How a prompt or candidate rated in one harm category.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    /// `category`, such as `HARM_CATEGORY_HARASSMENT`.
    pub category: Option<String>,
    /// `probability`, such as `NEGLIGIBLE`.
    pub probability: Option<String>,
    /// `blocked`: whether this rating blocked the content.
    pub blocked: Option<bool>,
}

/// The sources a candidate recites.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    /// `citationSources`.
    pub citation_sources: Option<Vec<CitationSource>>,
}

/// One recited source.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationSource {
    /// `startIndex` of the reciting text.
    pub start_index: Option<u64>,
    /// `endIndex` of the reciting text.
    pub end_index: Option<u64>,
    /// `uri` of the source.
    pub uri: Option<String>,
    /// `license` of the source.
    pub license: Option<String>,
}

/// What grounded a candidate in search or retrieval.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    /// `webSearchQueries`.
    pub web_search_queries: Option<Vec<String>>,
    /// `searchEntryPoint`.
    pub search_entry_point: Option<SearchEntryPoint>,
    /// `groundingChunks`.
    pub grounding_chunks: Option<Vec<GroundingChunk>>,
    /// `groundingSupports`.
    pub grounding_supports: Option<Vec<GroundingSupport>>,
}

/// The search suggestion to show with a grounded answer.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEntryPoint {
    /// `renderedContent`: HTML and CSS to embed.
    pub rendered_content: Option<String>,
}

/// One grounding source.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingChunk {
    /// `web`: a web page.
    pub web: Option<WebChunk>,
    /// `retrievedContext`: a retrieved document.
    pub retrieved_context: Option<RetrievedContext>,
}

/// A web page a candidate is grounded in.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct WebChunk {
    /// `uri`.
    pub uri: Option<String>,
    /// `title`.
    pub title: Option<String>,
}

/// A retrieved document a candidate is grounded in.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct RetrievedContext {
    /// `uri`.
    pub uri: Option<String>,
    /// `title`.
    pub title: Option<String>,
    /// `text`.
    pub text: Option<String>,
}

/// A span of the candidate and the chunks that support it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    /// `segment`.
    pub segment: Option<Segment>,
    /// `groundingChunkIndices`, into `groundingChunks`.
    pub grounding_chunk_indices: Option<Vec<u32>>,
    /// `confidenceScores`, one per index.
    pub confidence_scores: Option<Vec<f64>>,
}

/// A span of a candidate's content.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    /// `partIndex`.
    pub part_index: Option<u32>,
    /// `startIndex`, in bytes.
    pub start_index: Option<u64>,
    /// `endIndex`, in bytes.
    pub end_index: Option<u64>,
    /// `text`.
    pub text: Option<String>,
}

/// The URLs the URL context tool retrieved.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UrlContextMetadata {
    /// `urlMetadata`.
    pub url_metadata: Option<Vec<UrlMetadata>>,
}

/// One URL the URL context tool retrieved.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UrlMetadata {
    /// `retrievedUrl`.
    pub retrieved_url: Option<String>,
    /// `urlRetrievalStatus`, such as `URL_RETRIEVAL_STATUS_SUCCESS`.
    pub url_retrieval_status: Option<String>,
}

/// The log probabilities of a candidate's tokens.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogprobsResult {
    /// `topCandidates`, one entry per decoding step.
    pub top_candidates: Option<Vec<TopCandidates>>,
    /// `chosenCandidates`, one per decoding step.
    pub chosen_candidates: Option<Vec<LogprobsCandidate>>,
    /// `logProbabilitySum`.
    pub log_probability_sum: Option<f64>,
}

/// The most likely tokens of one decoding step.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct TopCandidates {
    /// `candidates`, most likely first.
    pub candidates: Option<Vec<LogprobsCandidate>>,
}

/// A token and its log probability.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogprobsCandidate {
    /// `token`.
    pub token: Option<String>,
    /// `tokenId`.
    pub token_id: Option<i64>,
    /// `logProbability`.
    pub log_probability: Option<f64>,
}

/// Where an interaction stands.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionStatus {
    /// `in_progress`.
    InProgress,
    /// `requires_action`.
    RequiresAction,
    /// `incomplete`.
    Incomplete,
    /// `budget_exceeded`.
    BudgetExceeded,
    /// `completed`.
    Completed,
    /// `failed`.
    Failed,
    /// `cancelled`.
    Cancelled,
    /// A status this crate does not know yet, as the API spelled it.
    #[serde(untagged)]
    Unknown(String),
}

/// A token count of one modality, as Interactions reports it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct ModalityTokens {
    /// `modality`, such as `text`.
    pub modality: Option<String>,
    /// `tokens`.
    pub tokens: Option<u64>,
}

/// How often a grounding tool ran.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct GroundingToolCount {
    /// `type`, such as `google_search`.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// `count`.
    pub count: Option<u64>,
    /// `search_query_count`.
    pub search_query_count: Option<u64>,
}

#[cfg(test)]
mod tests;
