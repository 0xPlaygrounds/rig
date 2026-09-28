//! The Interactions API's schema as Rust, maintained by hand: Google
//! publishes no discovery document for it. Every struct keeps the fields it
//! does not type in [`Unmodeled`], so what Gemini sends survives a read and a
//! re-encoding.
//!
//! Hosted-tool steps keep their `signature` and `search_type`: a step replayed
//! without either is refused.
//!
//! ```
//! use rig_core::providers::gemini::interactions_api::api;
//!
//! let settings = api::RequestSettings {
//!     service_tier: Some(api::ServiceTier::Deferred),
//!     generation_config: api::GenerationSettings {
//!         thinking_level: Some(api::ThinkingLevel::Low),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! # let _ = settings;
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::message::{NativePart, Sealed};
use crate::providers::gemini::api::{Mirrored, Unmodeled, mirror_enum};

/// The schema of a native Interactions step.
pub const STEP_SCHEMA: &str = "gemini.interactions.Step";
/// The schema of a native content item of a `model_output` step.
pub const CONTENT_SCHEMA: &str = "gemini.interactions.Content";
/// The schema of a text item's annotations, kept beside its text.
pub const ANNOTATIONS_SCHEMA: &str = "gemini.interactions.Annotations";

/// Implement [`Mirrored`] for a hand-maintained struct.
macro_rules! mirrored {
    ($name:ident [$($field:literal),* $(,)?]) => {
        impl Mirrored for $name {
            const NAME: &'static str = stringify!($name);
            const FIELDS: &'static [&'static str] = &[$($field),*];
            fn unmodeled_fields(&self, path: &str, out: &mut Vec<String>) {
                out.extend(self.unmodeled.keys().map(|key| format!("{path}.{key}")));
            }
        }
    };
}

mirror_enum! {
    ServiceTier {
        Standard => "standard",
        Flex => "flex",
        Priority => "priority",
        Deferred => "deferred",
    }
}

mirror_enum! {
    ThinkingLevel {
        Minimal => "minimal",
        Low => "low",
        Medium => "medium",
        High => "high",
    }
}

mirror_enum! {
    ThinkingSummaries {
        Auto => "auto",
        None => "none",
    }
}

mirror_enum! {
    ResponseModality {
        Text => "text",
        Image => "image",
        Audio => "audio",
    }
}

mirror_enum! {
    InteractionStatus {
        InProgress => "in_progress",
        RequiresAction => "requires_action",
        Incomplete => "incomplete",
        BudgetExceeded => "budget_exceeded",
        Completed => "completed",
        Failed => "failed",
        Cancelled => "cancelled",
    }
}

impl InteractionStatus {
    /// Whether polling can stop: every status but `in_progress`.
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::InProgress)
    }
}

/// Everything an interaction request carries that rig does not own: rig
/// owns the input, system instruction, model, stream flag, function tools,
/// tool choice, temperature and output limit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestSettings {
    /// A managed agent to run instead of a model, such as Deep Research.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// The agent's configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_config: Option<Value>,
    /// Run in the background; poll the interaction for its result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// Whether Gemini stores the interaction for chaining and retrieval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Continue from a stored interaction instead of resending history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_interaction_id: Option<String>,
    /// The output format, per modality.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// The output MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// The modalities the model may answer in.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_modalities: Vec<ResponseModality>,
    /// Standard, flex, priority or deferred serving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Generation options rig does not own.
    #[serde(default, skip_serializing_if = "GenerationSettings::is_empty")]
    pub generation_config: GenerationSettings,
    /// Hosted tools: Google Search, code execution, URL context and others.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<HostedTool>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(RequestSettings [
    "agent", "agent_config", "background", "store", "previous_interaction_id",
    "response_format", "response_mime_type", "response_modalities", "service_tier",
    "generation_config", "tools", "input", "model", "system_instruction", "stream",
    "safety_settings",
]);

/// Generation options rig does not own. Interactions has no
/// `media_resolution` here; resolution is per content item.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationSettings {
    /// How deeply the model thinks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
    /// Whether thought summaries are returned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_summaries: Option<ThinkingSummaries>,
    /// Nucleus sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// A decoding seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Sequences that end generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Voices for audio output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<Value>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(GenerationSettings [
    "thinking_level", "thinking_summaries", "top_p", "seed", "stop_sequences",
    "speech_config", "temperature", "max_output_tokens", "tool_choice", "media_resolution",
]);

impl GenerationSettings {
    /// Whether nothing is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A hosted tool, tagged by its `type`: `google_search`, `code_execution`,
/// `url_context`, `computer_use`, `mcp_server`, `file_search`, `google_maps`
/// or one Google adds later.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HostedTool {
    /// The tool's kind.
    #[serde(rename = "type")]
    pub kind: String,
    /// The tool's options, by Google's names.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(HostedTool["type"]);

impl HostedTool {
    /// A tool of `kind` with no options.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            unmodeled: Unmodeled::default(),
        }
    }
}

/// A function tool declaration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionTool {
    /// Always `function`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The function's name.
    pub name: String,
    /// What the function does.
    pub description: String,
    /// Its arguments' JSON Schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// The generation config rig sends: the settings plus the fields rig owns.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GenerationConfig {
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// The output limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Which tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// The settings' fields.
    #[serde(flatten)]
    pub settings: GenerationSettings,
}

/// Token usage of an interaction. Cached tokens are reported beside the
/// input they are part of.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_cached_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tool_use_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_thought_tokens: Option<u64>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(Usage [
    "total_tokens", "total_input_tokens", "total_cached_tokens", "total_output_tokens",
    "total_tool_use_tokens", "total_thought_tokens",
]);

/// An interaction resource.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Interaction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<InteractionStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<Step>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(Interaction ["id", "status", "model", "agent", "steps", "usage"]);

impl Interaction {
    /// Whether polling can stop.
    pub fn is_terminal(&self) -> bool {
        self.status
            .as_ref()
            .is_some_and(InteractionStatus::is_terminal)
    }

    /// The text of the model's output steps, joined.
    pub fn text(&self) -> String {
        self.steps
            .iter()
            .filter_map(|step| match step {
                Step::ModelOutput(output) => Some(output),
                _ => None,
            })
            .flat_map(|output| &output.content)
            .filter_map(|content| match content {
                Content::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// One step of an interaction, tagged by its `type`.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// What the user said.
    UserInput(Contents),
    /// What the model said.
    ModelOutput(Contents),
    /// The model's thought, and the signature that closes it.
    Thought(Thought),
    /// A function the model asks the client to run.
    FunctionCall(FunctionCall),
    /// The client's answer to a call.
    FunctionResult(FunctionResult),
    /// A hosted tool's call or result (`google_search_call`,
    /// `code_execution_result`, ...).
    Hosted(HostedStep),
    /// A step kind this mirror does not know yet, as Gemini sent it.
    Unknown(Map<String, Value>),
}

impl Step {
    /// The step's `type`.
    pub fn kind(&self) -> &str {
        match self {
            Self::UserInput(_) => "user_input",
            Self::ModelOutput(_) => "model_output",
            Self::Thought(_) => "thought",
            Self::FunctionCall(_) => "function_call",
            Self::FunctionResult(_) => "function_result",
            Self::Hosted(step) => &step.kind,
            Self::Unknown(map) => map.get("type").and_then(Value::as_str).unwrap_or_default(),
        }
    }
}

/// Whether `kind` names a hosted tool's call or result step.
pub fn is_hosted(kind: &str) -> bool {
    (kind.ends_with("_call") || kind.ends_with("_result"))
        && !matches!(kind, "function_call" | "function_result")
}

fn tagged<T: Serialize>(kind: &str, value: &T) -> Result<Value, serde_json::Error> {
    let mut value = serde_json::to_value(value)?;
    if let Value::Object(map) = &mut value {
        map.insert("type".to_owned(), Value::String(kind.to_owned()));
    }
    Ok(value)
}

impl Serialize for Step {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let value = match self {
            Self::UserInput(contents) => tagged("user_input", contents),
            Self::ModelOutput(contents) => tagged("model_output", contents),
            Self::Thought(thought) => tagged("thought", thought),
            Self::FunctionCall(call) => tagged("function_call", call),
            Self::FunctionResult(result) => tagged("function_result", result),
            Self::Hosted(step) => serde_json::to_value(step),
            Self::Unknown(map) => Ok(Value::Object(map.clone())),
        }
        .map_err(S::Error::custom)?;
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Step {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let mut map = Map::<String, Value>::deserialize(deserializer)?;
        let kind = map
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let read = |map: Map<String, Value>| Value::Object(map);
        Ok(match kind.as_str() {
            "user_input" => {
                map.remove("type");
                Self::UserInput(serde_json::from_value(read(map)).map_err(D::Error::custom)?)
            }
            "model_output" => {
                map.remove("type");
                Self::ModelOutput(serde_json::from_value(read(map)).map_err(D::Error::custom)?)
            }
            "thought" => {
                map.remove("type");
                Self::Thought(serde_json::from_value(read(map)).map_err(D::Error::custom)?)
            }
            "function_call" => {
                map.remove("type");
                Self::FunctionCall(serde_json::from_value(read(map)).map_err(D::Error::custom)?)
            }
            "function_result" => {
                map.remove("type");
                Self::FunctionResult(serde_json::from_value(read(map)).map_err(D::Error::custom)?)
            }
            kind if is_hosted(kind) => {
                Self::Hosted(serde_json::from_value(read(map)).map_err(D::Error::custom)?)
            }
            _ => Self::Unknown(map),
        })
    }
}

/// The content items of a `user_input` or `model_output` step.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Contents {
    #[serde(default)]
    pub content: Vec<Content>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(Contents["content"]);

/// A thought: its summary, and the signature Gemini needs back.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Thought {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summary: Vec<Content>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(Thought ["signature", "summary"]);

/// A function call.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(FunctionCall ["id", "name", "arguments", "signature"]);

/// A function call's result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(FunctionResult ["call_id", "name", "result", "is_error"]);

/// A hosted tool's call or result step. Its `signature` and, for a search,
/// its `search_type` must return with it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HostedStep {
    /// `google_search_call`, `code_execution_result`, ...
    #[serde(rename = "type")]
    pub kind: String,
    /// A call's id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The call a result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Which Google Search ran: `web_search` or `image_search`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(HostedStep [
    "type", "id", "call_id", "signature", "search_type", "arguments", "result", "is_error",
]);

impl HostedStep {
    /// Restore what a streamed step leaves out but its kind implies: a
    /// streamed `google_search_call` omits `search_type`.
    pub(crate) fn restore(&mut self) {
        if self.kind == "google_search_call" && self.search_type.is_none() {
            self.search_type = Some("web_search".to_owned());
        }
        if self.signature.as_deref() == Some("") {
            self.signature = None;
        }
    }
}

/// A content item, tagged by its `type`.
#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Text(TextContent),
    Image(MediaContent),
    Audio(MediaContent),
    Document(MediaContent),
    Video(MediaContent),
    /// A content kind this mirror does not know yet, as Gemini sent it.
    Unknown(Map<String, Value>),
}

impl Serialize for Content {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let value = match self {
            Self::Text(text) => tagged("text", text),
            Self::Image(media) => tagged("image", media),
            Self::Audio(media) => tagged("audio", media),
            Self::Document(media) => tagged("document", media),
            Self::Video(media) => tagged("video", media),
            Self::Unknown(map) => Ok(Value::Object(map.clone())),
        }
        .map_err(S::Error::custom)?;
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Content {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let mut map = Map::<String, Value>::deserialize(deserializer)?;
        let kind = map
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let media = |mut map: Map<String, Value>| {
            map.remove("type");
            serde_json::from_value::<MediaContent>(Value::Object(map)).map_err(D::Error::custom)
        };
        Ok(match kind.as_str() {
            "text" => {
                map.remove("type");
                Self::Text(serde_json::from_value(Value::Object(map)).map_err(D::Error::custom)?)
            }
            "image" => Self::Image(media(map)?),
            "audio" => Self::Audio(media(map)?),
            "document" => Self::Document(media(map)?),
            "video" => Self::Video(media(map)?),
            _ => Self::Unknown(map),
        })
    }
}

/// Text, and the citations Gemini attached to it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<Value>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(TextContent ["text", "annotations"]);

/// Inline or referenced media.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MediaContent {
    /// Base64 bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// A URI Gemini fetches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// `low`, `medium`, `high` or `ultra_high`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    /// Fields this mirror does not type yet.
    #[serde(flatten)]
    pub unmodeled: Unmodeled<Self>,
}

mirrored!(MediaContent ["data", "uri", "mime_type", "resolution"]);

/// One server-sent event of an interaction stream, tagged by `event_type`.
/// The step and delta stay raw: the decoder classifies them.
#[derive(Debug, Deserialize)]
pub struct Event {
    pub event_type: String,
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub interaction: Option<Interaction>,
    #[serde(default)]
    pub status: Option<InteractionStatus>,
    #[serde(default)]
    pub step: Option<Box<serde_json::value::RawValue>>,
    #[serde(default)]
    pub delta: Option<Box<serde_json::value::RawValue>>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub event_id: Option<String>,
}

/// A native Interactions part that does not read as a [`Step`].
#[derive(Debug, thiserror::Error)]
pub enum NativeError {
    /// Another service issued the part.
    #[error("native part was issued by `{0}`")]
    OtherIssuer(crate::message::Issuer),
    /// The part is not a step.
    #[error("native part follows `{0}`")]
    OtherSchema(String),
    /// The JSON is not a step.
    #[error(transparent)]
    Decode(#[from] serde_json::Error),
}

impl TryFrom<&Sealed<NativePart>> for Step {
    type Error = NativeError;

    fn try_from(native: &Sealed<NativePart>) -> Result<Self, Self::Error> {
        let part = native
            .open(&super::super::ISSUER)
            .ok_or_else(|| NativeError::OtherIssuer(native.issuer().clone()))?;
        if part.schema != STEP_SCHEMA {
            return Err(NativeError::OtherSchema(part.schema.to_string()));
        }
        Ok(serde_json::from_str(part.json())?)
    }
}
