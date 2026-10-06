//! Typed request options and reply extras for Anthropic's own Messages API.
//! [`Anthropic`] keys them by [`PROVIDER_NAME`](super::PROVIDER_NAME); the
//! Messages wire reads the entry only when its dialect is Anthropic, so a
//! gateway speaking the same format never receives them.
//!
//! A field marked "+ beta" needs its `anthropic-beta` flag set on the
//! provider with [`AnthropicConfig::with_beta`](super::AnthropicConfig::with_beta);
//! without it the API rejects the request.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::anthropic::extension::{Anthropic, AnthropicOptions, InferenceGeo};
//!
//! # fn run() -> Result<(), rig_core::completion::OptionsError> {
//! let options = AnthropicOptions::default()
//!     .metadata_user_id("user-7")
//!     .inference_geo(InferenceGeo::Us);
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<Anthropic>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use super::completion::{CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_OPUS_5_5, claude_spec};
use super::wire::MESSAGES_API;
use crate::catalog::Sampling;
use crate::completion::{
    CompletionRequest, ExtensionOptions, ProviderExtension, ReplayTarget, ReplyExtras,
};
use crate::message::Api;

/// The `anthropic-beta` flag fast mode needs.
pub const FAST_MODE_BETA: &str = "fast-mode-2026-02-01";
/// The `anthropic-beta` flag task budgets need.
pub const TASK_BUDGETS_BETA: &str = "task-budgets-2026-03-13";
/// The `anthropic-beta` flag [`Fallbacks::Default`] needs.
pub const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// The `anthropic-beta` flag [`Fallbacks::Models`] needs.
pub const SERVER_SIDE_FALLBACK_MODELS_BETA: &str = "server-side-fallback-2026-06-01";
/// The `anthropic-beta` flag context editing needs.
pub const CONTEXT_MANAGEMENT_BETA: &str = "context-management-2025-06-27";
/// The `anthropic-beta` flag the MCP connector needs.
pub const MCP_CLIENT_BETA: &str = "mcp-client-2025-11-20";
/// The `anthropic-beta` flag cache diagnostics need.
pub const CACHE_DIAGNOSIS_BETA: &str = "cache-diagnosis-2026-04-07";

/// The models fast mode runs on.
const FAST_MODELS: &[&str] = &[CLAUDE_OPUS_5_5, CLAUDE_OPUS_5, CLAUDE_OPUS_4_8];

/// Anthropic's typed options and extras.
#[derive(Clone, Copy, Debug, Default)]
pub struct Anthropic;

impl ProviderExtension for Anthropic {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = AnthropicOptions;
    type Extras = AnthropicExtras;
}

/// Anthropic's request options. The Messages API is its one route, so every
/// field sits in the shared section.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AnthropicOptions {
    /// The fields every route sends.
    #[serde(rename = "*")]
    pub shared: AnthropicShared,
}

/// The fields of [`AnthropicOptions`]. Each unset field is left out of the
/// body.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnthropicShared {
    /// `top_k`: sample from the `k` most likely tokens. Refused on models
    /// that fix their sampling.
    pub top_k: Option<u32>,
    /// `metadata.user_id`: an opaque id of the end user, at most 512
    /// characters.
    pub metadata_user_id: Option<String>,
    /// `inference_geo`: where inference runs.
    pub inference_geo: Option<InferenceGeo>,
    /// `speed`. [`Speed::Fast`] + beta [`FAST_MODE_BETA`], on Claude Opus
    /// 5.5, Opus 5 and Opus 4.8 only; refused on other listed models.
    pub speed: Option<Speed>,
    /// `output_config.task_budget` + beta [`TASK_BUDGETS_BETA`].
    pub task_budget: Option<TaskBudget>,
    /// `fallbacks`: the models a refused request is re-run on, + beta
    /// [`SERVER_SIDE_FALLBACK_BETA`] or [`SERVER_SIDE_FALLBACK_MODELS_BETA`].
    pub fallbacks: Option<Fallbacks>,
    /// `container`: the code-execution container to reuse, or the skills to
    /// load into it. It replaces the container the history names.
    pub container: Option<ContainerParam>,
    /// `context_management` + beta [`CONTEXT_MANAGEMENT_BETA`].
    pub context_management: Option<ContextManagement>,
    /// `mcp_servers` + beta [`MCP_CLIENT_BETA`]. Each server also needs an
    /// `mcp_toolset` tool naming it.
    pub mcp_servers: Vec<McpServer>,
    /// `diagnostics.previous_message_id` + beta [`CACHE_DIAGNOSIS_BETA`]:
    /// `Some(None)` sends `null` on a conversation's first request,
    /// `Some(Some(id))` the previous reply's id.
    pub diagnostics_previous_message_id: Option<Option<String>>,
}

impl Serialize for AnthropicShared {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self {
            top_k,
            metadata_user_id,
            inference_geo,
            speed,
            task_budget,
            fallbacks,
            container,
            context_management,
            mcp_servers,
            diagnostics_previous_message_id,
        } = self;
        let mut map = serializer.serialize_map(None)?;
        if let Some(top_k) = top_k {
            map.serialize_entry("top_k", top_k)?;
        }
        if let Some(user_id) = metadata_user_id {
            map.serialize_entry("metadata", &Nested("user_id", user_id))?;
        }
        if let Some(geo) = inference_geo {
            map.serialize_entry("inference_geo", geo)?;
        }
        if let Some(speed) = speed {
            map.serialize_entry("speed", speed)?;
        }
        if let Some(budget) = task_budget {
            map.serialize_entry("output_config", &Nested("task_budget", budget))?;
        }
        if let Some(fallbacks) = fallbacks {
            map.serialize_entry("fallbacks", fallbacks)?;
        }
        if let Some(container) = container {
            map.serialize_entry("container", container)?;
        }
        if let Some(context_management) = context_management {
            map.serialize_entry("context_management", context_management)?;
        }
        if !mcp_servers.is_empty() {
            map.serialize_entry("mcp_servers", mcp_servers)?;
        }
        if let Some(previous) = diagnostics_previous_message_id {
            map.serialize_entry("diagnostics", &Nested("previous_message_id", previous))?;
        }
        map.end()
    }
}

/// A one-key object: `{key: value}`.
pub(crate) struct Nested<'a, T>(pub(crate) &'static str, pub(crate) &'a T);

impl<T: Serialize> Serialize for Nested<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(self.0, self.1)?;
        map.end()
    }
}

impl AnthropicOptions {
    /// Set [`AnthropicShared::top_k`].
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.shared.top_k = Some(top_k);
        self
    }

    /// Set [`AnthropicShared::metadata_user_id`].
    pub fn metadata_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.shared.metadata_user_id = Some(user_id.into());
        self
    }

    /// Set [`AnthropicShared::inference_geo`].
    pub fn inference_geo(mut self, geo: InferenceGeo) -> Self {
        self.shared.inference_geo = Some(geo);
        self
    }

    /// Set [`AnthropicShared::speed`].
    pub fn speed(mut self, speed: Speed) -> Self {
        self.shared.speed = Some(speed);
        self
    }

    /// Set [`AnthropicShared::task_budget`].
    pub fn task_budget(mut self, budget: TaskBudget) -> Self {
        self.shared.task_budget = Some(budget);
        self
    }

    /// Set [`AnthropicShared::fallbacks`].
    pub fn fallbacks(mut self, fallbacks: Fallbacks) -> Self {
        self.shared.fallbacks = Some(fallbacks);
        self
    }

    /// Set [`AnthropicShared::container`].
    pub fn container(mut self, container: impl Into<ContainerParam>) -> Self {
        self.shared.container = Some(container.into());
        self
    }

    /// Set [`AnthropicShared::context_management`].
    pub fn context_management(mut self, context_management: ContextManagement) -> Self {
        self.shared.context_management = Some(context_management);
        self
    }

    /// Add one server to [`AnthropicShared::mcp_servers`].
    pub fn mcp_server(mut self, server: McpServer) -> Self {
        self.shared.mcp_servers.push(server);
        self
    }

    /// Set [`AnthropicShared::diagnostics_previous_message_id`].
    pub fn diagnostics_previous_message_id(mut self, previous: Option<String>) -> Self {
        self.shared.diagnostics_previous_message_id = Some(previous);
        self
    }
}

impl ExtensionOptions for AnthropicOptions {
    /// `top_k` on a model that fixes its sampling, and fast mode on a
    /// listed model that does not offer it. A model the catalog does not
    /// list is sent everything.
    fn unsupported(
        &self,
        target: &dyn ReplayTarget,
        request: &CompletionRequest,
    ) -> Vec<(&'static str, String)> {
        let model = request
            .model
            .as_deref()
            .filter(|model| !model.is_empty())
            .unwrap_or_else(|| target.model());
        let Some(spec) = claude_spec(model) else {
            return Vec::new();
        };
        let mut refused = Vec::new();
        if self.shared.top_k.is_some() && spec.sampling == Some(Sampling::Never) {
            refused.push(("top_k", "this model does not take `top_k`".to_owned()));
        }
        if self.shared.speed == Some(Speed::Fast) && !FAST_MODELS.contains(&spec.id.as_str()) {
            refused.push((
                "speed",
                "fast mode runs on Claude Opus 5.5, Opus 5 and Opus 4.8 only".to_owned(),
            ));
        }
        refused
    }
}

/// Where inference runs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceGeo {
    /// `"us"`: in the United States only.
    Us,
    /// `"global"`: anywhere.
    Global,
}

/// The output speed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Speed {
    /// `"standard"`.
    Standard,
    /// `"fast"`: fast mode, at its own price.
    Fast,
}

/// A token budget the model paces an agentic loop by, at least 20000
/// tokens.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskBudget {
    /// The whole budget.
    pub total: u32,
    /// What is left of it, when history was rewritten and the server can no
    /// longer count it. Unset, the server counts it.
    pub remaining: Option<u32>,
}

impl TaskBudget {
    /// A budget of `total` tokens the server counts down.
    pub fn new(total: u32) -> Self {
        Self {
            total,
            remaining: None,
        }
    }

    /// `self` with `remaining` tokens left.
    pub fn remaining(mut self, remaining: u32) -> Self {
        self.remaining = Some(remaining);
        self
    }
}

impl Serialize for TaskBudget {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self { total, remaining } = self;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", "tokens")?;
        map.serialize_entry("total", total)?;
        if let Some(remaining) = remaining {
            map.serialize_entry("remaining", remaining)?;
        }
        map.end()
    }
}

/// The models a refused request is re-run on.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fallbacks {
    /// `"default"`: Anthropic picks by refusal category. Beta
    /// [`SERVER_SIDE_FALLBACK_BETA`].
    Default,
    /// `[{"model": ..}, ..]`: these models, in order. Beta
    /// [`SERVER_SIDE_FALLBACK_MODELS_BETA`].
    Models(Vec<String>),
}

impl Serialize for Fallbacks {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Models(models) => {
                serializer.collect_seq(models.iter().map(|model| Nested("model", model)))
            }
        }
    }
}

/// The container a request runs code in.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ContainerParam {
    /// An existing container's id.
    Id(String),
    /// A container with skills loaded: `id` reuses one, unset starts one.
    Skills {
        /// The container to reuse.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The skills to load.
        skills: Vec<Skill>,
    },
}

impl From<String> for ContainerParam {
    fn from(id: String) -> Self {
        Self::Id(id)
    }
}

impl From<&str> for ContainerParam {
    fn from(id: &str) -> Self {
        Self::Id(id.to_owned())
    }
}

/// A skill loaded into a container.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Skill {
    /// Who publishes it.
    #[serde(rename = "type")]
    pub kind: SkillKind,
    /// Its id: a prebuilt skill's name (`"xlsx"`) or a custom skill's id.
    pub skill_id: String,
    /// Its version, such as `"latest"`. Unset, the latest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl Skill {
    /// The skill `skill_id` from `kind`, at its latest version.
    pub fn new(kind: SkillKind, skill_id: impl Into<String>) -> Self {
        Self {
            kind,
            skill_id: skill_id.into(),
            version: None,
        }
    }

    /// `self` at `version`.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
}

/// Who publishes a skill.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillKind {
    /// `"anthropic"`: a prebuilt skill.
    Anthropic,
    /// `"custom"`: one the organization uploaded.
    Custom,
}

/// Context editing: the edits the server applies before the model reads the
/// conversation.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ContextManagement {
    /// Each edit as the API spells it, such as
    /// `{"type": "clear_tool_uses_20250919"}`. Edits are versioned server
    /// types, so they are written as JSON.
    pub edits: Vec<Value>,
}

impl ContextManagement {
    /// The edits `edits`.
    pub fn new(edits: Vec<Value>) -> Self {
        Self { edits }
    }
}

/// A remote MCP server the API connects to.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpServer {
    /// The server's URL.
    pub url: String,
    /// The name an `mcp_toolset` tool refers to it by.
    pub name: String,
    /// An OAuth token the API sends to it.
    pub authorization_token: Option<String>,
}

impl McpServer {
    /// The server `name` at `url`, with no token.
    pub fn new(url: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            name: name.into(),
            authorization_token: None,
        }
    }

    /// `self` with `token` sent to the server.
    pub fn authorization_token(mut self, token: impl Into<String>) -> Self {
        self.authorization_token = Some(token.into());
        self
    }
}

impl Serialize for McpServer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self {
            url,
            name,
            authorization_token,
        } = self;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", "url")?;
        map.serialize_entry("url", url)?;
        map.serialize_entry("name", name)?;
        if let Some(token) = authorization_token {
            map.serialize_entry("authorization_token", token)?;
        }
        map.end()
    }
}

/// The fields of an Anthropic Messages reply rig does not normalize. Each
/// is `None` when the reply leaves it out.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnthropicExtras {
    /// `stop_reason`, verbatim.
    pub stop_reason: Option<String>,
    /// `stop_sequence`: the stop sequence the turn ended on.
    pub stop_sequence: Option<String>,
    /// `stop_details`: why a refusal was declined.
    pub stop_details: Option<StopDetails>,
    /// `usage.cache_creation`: the cache writes per lifetime.
    pub cache_creation: Option<CacheCreation>,
    /// `usage.service_tier`: `standard`, `priority` or `batch`.
    pub service_tier: Option<String>,
    /// `usage.inference_geo`: where inference ran.
    pub inference_geo: Option<String>,
    /// `usage.speed`: the speed the reply ran at.
    pub speed: Option<String>,
    /// `usage.server_tool_use`: the server tools' request counts.
    pub server_tool_use: Option<ServerToolUse>,
    /// `container`: the code-execution container the reply ran in.
    pub container: Option<Container>,
    /// The model a leading `fallback` block switched to, when the request
    /// was refused and re-run on another model.
    pub fallback_model: Option<String>,
}

/// Why a reply was refused.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct StopDetails {
    /// `type`: `"refusal"`.
    #[serde(rename = "type", default)]
    pub kind: String,
    /// The policy category, such as `"cyber"`.
    #[serde(default)]
    pub category: Option<String>,
    /// A readable explanation.
    #[serde(default)]
    pub explanation: Option<String>,
}

/// The input tokens written to the cache, per cache lifetime.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct CacheCreation {
    /// Written with a five-minute lifetime.
    #[serde(default)]
    pub ephemeral_5m_input_tokens: u64,
    /// Written with a one-hour lifetime.
    #[serde(default)]
    pub ephemeral_1h_input_tokens: u64,
}

/// How many requests the server tools made.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct ServerToolUse {
    /// Web searches.
    #[serde(default)]
    pub web_search_requests: u64,
    /// Web fetches.
    #[serde(default)]
    pub web_fetch_requests: u64,
}

/// A code-execution container.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct Container {
    /// Its id, which a later request reuses.
    #[serde(default)]
    pub id: String,
    /// When it expires, as an RFC 3339 time.
    #[serde(default)]
    pub expires_at: String,
}

/// The reply fields [`AnthropicExtras`] reads.
#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_sequence: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
    #[serde(default)]
    usage: Option<ReplyUsage>,
    #[serde(default)]
    container: Option<Container>,
    #[serde(default)]
    content: Vec<Value>,
}

#[derive(Default, Deserialize)]
struct ReplyUsage {
    #[serde(default)]
    cache_creation: Option<CacheCreation>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    inference_geo: Option<String>,
    #[serde(default)]
    speed: Option<String>,
    #[serde(default)]
    server_tool_use: Option<ServerToolUse>,
}

impl ReplyExtras for AnthropicExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() != MESSAGES_API {
            return Err(serde::de::Error::custom(format!(
                "Anthropic extras read a Messages reply, not one from `{api}`"
            )));
        }
        let Reply {
            stop_reason,
            stop_sequence,
            stop_details,
            usage,
            container,
            content,
        } = Reply::deserialize(raw)?;
        let usage = usage.unwrap_or_default();
        let fallback_model = content
            .first()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("fallback"))
            .and_then(|block| block.pointer("/to/model"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        Ok(Self {
            stop_reason,
            stop_sequence,
            stop_details,
            cache_creation: usage.cache_creation,
            service_tier: usage.service_tier,
            inference_geo: usage.inference_geo,
            speed: usage.speed,
            server_tool_use: usage.server_tool_use,
            container,
            fallback_model,
        })
    }
}

/// The reply fields every Messages-format dialect returns, which the
/// dialects' extras read on this route.
#[derive(Deserialize)]
pub(crate) struct MessagesStop {
    #[serde(default)]
    pub(crate) stop_reason: Option<String>,
    #[serde(default)]
    pub(crate) stop_sequence: Option<String>,
}

impl MessagesStop {
    /// The stop fields of `raw`, a reply on `api`, for `provider`'s extras.
    ///
    /// # Errors
    ///
    /// When `api` is not the Messages API or `raw` is not an object.
    pub(crate) fn read(provider: &str, api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() != MESSAGES_API {
            return Err(serde::de::Error::custom(format!(
                "{provider} extras read a Messages reply, not one from `{api}`"
            )));
        }
        Self::deserialize(raw)
    }
}

#[cfg(test)]
mod tests;
