//! The Responses section of OpenAI's options, and the reply fields only a
//! Responses reply carries.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::provider_options::reply_field;

/// The fields only OpenAI's Responses route sends, each at the top level of
/// the body. Serialize-only; an unset field is not sent.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpenAiResponsesOptions {
    /// `reasoning.summary`, `.mode` and `.context`, merged beside the
    /// `reasoning.effort` that `GenerationOptions::reasoning` sends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningOptions>,
    /// `include`: extra output to return. The wire adds
    /// `reasoning.encrypted_content` itself when it replays reasoning.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<Include>,
    /// `conversation`: the id of a stored conversation the request joins.
    /// The history before it is then the provider's to hold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    /// `truncation`: what the provider does when the input outgrows the
    /// context window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
    /// `context_management`: server-side compaction of the context.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_management: Vec<ContextManagement>,
    /// `prompt_cache_options.comparison_response_id`, merged beside the
    /// `ttl` and `mode` that `GenerationOptions::cache` sends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_options: Option<PromptCacheOptions>,
    /// `background`: run the response asynchronously. HTTP only: a
    /// WebSocket session refuses it through the request's
    /// [`OnUnsupported`](crate::completion::OnUnsupported) policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// `max_tool_calls`: the most built-in tool calls the response makes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    /// `top_logprobs`: 0 to 20 most likely tokens per position. A model
    /// that samples only without reasoning refuses it while it reasons.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    /// `access_programs`: the access programs the request runs under.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_programs: Option<AccessPrograms>,
}

impl OpenAiResponsesOptions {
    /// Send `reasoning.summary`.
    #[must_use]
    pub fn reasoning_summary(mut self, summary: ReasoningSummary) -> Self {
        self.reasoning.get_or_insert_with(Default::default).summary = Some(summary);
        self
    }

    /// Send `reasoning.mode`.
    #[must_use]
    pub fn reasoning_mode(mut self, mode: ReasoningMode) -> Self {
        self.reasoning.get_or_insert_with(Default::default).mode = Some(mode);
        self
    }

    /// Send `reasoning.context`.
    #[must_use]
    pub fn reasoning_context(mut self, context: ReasoningContext) -> Self {
        self.reasoning.get_or_insert_with(Default::default).context = Some(context);
        self
    }

    /// Add `include` entries.
    #[must_use]
    pub fn include(mut self, include: impl IntoIterator<Item = Include>) -> Self {
        self.include.extend(include);
        self
    }

    /// Send `conversation`.
    #[must_use]
    pub fn conversation(mut self, id: impl Into<String>) -> Self {
        self.conversation = Some(id.into());
        self
    }

    /// Send `truncation`.
    #[must_use]
    pub fn truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Add `context_management` entries.
    #[must_use]
    pub fn context_management(
        mut self,
        entries: impl IntoIterator<Item = ContextManagement>,
    ) -> Self {
        self.context_management.extend(entries);
        self
    }

    /// Send `prompt_cache_options.comparison_response_id`.
    #[must_use]
    pub fn prompt_cache_comparison(mut self, response_id: impl Into<String>) -> Self {
        self.prompt_cache_options = Some(PromptCacheOptions {
            comparison_response_id: Some(response_id.into()),
        });
        self
    }

    /// Send `background`.
    #[must_use]
    pub fn background(mut self, background: bool) -> Self {
        self.background = Some(background);
        self
    }

    /// Send `max_tool_calls`.
    #[must_use]
    pub fn max_tool_calls(mut self, max: u32) -> Self {
        self.max_tool_calls = Some(max);
        self
    }

    /// Send `top_logprobs`.
    #[must_use]
    pub fn top_logprobs(mut self, count: u8) -> Self {
        self.top_logprobs = Some(count);
        self
    }

    /// Send `access_programs`.
    #[must_use]
    pub fn access_programs(mut self, programs: AccessPrograms) -> Self {
        self.access_programs = Some(programs);
        self
    }
}

/// The `reasoning` fields beside its effort.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReasoningOptions {
    /// `reasoning.summary`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
    /// `reasoning.mode`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<ReasoningMode>,
    /// `reasoning.context`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ReasoningContext>,
}

/// How much of its reasoning the model summarizes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    /// The model's choice.
    Auto,
    /// A short summary.
    Concise,
    /// A detailed summary.
    Detailed,
}

/// The reasoning mode, independent of the effort. GPT-5.6 and later take
/// it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// The default mode.
    Standard,
    /// Pro mode.
    Pro,
}

/// Which earlier turns' reasoning the model reuses. GPT-5.6 and later take
/// it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningContext {
    /// The model's choice.
    Auto,
    /// Reasoning from every earlier turn.
    AllTurns,
    /// Only the current turn's reasoning.
    CurrentTurn,
}

/// Extra output a response returns.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Include {
    /// `file_search_call.results`.
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// `web_search_call.results`.
    #[serde(rename = "web_search_call.results")]
    WebSearchCallResults,
    /// `web_search_call.action.sources`.
    #[serde(rename = "web_search_call.action.sources")]
    WebSearchCallActionSources,
    /// `message.input_image.image_url`.
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    /// `computer_call_output.output.image_url`.
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputOutputImageUrl,
    /// `code_interpreter_call.outputs`.
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    /// `reasoning.encrypted_content`.
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    /// `message.output_text.logprobs`.
    #[serde(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
}

/// What the provider does when the input outgrows the context window.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    /// Drop input items from the middle of the conversation.
    Auto,
    /// Fail the request.
    Disabled,
}

/// One `context_management` entry.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextManagement {
    /// Compact the context once it reaches `compact_threshold` tokens, or
    /// at the provider's threshold.
    Compaction {
        /// The token count compaction starts at.
        #[serde(skip_serializing_if = "Option::is_none")]
        compact_threshold: Option<u32>,
    },
}

impl ContextManagement {
    /// Compaction at `threshold` tokens, or at the provider's threshold.
    pub fn compaction(threshold: Option<u32>) -> Self {
        Self::Compaction {
            compact_threshold: threshold,
        }
    }
}

/// The `prompt_cache_options` fields `GenerationOptions::cache` does not
/// send.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PromptCacheOptions {
    /// `comparison_response_id`: an earlier response whose cache use this
    /// one is compared with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison_response_id: Option<String>,
}

/// The access programs a request runs under.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AccessPrograms {
    /// `cyber`: the cyber-security access program.
    pub cyber: CyberAccess,
}

impl AccessPrograms {
    /// The `cyber` program at `access`.
    pub fn cyber(access: CyberAccess) -> Self {
        Self { cyber: access }
    }
}

/// The cyber-security access program's level.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CyberAccess {
    /// `standard`.
    Standard,
    /// `daybreak_blue`.
    DaybreakBlue,
    /// `daybreak_red`.
    DaybreakRed,
}

/// The `phase` of one `message` item of a Responses reply.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ItemPhase {
    /// The item's `id`.
    pub id: String,
    /// The item's `phase`, such as `final_answer`, when it names one.
    #[serde(default)]
    pub phase: Option<String>,
}

/// The fields a Responses reply's envelope carries, as the OpenAI and
/// ChatGPT extras read them.
pub(crate) struct Envelope {
    pub(crate) service_tier: Option<String>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) reasoning_summary: Option<String>,
    pub(crate) reasoning_mode: Option<String>,
    pub(crate) reasoning_context: Option<String>,
    pub(crate) prompt_cache_retention: Option<String>,
    pub(crate) incomplete_reason: Option<String>,
    pub(crate) phases: Option<Vec<ItemPhase>>,
}

impl Envelope {
    /// The envelope of `raw`. A `reasoning` that is not an object, as a
    /// compatible server's text reasoning, gives no reasoning fields.
    pub(crate) fn from_reply(raw: &Value) -> Result<Self, serde_json::Error> {
        let reasoning = raw.get("reasoning").filter(|value| value.is_object());
        let reasoning = |key: &str| match reasoning {
            Some(reasoning) => reply_field::<String>(reasoning, &format!("/{key}")),
            None => Ok(None),
        };
        let messages = raw
            .get("output")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
            .map(ItemPhase::deserialize)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            service_tier: reply_field(raw, "/service_tier")?,
            reasoning_effort: reasoning("effort")?,
            reasoning_summary: reasoning("summary")?,
            reasoning_mode: reasoning("mode")?,
            reasoning_context: reasoning("context")?,
            prompt_cache_retention: reply_field(raw, "/prompt_cache_retention")?,
            incomplete_reason: reply_field(raw, "/incomplete_details/reason")?,
            phases: (!messages.is_empty()).then_some(messages),
        })
    }
}

#[cfg(test)]
mod tests;
