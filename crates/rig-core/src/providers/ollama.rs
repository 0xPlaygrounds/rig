//! Ollama configuration, model identifiers, message conversion, and NDJSON decoding.
//!
//! ```no_run
//! use rig_core::providers::ollama;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = ollama::Ollama::new();
//!
//! let qwen = provider.completion("qwen2.5:14b");
//! let embeddings = provider.embedding(ollama::ALL_MINILM, Some(384));
//! # Ok(())
//! # }
//! ```
//!
//! Pair a wire with a transport in a [`crate::Model`] to execute it.
//! `Ollama::from_env` reads
//! `OLLAMA_API_BASE_URL` and `OLLAMA_API_KEY` for remote or authenticated daemons.
use crate::completion::Usage;
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::message::DocumentSourceKind;
use crate::model::ModelInfo;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal;
use crate::providers::openai::wire::dto::open_once;
use crate::wire::{Flow, Out};
use crate::{
    completion::{self, CompletionRequest},
    json_utils, message,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub mod wire;

pub use crate::client::ollama::Ollama;
pub use wire::{Chat, Embeddings, Models, OllamaConfig};

/// The address of a local daemon.
const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

/// Stable descriptor name recorded on normalized responses, streams, and
/// telemetry spans for this provider.
const PROVIDER_NAME: &str = "ollama";

/// The `all-minilm` embedding model.
pub const ALL_MINILM: &str = "all-minilm";
/// The `nomic-embed-text` embedding model.
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";
/// The `mxbai-embed-large` embedding model.
pub const MXBAI_EMBED_LARGE: &str = "mxbai-embed-large";
/// The `bge-m3` multilingual embedding model.
pub const BGE_M3: &str = "bge-m3";
/// The `embeddinggemma` embedding model.
pub const EMBEDDINGGEMMA: &str = "embeddinggemma";
/// The `qwen3-embedding` embedding model family; dimensions vary by size, so pass them explicitly.
pub const QWEN3_EMBEDDING: &str = "qwen3-embedding";

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        ALL_MINILM => Some(384),
        NOMIC_EMBED_TEXT => Some(768),
        MXBAI_EMBED_LARGE => Some(1024),
        BGE_M3 => Some(1024),
        EMBEDDINGGEMMA => Some(768),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
}

/// The `llama3.2` model.
pub const LLAMA3_2: &str = "llama3.2";
/// The `llama3.1` model.
pub const LLAMA3_1: &str = "llama3.1";
/// The `llama3.3` model.
pub const LLAMA3_3: &str = "llama3.3";
/// The `llama4` multimodal model.
pub const LLAMA4: &str = "llama4";
/// The `llava` vision model.
pub const LLAVA: &str = "llava";
/// The `mistral` model.
pub const MISTRAL: &str = "mistral";
/// The `mistral-small3.2` model.
pub const MISTRAL_SMALL3_2: &str = "mistral-small3.2";
/// The `gemma3` model.
pub const GEMMA3: &str = "gemma3";
/// The `gemma4` model.
pub const GEMMA4: &str = "gemma4";
/// The `qwen3` model.
pub const QWEN3: &str = "qwen3";
/// The `qwen3.5` model.
pub const QWEN3_5: &str = "qwen3.5";
/// The `qwen3.6` model.
pub const QWEN3_6: &str = "qwen3.6";
/// The `qwen3.8` model.
pub const QWEN3_8: &str = "qwen3.8";
/// The `qwen3-coder` model.
pub const QWEN3_CODER: &str = "qwen3-coder";
/// The `deepseek-r1` reasoning model.
pub const DEEPSEEK_R1: &str = "deepseek-r1";
/// The `deepseek-v3.1` model.
pub const DEEPSEEK_V3_1: &str = "deepseek-v3.1";
/// The `gpt-oss` model.
pub const GPT_OSS: &str = "gpt-oss";
/// The `phi4` model.
pub const PHI4: &str = "phi4";

/// One `/api/chat` record. Only `message` and `done` build the turn, so
/// every other field is optional, and a counter that is not a count is
/// unreported.
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub created_at: String,
    /// The record's assistant message, as Ollama sent it.
    pub message: serde_json::Map<String, Value>,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    pub total_duration: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    pub load_duration: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    pub prompt_eval_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    pub eval_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    pub eval_duration: Option<u64>,
}

/// Map Ollama's `done_reason` onto rig's normalized vocabulary: `stop` and
/// `length` as documented, and `load` and `unload` (a request that only
/// loaded or unloaded the model) as stops. Any other reason is kept
/// verbatim, which fails the turn.
pub(crate) fn map_done_reason(reason: &str) -> completion::FinishReason {
    match reason {
        "stop" | "load" | "unload" => completion::FinishReason::Stop,
        "length" => completion::FinishReason::Length,
        other => completion::FinishReason::Other(other.to_owned()),
    }
}

/// Ollama reports prompt and generation counts but no total; the total is
/// derived only when both are present.
fn ollama_usage(prompt_eval_count: Option<u64>, eval_count: Option<u64>) -> Usage {
    Usage {
        input_tokens: prompt_eval_count,
        output_tokens: eval_count,
        total_tokens: prompt_eval_count
            .zip(eval_count)
            .map(|(input, output)| input + output),
        ..Default::default()
    }
}

/// The reasoning and the visible text of content that opens with a
/// terminated reasoning block, both trimmed, or `None` when it does not.
/// When allowed, the Qwen prefilled-start boundary counts; ordinary mentions
/// of the markers do not.
fn split_legacy_thinking(content: &str, permits_omitted_start: bool) -> Option<(&str, &str)> {
    let trimmed = content.trim_start();
    let (reasoning, visible) = if let Some(reasoning_start) = trimmed.strip_prefix("<think>") {
        reasoning_start.split_once("</think>")?
    } else if permits_omitted_start {
        // Qwen's prefilled opening marker produces this exact blank-line
        // boundary. Requiring the full boundary avoids hiding ordinary visible
        // text that merely demonstrates a closing XML-like tag on its own line.
        trimmed.split_once("\n</think>\n\n")?
    } else {
        return None;
    };
    Some((reasoning.trim(), visible.trim_start()))
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaCompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<Think>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<schemars::Schema>,
    options: serde_json::Value,
}

impl TryFrom<(&str, CompletionRequest)> for OllamaCompletionRequest {
    type Error = EncodeError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let chat_history = req.chat_history_with_documents();
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        if req.tool_choice.is_some() {
            tracing::warn!("WARNING: `tool_choice` not supported for Ollama");
        }
        let full_history = chat_history
            .into_iter()
            .map(message::Message::try_into)
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        let mut think: Option<Think> = None;
        let mut keep_alive: Option<String> = None;

        // The native API has no top-level `temperature` or `max_tokens`;
        // both are model parameters that belong in `options` (`max_tokens`
        // is called `num_predict` there).
        let mut base_options = serde_json::Map::new();
        if let Some(temperature) = req.temperature {
            base_options.insert("temperature".to_string(), json!(temperature));
        }
        if let Some(max_tokens) = req.max_tokens {
            base_options.insert("num_predict".to_string(), json!(max_tokens));
        }
        let base_options = Value::Object(base_options);

        let options = if let Some(mut extra) = req.additional_params {
            // These controls belong at the request root, not in model options.
            if let Some(obj) = extra.as_object_mut() {
                if let Some(think_val) = obj.shift_remove("think") {
                    think = Some(match think_val {
                        Value::Bool(think) => Think::Bool(think),
                        Value::String(think) => Think::Level(match think.to_lowercase().as_str() {
                            "low" => Level::Low,
                            "medium" => Level::Medium,
                            "high" => Level::High,
                            "max" => Level::Max,
                            _ => {
                                return Err(EncodeError::request(
                                    "`think` must be a 'low', 'medium', 'high', 'max' or bool",
                                ));
                            }
                        }),
                        _ => {
                            return Err(EncodeError::request(
                                "`think` must be a 'low', 'medium', 'high', 'max' or bool",
                            ));
                        }
                    });
                }

                if let Some(keep_alive_val) = obj.shift_remove("keep_alive") {
                    keep_alive = Some(
                        keep_alive_val
                            .as_str()
                            .ok_or_else(|| EncodeError::request("`keep_alive` must be a string"))?
                            .to_string(),
                    );
                }
            }

            json_utils::merge(base_options, extra)
        } else {
            base_options
        };

        Ok(Self {
            model,
            messages: full_history,
            stream: false,
            think,
            keep_alive,
            format: req.output_schema,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            options,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum Think {
    Bool(bool),
    Level(Level),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Level {
    Low,
    Medium,
    High,
    Max,
}

/// Ollama's terminal stream record: the `done: true` line's counters as rig
/// parsed them, a streamed response's `raw`.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
    /// Provider-reported model identifier from the terminating NDJSON line.
    pub model: String,
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}

impl From<&StreamingCompletionResponse> for Usage {
    fn from(response: &StreamingCompletionResponse) -> Usage {
        ollama_usage(response.prompt_eval_count, response.eval_count)
    }
}

/// Ollama's `done: true` record as the provider's end of the reply.
fn finish_of(response: StreamingCompletionResponse) -> Finish {
    // Ollama's `/api/chat` stream assigns no response identifier, so the
    // origin's `response_id` stays unset.
    Finish {
        usage: Usage::from(&response),
        reason: response.done_reason.as_deref().map(map_done_reason),
        model: Some(response.model),
        ..Finish::default()
    }
}

/// The writer index of a reply's `n`th tool call.
const CALL_INDEX: usize = 1 << 24;

/// Decode `/api/chat` records, one whole reply or a stream of lines. Each
/// record's message is a delta of the turn's: its thinking and content grow
/// one reasoning and one text block, each tool call arrives whole, and the
/// assembled message is the turn's native. Only a `done: true` record ends
/// the reply; EOF alone does not.
///
/// A model that writes its reasoning into `content` (`<think>…</think>`, or
/// Qwen3 after its template prefilled the opening tag) is split the same way
/// in both modes: its content is held until the reasoning closes, or until
/// the reply ends for a Qwen3 model that never closes one, and its message
/// keeps the reasoning under `thinking`.
#[derive(Default)]
pub struct OllamaDecoder {
    /// The message as assembled so far, without its tool calls.
    message: serde_json::Map<String, Value>,
    tool_calls: Vec<Value>,
    reasoning: Option<usize>,
    text: Option<usize>,
    /// How many tool calls had arrived when the reasoning and the text block
    /// opened: a fragment after a later call opens the next block.
    opened_at: [usize; 2],
    /// The open reasoning block's text: each block holds its own part of
    /// the message's `thinking`.
    thinking: String,
    /// Content held while it may still open with inline reasoning.
    held: String,
    /// Whether the content's shape is known: explicit thinking arrived, or
    /// the inline reasoning closed, or the content cannot open with any.
    settled: bool,
}

impl OllamaDecoder {
    /// Append `fragment` to the reasoning or text block, opening it first.
    fn push(
        &mut self,
        reasoning: bool,
        fragment: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if fragment.is_empty() {
            return Ok(());
        }
        let calls = self.tool_calls.len();
        let [reasoning_at, text_at] = &mut self.opened_at;
        let (slot, opened_at, block) = if reasoning {
            (
                &mut self.reasoning,
                reasoning_at,
                Block::Reasoning { redacted: false },
            )
        } else {
            (&mut self.text, text_at, Block::Text)
        };
        if let Some(index) = *slot
            && *opened_at < calls
        {
            // A call ends the block: what follows it is the next block.
            *slot = None;
            if reasoning {
                let thinking = std::mem::take(&mut self.thinking);
                out.finish_with(index, json!({ "thinking": thinking }))?;
            } else {
                out.finish(index)?;
            }
        }
        if slot.is_none() {
            *opened_at = calls;
        }
        let index = open_once(slot, block, out)?;
        if reasoning {
            self.thinking.push_str(fragment);
        }
        out.push(index, fragment)
    }

    /// Content, held while it may open with inline reasoning and released
    /// once its shape is known, or when the reply ends.
    fn content(
        &mut self,
        fragment: &str,
        permits_omitted_start: bool,
        done: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if self.settled {
            return self.push(false, fragment, out);
        }
        self.held.push_str(fragment);
        let held = std::mem::take(&mut self.held);
        if let Some((reasoning, visible)) = split_legacy_thinking(&held, permits_omitted_start) {
            self.settled = true;
            self.push(true, reasoning, out)?;
            return self.push(false, visible, out);
        }
        let opening = held.trim_start();
        let may_open = permits_omitted_start
            || opening.starts_with("<think>")
            || "<think>".starts_with(opening);
        if done || !may_open {
            self.settled = true;
            return self.push(false, &held, out);
        }
        self.held = held;
        Ok(())
    }

    /// Write a record's content and calls, and end the reply when it is
    /// `done`.
    fn interpret_record(
        &mut self,
        response: CompletionResponse,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let mut message = response.message;
        if let Some(thinking) = message.get("thinking").and_then(Value::as_str)
            && !thinking.is_empty()
        {
            self.settled = true;
            self.push(true, thinking, &mut out)?;
        }
        let fragment = match message.get("content") {
            Some(Value::String(content)) => content.as_str(),
            None | Some(Value::Null) => "",
            Some(other) => {
                return Err(ProviderError::Response(format!(
                    "Ollama message content is not a string: {other}"
                )));
            }
        };
        let permits_omitted_start = response.model.to_ascii_lowercase().contains("qwen3");
        // A call ends the content before it: whatever is held is released.
        let calls_follow = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty());
        self.content(
            fragment,
            permits_omitted_start,
            response.done || calls_follow,
            &mut out,
        )?;
        if let Some(Value::Array(calls)) = message.shift_remove("tool_calls") {
            for call in calls {
                // Ollama sends each call whole, with object arguments (a
                // proxy may stringify them). A nameless call is dropped by
                // the writer; an id-less one gets an id rig issues, never
                // the tool name.
                let text = |pointer: &str| call.pointer(pointer).and_then(Value::as_str);
                let arguments = match call.pointer("/function/arguments") {
                    Some(Value::String(text)) => text.clone(),
                    Some(arguments) => arguments.to_string(),
                    None => String::new(),
                };
                let index = CALL_INDEX + self.tool_calls.len();
                out.fragment(
                    index,
                    CallFragment {
                        id: text("/id"),
                        name: text("/function/name"),
                        arguments: Some(&arguments),
                    },
                )?;
                out.finish_with(index, call.clone())?;
                self.tool_calls.push(call);
            }
        }
        crate::providers::openai::wire::dto::merge_fields(&mut self.message, &message);

        // Nonterminal counters do not establish successful turn completion.
        if !response.done {
            return Ok(Flow::More);
        }
        let mut message = std::mem::take(&mut self.message);
        // A reply that wrote its reasoning inline is kept in the shape Ollama
        // gives a thinking reply: the reasoning under `thinking`.
        let inline = message
            .get("content")
            .and_then(Value::as_str)
            .filter(|_| {
                message
                    .get("thinking")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            })
            .and_then(|content| split_legacy_thinking(content, permits_omitted_start))
            .map(|(reasoning, visible)| (reasoning.to_owned(), visible.to_owned()));
        if let Some((reasoning, visible)) = inline {
            if !reasoning.is_empty() {
                message.insert("thinking".to_owned(), reasoning.into());
            }
            message.insert("content".to_owned(), visible.into());
        }
        if let Some(index) = self.reasoning.take() {
            let thinking = std::mem::take(&mut self.thinking);
            out.finish_with(index, json!({ "thinking": thinking }))?;
        }
        if let Some(index) = self.text.take() {
            out.finish(index)?;
        }
        if !self.tool_calls.is_empty() {
            message.insert(
                "tool_calls".to_owned(),
                Value::Array(std::mem::take(&mut self.tool_calls)),
            );
        }
        out.message_native(Value::Object(message));
        let native = StreamingCompletionResponse {
            model: response.model,
            total_duration: response.total_duration,
            load_duration: response.load_duration,
            prompt_eval_count: response.prompt_eval_count,
            prompt_eval_duration: response.prompt_eval_duration,
            eval_count: response.eval_count,
            eval_duration: response.eval_duration,
            done_reason: response.done_reason,
        };
        out.raw(serde_json::to_value(&native)?);
        Ok(out.end(finish_of(native)))
    }

    /// Classify one NDJSON line. The wire has no discriminator at all: a
    /// line either decodes as the record shape or is corrupt.
    fn classify_line(frame: crate::wire::WireFrame) -> crate::wire::WireEvent<CompletionResponse> {
        match frame {
            crate::wire::WireFrame::Bytes(line) => internal::wire::classify_untyped_line(&line),
            crate::wire::WireFrame::Text(line) => {
                internal::wire::classify_untyped_line(line.as_bytes())
            }
        }
    }
}

/// EOF without a `done: true` record is truncation.
impl<'id> crate::wire::Decoder<'id, Completion> for OllamaDecoder {
    type Event = CompletionResponse;

    fn classify(
        &self,
        frame: crate::wire::WireFrame,
    ) -> crate::wire::WireEvent<CompletionResponse> {
        OllamaDecoder::classify_line(frame)
    }

    fn decode(
        &mut self,
        response: CompletionResponse,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.interpret_record(response, out)
    }
}

/// The reply of `GET /api/tags`: every model the daemon has pulled.
#[derive(Debug, Deserialize)]
pub struct ListModelsResponse {
    /// The installed models, in the daemon's own order.
    pub models: Vec<ListModelEntry>,
}

/// One installed model.
#[derive(Debug, Deserialize)]
pub struct ListModelEntry {
    /// The tag as the daemon displays it (`qwen3:4b`).
    pub name: String,
    /// The identifier a request addresses.
    pub model: String,
}

impl From<ListModelEntry> for ModelInfo {
    fn from(value: ListModelEntry) -> Self {
        ModelInfo::new(value.model, value.name)
    }
}

/// Ollama-required tool definition format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: completion::ToolDefinition,
}

/// Convert internal ToolDefinition (from the completion module) into Ollama's tool definition.
impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        ToolDefinition {
            type_field: "function".to_owned(),
            function: completion::ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    /// Optional call ID. Replayed exactly when a provider issued one (the
    /// daemon's own, or another wire's when a history is ported); a locally
    /// minted handle never travels upstream because the slot is optional and
    /// results still correlate by `tool_name`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub r#type: ToolType,
    pub function: Function,
}
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, deserialize_with = "json_utils::null_or_default")]
        tool_calls: Vec<ToolCall>,
    },
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        #[serde(rename = "tool_name")]
        name: String,
        content: String,
        /// The provider-issued id of the call this result answers, when one
        /// was issued.
        #[serde(
            rename = "tool_call_id",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        call_id: Option<String>,
    },
    /// An assistant message as the wire carries it: the daemon's own
    /// message, or one rebuilt from a turn's blocks. Request-only.
    #[serde(untagged, skip_deserializing)]
    Native(Value),
}

/// Combine text and supported image/document content into one user message.
/// Reject unsupported media sources and tool results.
fn user_message_from_content(
    content: Vec<crate::message::UserContent>,
) -> Result<Message, crate::message::MessageError> {
    let mut texts = Vec::new();
    let mut images = Vec::new();

    for content in content {
        match content {
            crate::message::UserContent::Text(crate::message::Text { text, .. }) => {
                texts.push(text);
            }
            crate::message::UserContent::Image(crate::message::Image {
                data: DocumentSourceKind::Base64(data),
                ..
            }) => images.push(data),
            crate::message::UserContent::Image(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama images must be base64 encoded data".into(),
                ));
            }
            crate::message::UserContent::Document(crate::message::Document {
                data: DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data),
                ..
            }) => texts.push(data),
            crate::message::UserContent::Document(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama documents must be string or base64 encoded data".into(),
                ));
            }
            crate::message::UserContent::Audio(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama does not support audio user content".into(),
                ));
            }
            crate::message::UserContent::Video(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama does not support video user content".into(),
                ));
            }
            crate::message::UserContent::ToolResult(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "tool results must be converted to a separate Ollama message".into(),
                ));
            }
        }
    }

    Ok(Message::User {
        content: texts.join(" "),
        images: (!images.is_empty()).then_some(images),
        name: None,
    })
}

/// Convert system, user, and assistant messages. User tool results become
/// separate name-keyed messages; unsupported media returns a conversion error.
impl TryFrom<crate::message::Message> for Vec<Message> {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        use crate::message::Message as InternalMessage;
        match internal_msg {
            InternalMessage::System { content } => Ok(vec![Message::System {
                content,
                images: None,
                name: None,
            }]),
            InternalMessage::User { content, .. } => {
                let mut messages = Vec::new();
                let mut pending_user_content = Vec::new();

                for content in content {
                    match content {
                        crate::message::UserContent::ToolResult(crate::message::ToolResult {
                            call,
                            name,
                            content,
                            is_error: _,
                        }) => {
                            let function_name = name;
                            if !pending_user_content.is_empty() {
                                messages.push(user_message_from_content(std::mem::take(
                                    &mut pending_user_content,
                                ))?);
                            }

                            let content = content
                                .into_iter()
                                .map(|content| match content {
                                    crate::message::ToolResultContent::Text(text) => Ok(text.text),
                                    crate::message::ToolResultContent::Json { value } => {
                                        Ok(value.to_string())
                                    }
                                    // The adapter moves a result's images to a user
                                    // message, since Ollama reads none here.
                                    crate::message::ToolResultContent::Image(_) => {
                                        Ok(crate::completion::history::TOOL_IMAGE_OMITTED
                                            .to_owned())
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .join("\n");
                            messages.push(Message::ToolResult {
                                name: function_name.into(),
                                content,
                                call_id: call.provider().map(|id| id.as_str().to_owned()),
                            });
                        }
                        content => pending_user_content.push(content),
                    }
                }

                if !pending_user_content.is_empty() {
                    messages.push(user_message_from_content(pending_user_content)?);
                }

                Ok(messages)
            }
            InternalMessage::Assistant(turn) => Ok(vec![assistant_message(turn)]),
        }
    }
}

/// One assistant turn as Ollama takes it, rebuilt from its blocks as pi
/// rebuilds a message: its text joined into `content`, its reasoning joined
/// into `thinking`, and each call with its canonical name and object
/// arguments. A rig-issued call id stays off the wire: the daemon pairs a
/// result with its call by tool name.
fn assistant_message(turn: crate::message::AssistantMessage) -> Message {
    use crate::providers::internal::rebuild::{Rebuilt, call_item};
    let rebuilt = Rebuilt::of(&turn);
    let mut message = serde_json::Map::new();
    message.insert("role".to_owned(), "assistant".into());
    message.insert("content".to_owned(), rebuilt.text().into());
    let thinking: Vec<String> = rebuilt
        .reasoning(Some("thinking"))
        .into_iter()
        .map(|(_, text)| text)
        .collect();
    if !thinking.is_empty() {
        message.insert("thinking".to_owned(), thinking.join("\n").into());
    }
    let calls: Vec<Value> = rebuilt
        .calls
        .into_iter()
        .map(|(call, item)| {
            let id = call.id.provider().map(|id| id.as_str().to_owned());
            call_item(&call, item, id, false)
        })
        .collect();
    if !calls.is_empty() {
        message.insert("tool_calls".to_owned(), calls.into());
    }
    Message::Native(Value::Object(message))
}

impl Message {
    /// Constructs a system message.
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            images: None,
            name: None,
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod history_tests;
