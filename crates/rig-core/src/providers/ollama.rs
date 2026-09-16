//! Ollama API integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::ollama;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // The local daemon, unauthenticated; `from_env()` reads
//! // `OLLAMA_API_BASE_URL` and `OLLAMA_API_KEY` when a proxied daemon
//! // needs them.
//! let provider = ollama::Ollama::new();
//!
//! let qwen = provider.chat("qwen2.5:14b");
//! let embeddings = provider.embeddings(ollama::ALL_MINILM, Some(384));
//! # Ok(())
//! # }
//! ```
//!
//! A wire says what to send and how to read the reply; `.bind(transport)`
//! joins it to a socket and yields the [`Bound`](crate::driver::Bound) that
//! implements the consumer-facing model traits.
use crate::completion::Usage;
use crate::message::DocumentSourceKind;
use crate::model::Model;
use crate::operation::Completion;
use crate::providers::internal;
use crate::streaming::{StreamFinal, ToolCallEnd};
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils, message,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub mod wire;

pub use wire::{Chat, Embeddings, Models, Ollama};

/// The address of a local daemon.
const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

/// Stable descriptor name recorded on normalized responses, streams, and
/// telemetry spans for this provider.
const PROVIDER_NAME: &str = "ollama";

// ---------- Embedding API ----------

// Model names follow <https://ollama.com/library>.

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

// ---------- Completion API ----------

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

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}
/// Map Ollama's `done_reason` onto rig's normalized vocabulary.
///
/// Ollama documents `stop` and `length`, but also emits operational reasons
/// such as `load`/`unload`; those are carried verbatim in Ollama's own spelling
/// rather than being flattened into a natural stop.
pub(crate) fn map_done_reason(reason: &str) -> completion::FinishReason {
    match reason {
        "stop" => completion::FinishReason::Stop,
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

/// Older reasoning models served by Ollama sometimes returned their reasoning
/// in `content` instead of `thinking`. Qwen can also omit the opening marker
/// because its chat template prefills it. Only split a leading, terminated
/// reasoning block so ordinary mentions of the marker remain untouched.
fn split_legacy_thinking(content: &str, permits_omitted_start: bool) -> (Option<&str>, &str) {
    let trimmed = content.trim_start();
    let split = if let Some(reasoning_start) = trimmed.strip_prefix("<think>") {
        reasoning_start.split_once("</think>")
    } else if permits_omitted_start {
        // Qwen's prefilled opening marker produces this exact blank-line
        // boundary. Requiring the full boundary avoids hiding ordinary visible
        // text that merely demonstrates a closing XML-like tag on its own line.
        trimmed.split_once("\n</think>\n\n")
    } else {
        None
    };
    let Some((reasoning, visible)) = split else {
        return (None, content);
    };

    let reasoning = reasoning.trim();
    if reasoning.is_empty() {
        return (None, visible.trim_start());
    }

    (Some(reasoning), visible.trim_start())
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
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let chat_history = req.chat_history_with_documents();
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        if req.tool_choice.is_some() {
            tracing::warn!("WARNING: `tool_choice` not supported for Ollama");
        }
        // Build up the order of messages.
        let mut partial_history = vec![];
        partial_history.extend(chat_history);
        // Ollama tool messages are name-keyed: cross-provider ingested
        // results arrive with an empty name and their call carries it.
        crate::providers::internal::resolve_empty_tool_result_names(&mut partial_history);

        let mut full_history: Vec<Message> = Vec::new();
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten(),
        );

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
            // Extract top-level parameters that should not be in `options`
            if let Some(obj) = extra.as_object_mut() {
                // Extract `think` parameter
                if let Some(think_val) = obj.remove("think") {
                    think = Some(match think_val {
                        Value::Bool(think) => Think::Bool(think),
                        Value::String(think) => Think::Level(match think.to_lowercase().as_str() {
                            "low" => Level::Low,
                            "medium" => Level::Medium,
                            "high" => Level::High,
                            "max" => Level::Max,
                            _ => {
                                return Err(CompletionError::RequestError(
                                    "`think` must be a 'low', 'medium', 'high', 'max' or bool"
                                        .into(),
                                ));
                            }
                        }),
                        _ => {
                            return Err(CompletionError::RequestError(
                                "`think` must be a 'low', 'medium', 'high', 'max' or bool".into(),
                            ));
                        }
                    });
                }

                // Extract `keep_alive` parameter
                if let Some(keep_alive_val) = obj.remove("keep_alive") {
                    keep_alive = Some(
                        keep_alive_val
                            .as_str()
                            .ok_or_else(|| {
                                CompletionError::RequestError(
                                    "`keep_alive` must be a string".into(),
                                )
                            })?
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

// ---------- CompletionModel Implementation ----------

/// Ollama's terminal stream record: the `done: true` line's counters as rig
/// parsed them, serialized onto [`StreamFinal::raw`] by the adapter's
/// terminal mapping.
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

/// The adapter's terminal mapping: Ollama's `done: true` record as a
/// normalized [`StreamFinal`] (the caller attaches `raw`).
fn stream_final(response: StreamingCompletionResponse) -> StreamFinal {
    // Ollama's `/api/chat` stream assigns no message identifier, so the
    // normalized `message_id` stays unset.
    StreamFinal::new(PROVIDER_NAME, Usage::from(&response))
        .with_optional_finish_reason(response.done_reason.as_deref().map(map_done_reason))
        .with_model(response.model)
}

/// The Ollama NDJSON wire's decoder, serving both replies.
///
/// Ollama answers `/api/chat` with the same record shape either way — a
/// stream is a sequence of them and a whole reply is one with `done: true`
/// — so the unary body needs no second variant here, and the two paths
/// cannot drift. Frame-triage policy (in-band `Err` on `Corrupt`, so a
/// later genuine `done: true` record can still complete the stream) lives
/// in the driver, not here.
pub struct OllamaDecoder {
    /// Owns the constant-key reasoning lifecycle: `thinking` deltas
    /// accumulate under the per-reply minted key, and the boundary end
    /// this wire never announces is derived, not hand-rolled here.
    reasoning: internal::chunk_lifecycle::MintedReasoningLifecycle,
    /// Per-reply minter for id-less tool-call keys. Counted across the
    /// whole reply, not per record — a per-record enumeration would hand
    /// two id-less calls in separate records the same `Minted(Tool, 0)`
    /// key, and one would silently swallow the other downstream.
    tool_ids: crate::streaming::SyntheticIds,
}

impl Default for OllamaDecoder {
    fn default() -> Self {
        Self {
            reasoning: internal::chunk_lifecycle::MintedReasoningLifecycle::new(
                crate::streaming::MintKind::Reasoning,
            ),
            tool_ids: crate::streaming::SyntheticIds::tool(),
        }
    }
}

impl OllamaDecoder {
    /// Interpret one `/api/chat` record: its content, its tool calls, and —
    /// when it says `done` — the terminal it carries.
    fn interpret_record(
        &mut self,
        response: CompletionResponse,
        out: &mut crate::operation::AdapterOutput,
    ) {
        let done = response.done;
        let model = response.model;
        if let Message::Assistant {
            content,
            thinking,
            tool_calls,
            ..
        } = response.message
        {
            // A daemon-issued call id keys the reply and travels as the
            // durable id; an id-less call (older daemons) keys by a
            // distinct minted identity and its durable id stays absent —
            // never the tool name, which would collide two same-tool calls
            // in one turn.
            let mut tool_events = crate::operation::AdapterOutput::new();
            for tool_call in tool_calls {
                let key = match tool_call
                    .id
                    .as_deref()
                    .and_then(crate::streaming::non_empty_id)
                {
                    Some(wire_id) => crate::streaming::BlockId::wire(wire_id.as_str()),
                    None => self.tool_ids.mint(),
                };
                let mut end =
                    ToolCallEnd::whole(tool_call.function.name, tool_call.function.arguments);
                if let Some(wire_id) = key.wire_str() {
                    end = end.with_tool_id(wire_id);
                }
                tool_events.tool_call(key, end);
            }

            // Older reasoning models put their reasoning in `content`
            // instead of `thinking`. Splitting it out needs the WHOLE
            // content, so only a record that completes the turn is a
            // candidate: a streamed delta carries a fragment, where a
            // leading `<think>` has no terminator yet and the content is
            // left alone (issue #1926 keeps the reasoning either way).
            let (reasoning, text) = match thinking.as_deref() {
                None | Some("") if done => {
                    let permits_omitted_think_start = model.to_ascii_lowercase().contains("qwen3");
                    let (legacy, visible) =
                        split_legacy_thinking(&content, permits_omitted_think_start);
                    (legacy.map(str::to_owned), visible.to_owned())
                }
                _ => (thinking, content),
            };

            // Declare what the record carried; the shared lifecycle derives
            // the canonical sequence (boundary end included).
            self.reasoning.emit_chunk(
                internal::chunk_lifecycle::ChunkParts {
                    reasoning,
                    reasoning_signature: None,
                    text: Some(text),
                    tool_events: tool_events
                        .into_items()
                        .into_iter()
                        .filter_map(Result::ok)
                        .collect(),
                },
                out,
            );
        }

        // Only a `done: true` record counts as the provider completing the
        // turn; the driver stops consuming after the terminal record, and
        // the span is the driver's to record.
        if done {
            let native = StreamingCompletionResponse {
                model,
                total_duration: response.total_duration,
                load_duration: response.load_duration,
                prompt_eval_count: response.prompt_eval_count,
                prompt_eval_duration: response.prompt_eval_duration,
                eval_count: response.eval_count,
                eval_duration: response.eval_duration,
                done_reason: response.done_reason,
            };
            match serde_json::to_value(&native) {
                Ok(raw) => out.final_record(stream_final(native).with_raw(raw)),
                Err(err) => out.error(err.into()),
            }
        }
    }

    /// Classify one NDJSON line. The wire has no discriminator at all: a
    /// line either decodes as the record shape or is corrupt.
    fn classify_line(
        frame: crate::wire::WireFrame,
    ) -> internal::wire::WireEvent<CompletionResponse> {
        match frame {
            crate::wire::WireFrame::Bytes(line) => internal::wire::classify_untyped_line(&line),
            crate::wire::WireFrame::Text(line) => {
                internal::wire::classify_untyped_line(line.as_bytes())
            }
        }
    }
}

impl crate::wire::Decoder<Completion> for OllamaDecoder {
    type Event = CompletionResponse;

    fn classify(
        &self,
        frame: crate::wire::WireFrame,
    ) -> internal::wire::WireEvent<CompletionResponse> {
        Self::classify_line(frame)
    }

    fn interpret(
        &mut self,
        response: CompletionResponse,
        out: &mut crate::operation::AdapterOutput,
    ) {
        self.interpret_record(response, out);
    }

    /// EOF without a `done: true` record is truncation: no terminal record
    /// may be synthesized.
    fn finish(&mut self, _out: &mut crate::operation::AdapterOutput) {}
}

// ---------- Model Listing  ----------

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

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        Model::new(value.model, value.name)
    }
}

// ---------- Tool Definition Conversion ----------

/// Ollama-required tool definition format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String, // Fixed as "function"
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
    /// The daemon-issued call id (`"id":"call_..."`), present on modern
    /// Ollama daemons and absent on older ones. Read when present — it is
    /// the durable handle that distinguishes two same-tool calls in one
    /// turn — but never serialized back: Ollama's request schema correlates
    /// tool messages by `tool_name`, and replayed histories predate the id.
    #[serde(default, skip_serializing)]
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

// ---------- Provider Message Definition ----------

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
    },
}

/// -----------------------------
/// Provider Message Conversions
/// -----------------------------
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

/// Conversion from an internal Rig message (crate::message::Message) to a provider Message.
/// (Only User and Assistant variants are supported.)
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
                            name,
                            content,
                            ..
                        }) => {
                            // The executed tool's name travels as required data.
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
                                    crate::message::ToolResultContent::Image(_) => {
                                        Err(crate::message::MessageError::ConversionError(
                                            "Ollama does not support images in tool results".into(),
                                        ))
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .join("\n");
                            messages.push(Message::ToolResult {
                                name: function_name,
                                content,
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
            InternalMessage::Assistant { content, .. } => {
                let mut thinking: Option<String> = None;
                let mut text_content = Vec::new();
                let mut tool_calls = Vec::new();

                for content in content.into_iter() {
                    match content {
                        crate::message::AssistantContent::Text(text) => {
                            text_content.push(text.text);
                        }
                        crate::message::AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(tool_call);
                        }
                        crate::message::AssistantContent::Reasoning(reasoning) => {
                            let display = reasoning.display_text();
                            if !display.is_empty() {
                                thinking = Some(display);
                            }
                        }
                        crate::message::AssistantContent::Image(_) => {
                            return Err(crate::message::MessageError::ConversionError(
                                "Ollama currently doesn't support images.".into(),
                            ));
                        }
                    }
                }

                // Both fields may be empty. This used to lean on the non-empty
                // content type to argue that at least one of them was populated;
                // content is a `Vec` now, so an assistant turn that carried
                // nothing renders as an Ollama message with empty text and no
                // tool calls, which is what such a turn actually was.
                Ok(vec![Message::Assistant {
                    content: text_content.join(" "),
                    thinking,
                    images: None,
                    name: None,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(std::convert::Into::into)
                        .collect::<Vec<_>>(),
                }])
            }
        }
    }
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

// ---------- Additional Message Types ----------

impl From<crate::message::ToolCall> for ToolCall {
    fn from(tool_call: crate::message::ToolCall) -> Self {
        Self {
            // Never serialized (replay correlates by `tool_name`); the
            // request shape is id-less regardless of what history holds.
            id: None,
            r#type: ToolType::Function,
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests;
