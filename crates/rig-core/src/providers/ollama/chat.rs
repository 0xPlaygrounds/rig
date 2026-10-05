//! The native Ollama chat wire: `POST /api/chat`, a whole reply or a stream
//! of NDJSON records, both read by [`ChatDecoder`].
//!
//! The request carries the daemon's own fields: `think` and `keep_alive` at
//! the top level, model parameters such as `num_ctx` in `options`, and the
//! output schema as `format`.
//!
//! ```
//! use rig_core::providers::ollama::OllamaConfig;
//!
//! let wire = OllamaConfig::new().client().native_completion("qwen3:4b").wire;
//! assert_eq!(wire.model, "qwen3:4b");
//! ```

use serde_json::{Map, Value, json};

use crate::completion::{
    Accepts, CompletionRequest, Media, Place, ProviderCapabilities, Replay, ReplayTarget,
};
use crate::error::EncodeError;
use crate::message::{
    AssistantContent, AssistantMessage, DocumentSourceKind as Source, Message, ToolCall,
    ToolResult, ToolResultContent, UserContent,
};
use crate::operation::Completion;
use crate::providers::internal::wire_ids::WireIds;
use crate::wire::{Body, Capabilities, Descriptor, Encoded, Framing, Mode, Wire};

use super::streaming::ChatDecoder;
use super::{OllamaConfig, PROVIDER_NAME};

/// The chat endpoint, relative to the daemon's address.
const CHAT_PATH: &str = "/api/chat";

/// The `additional_params` keys `/api/chat` reads at the top level of its
/// request. Every other key is a model parameter and goes in `options`.
const TOP_LEVEL: &[&str] = &[
    "format",
    "keep_alive",
    "logprobs",
    "top_logprobs",
    "truncate",
    "shift",
];

/// The thinking levels `think` takes besides a boolean.
const THINK_LEVELS: &[&str] = &["low", "medium", "high", "max"];

/// The native chat wire of an Ollama daemon.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Chat {
    /// The daemon's address and credential.
    pub provider: OllamaConfig,
    /// The model to address, e.g. [`QWEN3`](super::QWEN3).
    pub model: String,
}

impl Chat {
    /// The wire for `model` on `provider`.
    pub fn new(provider: OllamaConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// The `/api/chat` body `request` sends in `mode`.
    ///
    /// `additional_params` is split by where the daemon reads each key:
    /// `think` (a boolean or `low`, `medium`, `high` or `max`) and the keys
    /// in [`TOP_LEVEL`] go at the top level, `tools` join the request's
    /// tools, an `options` object merges into `options`, and every other key
    /// is an `options` entry, `reasoning_effort` included, with a warning. `temperature` and `max_tokens` (as
    /// `num_predict`) go in `options`, where a caller's own entries win.
    fn body(
        &self,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Map<String, Value>, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let params = match request.additional_params {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(params)) => params,
            Some(_) => {
                return Err(EncodeError::request(
                    "Ollama `additional_params` must be a JSON object",
                ));
            }
        };
        let messages = self.messages(&request.chat_history, &model)?;

        let mut tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({"type": "function", "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }})
            })
            .collect();
        if request.tool_choice.is_some() {
            tracing::warn!("Ollama has no tool choice; `tool_choice` is ignored");
        }

        let mut options = Map::new();
        if let Some(temperature) = request.temperature {
            options.insert("temperature".to_owned(), Value::from(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            options.insert("num_predict".to_owned(), Value::from(max_tokens));
        }
        let mut top = Map::new();
        for (key, value) in params {
            match key.as_str() {
                "think" => {
                    top.insert(key, think(value)?);
                }
                "reasoning_effort" => {
                    tracing::warn!(
                        "Ollama's `/api/chat` takes `think`, not `reasoning_effort`; \
                         it is sent as a model option"
                    );
                    options.insert(key, value);
                }
                "keep_alive" if !(value.is_string() || value.is_number()) => {
                    return Err(EncodeError::request(
                        "Ollama `keep_alive` must be a duration string or a number of seconds",
                    ));
                }
                "tools" => match value {
                    Value::Array(passthrough) => tools.extend(passthrough),
                    _ => {
                        return Err(EncodeError::request(
                            "Ollama `additional_params.tools` must be an array",
                        ));
                    }
                },
                "options" => match value {
                    Value::Object(entries) => options.extend(entries),
                    _ => {
                        return Err(EncodeError::request(
                            "Ollama `additional_params.options` must be an object",
                        ));
                    }
                },
                key if TOP_LEVEL.contains(&key) => {
                    top.insert(key.to_owned(), value);
                }
                _ => {
                    options.insert(key, value);
                }
            }
        }

        // Defer the schema until a tool result exists: a constrained reply
        // cannot call a tool.
        let answered = messages
            .iter()
            .any(|message| message.get("role").and_then(Value::as_str) == Some("tool"));
        let format = request
            .output_schema
            .filter(|_| tools.is_empty() || answered)
            .map(|schema| schema.to_value());

        let fields = [
            ("model", Some(Value::String(model))),
            ("messages", Some(Value::Array(messages))),
            ("tools", (!tools.is_empty()).then_some(Value::Array(tools))),
            ("format", format),
            (
                "options",
                (!options.is_empty()).then_some(Value::Object(options)),
            ),
            ("stream", Some(Value::Bool(mode == Mode::Streaming))),
        ];
        let mut body: Map<String, Value> = fields
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_owned(), value?)))
            .collect();
        body.extend(top);
        Ok(body)
    }

    /// The history as `/api/chat` messages, each call and result spelled by
    /// one [`WireIds`].
    fn messages(&self, history: &[Message], model: &str) -> Result<Vec<Value>, EncodeError> {
        let ids = WireIds::for_target(history, self, model);
        let mut messages = Vec::new();
        for message in history {
            match message {
                Message::System { content } => {
                    messages.push(json!({"role": "system", "content": content}));
                }
                Message::User { content } => {
                    let mut user = UserParts::default();
                    for part in content {
                        match part {
                            UserContent::ToolResult(result) => {
                                user.push_to(&mut messages);
                                messages.push(tool_message(result, &ids)?);
                            }
                            part => user.add(part)?,
                        }
                    }
                    user.push_to(&mut messages);
                }
                Message::Assistant(turn) => {
                    messages.extend(self.assistant(turn, &ids));
                }
            }
        }
        if messages.is_empty() {
            return Err(EncodeError::request(
                "Ollama chat request has no messages after conversion",
            ));
        }
        Ok(messages)
    }

    /// One assistant turn: its text joined as `content`, its reasoning as
    /// `thinking`, and its calls. A turn with none of them is `None`.
    fn assistant(&self, turn: &AssistantMessage, ids: &WireIds) -> Option<Value> {
        let (mut text, mut thinking, mut calls) = (String::new(), Vec::new(), Vec::new());
        for block in &turn.content {
            match block {
                AssistantContent::Text(block) => text.push_str(&block.text),
                AssistantContent::Reasoning(block) if !block.text.is_empty() => {
                    thinking.push(block.text.as_str());
                }
                AssistantContent::ToolCall(call) => {
                    calls.push(call_item(call, block.replay(self, ids), ids));
                }
                AssistantContent::Reasoning(_)
                | AssistantContent::Image(_)
                | AssistantContent::Opaque(_) => {}
            }
        }
        if text.is_empty() && thinking.is_empty() && calls.is_empty() {
            return None;
        }
        let mut message = Map::from_iter([
            ("role".to_owned(), Value::from("assistant")),
            ("content".to_owned(), Value::String(text)),
        ]);
        if !thinking.is_empty() {
            message.insert("thinking".to_owned(), Value::String(thinking.join("\n")));
        }
        if !calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(calls));
        }
        Some(Value::Object(message))
    }
}

/// `think` as the daemon takes it: a boolean, or one of [`THINK_LEVELS`].
fn think(value: Value) -> Result<Value, EncodeError> {
    match &value {
        Value::Bool(_) => Ok(value),
        Value::String(level) if THINK_LEVELS.contains(&level.to_ascii_lowercase().as_str()) => {
            Ok(Value::String(level.to_ascii_lowercase()))
        }
        _ => Err(EncodeError::request(format!(
            "Ollama `think` must be a boolean or one of {}",
            THINK_LEVELS.join(", ")
        ))),
    }
}

/// The text and images of one user message as they gather.
#[derive(Default)]
struct UserParts {
    texts: Vec<String>,
    images: Vec<String>,
}

impl UserParts {
    /// Add a part: text, or an image as base64 data. The adapter has
    /// replaced every other part ([`ReplayTarget::encodes`]).
    fn add(&mut self, part: &UserContent) -> Result<(), EncodeError> {
        match part {
            UserContent::Text(text) => self.texts.push(text.text.clone()),
            UserContent::Image(image) => match &image.data {
                Source::Base64(data) => self.images.push(data.clone()),
                _ => return Err(unsendable("an image that is not base64 data")),
            },
            UserContent::Document(document) => match &document.data {
                Source::String(text) => self.texts.push(text.clone()),
                _ => return Err(unsendable("a document")),
            },
            UserContent::Audio(_) => return Err(unsendable("audio")),
            UserContent::Video(_) => return Err(unsendable("video")),
            UserContent::ToolResult(_) => return Err(unsendable("a tool result as user content")),
        }
        Ok(())
    }

    /// Push the parts gathered so far as one user message, if there are any.
    fn push_to(&mut self, messages: &mut Vec<Value>) {
        if self.texts.is_empty() && self.images.is_empty() {
            return;
        }
        let mut message = Map::from_iter([
            ("role".to_owned(), Value::from("user")),
            ("content".to_owned(), Value::String(self.texts.join("\n"))),
        ]);
        if !self.images.is_empty() {
            message.insert(
                "images".to_owned(),
                Value::Array(self.images.drain(..).map(Value::String).collect()),
            );
        }
        self.texts.clear();
        messages.push(Value::Object(message));
    }
}

/// The error for content `/api/chat` cannot carry, which the adapter
/// replaces before a request is encoded.
fn unsendable(what: &str) -> EncodeError {
    EncodeError::request(format!("Ollama chat cannot carry {what}"))
}

/// A result as the `tool` message answering its call: its text joined into
/// one string, with the tool's name and the call's id.
fn tool_message(result: &ToolResult, ids: &WireIds) -> Result<Value, EncodeError> {
    let texts = result
        .content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => Ok(text.text.clone()),
            ToolResultContent::Json { value } => Ok(value.to_string()),
            ToolResultContent::Image(_) => Err(unsendable("an image in a tool result")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "role": "tool",
        "content": texts.join("\n"),
        "tool_name": result.name,
        "tool_call_id": ids.spell(&result.call),
    }))
}

/// `call` as a `tool_calls` entry: its item as the daemon sent it when it is
/// current, with the canonical name and arguments and the call's spelled id.
fn call_item(call: &ToolCall, replay: Replay<'_>, ids: &WireIds) -> Value {
    let mut item = match replay {
        Replay::Item(item) => match item.into_owned() {
            Value::Object(item) => item,
            _ => Map::new(),
        },
        Replay::Identity(identity) => identity,
        Replay::Rebuild => Map::new(),
    };
    item.insert("id".to_owned(), Value::String(ids.spell(&call.id)));
    let function = item
        .entry("function")
        .or_insert_with(|| Value::Object(Map::new()));
    if !function.is_object() {
        *function = Value::Object(Map::new());
    }
    if let Value::Object(function) = function {
        function.insert("name".to_owned(), Value::from(call.function.name.as_str()));
        function.insert(
            "arguments".to_owned(),
            Value::Object(call.function.arguments.clone()),
        );
    }
    Value::Object(item)
}

impl Wire for Chat {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ChatDecoder;

    /// The output schema is deferred while tools are unanswered, so it
    /// composes with tools.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(true),
            ))
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let body = Value::Object(self.body(request, mode)?);
        let target = match mode {
            Mode::Unary => crate::providers::internal::LogTarget::Completions,
            Mode::Streaming => crate::providers::internal::LogTarget::Streaming,
        };
        crate::providers::internal::trace_json(target, "Ollama chat request", &body);
        let request = self
            .provider
            .request(http::Method::POST, CHAT_PATH)
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        let framing = match mode {
            Mode::Unary => Framing::Whole,
            Mode::Streaming => Framing::Ndjson,
        };
        Ok(Encoded::new(request, framing)
            .with_route(Some(CHAT_PATH))
            .with_projection(ChatDecoder::project))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ChatDecoder::default()
    }
}

impl ReplayTarget for Chat {
    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("ollama.chat")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Images in user messages, and none in assistant turns or tool results,
    /// which the adapter moves to a user message.
    fn accepts(&self, _model: &str) -> Accepts {
        Accepts {
            assistant_images: false,
            tool_result_images: false,
            ..Accepts::ALL
        }
    }

    /// An image as base64 data in a user message. The daemon never fetches a
    /// URL, and reads no audio, video or document; a text document's text
    /// is sent by the adapter.
    fn encodes(&self, _model: &str, media: Media<'_>) -> bool {
        matches!(
            media,
            Media::Image(image, Place::User) if matches!(image.data, Source::Base64(_))
        )
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    /// A message needs content, thinking or calls.
    fn sends_alone(&self, block: &AssistantContent) -> bool {
        match block {
            AssistantContent::ToolCall(_) => true,
            AssistantContent::Text(text) => !text.text.is_empty(),
            AssistantContent::Reasoning(reasoning) => !reasoning.text.is_empty(),
            AssistantContent::Image(_) | AssistantContent::Opaque(_) => false,
        }
    }
}

#[cfg(test)]
mod tests;
