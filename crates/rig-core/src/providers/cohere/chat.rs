//! Cohere's native chat API, `POST /v2/chat`: the request built as JSON,
//! with the request's documents sent as Cohere `documents`. Its replies
//! are read by [`ChatDecoder`].
//!
//! ```
//! use rig_core::providers::cohere::{COMMAND_A_03_2025, CohereConfig, NativeChat};
//!
//! let wire = NativeChat::new(CohereConfig::new("key"), COMMAND_A_03_2025);
//! assert_eq!(wire.model, COMMAND_A_03_2025);
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::CohereConfig;
use super::streaming::ChatDecoder;
use crate::catalog::ModelFacts;
use crate::completion::options::{BaseInput, FinalBody, RawAt, request_params};
use crate::completion::{CompletionRequest, Document, ProviderCapabilities, Replay};
use crate::error::EncodeError;
use crate::json_utils::Lenient;
use crate::message::{
    AssistantContent, AssistantMessage, DocumentData, DocumentSourceKind as Source, Message,
    MimeType, ToolChoice, ToolResult, ToolResultContent, UserContent,
};
use crate::operation::Completion;
use crate::providers::internal::wire_ids::WireIds;
use crate::wire::{Capabilities, Descriptor, Encoded, Framing, Mode, Wire};

/// Where the native chat endpoint sits under the API root.
const CHAT_PATH: &str = "/v2/chat";

/// The replay API of turns made on the native chat endpoint.
pub(crate) const API: &str = "cohere.chat";

/// The native chat wire: a provider configuration, a model, and the
/// per-turn options the endpoint takes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NativeChat {
    /// Which provider, and how to reach it.
    pub provider: CohereConfig,
    /// The model this wire addresses.
    pub model: String,
    /// Whether requests ask Cohere to hold tool calls to their schemas
    /// (`strict_tools`).
    pub strict_tools: bool,
    /// The model facts the encoder reads and replies are priced by.
    #[serde(skip)]
    pub facts: ModelFacts,
}

impl NativeChat {
    /// The wire for `model` on `provider`, with every option off.
    pub fn new(provider: CohereConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            strict_tools: false,
            facts: ModelFacts::default(),
        }
    }

    /// The same wire, encoding with `facts` and pricing its replies by
    /// them.
    pub fn with_facts(mut self, facts: ModelFacts) -> Self {
        self.facts = facts;
        self
    }

    /// Ask Cohere to hold every tool call to its tool's schema.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// The request body: the wire's encoding of `request`, then the mapped
    /// options, then `additional_params`, merged key by key.
    fn body(&self, request: &CompletionRequest, mode: Mode) -> Result<FinalBody, EncodeError> {
        request_params(
            self,
            request,
            |input| self.base(request, mode, input),
            RawAt::Top,
            &[],
        )
    }

    /// The wire's own encoding of `request`.
    fn base(
        &self,
        request: &CompletionRequest,
        mode: Mode,
        input: &mut BaseInput<'_>,
    ) -> Result<Map<String, Value>, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let messages = self.messages(&request.chat_history, &model)?;
        let mut tools: Vec<Value> = request
            .tools
            .iter()
            .filter(|tool| match &request.tool_choice {
                // Cohere cannot name the tool it must call: only the named
                // tools are offered, and one of them is required.
                Some(ToolChoice::Specific { function_names }) => {
                    function_names.contains(&tool.name)
                }
                _ => true,
            })
            .map(|tool| {
                json!({"type": "function", "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }})
            })
            .collect();
        tools.extend(input.raw_tools()?);
        let tool_choice = match &request.tool_choice {
            None | Some(ToolChoice::Auto) => None,
            Some(ToolChoice::None) => Some("NONE"),
            Some(ToolChoice::Required | ToolChoice::Specific { .. }) => Some("REQUIRED"),
        }
        .filter(|_| !tools.is_empty());
        let documents: Vec<Value> = request
            .documents
            .iter()
            .enumerate()
            .map(|(position, document)| document_value(position, document))
            .collect();
        let response_format = request
            .output_schema
            .clone()
            .map(|schema| json!({"type": "json_object", "schema": schema.to_value()}));
        let fields = [
            ("model", Some(Value::String(model))),
            ("messages", Some(Value::Array(messages))),
            (
                "documents",
                (!documents.is_empty()).then_some(Value::Array(documents)),
            ),
            ("tools", (!tools.is_empty()).then_some(Value::Array(tools))),
            ("tool_choice", tool_choice.map(Value::from)),
            (
                "strict_tools",
                self.strict_tools.then_some(Value::Bool(true)),
            ),
            ("response_format", response_format),
            ("temperature", request.temperature.map(Value::from)),
            ("max_tokens", request.max_tokens.map(Value::from)),
            (
                "stream",
                (mode == Mode::Streaming).then_some(Value::Bool(true)),
            ),
        ];
        Ok(fields
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_owned(), value?)))
            .collect())
    }

    /// The history as Cohere messages, each call and result spelled by one
    /// [`WireIds`].
    fn messages(&self, history: &[Message], model: &str) -> Result<Vec<Value>, EncodeError> {
        let ids = WireIds::for_target(history, self, model);
        let mut messages = Vec::new();
        for message in history {
            match message {
                Message::System { content } => {
                    messages.push(json!({"role": "system", "content": content}));
                }
                Message::User { content } => {
                    let mut parts = Vec::new();
                    for part in content {
                        if let UserContent::ToolResult(result) = part {
                            user_message(&mut messages, &mut parts);
                            messages.push(tool_message(result, &ids));
                        } else {
                            parts.push(user_part(part)?);
                        }
                    }
                    user_message(&mut messages, &mut parts);
                }
                Message::Assistant(turn) => messages.extend(self.assistant(turn, &ids)),
            }
        }
        if messages.is_empty() {
            return Err(EncodeError::request(
                "Cohere chat request has no messages after conversion",
            ));
        }
        Ok(messages)
    }

    /// One assistant turn: text, thinking and unknown parts as content
    /// parts, a tool plan as `tool_plan`, each call under `tool_calls`, and
    /// the citations each block's item holds, pointed at the part they
    /// cite. A current item goes back as it came, less its citations. A
    /// turn with nothing to send is `None`.
    fn assistant(&self, turn: &AssistantMessage, ids: &WireIds) -> Option<Value> {
        let (mut content, mut plan, mut calls, mut citations) =
            (Vec::new(), String::new(), Vec::new(), Vec::new());
        for block in &turn.content {
            let replay = block.replay(self, ids);
            let kind = replay_kind(&replay);
            let mut item = match replay {
                Replay::Item(item) => match item.into_owned() {
                    Value::Object(item) => item,
                    _ => Map::new(),
                },
                Replay::Identity(_) | Replay::Rebuild => Map::new(),
            };
            let cited = match item.shift_remove("citations") {
                Some(Value::Array(cited)) => cited,
                _ => Vec::new(),
            };
            let part = match block {
                AssistantContent::Text(text) if !text.text.is_empty() => {
                    json!({"type": "text", "text": text.text})
                }
                AssistantContent::Reasoning(reasoning) if kind.as_deref() == Some(PLAN) => {
                    citations.extend(pointed(cited, None));
                    plan.push_str(&reasoning.text);
                    continue;
                }
                AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                    json!({"type": "thinking", "thinking": reasoning.text})
                }
                AssistantContent::Opaque(opaque)
                    if opaque.replay && opaque.item.get("type").is_some() =>
                {
                    content.push(opaque.item.clone());
                    continue;
                }
                AssistantContent::ToolCall(call) => {
                    item.entry("type")
                        .or_insert_with(|| Value::from("function"));
                    item.insert("id".to_owned(), Value::String(ids.spell(&call.id)));
                    let function = item
                        .entry("function")
                        .or_insert_with(|| Value::Object(Map::new()));
                    if !function.is_object() {
                        *function = Value::Object(Map::new());
                    }
                    if let Value::Object(function) = function {
                        function
                            .insert("name".to_owned(), Value::from(call.function.name.as_str()));
                        function.insert(
                            "arguments".to_owned(),
                            Value::String(call.function.arguments_value().to_string()),
                        );
                    }
                    calls.push(Value::Object(item));
                    continue;
                }
                AssistantContent::Text(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_)
                | AssistantContent::Opaque(_) => continue,
            };
            citations.extend(pointed(cited, Some(content.len())));
            content.push(if item.is_empty() {
                part
            } else {
                Value::Object(item)
            });
        }
        if content.is_empty() && calls.is_empty() {
            return None;
        }
        let fields = [
            ("role", Some(Value::from("assistant"))),
            (
                "content",
                (!content.is_empty()).then_some(Value::Array(content)),
            ),
            (
                "tool_plan",
                (!plan.is_empty()).then_some(Value::String(plan)),
            ),
            (
                "tool_calls",
                (!calls.is_empty()).then_some(Value::Array(calls)),
            ),
            (
                "citations",
                (!citations.is_empty()).then_some(Value::Array(citations)),
            ),
        ];
        Some(Value::Object(
            fields
                .into_iter()
                .filter_map(|(key, value)| Some((key.to_owned(), value?)))
                .collect(),
        ))
    }
}

/// The item kind a tool plan's reasoning block holds.
pub(crate) const PLAN: &str = "tool_plan";

/// The kind of item replay hands the encoder for a block: the current
/// item's `type`, or the one an edited block kept.
fn replay_kind(replay: &Replay<'_>) -> Option<String> {
    match replay {
        Replay::Item(item) => item.str("type").map(str::to_owned),
        Replay::Identity(identity) => identity
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_owned),
        Replay::Rebuild => None,
    }
}

/// The kind of item `block` replays as on `target`.
fn kind(block: &AssistantContent, target: &NativeChat) -> Option<String> {
    replay_kind(&block.replay(target, &WireIds::default()))
}

/// `citations` as a request sends them back: each at `content_index`, the
/// position of the part it cites, or with none for a tool plan's.
fn pointed(citations: Vec<Value>, content_index: Option<usize>) -> Vec<Value> {
    citations
        .into_iter()
        .map(|mut citation| {
            if let (Some(citation), Some(index)) = (citation.as_object_mut(), content_index) {
                citation.insert("content_index".to_owned(), Value::from(index));
            }
            citation
        })
        .collect()
}

/// `document` as a Cohere document: its metadata and `text` under `data`,
/// keys sorted so the same document always sends the same bytes, under its
/// id, or `doc_<position>` when it has none.
fn document_value(position: usize, document: &Document) -> Value {
    let mut data: BTreeMap<&str, &str> = document
        .additional_props
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    data.insert("text", &document.text);
    let id = if document.id.is_empty() {
        format!("doc_{position}")
    } else {
        document.id.clone()
    };
    json!({"id": id, "data": data})
}

/// One user content part as Cohere carries it: text, a text document's
/// text, or an image by URL. Replay leaves no other part.
fn user_part(part: &UserContent) -> Result<Value, EncodeError> {
    Ok(match part {
        UserContent::Text(text) => json!({"type": "text", "text": text.text}),
        UserContent::Image(image) => {
            let mime = image.media_type.as_ref().map(MimeType::to_mime_type);
            let url = match (&image.data, mime) {
                (Source::Url(url), _) => url.clone(),
                (Source::Base64(data), Some(mime)) => format!("data:{mime};base64,{data}"),
                _ => return Err(unsendable("an image")),
            };
            json!({"type": "image_url", "image_url": {"url": url}})
        }
        UserContent::Document(document) => match &document.data {
            DocumentData::Text(text) => json!({"type": "text", "text": text}),
            DocumentData::File(_) => return Err(unsendable("a document")),
        },
        UserContent::Audio(_) => return Err(unsendable("audio")),
        UserContent::Video(_) => return Err(unsendable("a video")),
        UserContent::ToolResult(_) => return Err(unsendable("a tool result as a content part")),
    })
}

/// The error for content Cohere chat cannot carry, which replay replaces
/// before a request is encoded.
fn unsendable(what: &str) -> EncodeError {
    EncodeError::request(format!("Cohere chat cannot carry {what} in this form"))
}

/// Push the user parts gathered so far as one message.
fn user_message(messages: &mut Vec<Value>, parts: &mut Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    messages.push(json!({"role": "user", "content": std::mem::take(parts)}));
}

/// A result as the `tool` message that answers its call, its parts as text.
fn tool_message(result: &ToolResult, ids: &WireIds) -> Value {
    let content: Vec<Value> = result
        .content
        .iter()
        .filter_map(|part| match part {
            ToolResultContent::Text(text) => Some(text.text.clone()),
            ToolResultContent::Json { value } => Some(value.to_string()),
            ToolResultContent::Image(_) => None,
        })
        .map(|text| json!({"type": "text", "text": text}))
        .collect();
    json!({"role": "tool", "tool_call_id": ids.spell(&result.call), "content": content})
}

impl Wire for NativeChat {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ChatDecoder;
    type Reassembler = super::streaming::document::ChatResponse;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
            .model(self.model.as_str())
            .facts(&self.facts)
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(true),
            ))
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let body = self.body(&request, mode)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Cohere chat request",
            &body,
        );
        let request = self.provider.post(CHAT_PATH).body(body.into_body())?;
        let framing = match mode {
            Mode::Streaming => Framing::Sse,
            Mode::Unary => Framing::Whole,
        };
        Ok(Encoded::new(request, framing)
            .with_request_id_header(Some(REQUEST_ID_HEADER))
            .with_projection(ChatDecoder::project)
            .with_route(Some(CHAT_PATH)))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ChatDecoder::default()
    }
}

/// The reply header carrying Cohere's request id.
const REQUEST_ID_HEADER: &str = "x-debug-trace-id";

impl crate::completion::ReplayTarget for NativeChat {
    /// Section 6.5 of the typed-options design, for the native API.
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        use crate::completion::options::{Mapping, OptionFields, OptionMap};
        use crate::completion::{CacheRetention, Effort, Reasoning};
        let OptionFields {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
            cache_key,
        } = fields;
        let model = request.model.as_deref().unwrap_or(&self.model);
        // A model that thinks does so by default. An id the catalog does
        // not list thinks when its name says `reasoning`.
        let thinks = super::thinks(&self.facts, model);
        let reasons = thinks.unwrap_or_else(|| model.contains("reasoning"));
        const NO_FIELD: &str = "Cohere's chat API has no such field";
        OptionMap {
            reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
                Reasoning::Off if reasons => {
                    Mapping::Send(json!({"thinking": {"type": "disabled"}}))
                }
                Reasoning::Off => Mapping::Omit("the model does not think"),
                Reasoning::Effort(_) | Reasoning::Budget { .. } if thinks == Some(false) => {
                    Mapping::unsupported("the model does not think")
                }
                Reasoning::Effort(Effort::High) => {
                    Mapping::Send(json!({"thinking": {"type": "enabled"}}))
                }
                Reasoning::Effort(effort) => Mapping::unsupported(format!(
                    "Cohere takes thinking on or a token budget, not `{}`",
                    effort.as_str()
                )),
                Reasoning::Budget { tokens } => Mapping::Send(json!({
                    "thinking": {"type": "enabled", "token_budget": tokens},
                })),
            }),
            cache: Mapping::of(cache, |cache| match cache {
                CacheRetention::None => Mapping::Omit("Cohere does not cache prompts"),
                CacheRetention::Short | CacheRetention::Long => {
                    Mapping::unsupported("Cohere has no prompt cache")
                }
            }),
            service_tier: Mapping::of(service_tier, |_| {
                Mapping::unsupported("Cohere's `priority` is a queue position, not a tier")
            }),
            verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(NO_FIELD)),
            parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
                Mapping::unsupported(NO_FIELD)
            }),
            top_p: Mapping::of(top_p, |top_p| {
                if (0.01..=0.99).contains(&top_p) {
                    Mapping::Send(json!({ "p": top_p }))
                } else {
                    Mapping::unsupported("Cohere takes `p` from 0.01 to 0.99")
                }
            }),
            seed: Mapping::of(seed, |seed| Mapping::Send(json!({ "seed": seed }))),
            stop: Mapping::of_stop(stop, |stop| match stop.len() {
                0..=5 => Mapping::Send(json!({ "stop_sequences": stop })),
                _ => Mapping::unsupported("Cohere takes at most 5 stop sequences"),
            }),
            cache_key: Mapping::unrouted(cache_key),
        }
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static(API)
    }

    fn facts(&self) -> Option<&ModelFacts> {
        Some(&self.facts)
    }

    fn provider(&self) -> &str {
        super::PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Cohere's vision models read user images; no model reads images in
    /// assistant turns or tool results.
    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        crate::completion::Accepts {
            user_images: self.facts.reads_images_or(
                super::PROVIDER_NAME,
                model,
                super::reads_images,
            ),
            assistant_images: false,
            tool_result_images: false,
            tools: true,
        }
    }

    /// The encoder carries a user image by URL or typed data, and a text
    /// document as its text. It carries no other media.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        use crate::completion::{Media, Place};
        match media {
            Media::Image(image, Place::User) => match &image.data {
                Source::Url(_) => true,
                Source::Base64(_) => image.media_type.is_some(),
                _ => false,
            },
            Media::Document(document) => matches!(document.data, DocumentData::Text(_)),
            Media::Image(..) | Media::Audio(_) | Media::Video(_) => false,
        }
    }

    /// An edited block keeps its kind: a tool plan stays a plan.
    fn identity(&self, item: &Value) -> Map<String, Value> {
        item.str("type")
            .filter(|kind| *kind == PLAN)
            .map(|kind| Map::from_iter([("type".to_owned(), Value::from(kind))]))
            .unwrap_or_default()
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    fn takes_documents(&self) -> bool {
        true
    }

    /// Text and thinking with text are content parts, an opaque item with
    /// a `type` is a part, and a call is a call. A tool plan rides with its
    /// calls, and images are never sent.
    fn sends_alone(&self, block: &AssistantContent) -> bool {
        match block {
            AssistantContent::Text(text) => !text.text.is_empty(),
            AssistantContent::Reasoning(reasoning) => {
                !reasoning.text.is_empty() && kind(block, self).as_deref() != Some(PLAN)
            }
            AssistantContent::Opaque(opaque) => opaque.item.get("type").is_some(),
            AssistantContent::ToolCall(_) => true,
            AssistantContent::Image(_) => false,
        }
    }
}

#[cfg(test)]
mod tests;
