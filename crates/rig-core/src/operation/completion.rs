//! Generating an assistant turn. A completion decoder writes its reply one
//! provider item at a time, keyed by the item's wire index: the item opens
//! when its first event arrives and takes the next position in the reply,
//! its fragments grow it, and closing it finalizes its block with the
//! provider's item as the block's native. Nothing regroups: blocks keep the
//! order their items opened in. The provider's end of the reply is a
//! [`Finish`], which the fold needs to produce the response.
//!
//! ```
//! use rig_core::operation::Finish;
//! use rig_core::completion::FinishReason;
//!
//! let finish = Finish {
//!     reason: Some(FinishReason::Stop),
//!     ..Finish::default()
//! };
//! assert_eq!(finish.reason, Some(FinishReason::Stop));
//! ```

use std::collections::{BTreeMap, HashSet};

use crate::completion::{CompletionRequest, CompletionResponse, FinishReason, Usage};
use crate::error::{MalformedToolInput, ProviderError};
use crate::message::{
    Api, AssistantContent, AssistantMessage, CallId, LocalCallId, Opaque, Origin, Reasoning, Text,
    ToolCall, ToolFunction, ToolName,
};
use crate::streaming::{Item, Part, PartKind, StreamEvent};
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Assembled, Call, Descriptor, Emit, Fold, Mode, Operation, Out, Reply, Shared};

/// Generating an assistant turn, unary or streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion;

impl Operation for Completion {
    type Request = CompletionRequest;
    type Event = StreamEvent;
    type End = Finish;
    type Response = CompletionResponse;
    type Fold = Turn;
    type Emit = Assembled;

    /// The call's span names the model the request overrides to, when it
    /// names one: every wire honours the override on encode.
    fn fold(request: &Self::Request, call: &mut Call<'_>) -> Self::Fold {
        let telemetry = call.wire.telemetry.map_or_else(
            || match call.mode {
                Mode::Unary => GenAiOperation::Chat,
                Mode::Streaming => GenAiOperation::ChatStreaming,
            },
            |telemetry| telemetry(call.mode),
        );
        debug_assert!(telemetry.is_completion());
        let model = request
            .model
            .as_deref()
            .or(call.wire.model)
            .unwrap_or_default();
        let span = SpanBuilder::new(call.wire.name, model, telemetry)
            .system_instructions(
                request.system_instructions(),
                request.record_telemetry_content,
            )
            .build();
        call.instrument(span.clone());
        let api = call.wire.replay.map_or_else(
            || Api::from(call.wire.name.to_owned()),
            |target| target.api(),
        );
        Turn {
            span,
            ..Turn::new(Origin::new(api, call.wire.name, model))
        }
    }

    /// [`adapt`](crate::completion::adapt) the history for the wire's
    /// replay target, then [`CompletionRequest::validate_message_content`].
    fn prepare(
        mut request: Self::Request,
        wire: &Descriptor<'_>,
    ) -> Result<Self::Request, ProviderError> {
        let Some(target) = wire.replay else {
            return Err(ProviderError::request(format!(
                "completion wire `{}` names no replay target",
                wire.name
            )));
        };
        request.chat_history = crate::completion::history::adapt_for_model(
            &request.chat_history,
            target,
            request.model.as_deref(),
        );
        request.validate_message_content()?;
        Ok(request)
    }
}

impl crate::wire::reply::Closing<Completion> for Assembled {
    fn close(shared: &mut Shared<Completion>) {
        let Shared { fold, items, .. } = shared;
        fold.close_open(items);
    }
}

impl Emit<Completion> for Assembled {}

/// What the provider sends when it ends a completion reply.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Finish {
    /// Token usage the provider reported. A counter it did not report is
    /// `None`.
    pub usage: Usage,
    /// Why the model stopped, when the provider said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<FinishReason>,
    /// The response id.
    pub response_id: Option<String>,
    /// The model the provider reports.
    pub model: Option<String>,
    /// The provider's report that the turn failed, such as a refusal's
    /// explanation. The turn is then never replayed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What a provider item becomes when it opens.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// Answer text.
    Text,
    /// Reasoning, possibly withheld by the provider.
    Reasoning {
        /// Whether the provider withheld the text.
        redacted: bool,
    },
    /// A tool call with its id and name.
    Call {
        /// The call's id.
        id: CallId,
        /// The tool's name.
        name: ToolName,
    },
    /// An item with no canonical meaning.
    Opaque {
        /// Whether the item goes back to the model that produced it.
        replay: bool,
    },
}

/// The completion fold. The driver and the bus writer build it; no other
/// code can feed one.
///
/// It is both sides of one reply: the writer state a decoder writes through
/// (the open items by wire index) and the blocks the consumer has taken, in
/// their position.
pub struct Turn {
    span: tracing::Span,
    /// Who the reply is from; the end adds the provider's model and id.
    origin: Origin,
    // The writer.
    open: BTreeMap<usize, Draft>,
    next_part: u32,
    call_ids: HashSet<CallId>,
    native: Option<serde_json::Value>,
    /// The index of the block a boundary-less wire is streaming.
    run: Option<usize>,
    next_auto: usize,
    // The fold.
    choice: Vec<Option<AssistantContent>>,
    /// The text the consumer took of parts still open, by position.
    open_text: BTreeMap<usize, String>,
}

/// One open provider item.
struct Draft {
    part: Part,
    /// Whether its start was emitted.
    started: bool,
    /// The provider's item as assembled so far; `Null` for none.
    item: serde_json::Value,
    body: Body,
}

enum Body {
    Text(String),
    Reasoning {
        text: String,
        redacted: bool,
    },
    Call {
        id: Option<CallId>,
        name: String,
        arguments: Arguments,
    },
    Opaque {
        replay: bool,
    },
}

impl Body {
    fn kind(&self) -> PartKind {
        match self {
            Self::Text(_) => PartKind::Text,
            Self::Reasoning { .. } => PartKind::Reasoning,
            Self::Call { .. } => PartKind::ToolCall,
            Self::Opaque { .. } => PartKind::Opaque,
        }
    }
}

/// A tool call's argument text as it arrives.
#[derive(Default)]
struct Arguments {
    text: String,
    overflowed: bool,
    /// Whether any fragment carried a non-blank byte.
    substantive: bool,
    /// Arguments the provider announced when the call opened, used only
    /// when no fragment arrives.
    announced: Option<serde_json::Value>,
}

/// The most argument bytes one call accumulates.
const MAX_TOOL_INPUT_BYTES: usize = 32 * 1024 * 1024;

impl Arguments {
    fn push(&mut self, fragment: &str, name: &str) {
        self.substantive |= !fragment.trim().is_empty();
        // Some OpenAI-compatible gateways send a literal `null` before the
        // real fragments; a non-blank fragment supersedes it.
        if self.text.trim() == "null" && !fragment.trim().is_empty() {
            self.text.clear();
        }
        if self.text.len().saturating_add(fragment.len()) > MAX_TOOL_INPUT_BYTES {
            if !self.overflowed {
                self.overflowed = true;
                tracing::warn!(
                    tool = name,
                    "streamed tool-call input exceeded the accumulation bound; truncating"
                );
            }
        } else {
            self.text.push_str(fragment);
        }
    }

    /// The arguments as JSON: the announced ones when no fragment arrived.
    fn parse(&self) -> Result<serde_json::Value, serde_json::Error> {
        if self.text.is_empty()
            && let Some(announced) = &self.announced
        {
            return Ok(announced.clone());
        }
        let arguments = crate::json_utils::parse_tool_arguments(&self.text)?;
        if self.overflowed {
            return Err(serde::de::Error::custom(
                "tool-call input exceeded the accumulation bound",
            ));
        }
        Ok(arguments)
    }
}

/// What to do with a call whose arguments do not parse when it closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfMalformed {
    /// Fail the reply with [`ProviderError::MalformedToolInput`]: the
    /// provider said the call was complete.
    Fail,
    /// Deliver the call with `{}` arguments: the provider superseded it
    /// mid-assembly.
    EmptyObject,
    /// Drop it: its input never fully arrived.
    Drop,
    /// Leave it open: the close was a probe, and more input may follow.
    KeepOpen,
}

/// What one fragment of a buffered tool call carries.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallFragment<'a> {
    /// The provider's id for the call.
    pub id: Option<&'a str>,
    /// The tool's name.
    pub name: Option<&'a str>,
    /// A fragment of the argument JSON.
    pub arguments: Option<&'a str>,
}

/// The first index the writer hands out itself ([`Out::fresh_index`]); a
/// provider's wire indices stay below it.
pub const AUTO_INDEX: usize = 1 << 48;

fn not_open(index: usize) -> ProviderError {
    ProviderError::Response(format!(
        "the reply wrote to item {index}, which is not open"
    ))
}

impl Turn {
    /// The writer of a reply from `origin`.
    pub(crate) fn new(origin: Origin) -> Self {
        Self {
            span: tracing::Span::none(),
            origin,
            open: BTreeMap::new(),
            next_part: 0,
            call_ids: HashSet::new(),
            native: None,
            run: None,
            next_auto: AUTO_INDEX,
            choice: Vec::new(),
            open_text: BTreeMap::new(),
        }
    }

    /// The fold of a stream relayed from another fold, which built its
    /// events: it only collects them.
    pub(crate) fn relayed(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        Self::new(Origin::new(provider.clone(), provider, ""))
    }

    /// The next position in the choice.
    fn next(&mut self) -> Part {
        let part = Part::new(self.next_part);
        self.next_part += 1;
        part
    }

    fn fresh_index(&mut self) -> usize {
        let index = self.next_auto;
        self.next_auto += 1;
        index
    }

    fn draft(&mut self, index: usize) -> Result<&mut Draft, ProviderError> {
        self.open.get_mut(&index).ok_or_else(|| not_open(index))
    }

    pub(crate) fn open_item(
        &mut self,
        index: usize,
        block: Block,
        item: serde_json::Value,
    ) -> Result<(), ProviderError> {
        if self.open.contains_key(&index) {
            return Err(ProviderError::Response(format!(
                "the reply opened item {index} twice"
            )));
        }
        let body = match block {
            Block::Text => Body::Text(String::new()),
            Block::Reasoning { redacted } => Body::Reasoning {
                text: String::new(),
                redacted,
            },
            Block::Call { id, name } => Body::Call {
                id: Some(id),
                name: name.into(),
                arguments: Arguments::default(),
            },
            Block::Opaque { replay } => Body::Opaque { replay },
        };
        self.insert(index, body, item);
        Ok(())
    }

    fn insert(&mut self, index: usize, body: Body, item: serde_json::Value) {
        let part = self.next();
        let started = false;
        self.open.insert(
            index,
            Draft {
                part,
                started,
                item,
                body,
            },
        );
    }

    pub(crate) fn push_item(
        &mut self,
        items: &mut Items,
        index: usize,
        fragment: &str,
    ) -> Result<(), ProviderError> {
        if fragment.is_empty() {
            return Ok(());
        }
        let draft = self.draft(index)?;
        let event = match &mut draft.body {
            Body::Text(text) => {
                text.push_str(fragment);
                StreamEvent::Text {
                    part: draft.part,
                    text: fragment.to_owned(),
                }
            }
            Body::Reasoning { text, .. } => {
                text.push_str(fragment);
                StreamEvent::Reasoning {
                    part: draft.part,
                    text: fragment.to_owned(),
                }
            }
            Body::Call {
                name, arguments, ..
            } => {
                arguments.push(fragment, name);
                return Ok(());
            }
            Body::Opaque { .. } => {
                return Err(ProviderError::Response(format!(
                    "the reply wrote text to the opaque item {index}"
                )));
            }
        };
        if !draft.started {
            draft.started = true;
            emit(
                items,
                StreamEvent::Start {
                    part: draft.part,
                    kind: draft.body.kind(),
                },
            );
        }
        emit(items, event);
        Ok(())
    }

    /// Close the item at `index`: its block becomes visible with the
    /// provider's item as its native, or `if_malformed` decides for a call
    /// whose arguments do not parse.
    pub(crate) fn close_item(
        &mut self,
        items: &mut Items,
        index: usize,
        if_malformed: IfMalformed,
    ) -> Result<(), ProviderError> {
        let draft = self.open.remove(&index).ok_or_else(|| not_open(index))?;
        if self.run == Some(index) {
            self.run = None;
        }
        let Draft {
            part,
            started,
            item,
            body,
        } = draft;
        let content = match body {
            Body::Text(text) => {
                if text.is_empty() && item.is_null() {
                    return Ok(());
                }
                with_item(AssistantContent::Text(Text::new(text)), item)
            }
            Body::Reasoning { text, redacted } => {
                if text.is_empty() && !redacted && item.is_null() {
                    return Ok(());
                }
                let reasoning = Reasoning {
                    text,
                    redacted,
                    native: None,
                };
                with_item(AssistantContent::Reasoning(reasoning), item)
            }
            Body::Opaque { replay } => AssistantContent::Opaque(Opaque { item, replay }),
            Body::Call {
                id,
                name,
                arguments,
            } => {
                let Ok(name) = ToolName::new(name) else {
                    return Ok(());
                };
                let parsed = match (arguments.parse(), if_malformed) {
                    (Ok(parsed), _) => parsed,
                    (Err(_), IfMalformed::KeepOpen) => {
                        self.open.insert(
                            index,
                            Draft {
                                part,
                                started,
                                item,
                                body: Body::Call {
                                    id,
                                    name: name.into(),
                                    arguments,
                                },
                            },
                        );
                        return Ok(());
                    }
                    (Err(_), IfMalformed::EmptyObject) => {
                        serde_json::Value::Object(Default::default())
                    }
                    (Err(_), IfMalformed::Drop) => return Ok(()),
                    (Err(error), IfMalformed::Fail) => {
                        return Err(ProviderError::MalformedToolInput(MalformedToolInput {
                            name: name.into(),
                            id: id.unwrap_or_else(|| CallId::Local(LocalCallId::new())),
                            raw: arguments.text,
                            error: error.to_string(),
                        }));
                    }
                };
                let id = id.unwrap_or_else(|| CallId::Local(LocalCallId::new()));
                if !self.call_ids.insert(id.clone()) {
                    return Err(ProviderError::DuplicateCallId(id));
                }
                let json = if arguments.text.is_empty() {
                    parsed.to_string()
                } else {
                    arguments.text
                };
                emit(
                    items,
                    StreamEvent::Start {
                        part,
                        kind: PartKind::ToolCall,
                    },
                );
                emit(items, StreamEvent::Arguments { part, json });
                let call = ToolCall::new(id, ToolFunction::new(name, parsed));
                let content = with_item(AssistantContent::ToolCall(call), item);
                emit(items, StreamEvent::End { part, content });
                return Ok(());
            }
        };
        if !started {
            emit(
                items,
                StreamEvent::Start {
                    part,
                    kind: kind_of(&content),
                },
            );
        }
        emit(items, StreamEvent::End { part, content });
        Ok(())
    }

    pub(crate) fn run_item(
        &mut self,
        items: &mut Items,
        block: Block,
        fragment: &str,
    ) -> Result<usize, ProviderError> {
        let current = self.run.filter(|index| {
            self.open.get(index).is_some_and(|draft| {
                matches!(
                    (&draft.body, &block),
                    (Body::Text(_), Block::Text)
                        | (
                            Body::Reasoning {
                                redacted: false,
                                ..
                            },
                            Block::Reasoning { redacted: false }
                        )
                )
            })
        });
        let index = match current {
            Some(index) => index,
            None => {
                self.end_run(items)?;
                let index = self.fresh_index();
                self.open_item(index, block, serde_json::Value::Null)?;
                self.run = Some(index);
                index
            }
        };
        self.push_item(items, index, fragment)?;
        Ok(index)
    }

    pub(crate) fn end_run(&mut self, items: &mut Items) -> Result<(), ProviderError> {
        match self.run.take() {
            Some(index) => self.close_item(items, index, IfMalformed::Drop),
            None => Ok(()),
        }
    }

    /// One whole block at the next position, its native kept as given.
    pub(crate) fn write_content(
        &mut self,
        items: &mut Items,
        content: AssistantContent,
    ) -> Result<(), ProviderError> {
        if let AssistantContent::ToolCall(call) = &content
            && !self.call_ids.insert(call.id.clone())
        {
            return Err(ProviderError::DuplicateCallId(call.id.clone()));
        }
        let part = self.next();
        emit(
            items,
            StreamEvent::Start {
                part,
                kind: kind_of(&content),
            },
        );
        match &content {
            AssistantContent::Text(text) if !text.text.is_empty() => emit(
                items,
                StreamEvent::Text {
                    part,
                    text: text.text.clone(),
                },
            ),
            AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => emit(
                items,
                StreamEvent::Reasoning {
                    part,
                    text: reasoning.text.clone(),
                },
            ),
            AssistantContent::ToolCall(call) => emit(
                items,
                StreamEvent::Arguments {
                    part,
                    json: call.function.arguments.to_string(),
                },
            ),
            _ => {}
        }
        emit(items, StreamEvent::End { part, content });
        Ok(())
    }

    /// Close every item still open, in the order they opened. A call closes
    /// when it has an id and its arguments parse; one whose input never
    /// completed is dropped.
    pub(crate) fn close_open(&mut self, items: &mut Items) {
        let mut open: Vec<(Part, usize)> = self
            .open
            .iter()
            .map(|(index, draft)| (draft.part, *index))
            .collect();
        open.sort();
        for (_, index) in open {
            if matches!(
                self.open.get(&index),
                Some(Draft {
                    body: Body::Call { id: None, .. },
                    ..
                })
            ) {
                self.open.remove(&index);
                continue;
            }
            let _ = self.close_item(items, index, IfMalformed::Drop);
        }
    }

    /// The parts taken so far, in their position; a part that has not
    /// ended is not among them.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.choice.iter().flatten().cloned().collect()
    }

    /// Who the reply is from.
    pub fn origin(&self) -> &Origin {
        &self.origin
    }

    /// What arrived so far as a response: every part that ended, the text
    /// the consumer already took of a text part still open, and the
    /// provider's end when it arrived.
    pub(crate) fn partial(&self, end: Option<&Finish>, reply: &Reply) -> CompletionResponse {
        let mut response = self.response(end.cloned().unwrap_or_default(), reply.clone());
        response.choice = self
            .choice
            .iter()
            .enumerate()
            .filter_map(|(index, part)| {
                part.clone().or_else(|| {
                    self.open_text
                        .get(&index)
                        .map(|text| AssistantContent::text(text.clone()))
                })
            })
            .collect();
        response
    }

    fn response(&self, end: Finish, reply: Reply) -> CompletionResponse {
        let Finish {
            usage,
            reason,
            response_id,
            model,
            error,
        } = end;
        use crate::provider_response::reported;
        let mut origin = self.origin.clone();
        origin.provider = reply.provider;
        origin.response_model = reported(model);
        origin.response_id = reported(response_id);
        let choice = self.snapshot();
        let native = self.native.clone().and_then(|item| {
            AssistantMessage::new(choice.clone())
                .with_native(item)
                .native
        });
        let mut response = CompletionResponse::new(choice, usage, origin, reply.raw)
            .with_optional_finish_reason(reason);
        response.native = native;
        response.error = error;
        response.provider_request_id = reported(reply.provider_request_id);
        response
    }
}

/// `block` holding `item` as its native, unless there is no item.
fn with_item(block: AssistantContent, item: serde_json::Value) -> AssistantContent {
    if item.is_null() {
        block
    } else {
        block.with_native(item)
    }
}

fn kind_of(content: &AssistantContent) -> PartKind {
    match content {
        AssistantContent::Text(_) => PartKind::Text,
        AssistantContent::Reasoning(_) => PartKind::Reasoning,
        AssistantContent::ToolCall(_) => PartKind::ToolCall,
        AssistantContent::Image(_) => PartKind::Image,
        AssistantContent::Opaque(_) => PartKind::Opaque,
    }
}

pub(crate) type Items = std::collections::VecDeque<Result<Item<StreamEvent>, ProviderError>>;

fn emit(items: &mut Items, event: StreamEvent) {
    items.push_back(Ok(Item::Event(event)));
}

/// Merge `delta` into `item`: each string field but `type` appends to the
/// item's string of that key, an array extends the item's array, and any
/// other value replaces the key. A delta kind rig has never seen still
/// lands in the item.
fn merge_delta(item: &mut serde_json::Value, delta: &serde_json::Map<String, serde_json::Value>) {
    use serde_json::Value;
    if !item.is_object() {
        *item = Value::Object(serde_json::Map::new());
    }
    let Value::Object(item) = item else {
        return;
    };
    for (key, value) in delta {
        if key == "type" {
            continue;
        }
        match (item.get_mut(key), value) {
            (Some(Value::String(existing)), Value::String(fragment)) => {
                existing.push_str(fragment);
            }
            (Some(Value::Array(existing)), Value::Array(more)) => {
                existing.extend(more.iter().cloned());
            }
            _ => {
                item.insert(key.clone(), value.clone());
            }
        }
    }
}

impl Fold<Completion> for Turn {
    fn absorb(&mut self, event: &StreamEvent) -> Result<(), ProviderError> {
        match event {
            StreamEvent::Start { part, .. } => {
                if self.choice.len() <= part.index() {
                    self.choice.resize(part.index() + 1, None);
                }
            }
            StreamEvent::End { part, content } => {
                if self.choice.len() <= part.index() {
                    self.choice.resize(part.index() + 1, None);
                }
                if let Some(slot) = self.choice.get_mut(part.index()) {
                    *slot = Some(content.clone());
                }
                self.open_text.remove(&part.index());
            }
            StreamEvent::Text { part, text } => {
                self.open_text
                    .entry(part.index())
                    .or_default()
                    .push_str(text);
            }
            StreamEvent::Reasoning { .. } | StreamEvent::Arguments { .. } => {}
        }
        Ok(())
    }

    fn finish(self, end: Finish, reply: Reply) -> Result<CompletionResponse, ProviderError> {
        let response = self.response(end, reply);
        self.span
            .record_response(response.response_id(), response.model(), &response.usage);
        Ok(response)
    }
}

impl<'id> Out<'id, Completion> {
    /// Open the provider item at wire `index` as `block`: it takes the next
    /// position in the reply. `item` is the item as its first event states
    /// it, the base its deltas merge into; `Null` when the block keeps no
    /// provider item. An index already open is an error.
    pub fn open(
        &mut self,
        index: usize,
        block: Block,
        item: serde_json::Value,
    ) -> Result<(), ProviderError> {
        self.lock().fold.open_item(index, block, item)
    }

    /// Append a fragment to the open item at `index`: text, reasoning, or a
    /// call's argument JSON, by the item's block.
    pub fn push(&mut self, index: usize, fragment: &str) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.push_item(items, index, fragment)
    }

    /// Merge a provider delta into the item at `index`: its string fields
    /// append by key (so signature fragments concatenate), arrays extend,
    /// other values replace. The delta's `type` is not merged.
    pub fn merge(
        &mut self,
        index: usize,
        delta: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), ProviderError> {
        merge_delta(&mut self.lock().fold.draft(index)?.item, delta);
        Ok(())
    }

    /// Edit the item at `index` in place, for a delta [`Self::merge`] does
    /// not model or the whole item a provider restates when it finishes.
    pub fn edit(
        &mut self,
        index: usize,
        edit: impl FnOnce(&mut serde_json::Value),
    ) -> Result<(), ProviderError> {
        edit(&mut self.lock().fold.draft(index)?.item);
        Ok(())
    }

    /// Close the item at `index`: its block becomes visible. Arguments that
    /// do not parse are handled by `if_malformed`; a reused provider call
    /// id is [`ProviderError::DuplicateCallId`]. Empty text and reasoning
    /// with no provider item are dropped, and so is a call with no name. A
    /// call that never got an id gets one rig issues.
    pub fn close(&mut self, index: usize, if_malformed: IfMalformed) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.close_item(items, index, if_malformed)
    }

    /// Open and close the item at `index` in one step: a whole block a
    /// provider states in one piece, `text` its text, reasoning or argument
    /// JSON.
    pub fn whole(
        &mut self,
        index: usize,
        block: Block,
        item: serde_json::Value,
        text: &str,
    ) -> Result<(), ProviderError> {
        self.open(index, block, item)?;
        self.push(index, text)?;
        self.close(index, IfMalformed::Fail)
    }

    /// Buffer one fragment of the tool call the provider streams under
    /// `index`, opening it at its first fragment. Its id and name may arrive
    /// in any fragment; the call becomes visible when it closes.
    pub fn fragment(
        &mut self,
        index: usize,
        fragment: CallFragment<'_>,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let turn = &mut shared.fold;
        if !turn.open.contains_key(&index) {
            let body = Body::Call {
                id: None,
                name: String::new(),
                arguments: Arguments::default(),
            };
            turn.insert(index, body, serde_json::Value::Null);
        }
        let Body::Call {
            id,
            name,
            arguments,
        } = &mut turn.draft(index)?.body
        else {
            return Err(ProviderError::Response(format!(
                "the reply sent a call fragment for item {index}, which is not a call"
            )));
        };
        if let Some(call_id) = fragment.id.filter(|id| !id.is_empty()) {
            *id = Some(CallId::from_wire(call_id));
        }
        if let Some(fragment) = fragment.name.filter(|name| !name.is_empty()) {
            fragment.clone_into(name);
        }
        if let Some(fragment) = fragment.arguments {
            arguments.push(fragment, name);
        }
        Ok(())
    }

    /// Arguments the provider announced for the call at `index`, used only
    /// if no fragment arrives.
    pub fn announce(
        &mut self,
        index: usize,
        announced: serde_json::Value,
    ) -> Result<(), ProviderError> {
        if let Body::Call { arguments, .. } = &mut self.lock().fold.draft(index)?.body {
            arguments.announced = Some(announced);
        }
        Ok(())
    }

    /// Drop the item at `index`: it never becomes visible.
    pub fn discard(&mut self, index: usize) {
        let mut shared = self.lock();
        shared.fold.open.remove(&index);
        if shared.fold.run == Some(index) {
            shared.fold.run = None;
        }
    }

    /// An index no provider item uses, for a wire that indexes nothing.
    pub fn fresh_index(&mut self) -> usize {
        self.lock().fold.fresh_index()
    }

    /// A fragment on a wire that marks no item boundaries: it continues the
    /// block the last fragment went to while `block` is the same kind, and
    /// otherwise closes that block and opens a new one. Returns the block's
    /// index, for merging its provider item.
    pub fn run(&mut self, block: Block, fragment: &str) -> Result<usize, ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.run_item(items, block, fragment)
    }

    /// Close the block [`Self::run`] is extending, if any: output of
    /// another kind interleaved it.
    pub fn end_run(&mut self) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.end_run(items)
    }

    /// Whether the item at `index` is open.
    pub fn is_open(&self, index: usize) -> bool {
        self.lock().fold.open.contains_key(&index)
    }

    /// The wire indices of the open items, in index order.
    pub fn open_items(&self) -> Vec<usize> {
        self.lock().fold.open.keys().copied().collect()
    }

    /// A whole block of an already assembled response, at the next
    /// position, its native kept as given.
    pub fn content(&mut self, content: AssistantContent) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.write_content(items, content)
    }

    /// The provider's whole assistant message, for a wire whose output item
    /// is the message. It becomes the turn's message-level native.
    pub fn message_native(&mut self, item: serde_json::Value) {
        self.lock().fold.native = Some(item);
    }
}

/// The events a stream of `response` would have carried: each part whole,
/// in order.
pub(crate) fn events_of(
    response: &CompletionResponse,
) -> Result<Vec<Item<StreamEvent>>, ProviderError> {
    let mut turn = Turn::new(response.origin.clone());
    let mut items = Items::new();
    for content in &response.choice {
        turn.write_content(&mut items, content.clone())?;
    }
    items.into_iter().collect()
}

#[cfg(test)]
mod tests;
