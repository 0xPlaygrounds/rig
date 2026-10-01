//! Generating an assistant turn. A completion decoder writes its reply
//! through move-only part handles, and the writer emits each part's events
//! in order: a fragment needs its part's handle, closing a part consumes it,
//! and ending the reply closes every part still open, in the order they
//! opened. The provider's end of the reply is a [`Finish`], which the fold
//! needs to produce the response.
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
use std::marker::PhantomData;

use crate::completion::{CompletionRequest, CompletionResponse, FinishReason, Usage};
use crate::error::{MalformedToolInput, ProviderError};
use crate::message::{
    AdditionalParams, AssistantContent, CallId, Image, Issuer, LocalCallId, Native, ProviderCallId,
    Reasoning, ReasoningContent, Sealed, Text, ToolCall, ToolFunction, ToolName,
};
use crate::streaming::{Item, Part, PartKind, StreamEvent};
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Assembled, Call, Emit, Fold, Mode, Operation, Out, Reply, Shared};

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
        Turn {
            span,
            ..Turn::new(call.wire.name)
        }
    }

    /// [`CompletionRequest::validate_message_content`].
    fn validate(request: &Self::Request) -> Result<(), ProviderError> {
        request.validate_message_content()
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
    /// The assistant message id, for replay. A message id the decoder
    /// recorded while the reply was open outranks it.
    pub message_id: Option<String>,
    /// The response id. Never replayed as a message id.
    pub response_id: Option<String>,
    /// The model the provider reports.
    pub model: Option<String>,
}

/// The completion fold. The driver and the bus writer build it; no other
/// code can feed one.
///
/// It is both sides of one reply: the writer state a decoder writes through
/// (open parts, the pending-call buffer, the issuer of the reply's
/// reasoning) and the parts the consumer has taken, in their position.
pub struct Turn {
    span: tracing::Span,
    /// The provider the reply is from, and the default issuer of its
    /// reasoning.
    provider: String,
    // The writer.
    drafts: Vec<Draft>,
    next_part: u32,
    issuer: Option<Issuer>,
    message_id: Option<String>,
    pending: BTreeMap<usize, Pending>,
    provider_ids: HashSet<ProviderCallId>,
    // The fold.
    choice: Vec<Option<AssistantContent>>,
    /// The text the consumer took of parts still open, by position.
    open_text: BTreeMap<usize, String>,
}

/// A part a handle names, as the writer holds it.
enum Draft {
    Text {
        part: Option<Part>,
        text: String,
        params: Option<AdditionalParams>,
    },
    Reasoning {
        part: Option<Part>,
        text: String,
    },
    Call {
        id: CallId,
        name: ToolName,
        arguments: Arguments,
        signature: Option<String>,
        additional_params: Option<serde_json::Value>,
    },
    Closed,
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

/// One tool call the provider streams under a wire index, until its id and
/// name are both known and it opens.
#[derive(Default)]
struct Pending {
    id: Option<String>,
    item_id: Option<String>,
    name: String,
    arguments: Arguments,
    signature: Option<String>,
    additional_params: Option<serde_json::Value>,
    /// The call's handle, once it opened.
    open: Option<usize>,
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
    /// The output-item id a dual-identifier wire issues beside it.
    pub item_id: Option<&'a str>,
    /// The tool's name.
    pub name: Option<&'a str>,
    /// A fragment of the argument JSON.
    pub arguments: Option<&'a str>,
}

/// How a closed reasoning part ends: its provider id, a signature, or the
/// provider's whole restatement of it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Seal {
    /// The provider's id for the reasoning item.
    pub id: Option<String>,
    /// A signature for the reasoning text.
    pub signature: Option<String>,
    /// The provider's authoritative restatement, which supersedes the
    /// fragments.
    pub restated: Option<Reasoning>,
}

type Brand<'id> = PhantomData<fn(&'id ()) -> &'id ()>;

/// An open text part of one reply. Only opening the part gives one, so a
/// fragment cannot precede its part's start:
///
/// ```compile_fail,E0599
/// use rig_core::operation::{Completion, TextPart};
/// use rig_core::wire::Out;
///
/// fn early(out: &mut Out<'_, Completion>) {
///     out.push_text(&TextPart::default(), "before the start");
/// }
/// ```
///
/// Nor can one be built from its fields to write into a part:
///
/// ```compile_fail,E0451
/// use rig_core::operation::{Completion, TextPart};
/// use rig_core::wire::Out;
///
/// fn forge<'id>(out: &mut Out<'id, Completion>) {
///     let part = TextPart { slot: 0, brand: std::marker::PhantomData };
///     out.push_text(&part, "into a part this code never opened");
/// }
/// ```
#[must_use = "an open part is closed when the reply ends"]
#[derive(Debug)]
pub struct TextPart<'id> {
    slot: usize,
    brand: Brand<'id>,
}

/// An open reasoning part of one reply.
#[must_use = "an open part is closed when the reply ends"]
#[derive(Debug)]
pub struct ReasoningPart<'id> {
    slot: usize,
    brand: Brand<'id>,
}

/// An open tool call of one reply. It has its id and its name, and becomes
/// visible when it closes.
///
/// ```compile_fail,E0382
/// use rig_core::operation::{CallPart, Completion};
/// use rig_core::wire::Out;
///
/// // A closed call cannot be written again.
/// fn twice<'id>(out: &mut Out<'id, Completion>, part: CallPart<'id>) {
///     let _ = out.close_call(part);
///     out.push_arguments(&part, "{}");
/// }
/// ```
#[must_use = "an open call that is not closed never becomes visible"]
#[derive(Debug)]
pub struct CallPart<'id> {
    slot: usize,
    brand: Brand<'id>,
}

impl Turn {
    /// The writer of a reply from `provider`.
    pub(crate) fn new(provider: impl Into<String>) -> Self {
        Self {
            span: tracing::Span::none(),
            provider: provider.into(),
            drafts: Vec::new(),
            next_part: 0,
            issuer: None,
            message_id: None,
            pending: BTreeMap::new(),
            provider_ids: HashSet::new(),
            choice: Vec::new(),
            open_text: BTreeMap::new(),
        }
    }

    /// The issuer this reply's reasoning is sealed to.
    fn issuer(&self) -> Issuer {
        self.issuer
            .clone()
            .unwrap_or_else(|| Issuer::from(self.provider.clone()))
    }

    fn draft(&mut self, draft: Draft) -> usize {
        self.drafts.push(draft);
        self.drafts.len() - 1
    }

    /// The next position in the choice.
    fn next(&mut self) -> Part {
        let part = Part::new(self.next_part);
        self.next_part += 1;
        part
    }

    fn start(&mut self, items: &mut Items, kind: PartKind) -> Part {
        let part = self.next();
        emit(items, StreamEvent::Start { part, kind });
        part
    }

    fn push_text(&mut self, items: &mut Items, slot: usize, fragment: &str) {
        if fragment.is_empty() {
            return;
        }
        let started = match self.drafts.get(slot) {
            Some(Draft::Text { part, .. }) => *part,
            _ => return,
        };
        let part = match started {
            Some(part) => part,
            None => {
                let part = self.start(items, PartKind::Text);
                if let Some(Draft::Text {
                    part: slot_part, ..
                }) = self.drafts.get_mut(slot)
                {
                    *slot_part = Some(part);
                }
                part
            }
        };
        if let Some(Draft::Text { text, .. }) = self.drafts.get_mut(slot) {
            text.push_str(fragment);
        }
        emit(
            items,
            StreamEvent::Text {
                part,
                text: fragment.to_owned(),
            },
        );
    }

    fn close_text(&mut self, items: &mut Items, slot: usize) {
        let Some(Draft::Text { part, text, params }) = self
            .drafts
            .get_mut(slot)
            .map(|draft| std::mem::replace(draft, Draft::Closed))
        else {
            return;
        };
        // A text part survives with text or with the metadata it carries.
        if text.is_empty() && params.is_none() {
            return;
        }
        let part = part.unwrap_or_else(|| self.start(items, PartKind::Text));
        emit(
            items,
            StreamEvent::End {
                part,
                content: AssistantContent::Text(Text {
                    text,
                    additional_params: params,
                }),
            },
        );
    }

    fn push_reasoning(&mut self, items: &mut Items, slot: usize, fragment: &str) {
        if fragment.is_empty() {
            return;
        }
        let started = match self.drafts.get(slot) {
            Some(Draft::Reasoning { part, .. }) => *part,
            _ => return,
        };
        let part = match started {
            Some(part) => part,
            None => {
                let part = self.start(items, PartKind::Reasoning);
                if let Some(Draft::Reasoning {
                    part: slot_part, ..
                }) = self.drafts.get_mut(slot)
                {
                    *slot_part = Some(part);
                }
                part
            }
        };
        if let Some(Draft::Reasoning { text, .. }) = self.drafts.get_mut(slot) {
            text.push_str(fragment);
        }
        emit(
            items,
            StreamEvent::Reasoning {
                part,
                text: fragment.to_owned(),
            },
        );
    }

    fn close_reasoning(&mut self, items: &mut Items, slot: usize, seal: Seal) {
        let Some(Draft::Reasoning { part, text }) = self
            .drafts
            .get_mut(slot)
            .map(|draft| std::mem::replace(draft, Draft::Closed))
        else {
            return;
        };
        let Seal {
            id,
            signature,
            restated,
        } = seal;
        let reasoning = match restated {
            Some(mut restated) => {
                // An omitted restatement id must not erase an established one.
                if restated.id.is_none() {
                    restated.id = id;
                }
                if let Some(signature) = signature {
                    attach_signature(&mut restated, signature);
                }
                restated
            }
            None if !text.is_empty() => Reasoning {
                id,
                content: vec![ReasoningContent::Text { text, signature }],
            },
            // A signature with nothing streamed to sign is replay state of
            // its own.
            None => match signature {
                Some(signature) => Reasoning {
                    id,
                    content: vec![ReasoningContent::Text {
                        text: String::new(),
                        signature: Some(signature),
                    }],
                },
                None => return,
            },
        };
        let part = part.unwrap_or_else(|| self.start(items, PartKind::Reasoning));
        let content = AssistantContent::Reasoning(reasoning.sealed(self.issuer()));
        emit(items, StreamEvent::End { part, content });
    }

    fn open_call(&mut self, id: CallId, name: ToolName) -> Result<usize, ProviderError> {
        if let Some(provider) = id.provider()
            && !self.provider_ids.insert(provider.clone())
        {
            return Err(ProviderError::DuplicateCallId(id));
        }
        Ok(self.draft(Draft::Call {
            id,
            name,
            arguments: Arguments::default(),
            signature: None,
            additional_params: None,
        }))
    }

    fn push_arguments(&mut self, slot: usize, fragment: &str) {
        if let Some(Draft::Call {
            name, arguments, ..
        }) = self.drafts.get_mut(slot)
        {
            arguments.push(fragment, name.as_str());
        }
    }

    /// Close the call in `slot`: it becomes visible with its arguments, or
    /// `if_malformed` decides.
    fn close_call(
        &mut self,
        items: &mut Items,
        slot: usize,
        if_malformed: IfMalformed,
    ) -> Result<(), ProviderError> {
        let parsed = match self.drafts.get(slot) {
            Some(Draft::Call { arguments, .. }) => arguments.parse(),
            _ => return Ok(()),
        };
        let parsed = match (parsed, if_malformed) {
            (Ok(arguments), _) => arguments,
            (Err(_), IfMalformed::KeepOpen) => return Ok(()),
            (Err(_), IfMalformed::EmptyObject) => serde_json::Value::Object(Default::default()),
            (Err(_), IfMalformed::Drop) => {
                if let Some(draft) = self.drafts.get_mut(slot) {
                    *draft = Draft::Closed;
                }
                return Ok(());
            }
            (Err(error), IfMalformed::Fail) => {
                let Some(Draft::Call {
                    id,
                    name,
                    arguments,
                    ..
                }) = self
                    .drafts
                    .get_mut(slot)
                    .map(|draft| std::mem::replace(draft, Draft::Closed))
                else {
                    return Ok(());
                };
                return Err(ProviderError::MalformedToolInput(MalformedToolInput {
                    name: name.into(),
                    id,
                    raw: arguments.text,
                    error: error.to_string(),
                }));
            }
        };
        let Some(Draft::Call {
            id,
            name,
            arguments,
            signature,
            additional_params,
        }) = self
            .drafts
            .get_mut(slot)
            .map(|draft| std::mem::replace(draft, Draft::Closed))
        else {
            return Ok(());
        };
        let json = if arguments.text.is_empty() {
            parsed.to_string()
        } else {
            arguments.text
        };
        let part = self.start(items, PartKind::ToolCall);
        emit(items, StreamEvent::Arguments { part, json });
        emit(
            items,
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(ToolCall {
                    id,
                    function: ToolFunction {
                        name,
                        arguments: parsed,
                    },
                    signature,
                    additional_params,
                }),
            },
        );
        Ok(())
    }

    /// The buffered call at `index`, opened when its id and name are both
    /// known. A wire that sends no id gets one rig issues when the call
    /// closes (`issue`).
    fn open_pending(&mut self, index: usize, issue: bool) -> Result<Option<usize>, ProviderError> {
        let Some(pending) = self.pending.get(&index) else {
            return Ok(None);
        };
        if let Some(slot) = pending.open {
            return Ok(Some(slot));
        }
        let Ok(name) = ToolName::new(pending.name.clone()) else {
            return Ok(None);
        };
        let id = match (&pending.id, &pending.item_id) {
            (Some(call_id), item_id) => match ProviderCallId::new(call_id.clone()) {
                Some(provider) => CallId::Provider(match item_id {
                    Some(item_id) => provider.with_item_id(item_id.clone()),
                    None => provider,
                }),
                None => CallId::Local(LocalCallId::new()),
            },
            (None, _) if issue => CallId::Local(LocalCallId::new()),
            (None, _) => return Ok(None),
        };
        let slot = self.open_call(id, name)?;
        let Some(pending) = self.pending.get_mut(&index) else {
            return Ok(None);
        };
        pending.open = Some(slot);
        let arguments = std::mem::take(&mut pending.arguments);
        let signature = pending.signature.take();
        let additional_params = pending.additional_params.take();
        if let Some(Draft::Call {
            arguments: open,
            signature: open_signature,
            additional_params: open_params,
            ..
        }) = self.drafts.get_mut(slot)
        {
            *open = arguments;
            *open_signature = signature;
            *open_params = additional_params;
        }
        Ok(Some(slot))
    }

    /// Close every part still open, in the order they opened. A call closes
    /// when its arguments parse; one whose input never completed is
    /// dropped, and so is every call still buffered.
    pub(crate) fn close_open(&mut self, items: &mut Items) {
        self.pending.clear();
        for slot in 0..self.drafts.len() {
            match self.drafts.get(slot) {
                Some(Draft::Text { .. }) => self.close_text(items, slot),
                Some(Draft::Reasoning { .. }) => self.close_reasoning(items, slot, Seal::default()),
                Some(Draft::Call { .. }) => {
                    let _ = self.close_call(items, slot, IfMalformed::Drop);
                }
                Some(Draft::Closed) | None => {}
            }
        }
    }

    /// The parts taken so far, in their position; a part that has not
    /// ended is not among them.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.choice.iter().flatten().cloned().collect()
    }

    /// The assistant message id the decoder recorded, if any.
    pub fn message_id(&self) -> Option<&str> {
        self.message_id.as_deref()
    }

    /// The issuer this reply's reasoning is sealed to.
    pub fn reasoning_issuer(&self) -> Issuer {
        self.issuer()
    }

    /// The provider the reply is from.
    pub fn provider(&self) -> &str {
        &self.provider
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
            .map(|part| reseal(part, &self.issuer()))
            .collect();
        response
    }

    fn response(&self, end: Finish, reply: Reply) -> CompletionResponse {
        let issuer = self.issuer();
        let choice = self
            .snapshot()
            .into_iter()
            .map(|part| reseal(part, &issuer))
            .collect();
        let Finish {
            usage,
            reason,
            message_id,
            response_id,
            model,
        } = end;
        use crate::provider_response::reported;
        let mut response = CompletionResponse::new(choice, usage, reply.provider, reply.raw)
            .with_optional_finish_reason(reason);
        // A message id the decoder recorded outranks the end's.
        response.message_id = reported(self.message_id.clone().or(message_id));
        response.response_id = reported(response_id);
        response.model = reported(model);
        response.provider_request_id = reported(reply.provider_request_id);
        response
    }
}

/// `part` sealed to the reply's issuer, when it is sealed at all.
fn reseal(part: AssistantContent, issuer: &Issuer) -> AssistantContent {
    match part {
        AssistantContent::Reasoning(reasoning) => {
            AssistantContent::Reasoning(reasoning.reseal(issuer.clone()))
        }
        AssistantContent::Native(native) => AssistantContent::Native(native.reseal(issuer.clone())),
        part @ (AssistantContent::Text(_)
        | AssistantContent::ToolCall(_)
        | AssistantContent::Image(_)) => part,
    }
}

pub(crate) type Items = std::collections::VecDeque<Result<Item<StreamEvent>, ProviderError>>;

fn emit(items: &mut Items, event: StreamEvent) {
    items.push_back(Ok(Item::Event(event)));
}

/// Attach a signature to the last unsigned reasoning text, or add a
/// signature-only text: replay needs every signature.
fn attach_signature(reasoning: &mut Reasoning, signature: String) {
    match reasoning
        .content
        .iter_mut()
        .rev()
        .find_map(|content| match content {
            ReasoningContent::Text {
                signature: slot @ None,
                ..
            } => Some(slot),
            _ => None,
        }) {
        Some(slot) => *slot = Some(signature),
        None => reasoning.content.push(ReasoningContent::Text {
            text: String::new(),
            signature: Some(signature),
        }),
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
                // A relayed reply's sealed parts name their issuer.
                if self.issuer.is_none() {
                    match content {
                        AssistantContent::Reasoning(reasoning) => {
                            self.issuer = Some(reasoning.issuer().clone());
                        }
                        AssistantContent::Native(native) => {
                            self.issuer = Some(native.issuer().clone());
                        }
                        AssistantContent::Text(_)
                        | AssistantContent::ToolCall(_)
                        | AssistantContent::Image(_) => {}
                    }
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
        self.span.record_response(
            response
                .response_id
                .as_deref()
                .or(response.message_id.as_deref()),
            response.model.as_deref(),
            &response.usage,
        );
        Ok(response)
    }
}

impl<'id> Out<'id, Completion> {
    /// Open a text part. Nothing is emitted until its first fragment.
    pub fn text(&mut self) -> TextPart<'id> {
        let slot = self.lock().fold.draft(Draft::Text {
            part: None,
            text: String::new(),
            params: None,
        });
        TextPart {
            slot,
            brand: PhantomData,
        }
    }

    /// Append to an open text part.
    pub fn push_text(&mut self, part: &TextPart<'id>, text: &str) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.push_text(items, part.slot, text);
    }

    /// Merge provider metadata into an open text part. Metadata is content:
    /// the part starts here if no text started it.
    pub fn text_params(&mut self, part: &TextPart<'id>, additional_params: AdditionalParams) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let unstarted = matches!(
            fold.drafts.get(part.slot),
            Some(Draft::Text { part: None, .. })
        );
        if unstarted {
            let started = fold.start(items, PartKind::Text);
            if let Some(Draft::Text { part, .. }) = fold.drafts.get_mut(part.slot) {
                *part = Some(started);
            }
        }
        if let Some(Draft::Text { params, .. }) = fold.drafts.get_mut(part.slot) {
            match params {
                Some(params) => params.merge(additional_params),
                None => *params = Some(additional_params),
            }
        }
    }

    /// Close a text part. One with neither text nor metadata is dropped.
    pub fn close_text(&mut self, part: TextPart<'id>) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.close_text(items, part.slot);
    }

    /// Append to the text part `open` holds, opening one there first. For a
    /// wire whose text chunks continue one part until other output
    /// interleaves it.
    pub fn extend_text<'p>(
        &mut self,
        open: &'p mut Option<TextPart<'id>>,
        text: &str,
    ) -> &'p TextPart<'id> {
        let part = open.get_or_insert_with(|| self.text());
        self.push_text(part, text);
        part
    }

    /// Close the text part `open` holds, if any, leaving it empty.
    pub fn close_open_text(&mut self, open: &mut Option<TextPart<'id>>) {
        if let Some(part) = open.take() {
            self.close_text(part);
        }
    }

    /// Open a reasoning part. Nothing is emitted until its first fragment.
    pub fn reasoning(&mut self) -> ReasoningPart<'id> {
        let slot = self.lock().fold.draft(Draft::Reasoning {
            part: None,
            text: String::new(),
        });
        ReasoningPart {
            slot,
            brand: PhantomData,
        }
    }

    /// Append to an open reasoning part.
    pub fn push_reasoning(&mut self, part: &ReasoningPart<'id>, text: &str) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.push_reasoning(items, part.slot, text);
    }

    /// Close a reasoning part, sealed to the reply's issuer. One with
    /// nothing to replay is dropped.
    pub fn close_reasoning(&mut self, part: ReasoningPart<'id>, seal: Seal) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.close_reasoning(items, part.slot, seal);
    }

    /// A whole reasoning part the provider sent in one piece.
    pub fn reasoning_block(&mut self, reasoning: Reasoning) {
        let part = self.reasoning();
        self.close_reasoning(
            part,
            Seal {
                restated: Some(reasoning),
                ..Seal::default()
            },
        );
    }

    /// Open a tool call with its id and name. A provider id already used by
    /// another call of this reply is [`ProviderError::DuplicateCallId`].
    ///
    /// Both are required, so no call opens without an id:
    ///
    /// ```compile_fail,E0308
    /// use rig_core::message::ToolName;
    /// use rig_core::operation::Completion;
    /// use rig_core::wire::Out;
    ///
    /// fn idless(out: &mut Out<'_, Completion>, name: ToolName) {
    ///     let _ = out.call(None, name);
    /// }
    /// ```
    ///
    /// or without a name:
    ///
    /// ```compile_fail,E0308
    /// use rig_core::message::CallId;
    /// use rig_core::operation::Completion;
    /// use rig_core::wire::Out;
    ///
    /// fn nameless(out: &mut Out<'_, Completion>, id: CallId) {
    ///     let _ = out.call(id, "");
    /// }
    /// ```
    pub fn call(&mut self, id: CallId, name: ToolName) -> Result<CallPart<'id>, ProviderError> {
        let slot = self.lock().fold.open_call(id, name)?;
        Ok(CallPart {
            slot,
            brand: PhantomData,
        })
    }

    /// Append a fragment of an open call's argument JSON.
    pub fn push_arguments(&mut self, part: &CallPart<'id>, json: &str) {
        self.lock().fold.push_arguments(part.slot, json);
    }

    /// Attach a provider signature and metadata to an open call.
    pub fn decorate_call(
        &mut self,
        part: &CallPart<'id>,
        signature: Option<String>,
        additional_params: Option<serde_json::Value>,
    ) {
        if let Some(Draft::Call {
            signature: open_signature,
            additional_params: open_params,
            ..
        }) = self.lock().fold.drafts.get_mut(part.slot)
        {
            if signature.is_some() {
                *open_signature = signature;
            }
            if additional_params.is_some() {
                *open_params = additional_params;
            }
        }
    }

    /// Close a call: it becomes visible. Arguments that do not parse are
    /// [`ProviderError::MalformedToolInput`].
    pub fn close_call(&mut self, part: CallPart<'id>) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.close_call(items, part.slot, IfMalformed::Fail)
    }

    /// Drop an open call: it never becomes visible.
    pub fn abandon_call(&mut self, part: CallPart<'id>) {
        if let Some(draft) = self.lock().fold.drafts.get_mut(part.slot) {
            *draft = Draft::Closed;
        }
    }

    /// A whole tool call the provider sent in one piece.
    pub fn tool_call(&mut self, call: ToolCall) -> Result<(), ProviderError> {
        let ToolCall {
            id,
            function,
            signature,
            additional_params,
        } = call;
        let part = self.call(id, function.name)?;
        self.decorate_call(&part, signature, additional_params);
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        if let Some(Draft::Call { arguments, .. }) = fold.drafts.get_mut(part.slot) {
            arguments.announced = Some(function.arguments);
        }
        fold.close_call(items, part.slot, IfMalformed::Fail)
    }

    /// An image part.
    pub fn image(&mut self, image: Image) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let part = fold.start(items, PartKind::Image);
        emit(
            items,
            StreamEvent::End {
                part,
                content: AssistantContent::Image(image),
            },
        );
    }

    /// A provider item with no canonical meaning, as one whole part sealed
    /// to the reply's issuer.
    pub fn native(&mut self, native: Native) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let part = fold.start(items, PartKind::Native);
        let issuer = fold.issuer();
        emit(
            items,
            StreamEvent::End {
                part,
                content: AssistantContent::Native(Sealed::new(issuer, native)),
            },
        );
    }

    /// A whole part of an already assembled response.
    pub fn content(&mut self, content: AssistantContent) -> Result<(), ProviderError> {
        match content {
            AssistantContent::Text(text) => {
                let part = self.text();
                self.push_text(&part, &text.text);
                if let Some(params) = text.additional_params {
                    self.text_params(&part, params);
                }
                self.close_text(part);
            }
            AssistantContent::Reasoning(reasoning) => {
                let issuer = reasoning.issuer().clone();
                self.issued_by(issuer.clone());
                if let Some(reasoning) = reasoning.open(&issuer) {
                    self.reasoning_block(reasoning.clone());
                }
            }
            AssistantContent::ToolCall(call) => self.tool_call(call)?,
            AssistantContent::Image(image) => self.image(image),
            AssistantContent::Native(native) => {
                let issuer = native.issuer().clone();
                self.issued_by(issuer.clone());
                if let Some(native) = native.open(&issuer) {
                    self.native(native.clone());
                }
            }
        }
        Ok(())
    }

    /// Record the assistant message id. It outranks the one the end names.
    pub fn message_id(&mut self, id: impl Into<String>) {
        let id = id.into();
        if !id.is_empty() {
            self.lock().fold.message_id = Some(id);
        }
    }

    /// Name the issuer of this reply's reasoning: a gateway relaying
    /// another provider's models.
    pub fn issued_by(&mut self, issuer: impl Into<Issuer>) {
        self.lock().fold.issuer = Some(issuer.into());
    }

    /// Buffer one fragment of the tool call the provider streams under
    /// `index`. The call opens when its id and name are both known; a
    /// provider id another call already has is
    /// [`ProviderError::DuplicateCallId`].
    pub fn call_fragment(
        &mut self,
        index: usize,
        fragment: CallFragment<'_>,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let turn = &mut shared.fold;
        let pending = turn.pending.entry(index).or_default();
        if let Some(id) = fragment.id.filter(|id| !id.is_empty()) {
            pending.id = Some(id.to_owned());
        }
        if let Some(item_id) = fragment.item_id.filter(|id| !id.is_empty()) {
            pending.item_id = Some(item_id.to_owned());
        }
        if let Some(name) = fragment.name.filter(|name| !name.is_empty()) {
            name.clone_into(&mut pending.name);
        }
        let open = pending.open;
        if let Some(arguments) = fragment.arguments.filter(|arguments| !arguments.is_empty()) {
            match open {
                Some(slot) => turn.push_arguments(slot, arguments),
                None => {
                    let name = pending.name.clone();
                    pending.arguments.push(arguments, &name);
                }
            }
        }
        turn.open_pending(index, false)?;
        Ok(())
    }

    /// Arguments the provider announced for the buffered call at `index`,
    /// used only if no fragment arrives.
    pub fn announce_pending(&mut self, index: usize, arguments: serde_json::Value) {
        let mut shared = self.lock();
        let turn = &mut shared.fold;
        let Some(pending) = turn.pending.get_mut(&index) else {
            return;
        };
        match pending.open {
            Some(slot) => {
                if let Some(Draft::Call {
                    arguments: open, ..
                }) = turn.drafts.get_mut(slot)
                {
                    open.announced = Some(arguments);
                }
            }
            None => pending.arguments.announced = Some(arguments),
        }
    }

    /// Attach a signature and metadata to the buffered call the provider
    /// names `provider_id`. What it already has wins.
    pub fn decorate_pending(
        &mut self,
        provider_id: &str,
        signature: Option<String>,
        additional_params: Option<serde_json::Value>,
    ) {
        if provider_id.is_empty() {
            return;
        }
        let mut shared = self.lock();
        let turn = &mut shared.fold;
        let Some(pending) = turn
            .pending
            .values_mut()
            .find(|pending| pending.id.as_deref() == Some(provider_id))
        else {
            return;
        };
        match pending.open {
            Some(slot) => {
                if let Some(Draft::Call {
                    signature: open_signature,
                    additional_params: open_params,
                    ..
                }) = turn.drafts.get_mut(slot)
                {
                    if open_signature.is_none() {
                        *open_signature = signature;
                    }
                    if open_params.is_none() {
                        *open_params = additional_params;
                    }
                }
            }
            None => {
                if pending.signature.is_none() {
                    pending.signature = signature;
                }
                if pending.additional_params.is_none() {
                    pending.additional_params = additional_params;
                }
            }
        }
    }

    /// Attach a signature and metadata to the buffered call at `index`.
    pub fn decorate_pending_at(
        &mut self,
        index: usize,
        signature: Option<String>,
        additional_params: Option<serde_json::Value>,
    ) {
        let mut shared = self.lock();
        let turn = &mut shared.fold;
        let Some(pending) = turn.pending.get_mut(&index) else {
            return;
        };
        match pending.open {
            Some(slot) => {
                if let Some(Draft::Call {
                    signature: open_signature,
                    additional_params: open_params,
                    ..
                }) = turn.drafts.get_mut(slot)
                {
                    if signature.is_some() {
                        *open_signature = signature;
                    }
                    if additional_params.is_some() {
                        *open_params = additional_params;
                    }
                }
            }
            None => {
                if signature.is_some() {
                    pending.signature = signature;
                }
                if additional_params.is_some() {
                    pending.additional_params = additional_params;
                }
            }
        }
    }

    /// The wire indices of the buffered calls, in order.
    pub fn pending_calls(&self) -> Vec<usize> {
        self.lock().fold.pending.keys().copied().collect()
    }

    /// The provider id of the buffered call at `index`, when it has one.
    pub fn pending_id(&self, index: usize) -> Option<String> {
        self.lock()
            .fold
            .pending
            .get(&index)
            .and_then(|pending| pending.id.clone())
    }

    /// The tool name the buffered call at `index` has so far.
    pub fn pending_name(&self, index: usize) -> String {
        self.lock()
            .fold
            .pending
            .get(&index)
            .map(|pending| pending.name.clone())
            .unwrap_or_default()
    }

    /// Whether the buffered call at `index` received argument bytes that are
    /// not blank, or announced arguments.
    pub fn pending_has_arguments(&self, index: usize) -> bool {
        let shared = self.lock();
        let turn = &shared.fold;
        let Some(pending) = turn.pending.get(&index) else {
            return false;
        };
        let arguments = match pending.open {
            Some(slot) => match turn.drafts.get(slot) {
                Some(Draft::Call { arguments, .. }) => arguments,
                _ => return false,
            },
            None => &pending.arguments,
        };
        arguments.substantive || arguments.announced.is_some()
    }

    /// Close the buffered call at `index`: it becomes visible, under the
    /// provider's id or, for a wire that sent none, one rig issues. A call
    /// with no name is dropped; `if_malformed` decides for one whose
    /// arguments do not parse.
    pub fn close_pending(
        &mut self,
        index: usize,
        if_malformed: IfMalformed,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let Some(slot) = fold.open_pending(index, true)? else {
            fold.pending.remove(&index);
            return Ok(());
        };
        let result = fold.close_call(items, slot, if_malformed);
        let kept_open = matches!(fold.drafts.get(slot), Some(Draft::Call { .. }));
        if !kept_open {
            fold.pending.remove(&index);
        }
        result
    }

    /// Drop the buffered call at `index`: it never becomes visible.
    pub fn drop_pending(&mut self, index: usize) {
        let mut shared = self.lock();
        let turn = &mut shared.fold;
        if let Some(pending) = turn.pending.remove(&index)
            && let Some(slot) = pending.open
            && let Some(draft) = turn.drafts.get_mut(slot)
        {
            *draft = Draft::Closed;
        }
    }
}

impl Turn {
    /// The fold of a stream relayed from another fold, which built its
    /// events: it only collects them.
    pub(crate) fn relayed(provider: impl Into<String>) -> Self {
        Self::new(provider)
    }

    // The bus writer writes through these, holding the parts it opened by
    // their slot.

    pub(crate) fn open_text(&mut self) -> usize {
        self.draft(Draft::Text {
            part: None,
            text: String::new(),
            params: None,
        })
    }

    pub(crate) fn write_text(&mut self, items: &mut Items, slot: usize, text: &str) {
        self.push_text(items, slot, text);
    }

    pub(crate) fn end_text(&mut self, items: &mut Items, slot: usize) {
        self.close_text(items, slot);
    }

    pub(crate) fn open_reasoning(&mut self) -> usize {
        self.draft(Draft::Reasoning {
            part: None,
            text: String::new(),
        })
    }

    pub(crate) fn write_reasoning(&mut self, items: &mut Items, slot: usize, text: &str) {
        self.push_reasoning(items, slot, text);
    }

    pub(crate) fn end_reasoning(&mut self, items: &mut Items, slot: usize) {
        self.close_reasoning(items, slot, Seal::default());
    }

    pub(crate) fn write_call(
        &mut self,
        items: &mut Items,
        call: ToolCall,
    ) -> Result<(), ProviderError> {
        let ToolCall {
            id,
            function,
            signature,
            additional_params,
        } = call;
        let slot = self.open_call(id, function.name)?;
        if let Some(Draft::Call {
            arguments,
            signature: open_signature,
            additional_params: open_params,
            ..
        }) = self.drafts.get_mut(slot)
        {
            arguments.announced = Some(function.arguments);
            *open_signature = signature;
            *open_params = additional_params;
        }
        self.close_call(items, slot, IfMalformed::Fail)
    }
}

/// The events a stream of `response` would have carried: each part whole,
/// in order.
pub(crate) fn events_of(
    response: &CompletionResponse,
) -> Result<Vec<Item<StreamEvent>>, ProviderError> {
    let shared = std::sync::Mutex::new(Shared::new(Turn::new(response.provider.clone())));
    {
        let mut out = Out::new(&shared);
        for content in &response.choice {
            out.content(content.clone())?;
        }
    }
    shared
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .items
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests;
