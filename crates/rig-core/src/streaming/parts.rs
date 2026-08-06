//! Accumulation of streamed events into the ordered parts of one assistant
//! choice.
//!
//! Reference design: pydantic-ai's `ModelResponsePartsManager`
//! (`pydantic_ai/_parts_manager.py`) — an ordered part list plus a
//! vendor-id-to-index map, with replacement centralized in one place
//! (`_replace_part`: "Fully replace a part and discard any buffered deltas").
//! This accumulator adds one refinement the OpenAI Responses wire requires:
//! reasoning parts key by `(item_id, ordinal)` rather than item id alone,
//! because one reasoning item may carry several sibling parts (summary parts,
//! visible text, encrypted content) that arrive as separate full blocks under
//! a single item id. A full block therefore supersedes only the delta
//! accumulation for its own part — never a completed sibling.

use std::collections::{HashMap, HashSet};

use crate::completion::CompletionError;
use crate::message::{AssistantContent, Reasoning, ReasoningContent, ToolCall, ToolFunction};
use crate::streaming::identity::{PartId, SyntheticIds, WireId};
use crate::streaming::{ToolInputEnd, UnparseableToolInput};

/// Identity of a reasoning part: the part identity plus an ordinal
/// distinguishing sibling parts of one multi-part item.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct PartKey {
    item_id: PartId,
    ordinal: u32,
}

/// A part of the choice under accumulation, tagged with how it was produced.
///
/// The tag decides replacement: a full block replaces only a `DeltaBuilt`
/// part with its exact key; a `Complete` part is never overwritten — a
/// same-key full block is a sibling and appends under the next ordinal.
#[derive(Debug)]
enum ManagedPart {
    /// Built from deltas; superseded wholesale when the full part arrives.
    DeltaBuilt(AssistantContent),
    /// Arrived whole, or already superseded its deltas.
    Complete(AssistantContent),
}

impl ManagedPart {
    fn into_content(self) -> AssistantContent {
        match self {
            Self::DeltaBuilt(content) | Self::Complete(content) => content,
        }
    }
}

/// Accumulates the streamed parts of one assistant choice, in arrival order.
///
/// Owns every aggregation decision `StreamingCompletionResponse` makes:
/// id-keyed text-block assembly, keyed reasoning delta/replace/sibling
/// semantics, and tool-call ordering. Consumers feed normalized events and
/// call [`PartsAccumulator::finish`] once the stream ends.
pub(crate) struct PartsAccumulator {
    parts: Vec<ManagedPart>,
    /// Reasoning part identity → index in `parts`. Invariant: every mapped
    /// index holds an `AssistantContent::Reasoning` part.
    reasoning_index: HashMap<PartKey, usize>,
    /// Item id → ordinal of that item's currently open (latest) part.
    open_ordinal: HashMap<PartId, u32>,
    /// Boundary-minted reasoning ids with a part at their currently open
    /// ordinal — exactly the ids [`PartsAccumulator::close_minted_reasoning`]
    /// would bump. Every text token and completed tool call is a boundary, so
    /// the common case (no open minted item) must not pay for a scan of every
    /// reasoning key (#2258 G5). Whether an id is minted is its [`PartId`]
    /// discriminant — lifecycle bookkeeping never parses an id string.
    open_minted_reasoning: Vec<PartId>,
    /// Active text block; text deltas and metadata merge here until the block
    /// is closed by a text start, a reasoning event, or a tool call.
    text_index: Option<usize>,
    /// Text-block identity → index in `parts` (the reasoning keying applied
    /// to text): a `TextStart` whose id was already seen reactivates that
    /// block instead of opening a duplicate, so a wire item's text keeps
    /// collapsing across interleaved output. Invariant: every mapped index
    /// holds an `AssistantContent::Text` part.
    text_ids: HashMap<PartId, usize>,
    /// Identity announced by the latest `TextStart` whose block has not yet
    /// opened; blocks open lazily on the first delta or metadata so a
    /// content-less `TextStart` never leaves an empty text part behind.
    pending_text_id: Option<PartId>,
    /// Mints identities for blocks a bare `Message` opens on wires that
    /// never announce text boundaries. Purely internal bookkeeping: text
    /// parts carry no id in the aggregated choice.
    minted_text_ids: SyntheticIds,
    /// Tool calls under delta assembly, keyed by the fragment id, in start
    /// order. A completed call leaves this list, so a reused id after
    /// completion opens a fresh call (the ordinal collapse the reasoning
    /// keying does explicitly).
    open_tool_inputs: Vec<OpenToolInput>,
    /// Assembly ids closed by a full [`PartsAccumulator::tool_call`] rather
    /// than by their own end event: a wire that restates a fragmented call as
    /// one complete block already finalized it, so a trailing
    /// [`PartsAccumulator::tool_input_end`] for that id must not finalize a
    /// duplicate part (#2258 F1). Cleared for an id when fragments reopen it,
    /// so a reused id still assembles a fresh call.
    closed_by_full_call: HashSet<PartId>,
    /// Whether any completed tool call was recorded; the streaming
    /// counterpart of the unary path's finish-reason reconciliation input.
    saw_tool_call: bool,
}

/// A tool call whose input is still being assembled from streamed fragments.
///
/// Reference designs: pydantic-ai `handle_tool_call_delta`
/// (`_parts_manager.py:358`) and vercel's shared
/// `streaming-tool-call-tracker` — fragment buffering, keying, and the
/// delta-to-part transition live in the one shared component, never in a
/// provider.
struct OpenToolInput {
    /// Assembly identity: every fragment of one call carries this id.
    id: PartId,
    /// Rig-minted correlation id, created when the call opens.
    internal_call_id: String,
    /// Tool name; a later non-empty name fragment replaces it.
    name: String,
    /// Concatenated raw argument fragments. `None` until a fragment arrives —
    /// a call that streamed no arguments is a parameterless invocation.
    buffer: Option<String>,
    /// Whether the buffer hit [`MAX_TOOL_INPUT_BYTES`] and stopped
    /// accumulating. The truncated JSON then fails to parse at finalization
    /// and flows through the wire's `UnparseableToolInput` policy — no
    /// runaway wire can grow a fragment buffer without bound.
    overflowed: bool,
}

/// Upper bound on one streamed tool call's accumulated argument bytes.
///
/// Far beyond any real tool input; a wire that exceeds it is defective, and
/// its call finalizes through the unparseable-input policy instead of
/// growing memory without bound.
const MAX_TOOL_INPUT_BYTES: usize = 32 * 1024 * 1024;

impl Default for PartsAccumulator {
    fn default() -> Self {
        Self {
            parts: Vec::new(),
            reasoning_index: HashMap::new(),
            open_ordinal: HashMap::new(),
            open_minted_reasoning: Vec::new(),
            text_index: None,
            text_ids: HashMap::new(),
            pending_text_id: None,
            minted_text_ids: SyntheticIds::text(),
            open_tool_inputs: Vec::new(),
            closed_by_full_call: HashSet::new(),
            saw_tool_call: false,
        }
    }
}

impl PartsAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append streamed text to the active text block, opening one if none is
    /// active.
    pub(crate) fn text_delta(&mut self, text: &str) {
        self.close_minted_reasoning();
        let index = self.ensure_text_block();
        if let Some(ManagedPart::DeltaBuilt(AssistantContent::Text(existing))) =
            self.parts.get_mut(index)
        {
            existing.text.push_str(text);
        }
    }

    /// Open (or reactivate) the text block identified by `id`.
    ///
    /// `id` must be non-empty ([`crate::streaming::RawStreamingChoice::TextStart`]'s
    /// mandatory-identity contract): a previously seen id reactivates its
    /// block — a wire item's text keeps collapsing across interleaved output,
    /// the reasoning keying applied to text. An unseen id closes the active
    /// block and opens the new one lazily, on its first delta or metadata.
    pub(crate) fn text_start(&mut self, id: &PartId, additional_params: Option<serde_json::Value>) {
        self.close_minted_reasoning();
        match self.text_ids.get(id) {
            Some(&index) => {
                self.text_index = Some(index);
                self.pending_text_id = None;
            }
            None => {
                self.text_index = None;
                self.pending_text_id = Some(id.clone());
            }
        }
        if let Some(additional_params) = additional_params {
            self.text_additional_params(additional_params);
        }
    }

    /// Merge provider metadata into the active text block, opening an empty
    /// block if none is active.
    pub(crate) fn text_additional_params(&mut self, additional_params: serde_json::Value) {
        if additional_params.is_null() {
            return;
        }

        // Metadata targets a text block, so it is a text entry point like
        // `text_delta`/`text_start`: any minted (id-less) reasoning run ends
        // here, or interleaved reasoning after this metadata would merge into
        // the pre-metadata part across the text boundary.
        self.close_minted_reasoning();
        let index = self.ensure_text_block();
        let Some(ManagedPart::DeltaBuilt(AssistantContent::Text(text))) = self.parts.get_mut(index)
        else {
            return;
        };

        match text.additional_params.as_mut() {
            Some(existing) => merge_text_additional_params(existing, additional_params),
            None => text.additional_params = Some(additional_params),
        }
    }

    /// Append reasoning delta text to the item's open part.
    ///
    /// Deltas key strictly by item id (ids are mandatory on the raw grammar,
    /// so identity is exact, not heuristic); the delta extends the item's
    /// currently open part. A delta arriving after that part was completed by
    /// a full block opens a fresh part under the next ordinal — it never
    /// resurrects a superseded buffer.
    pub(crate) fn reasoning_delta(&mut self, id: &PartId, text: &str) {
        self.text_index = None;

        let ordinal = self.open_ordinal.get(id).copied().unwrap_or(0);
        let key = PartKey {
            item_id: id.clone(),
            ordinal,
        };

        if let Some(&index) = self.reasoning_index.get(&key) {
            match self.parts.get_mut(index) {
                Some(ManagedPart::DeltaBuilt(AssistantContent::Reasoning(existing))) => {
                    if let Some(ReasoningContent::Text {
                        text: existing_text,
                        ..
                    }) = existing.content.last_mut()
                    {
                        existing_text.push_str(text);
                    } else {
                        existing.content.push(ReasoningContent::Text {
                            text: text.to_owned(),
                            signature: None,
                        });
                    }
                    return;
                }
                // The open part is already Complete: this delta belongs to
                // the item's next part.
                _ => {
                    let next = ordinal + 1;
                    self.open_ordinal.insert(id.clone(), next);
                    self.push_reasoning(
                        PartKey {
                            item_id: id.clone(),
                            ordinal: next,
                        },
                        ManagedPart::DeltaBuilt(delta_reasoning(id, text)),
                    );
                    return;
                }
            }
        }

        self.push_reasoning(key, ManagedPart::DeltaBuilt(delta_reasoning(id, text)));
    }

    /// Record a full reasoning block.
    ///
    /// Replacement is keyed, never positional: the block supersedes only a
    /// delta-built part under its own `(item_id, ordinal)` key — the deltas
    /// are a fallback for providers that never send the completed block, so
    /// the delta accumulation is *replaced* and its buffers discarded
    /// (pydantic-ai `_replace_part` semantics). A same-key part that is
    /// already complete is a sibling part of a multi-part item (OpenAI
    /// Responses: summary parts, text, encrypted content under one item id)
    /// and the block appends under the item's next ordinal.
    ///
    /// Known limitation (#2258 F5, accepted): siblings append at the END of
    /// the part list, not adjacent to the part they follow. If a multi-part
    /// done item arrives after interleaved output (part 1 replaces its
    /// delta-built slot in place, parts 2..n append after the interleaved
    /// content), the item's parts fragment around that content. The real
    /// Responses wire never produces this — a done item's sibling blocks
    /// arrive consecutively, before any later output — so adjacency insertion
    /// (which would shift every stored part index) is not worth its
    /// complexity.
    pub(crate) fn reasoning_full(&mut self, id: &PartId, reasoning: Reasoning) {
        self.text_index = None;

        let ordinal = self.open_ordinal.get(id).copied().unwrap_or(0);
        let key = PartKey {
            item_id: id.clone(),
            ordinal,
        };

        if let Some(&index) = self.reasoning_index.get(&key) {
            match self.parts.get_mut(index) {
                Some(part @ ManagedPart::DeltaBuilt(_)) => {
                    *part = ManagedPart::Complete(AssistantContent::Reasoning(reasoning));
                    return;
                }
                // Sibling: same item id, next part.
                _ => {
                    let next = ordinal + 1;
                    self.open_ordinal.insert(id.clone(), next);
                    self.push_reasoning(
                        PartKey {
                            item_id: id.clone(),
                            ordinal: next,
                        },
                        ManagedPart::Complete(AssistantContent::Reasoning(reasoning)),
                    );
                    return;
                }
            }
        }

        self.push_reasoning(
            key,
            ManagedPart::Complete(AssistantContent::Reasoning(reasoning)),
        );
    }

    /// Append a tool call the wire delivered whole, reconciling it with an
    /// open delta assembly of the same id. Returns the internal correlation id
    /// the completed call must be published under.
    ///
    /// A wire may fragment a call's input *and* restate it as one complete
    /// block (an out-of-tree adapter emitting `ToolCallDelta`s followed by a
    /// full `ToolCall`). The fragments already published a minted
    /// `internal_call_id` to the consumer, and
    /// [`StreamedAssistantContent::ToolCall`](crate::streaming::StreamedAssistantContent::ToolCall)
    /// promises that id correlates the completed call with its deltas — so the
    /// assembly's id is **adopted**, never replaced, and `minted_internal_call_id`
    /// (freshly generated for a call that arrived with no assembly) is returned
    /// only when there was nothing to adopt.
    ///
    /// Adoption also closes the assembly: its slot is removed and its id is
    /// marked closed, so a trailing [`PartsAccumulator::tool_input_end`] for
    /// that id finalizes nothing instead of appending a duplicate part
    /// (#2258 F1).
    pub(crate) fn tool_call(
        &mut self,
        id: &PartId,
        tool_call: ToolCall,
        minted_internal_call_id: String,
    ) -> String {
        let position = self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == *id)
            .or_else(|| {
                // A wire that fragmented under a minted identity and then
                // restates the call under its late-arriving wire id: two
                // wire ids disagreeing is a veto, but minted-vs-wire is not
                // — when exactly one minted assembly is open, the
                // restatement is that assembly completed. More than one
                // open minted assembly is ambiguous; don't guess.
                if id.is_minted() {
                    return None;
                }
                let mut minted = self
                    .open_tool_inputs
                    .iter()
                    .enumerate()
                    .filter(|(_, input)| input.id.is_minted());
                match (minted.next(), minted.next()) {
                    (Some((index, _)), None) => Some(index),
                    _ => None,
                }
            });
        let adopted = position.map(|index| {
            let input = self.open_tool_inputs.remove(index);
            self.closed_by_full_call.insert(input.id);
            input.internal_call_id
        });
        self.push_tool_call(tool_call);
        adopted.unwrap_or(minted_internal_call_id)
    }

    /// Append a completed tool call as the next part, closing the active text
    /// and minted-reasoning blocks (a completed call is a part boundary).
    fn push_tool_call(&mut self, tool_call: ToolCall) {
        self.close_minted_reasoning();
        self.text_index = None;
        self.saw_tool_call = true;
        self.parts
            .push(ManagedPart::Complete(AssistantContent::ToolCall(tool_call)));
    }

    /// Whether any completed tool call was recorded on this stream.
    pub(crate) fn saw_tool_call(&self) -> bool {
        self.saw_tool_call
    }

    /// Record a streamed tool name fragment, opening the call if `id` has no
    /// open call. Returns the call's minted internal id.
    ///
    /// `id` must be non-empty ([`RawStreamingChoice::ToolCallDelta`]'s
    /// mandatory-identity contract, `crate::streaming`): emitters mint
    /// `tool-{index}` when the wire omits an id, because a shared empty id
    /// would interleave parallel calls' fragments into one corrupted
    /// assembly here.
    ///
    /// A later non-empty name replaces the recorded one (OpenAI-compatible
    /// wire semantics: the established name is the last non-empty value).
    /// Open-tool state deliberately does not close the active text or
    /// reasoning block — only the *completed* call is a part boundary,
    /// matching the pre-assembly behavior where deltas bypassed accumulation.
    pub(crate) fn tool_name_delta(&mut self, id: &PartId, name: &str) -> String {
        let index = self.ensure_open_tool_input(id);
        match self.open_tool_inputs.get_mut(index) {
            Some(input) => {
                // Last-*non-empty* semantics (doc above, and `ToolCallBridge`'s
                // matching filter): an empty fragment must not erase an
                // established name, or finalization would drop the call as
                // nameless.
                if !name.is_empty() {
                    name.clone_into(&mut input.name);
                }
                input.internal_call_id.clone()
            }
            // Unreachable (`ensure` returns a live index); degrade to a fresh
            // id rather than panic.
            None => crate::id::generate(),
        }
    }

    /// Append a streamed argument fragment to the call's buffer, opening the
    /// call if `id` has no open call. Returns the call's minted internal id.
    ///
    /// `id` must be non-empty; see [`PartsAccumulator::tool_name_delta`].
    pub(crate) fn tool_args_delta(&mut self, id: &PartId, fragment: &str) -> String {
        let index = self.ensure_open_tool_input(id);
        match self.open_tool_inputs.get_mut(index) {
            Some(input) => {
                match input.buffer.as_mut() {
                    Some(buffer) => {
                        // Some OpenAI-compatible gateways emit a literal
                        // `null` placeholder before streaming the real JSON
                        // argument fragments; a later non-empty fragment
                        // supersedes it.
                        if buffer.trim() == "null" && !fragment.trim().is_empty() {
                            buffer.clear();
                        }
                        if buffer.len().saturating_add(fragment.len()) > MAX_TOOL_INPUT_BYTES {
                            if !input.overflowed {
                                input.overflowed = true;
                                tracing::warn!(
                                    tool = %input.name,
                                    "streamed tool-call input exceeded the accumulation bound; \
                                     truncating — the call will finalize through the wire's \
                                     unparseable-input policy"
                                );
                            }
                        } else {
                            buffer.push_str(fragment);
                        }
                    }
                    None => input.buffer = Some(fragment.to_owned()),
                }
                input.internal_call_id.clone()
            }
            // Unreachable (`ensure` returns a live index); degrade to a fresh
            // id rather than panic.
            None => crate::id::generate(),
        }
    }

    /// Close a streamed tool call's input and finalize it into a completed
    /// call.
    ///
    /// Returns the completed call and its internal id, `None` when the call
    /// is dropped (nameless, or unparseable under
    /// [`UnparseableToolInput::Drop`]), or an error item under
    /// [`UnparseableToolInput::Error`]. Authoritative fields on the end event
    /// (a wire's completed item) supersede the assembled state. An end with
    /// no open call opens and completes one from the event alone. Out-of-tree
    /// adapters beware: this means a `Keep`-mode end carrying an authoritative
    /// name for an id that never opened still finalizes a call with `{}`
    /// arguments from the end event alone — *unless* a full
    /// [`PartsAccumulator::tool_call`] already closed that id, in which case
    /// the end is the wire restating a call that is already a part and
    /// finalizes nothing.
    pub(crate) fn tool_input_end(
        &mut self,
        end: ToolInputEnd,
    ) -> Result<Option<(ToolCall, String)>, CompletionError> {
        let position = self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == end.id);
        // A full block already finalized this id and consumed its assembly
        // (#2258 F1); ending it again would append a duplicate part from the
        // event's authoritative payload.
        if position.is_none() && self.closed_by_full_call.contains(&end.id) {
            return Ok(None);
        }
        let open = position.map(|index| self.open_tool_inputs.remove(index));
        // Restores an open call that a `Keep`-mode probe could not finalize,
        // preserving its start-order slot.
        let keep_open = |accumulator: &mut Self, input: Option<OpenToolInput>| {
            if let (Some(index), Some(input)) = (position, input) {
                accumulator.open_tool_inputs.insert(index, input);
            }
        };

        let (internal_call_id, opened_id, mut name, buffer) = match open.as_ref() {
            Some(input) => (
                input.internal_call_id.clone(),
                input.id.clone(),
                input.name.clone(),
                input.buffer.clone(),
            ),
            None => (crate::id::generate(), end.id.clone(), String::new(), None),
        };
        let overflowed = open.as_ref().is_some_and(|input| input.overflowed);
        // Only a wire identity becomes the durable tool-call id; an assembly
        // opened under a minted key yields the empty string, which every
        // serializer treats as absent.
        let opened_id = opened_id
            .into_wire_id()
            .map(WireId::into_string)
            .unwrap_or_default();
        // An authoritative end-event name supersedes assembly, but an *empty*
        // one is filtered like the fragment path (`tool_name_delta`'s
        // last-non-empty semantics): it must not erase an established name and
        // turn a real call into a nameless drop.
        if let Some(final_name) = end.name.clone().filter(|final_name| !final_name.is_empty()) {
            name = final_name;
        }
        // A call whose name never arrived is not a call the model made
        // (OpenAI-compatible flush semantics: nameless entries drop).
        if name.is_empty() {
            if matches!(end.on_unparseable, UnparseableToolInput::Keep) {
                keep_open(self, open);
            }
            return Ok(None);
        }

        let arguments = match end.arguments {
            // The wire's completed item is authoritative over assembly.
            Some(arguments) => arguments,
            None => match buffer {
                // No streamed arguments: a parameterless invocation.
                None => serde_json::Value::Object(serde_json::Map::new()),
                // A capped (overflowed) buffer is truncated by definition —
                // the lenient partial-JSON parse could still "succeed" on it
                // and fabricate a silently corrupted call, so overflow forces
                // the unparseable path.
                Some(buffer) => {
                    match crate::json_utils::parse_tool_arguments(&buffer).and_then(|arguments| {
                        if overflowed {
                            Err(serde::de::Error::custom(
                                "tool-call input exceeded the accumulation bound",
                            ))
                        } else {
                            Ok(arguments)
                        }
                    }) {
                        Ok(arguments) => arguments,
                        Err(err) => match end.on_unparseable {
                            // Partial input (truncation): the call never fully
                            // arrived, so it must not reach the consumer.
                            UnparseableToolInput::Drop => {
                                tracing::debug!(
                                    tool = %name,
                                    "dropping streamed tool call whose arguments never fully arrived"
                                );
                                return Ok(None);
                            }
                            // The wire superseded this call mid-assembly; deliver
                            // it with empty arguments rather than losing it.
                            UnparseableToolInput::EmptyObject => {
                                serde_json::Value::Object(serde_json::Map::new())
                            }
                            // The wire promised a complete block; malformed input
                            // is a response defect, never a silent drop.
                            UnparseableToolInput::Error => {
                                return Err(CompletionError::ResponseError(format!(
                                    "tool call `{name}` arrived with malformed JSON input: {err}"
                                )));
                            }
                            // A completion probe: the input may still be extended.
                            UnparseableToolInput::Keep => {
                                keep_open(self, open);
                                return Ok(None);
                            }
                        },
                    }
                }
            },
        };

        let tool_call = ToolCall {
            id: end.tool_id.unwrap_or(opened_id),
            call_id: end.call_id,
            function: ToolFunction { name, arguments },
            signature: end.signature,
            additional_params: end.additional_params,
        };
        self.push_tool_call(tool_call.clone());
        Ok(Some((tool_call, internal_call_id)))
    }

    /// Index of the active text block, opening one if none is active.
    ///
    /// A newly opened block takes the identity the latest `TextStart`
    /// announced, or a boundary-minted `text-{n}` id when a bare `Message`
    /// opens it — every text block is keyed, however it opened.
    fn ensure_text_block(&mut self) -> usize {
        if let Some(index) = self.text_index
            && matches!(
                self.parts.get(index),
                Some(ManagedPart::DeltaBuilt(AssistantContent::Text(_)))
            )
        {
            return index;
        }

        self.parts
            .push(ManagedPart::DeltaBuilt(AssistantContent::text("")));
        let index = self.parts.len() - 1;
        self.text_index = Some(index);
        let id = self
            .pending_text_id
            .take()
            .unwrap_or_else(|| self.minted_text_ids.mint());
        self.text_ids.insert(id, index);
        index
    }

    /// Index of the open call for `id`, opening one if none exists.
    fn ensure_open_tool_input(&mut self, id: &PartId) -> usize {
        match self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == *id)
        {
            Some(index) => index,
            None => {
                // Fragments for an id a full block closed are a *new* call
                // reusing the id, not a continuation of the finalized one:
                // drop the mark so its end event finalizes normally.
                self.closed_by_full_call.remove(id);
                self.open_tool_inputs.push(OpenToolInput {
                    id: id.clone(),
                    internal_call_id: crate::id::generate(),
                    name: String::new(),
                    buffer: None,
                    overflowed: false,
                });
                self.open_tool_inputs.len() - 1
            }
        }
    }

    /// Attach a provider signature to the item's latest reasoning part.
    ///
    /// The wire shape this serves: Gemini delivers `thoughtSignature` on a
    /// trailing part *after* other output already closed the thought block
    /// (and, under the minted-id boundary, bumped its ordinal). The
    /// signature is lifecycle metadata for the block that was streamed, so
    /// it lands on the item's most recent part — wherever its ordinal ended
    /// up — never on a fresh empty sibling. With no part to sign (a
    /// signature-only stream), a signature-only part is recorded so the
    /// replay-required provider state still reaches history.
    pub(crate) fn reasoning_signature(&mut self, id: &PartId, signature: String) {
        let latest = self
            .reasoning_index
            .iter()
            .filter(|(key, _)| key.item_id == *id)
            .max_by_key(|(key, _)| key.ordinal)
            .map(|(_, &index)| index);
        if let Some(index) = latest
            && let Some(
                ManagedPart::DeltaBuilt(AssistantContent::Reasoning(reasoning))
                | ManagedPart::Complete(AssistantContent::Reasoning(reasoning)),
            ) = self.parts.get_mut(index)
        {
            match reasoning
                .content
                .iter_mut()
                .rev()
                .find_map(|content| match content {
                    ReasoningContent::Text { signature, .. } => Some(signature),
                    _ => None,
                }) {
                Some(slot) => *slot = Some(signature),
                None => reasoning.content.push(ReasoningContent::Text {
                    text: String::new(),
                    signature: Some(signature),
                }),
            }
            return;
        }
        // Nothing streamed for this item: record the signature alone.
        self.reasoning_full(
            id,
            Reasoning {
                id: id.as_wire().map(str::to_owned),
                content: vec![ReasoningContent::Text {
                    text: String::new(),
                    signature: Some(signature),
                }],
            },
        );
    }

    /// Consume the accumulated state into the ordered choice parts.
    ///
    /// Never empty: a stream that produced no content yields the single empty
    /// text part the aggregated choice has always defaulted to.
    pub(crate) fn finish(&mut self) -> Vec<AssistantContent> {
        let mut parts: Vec<AssistantContent> = std::mem::take(&mut self.parts)
            .into_iter()
            .map(ManagedPart::into_content)
            .collect();
        self.reasoning_index.clear();
        self.open_ordinal.clear();
        self.open_minted_reasoning.clear();
        self.text_index = None;
        self.text_ids.clear();
        self.pending_text_id = None;
        self.minted_text_ids = SyntheticIds::text();
        // Calls still open at stream end never fully arrived (no end event,
        // no adapter flush): truncated input drops, per the settled contract.
        self.open_tool_inputs.clear();
        self.closed_by_full_call.clear();
        self.saw_tool_call = false;
        if parts.is_empty() {
            parts.push(AssistantContent::text(""));
        }
        parts
    }

    /// Close every open reasoning item whose id is boundary-minted.
    ///
    /// A minted id (`reasoning-0`, `block-{n}`, `output-{n}`) is a per-stream
    /// constant, not a wire item boundary, so the only interleaving signal
    /// those wires have is other output arriving: bumping the ordinal makes a
    /// later delta or full block open (or complete) a *new* part instead of
    /// extending — or replacing — the one from before the boundary. Real wire
    /// ids (OpenAI Responses `rs_*`) are exact item identities and must keep
    /// collapsing across interleaved output, so they are never bumped.
    ///
    /// Every boundary event calls this, so the no-open-minted-item case — the
    /// overwhelmingly common one, including *every* text token after the first
    /// boundary — must not scan the reasoning keys (#2258 G5). The
    /// `open_minted_reasoning` list holds exactly the ids the scan would find,
    /// so an empty list is a proof that the scan is vacuous.
    fn close_minted_reasoning(&mut self) {
        // The list *is* the lifecycle state: it holds exactly the minted ids
        // with a part at their open ordinal (`push_reasoning` maintains it),
        // so closing consumes the list directly — no scan over the reasoning
        // keys, and no property recovered from an id's shape. O(open minted
        // items), which is O(0) for every boundary after the first.
        for item_id in std::mem::take(&mut self.open_minted_reasoning) {
            let ordinal = self.open_ordinal.get(&item_id).copied().unwrap_or(0);
            self.open_ordinal.insert(item_id, ordinal + 1);
        }
    }

    fn push_reasoning(&mut self, key: PartKey, part: ManagedPart) {
        // The pushed part is always the item's open ordinal, so a minted id
        // becomes closeable here and nowhere else.
        if key.item_id.is_minted() && !self.open_minted_reasoning.contains(&key.item_id) {
            self.open_minted_reasoning.push(key.item_id.clone());
        }
        self.parts.push(part);
        self.reasoning_index.insert(key, self.parts.len() - 1);
    }
}

fn delta_reasoning(id: &PartId, text: &str) -> AssistantContent {
    AssistantContent::Reasoning(Reasoning {
        // The provenance funnel: only a wire identity becomes the durable id.
        id: id.as_wire().map(str::to_owned),
        content: vec![ReasoningContent::Text {
            text: text.to_owned(),
            signature: None,
        }],
    })
}

/// Deep-merge streamed text metadata: arrays concatenate (citation deltas),
/// objects merge recursively, scalars take the incoming value.
fn merge_text_additional_params(existing: &mut serde_json::Value, incoming: serde_json::Value) {
    match (existing, incoming) {
        (serde_json::Value::Object(existing_map), serde_json::Value::Object(incoming_map)) => {
            for (key, incoming_value) in incoming_map {
                match existing_map.get_mut(&key) {
                    Some(existing_value) => match (existing_value, incoming_value) {
                        (
                            serde_json::Value::Array(existing_array),
                            serde_json::Value::Array(mut incoming_array),
                        ) => existing_array.append(&mut incoming_array),
                        (existing_value, incoming_value) => {
                            merge_text_additional_params(existing_value, incoming_value);
                        }
                    },
                    None => {
                        existing_map.insert(key, incoming_value);
                    }
                }
            }
        }
        (existing, incoming) => {
            *existing = incoming;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture-syntax part id: the legacy minted renderings (`reasoning-0`,
    /// `block-3`, `tool-1`, ...) decode to a [`PartId::Minted`] of that kind
    /// and index; anything else is a wire id.
    fn pid(id: &str) -> PartId {
        use crate::streaming::MintKind;
        for (namespace, kind) in [
            ("reasoning-", MintKind::Reasoning),
            ("block-", MintKind::Block),
            ("output-", MintKind::Output),
            ("tool-", MintKind::Tool),
            ("text-", MintKind::Text),
        ] {
            if let Some(rest) = id.strip_prefix(namespace)
                && let Ok(index) = rest.parse::<u64>()
            {
                return kind.for_wire_index(index);
            }
        }
        PartId::wire(id)
    }

    fn full(id: &str, content: ReasoningContent) -> Reasoning {
        Reasoning {
            id: Some(id.to_owned()),
            content: vec![content],
        }
    }

    fn summary(text: &str) -> ReasoningContent {
        ReasoningContent::Summary(text.to_owned())
    }

    fn reasoning_text(text: &str) -> ReasoningContent {
        ReasoningContent::Text {
            text: text.to_owned(),
            signature: None,
        }
    }

    /// Text of every reasoning part, flattened in part order.
    fn reasoning_texts(parts: &[AssistantContent]) -> Vec<String> {
        parts
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Reasoning(reasoning) => Some(reasoning.content.iter()),
                _ => None,
            })
            .flatten()
            .map(|content| match content {
                ReasoningContent::Summary(text) => text.clone(),
                ReasoningContent::Text { text, .. } => text.clone(),
                ReasoningContent::Encrypted(data) => data.clone(),
                ReasoningContent::Redacted { data } => data.clone(),
            })
            .collect()
    }

    #[test]
    fn a_full_block_replaces_its_delta_accumulation() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), "partial ");
        accumulator.reasoning_delta(&pid("rs_1"), "thought");
        accumulator.reasoning_full(
            &pid("rs_1"),
            full("rs_1", reasoning_text("the complete chain")),
        );

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["the complete chain"]);
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn replacement_is_keyed_across_interleaved_output() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), "partial");
        accumulator.tool_call(
            &pid("call_1"),
            ToolCall {
                id: "call_1".to_owned(),
                call_id: None,
                function: crate::message::ToolFunction {
                    name: "probe".to_owned(),
                    arguments: serde_json::json!({}),
                },
                signature: None,
                additional_params: None,
            },
            "internal-probe".to_owned(),
        );
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", reasoning_text("full block")));

        let parts = accumulator.finish();
        assert_eq!(
            parts.len(),
            2,
            "reasoning replaced in place, tool call kept"
        );
        // Replacement is in place: the reasoning part keeps its arrival slot
        // ahead of the tool call.
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(reasoning))
                if matches!(reasoning.content.first(), Some(ReasoningContent::Text { text, .. }) if text == "full block")
        ));
        assert!(matches!(parts.get(1), Some(AssistantContent::ToolCall(_))));
    }

    /// The P1-2 shape: one item id carrying two summary parts, visible text,
    /// and encrypted content as consecutive full blocks. Siblings append —
    /// every part survives, in arrival order.
    #[test]
    fn same_key_full_blocks_are_siblings_and_all_survive() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", summary("s1")));
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", summary("s2")));
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", reasoning_text("visible")));
        accumulator.reasoning_full(
            &pid("rs_1"),
            full("rs_1", ReasoningContent::Encrypted("enc".to_owned())),
        );

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "visible", "enc"]);
    }

    /// Deltas then the item's multi-part done block: the first full block
    /// supersedes the delta accumulation, the rest append as siblings.
    #[test]
    fn deltas_then_sibling_full_blocks_keep_each_part_once() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), "s1");
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", summary("s1")));
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", summary("s2")));
        accumulator.reasoning_full(
            &pid("rs_1"),
            full("rs_1", ReasoningContent::Encrypted("enc".to_owned())),
        );

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "enc"]);
    }

    #[test]
    fn a_delta_after_replacement_opens_a_fresh_part() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), "first");
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", reasoning_text("first complete")));
        accumulator.reasoning_delta(&pid("rs_1"), "second");

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["first complete", "second"]);
    }

    #[test]
    fn distinct_item_ids_never_interact() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), "first item deltas");
        accumulator.reasoning_full(
            &pid("rs_2"),
            full("rs_2", reasoning_text("a different item")),
        );

        let parts = accumulator.finish();
        assert_eq!(
            reasoning_texts(&parts),
            vec!["first item deltas", "a different item"]
        );
    }

    #[test]
    fn text_and_reasoning_interleave_in_arrival_order() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_delta("intro");
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", summary("thinking")));
        accumulator.text_delta("out");
        accumulator.text_delta("ro");

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 3);
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Text(text)) if text.text == "intro"
        ));
        assert!(matches!(parts.get(1), Some(AssistantContent::Reasoning(_))));
        // The reasoning event closed the first block; later deltas open a new
        // one rather than merging backwards.
        assert!(matches!(
            parts.get(2),
            Some(AssistantContent::Text(text)) if text.text == "outro"
        ));
    }

    /// The item-4 (#2258) shape: two distinct text identities (two OpenAI
    /// Responses `message` items, two Anthropic text blocks) must aggregate
    /// as two distinct text parts, in order — never concatenate.
    #[test]
    fn distinct_text_start_ids_open_distinct_parts() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.text_delta("first");
        accumulator.text_start(&pid("msg_2"), None);
        accumulator.text_delta("second");

        let parts = accumulator.finish();
        let texts: Vec<&str> = parts
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "second"]);
    }

    /// The reasoning keying applied to text: a `TextStart` whose id was
    /// already seen reactivates that block across interleaved output instead
    /// of opening a duplicate part.
    #[test]
    fn a_seen_text_start_id_reactivates_its_block() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.text_delta("collapsing ");
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", summary("thinking")));
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.text_delta("text");

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 2, "one text part, one reasoning part");
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Text(text)) if text.text == "collapsing text"
        ));
    }

    /// A `TextStart` that never receives content leaves no empty text part
    /// behind (blocks open lazily on the first delta or metadata).
    #[test]
    fn a_content_less_text_start_leaves_no_empty_part() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", summary("thinking")));

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 1);
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(_))
        ));
    }

    fn probe_tool_call() -> ToolCall {
        ToolCall {
            id: "call_1".to_owned(),
            call_id: None,
            function: crate::message::ToolFunction {
                name: "probe".to_owned(),
                arguments: serde_json::json!({}),
            },
            signature: None,
            additional_params: None,
        }
    }

    /// The F1b boundary contract for constant-id wires: a tool call closes
    /// the open minted-id reasoning item, so a later delta opens a new part in
    /// arrival order — never merging backwards across the boundary.
    #[test]
    fn other_output_closes_a_minted_id_reasoning_item() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("reasoning-0"), "A");
        accumulator.tool_call(
            &pid("call_1"),
            probe_tool_call(),
            "internal-probe".to_owned(),
        );
        accumulator.reasoning_delta(&pid("reasoning-0"), "B");

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 3);
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(reasoning))
                if matches!(reasoning.content.first(), Some(ReasoningContent::Text { text, .. }) if text == "A")
        ));
        assert!(matches!(parts.get(1), Some(AssistantContent::ToolCall(_))));
        assert!(matches!(
            parts.get(2),
            Some(AssistantContent::Reasoning(reasoning))
                if matches!(reasoning.content.first(), Some(ReasoningContent::Text { text, .. }) if text == "B")
        ));
    }

    /// The F1b erasure route: a full block after the boundary must not
    /// replace-and-discard the pre-boundary delta buffer of a minted id.
    #[test]
    fn a_full_block_after_the_boundary_does_not_erase_prior_minted_id_deltas() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("reasoning-0"), "A");
        accumulator.tool_call(
            &pid("call_1"),
            probe_tool_call(),
            "internal-probe".to_owned(),
        );
        accumulator.reasoning_full(
            &pid("reasoning-0"),
            full("reasoning-0", reasoning_text("B")),
        );

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
        assert!(matches!(parts.get(1), Some(AssistantContent::ToolCall(_))));
    }

    /// Text output is a boundary too: minted-id reasoning around streamed text
    /// stays two parts in arrival order.
    #[test]
    fn text_output_closes_a_minted_id_reasoning_item() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("reasoning-0"), "A");
        accumulator.text_delta("visible");
        accumulator.reasoning_delta(&pid("reasoning-0"), "B");

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
        assert!(matches!(
            parts.get(1),
            Some(AssistantContent::Text(text)) if text.text == "visible"
        ));
    }

    /// Standalone text metadata is a boundary too: a `TextAdditionalParams`
    /// with no surrounding text deltas still targets a text block, so
    /// minted-id reasoning around it stays two parts in arrival order.
    #[test]
    fn text_metadata_closes_a_minted_id_reasoning_item() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("reasoning-0"), "A");
        accumulator.text_additional_params(serde_json::json!({"citations": ["c1"]}));
        accumulator.reasoning_delta(&pid("reasoning-0"), "B");

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
        assert!(matches!(
            parts.get(1),
            Some(AssistantContent::Text(text))
                if text.additional_params == Some(serde_json::json!({"citations": ["c1"]}))
        ));
    }

    /// #2258 G5: once a minted reasoning item is closed, every following text
    /// token must be a no-op for the boundary machinery — the pre-fix code
    /// rescanned every reasoning key per token and found nothing, and the
    /// guard must not change what the tokens produce.
    ///
    /// Not inducible from a recorded provider turn: the defect is the *cost*
    /// of a scan whose result is already fixed, so no wire shape distinguishes
    /// the two implementations. Pinned here on the accumulator state instead.
    #[test]
    fn repeated_text_deltas_after_a_closed_minted_block_are_a_no_op() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("reasoning-0"), "A");
        assert_eq!(accumulator.open_minted_reasoning, vec![pid("reasoning-0")]);

        accumulator.text_delta("one");
        assert!(
            accumulator.open_minted_reasoning.is_empty(),
            "the first text token closes the minted item, emptying the guard"
        );
        accumulator.text_delta(" two");
        accumulator.text_delta(" three");
        assert!(
            accumulator.open_minted_reasoning.is_empty(),
            "later tokens have nothing left to close"
        );

        // Behavior is unchanged by the guard: one boundary, one text block,
        // and a later delta still opens a fresh reasoning part.
        accumulator.reasoning_delta(&pid("reasoning-0"), "B");
        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
        assert_eq!(parts.len(), 3);
        assert!(matches!(
            parts.get(1),
            Some(AssistantContent::Text(text)) if text.text == "one two three"
        ));
    }

    /// A wire-supplied reasoning id never enters the boundary guard: it is an
    /// exact item identity that must keep collapsing across interleaved
    /// output, so the guard stays empty and the scan stays skipped.
    #[test]
    fn wire_reasoning_ids_never_enter_the_minted_boundary_guard() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), "thinking");
        assert!(accumulator.open_minted_reasoning.is_empty());
    }

    /// Wire-supplied ids are exact item identities: the Responses replay case
    /// must keep collapsing a done block onto its deltas across interleaved
    /// output (the boundary bump applies to minted namespaces only).
    #[test]
    fn wire_id_full_blocks_still_collapse_across_interleaved_output() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), "thinking");
        accumulator.tool_call(
            &pid("call_1"),
            probe_tool_call(),
            "internal-probe".to_owned(),
        );
        accumulator.reasoning_full(&pid("rs_1"), full("rs_1", reasoning_text("full reasoning")));

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["full reasoning"]);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn finish_on_an_empty_stream_yields_one_empty_text_part() {
        let mut accumulator = PartsAccumulator::new();
        let parts = accumulator.finish();
        assert_eq!(parts, vec![AssistantContent::text("")]);
    }

    // --- Tool-input assembly: the settled semantics the four provider
    // trackers (compat, Responses, Anthropic, Bedrock) used to hand-roll. ---

    fn end(id: &str, mode: UnparseableToolInput) -> ToolInputEnd {
        ToolInputEnd::new(pid(id), mode)
    }

    #[test]
    fn fragments_assemble_into_a_completed_tool_call_with_a_stable_internal_id() {
        let mut accumulator = PartsAccumulator::new();
        let first = accumulator.tool_name_delta(&pid("call_1"), "get_weather");
        let second = accumulator.tool_args_delta(&pid("call_1"), "{\"location\":");
        let third = accumulator.tool_args_delta(&pid("call_1"), "\"Paris\",");
        accumulator.tool_args_delta(&pid("call_1"), "\"temp\":\"20C\"}");
        assert_eq!(first, second);
        assert_eq!(second, third);

        let (tool_call, internal_call_id) = accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("call must finalize");
        assert_eq!(internal_call_id, first, "internal id is minted at start");
        assert_eq!(tool_call.id, "call_1");
        assert_eq!(tool_call.function.name, "get_weather");
        assert_eq!(
            tool_call.function.arguments,
            serde_json::json!({"location": "Paris", "temp": "20C"})
        );
        assert!(accumulator.saw_tool_call());
        assert!(matches!(
            accumulator.finish().first(),
            Some(AssistantContent::ToolCall(_))
        ));
    }

    /// #2258 review probe: an empty name fragment after an established name
    /// must not erase it (last-*non-empty* semantics) — the call still
    /// finalizes under the established name instead of dropping as nameless.
    #[test]
    fn an_empty_name_fragment_does_not_erase_an_established_name() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "get_weather");
        accumulator.tool_name_delta(&pid("call_1"), "");
        accumulator.tool_args_delta(&pid("call_1"), "{}");

        let (tool_call, _) = accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("call must finalize under the established name");
        assert_eq!(tool_call.function.name, "get_weather");
    }

    /// The fragment filter applied to the authoritative path: an end event
    /// carrying `Some("")` as its name must not erase the established name
    /// and drop the call as nameless.
    #[test]
    fn an_empty_authoritative_end_name_does_not_erase_an_established_name() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "get_weather");
        accumulator.tool_args_delta(&pid("call_1"), "{}");

        let mut done = end("call_1", UnparseableToolInput::Drop);
        done.name = Some(String::new());
        let (tool_call, _) = accumulator
            .tool_input_end(done)
            .expect("no error")
            .expect("call must finalize under the established name");
        assert_eq!(tool_call.function.name, "get_weather");
    }

    #[test]
    fn a_call_with_no_streamed_arguments_is_a_parameterless_invocation() {
        for fragments in [Vec::new(), vec![""]] {
            let mut accumulator = PartsAccumulator::new();
            accumulator.tool_name_delta(&pid("call_1"), "ping");
            for fragment in fragments {
                accumulator.tool_args_delta(&pid("call_1"), fragment);
            }
            let (tool_call, _) = accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .expect("parameterless calls are preserved");
            assert_eq!(tool_call.function.arguments, serde_json::json!({}));
        }
    }

    /// Truncation contract: a partial argument payload (or a nameless entry)
    /// never reaches the consumer under the `Drop` flush.
    #[test]
    fn drop_mode_drops_partial_arguments_and_nameless_calls() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "ping");
        accumulator.tool_args_delta(&pid("call_1"), "{\"x\":");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none()
        );

        accumulator.tool_args_delta(&pid("call_2"), "{\"y\":1}");
        assert!(
            accumulator
                .tool_input_end(end("call_2", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none(),
            "a call whose name never arrived is dropped"
        );
        assert!(!accumulator.saw_tool_call());
    }

    /// Eviction contract: a superseded call is delivered even when its
    /// arguments never parsed — normalized to `{}`, never a bare string.
    #[test]
    fn empty_object_mode_delivers_unparseable_input_as_an_empty_object() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "memory_search");
        accumulator.tool_args_delta(&pid("call_1"), "{\"query\":");
        let (tool_call, _) = accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::EmptyObject))
            .expect("no error")
            .expect("evicted calls are preserved");
        assert_eq!(tool_call.function.arguments, serde_json::json!({}));
    }

    /// Complete-block wires (Anthropic/Bedrock stop events): malformed input
    /// surfaces as an error item rather than a silent drop.
    #[test]
    fn error_mode_surfaces_malformed_input_as_an_error() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "get_weather");
        accumulator.tool_args_delta(&pid("call_1"), "{\"location\": not-json");
        let err = accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Error))
            .expect_err("malformed complete input must error");
        assert!(err.to_string().contains("get_weather"));
    }

    /// Completion-probe contract (single-chunk immediate emission): a probe
    /// that cannot finalize leaves the call open, later fragments extend it,
    /// and the internal id survives.
    #[test]
    fn keep_mode_leaves_the_call_open_for_later_fragments() {
        let mut accumulator = PartsAccumulator::new();
        let internal = accumulator.tool_name_delta(&pid("call_1"), "search");
        accumulator.tool_args_delta(&pid("call_1"), "{\"q\":\"ru");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Keep))
                .expect("no error")
                .is_none()
        );

        accumulator.tool_args_delta(&pid("call_1"), "st\"}");
        let (tool_call, internal_after) = accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("extended call finalizes");
        assert_eq!(internal_after, internal);
        assert_eq!(
            tool_call.function.arguments,
            serde_json::json!({"q": "rust"})
        );
    }

    /// Responses contract: the done item's fields are authoritative over the
    /// assembled fragments, and correlation with the fragment-minted internal
    /// id comes from the shared item id.
    #[test]
    fn authoritative_end_fields_supersede_assembly() {
        let mut accumulator = PartsAccumulator::new();
        let internal = accumulator.tool_name_delta(&pid("fc_1"), "provisional");
        accumulator.tool_args_delta(&pid("fc_1"), "{\"partial\":");

        let mut done = end("fc_1", UnparseableToolInput::Drop);
        done.name = Some("final_name".to_owned());
        done.arguments = Some(serde_json::json!({"x": 1}));
        done.call_id = Some("call_abc".to_owned());
        let (tool_call, internal_after) = accumulator
            .tool_input_end(done)
            .expect("no error")
            .expect("authoritative payload finalizes");
        assert_eq!(internal_after, internal);
        assert_eq!(tool_call.id, "fc_1");
        assert_eq!(tool_call.call_id.as_deref(), Some("call_abc"));
        assert_eq!(tool_call.function.name, "final_name");
        assert_eq!(tool_call.function.arguments, serde_json::json!({"x": 1}));
    }

    /// Responses replay path: a done item with no preceding fragments opens
    /// and completes the call from the event alone.
    #[test]
    fn an_end_with_no_open_call_creates_the_call_from_its_payload() {
        let mut accumulator = PartsAccumulator::new();
        let mut done = end("fc_1", UnparseableToolInput::Drop);
        done.name = Some("add".to_owned());
        done.arguments = Some(serde_json::json!({"x": 2}));
        let (tool_call, _) = accumulator
            .tool_input_end(done)
            .expect("no error")
            .expect("whole done items finalize");
        assert_eq!(tool_call.function.name, "add");
    }

    /// An end with neither an open call nor an authoritative name is a stale
    /// flush of an already-finalized key: silently ignored.
    #[test]
    fn a_stale_end_for_a_finalized_key_is_a_no_op() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "ping");
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none()
        );
        assert_eq!(
            accumulator
                .finish()
                .iter()
                .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
                .count(),
            1
        );
    }

    /// Gateway quirk: a literal `null` placeholder fragment is superseded by
    /// the real JSON fragments that follow.
    #[test]
    fn null_placeholder_is_replaced_by_following_json_fragments() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_123"), "web_search");
        accumulator.tool_args_delta(&pid("call_123"), "null");
        accumulator.tool_args_delta(&pid("call_123"), "{\"query\": \"META");
        accumulator.tool_args_delta(&pid("call_123"), " Platforms news\"}");
        let (tool_call, _) = accumulator
            .tool_input_end(end("call_123", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("call must finalize");
        assert_eq!(
            tool_call.function.arguments,
            serde_json::json!({"query": "META Platforms news"})
        );
    }

    /// Valid scalar and array argument payloads are canonical JSON and
    /// survive every delivery-mode finalization unchanged.
    #[test]
    fn scalar_and_array_arguments_survive_finalization_in_every_delivery_mode() {
        for (encoded, expected) in [
            ("5", serde_json::json!(5)),
            (r#""value""#, serde_json::json!("value")),
            ("[1,2]", serde_json::json!([1, 2])),
        ] {
            for mode in [
                UnparseableToolInput::Drop,
                UnparseableToolInput::EmptyObject,
            ] {
                let mut accumulator = PartsAccumulator::new();
                accumulator.tool_name_delta(&pid("call_1"), "tool");
                accumulator.tool_args_delta(&pid("call_1"), encoded);
                let (tool_call, _) = accumulator
                    .tool_input_end(end("call_1", mode))
                    .expect("no error")
                    .expect("valid JSON must survive finalization");
                assert_eq!(tool_call.function.arguments, expected, "changed {encoded}");
            }
        }
    }

    /// Parallel assembly: fragments interleaved across two ids stay separate,
    /// and finalization order follows the end events, not the starts.
    #[test]
    fn parallel_calls_assemble_independently() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_a"), "get_weather");
        accumulator.tool_name_delta(&pid("call_b"), "get_time");
        accumulator.tool_args_delta(&pid("call_a"), "{\"location\":\"Paris\"}");
        accumulator.tool_args_delta(&pid("call_b"), "{\"zone\":\"UTC\"}");
        let (call_b, _) = accumulator
            .tool_input_end(end("call_b", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        let (call_a, _) = accumulator
            .tool_input_end(end("call_a", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        assert_eq!(
            call_b.function.arguments,
            serde_json::json!({"zone": "UTC"})
        );
        assert_eq!(
            call_a.function.arguments,
            serde_json::json!({"location": "Paris"})
        );
    }

    /// The item-0 identity pin (#2258): two id-less parallel calls arrive
    /// under distinct boundary-minted `tool-{index}` ids, so their
    /// interleaved fragments assemble into two distinct uncorrupted calls —
    /// a shared empty id would merge them into one garbled argument buffer.
    #[test]
    fn minted_tool_index_ids_keep_id_less_parallel_calls_distinct() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("tool-0"), "get_weather");
        accumulator.tool_name_delta(&pid("tool-1"), "get_time");
        accumulator.tool_args_delta(&pid("tool-0"), "{\"city\":");
        accumulator.tool_args_delta(&pid("tool-1"), "{\"zone\":");
        accumulator.tool_args_delta(&pid("tool-0"), "\"Tokyo\"}");
        accumulator.tool_args_delta(&pid("tool-1"), "\"UTC\"}");
        let (first, _) = accumulator
            .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        let (second, _) = accumulator
            .tool_input_end(end("tool-1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        // Minted assembly identities never become durable provider ids: the
        // finalized calls carry the absent (empty) id every serializer omits.
        assert_eq!(first.id, "");
        assert_eq!(second.id, "");
        assert_eq!(
            first.function.arguments,
            serde_json::json!({"city": "Tokyo"})
        );
        assert_eq!(
            second.function.arguments,
            serde_json::json!({"zone": "UTC"})
        );
    }

    /// Truncation contract at stream end: calls still open when the stream
    /// finishes never became content and are discarded.
    #[test]
    fn finish_discards_calls_still_open_at_stream_end() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "ping");
        accumulator.tool_args_delta(&pid("call_1"), "{\"x\":");
        let parts = accumulator.finish();
        assert_eq!(parts, vec![AssistantContent::text("")]);
        assert!(!accumulator.saw_tool_call());
    }

    /// The tool-id override: a call opened under an id-less key still reports
    /// the provider id its wire established later.
    #[test]
    fn the_tool_id_override_supersedes_the_assembly_key() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid(""), "ping");
        let mut done = end("", UnparseableToolInput::Drop);
        done.tool_id = Some("call_late".to_owned());
        let (tool_call, _) = accumulator
            .tool_input_end(done)
            .expect("no error")
            .expect("finalizes");
        assert_eq!(tool_call.id, "call_late");
    }

    // --- #2258 F1: a full `ToolCall` reconciling with an open delta
    // assembly of the same id.
    //
    // Not inducible from a recorded provider turn: no in-tree wire mixes
    // fragments with a full restatement of the same call (each family emits
    // one shape or the other), so these are pinned as unit probes of the
    // shape an out-of-tree adapter can produce. ---

    fn call_named(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            call_id: None,
            function: crate::message::ToolFunction {
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            },
            signature: None,
            additional_params: None,
        }
    }

    /// The correlation contract: the deltas already published their minted
    /// internal id, so the full call adopts it instead of publishing a fresh
    /// one the consumer cannot match to anything it saw.
    #[test]
    fn a_full_call_adopts_the_internal_id_its_deltas_published() {
        let mut accumulator = PartsAccumulator::new();
        let published = accumulator.tool_name_delta(&pid("tc1"), "add");
        accumulator.tool_args_delta(&pid("tc1"), "{\"x\":1}");

        let adopted = accumulator.tool_call(
            &pid("tc1"),
            call_named("tc1", "add"),
            "freshly-minted".to_owned(),
        );
        assert_eq!(
            adopted, published,
            "the completed call must correlate with its own deltas"
        );

        let parts = accumulator.finish();
        assert_eq!(
            parts
                .iter()
                .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
                .count(),
            1
        );
    }

    /// Adoption consumes the assembly, so the wire's trailing end event for
    /// the same id finalizes nothing — pre-fix it appended a duplicate part
    /// from its own authoritative payload.
    #[test]
    fn an_end_after_a_full_call_for_the_same_id_is_a_no_op() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("tc1"), "add");
        accumulator.tool_args_delta(&pid("tc1"), "{\"x\":1}");
        accumulator.tool_call(
            &pid("tc1"),
            call_named("tc1", "add"),
            "freshly-minted".to_owned(),
        );

        let mut done = end("tc1", UnparseableToolInput::Drop);
        done.name = Some("add".to_owned());
        done.arguments = Some(serde_json::json!({"x": 1}));
        assert!(
            accumulator
                .tool_input_end(done)
                .expect("no error")
                .is_none(),
            "the call is already a part; ending it again must not duplicate it"
        );

        let parts = accumulator.finish();
        assert_eq!(
            parts
                .iter()
                .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
                .count(),
            1
        );
    }

    /// The mark is per-assembly, not permanent: fragments reusing a closed id
    /// open a genuinely new call with its own internal id, and its end event
    /// finalizes normally.
    #[test]
    fn fragments_reusing_a_closed_id_open_a_fresh_call() {
        let mut accumulator = PartsAccumulator::new();
        let first = accumulator.tool_name_delta(&pid("tc1"), "add");
        let adopted = accumulator.tool_call(
            &pid("tc1"),
            call_named("tc1", "add"),
            "freshly-minted".to_owned(),
        );
        assert_eq!(adopted, first);

        let second = accumulator.tool_name_delta(&pid("tc1"), "subtract");
        assert_ne!(second, first, "the reused id opens a distinct call");
        accumulator.tool_args_delta(&pid("tc1"), "{\"y\":2}");
        let (tool_call, internal) = accumulator
            .tool_input_end(end("tc1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("the reused id's call finalizes");
        assert_eq!(internal, second);
        assert_eq!(tool_call.function.name, "subtract");

        let parts = accumulator.finish();
        assert_eq!(
            parts
                .iter()
                .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
                .count(),
            2
        );
    }

    /// The mint-then-restate wire: fragments keyed under a minted identity,
    /// then the call restated whole under its late-arriving wire id. With
    /// exactly one minted assembly open, the restatement adopts it — the
    /// published internal id correlates, and no duplicate part appears.
    #[test]
    fn a_wire_restatement_adopts_the_single_open_minted_assembly() {
        let mut accumulator = PartsAccumulator::new();
        let published = accumulator.tool_name_delta(&pid("tool-0"), "add");
        accumulator.tool_args_delta(&pid("tool-0"), "{\"x\":1}");

        let adopted = accumulator.tool_call(
            &pid("call_late"),
            call_named("call_late", "add"),
            "freshly-minted".to_owned(),
        );
        assert_eq!(adopted, published, "the restatement must correlate");

        // The stale end for the minted key finalizes nothing.
        assert!(
            accumulator
                .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none()
        );
        let parts = accumulator.finish();
        assert_eq!(
            parts
                .iter()
                .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
                .count(),
            1
        );
    }

    /// Two open minted assemblies make adoption ambiguous: the restatement
    /// must not guess, and both assemblies stay open for their own ends.
    #[test]
    fn a_wire_restatement_never_guesses_between_two_minted_assemblies() {
        let mut accumulator = PartsAccumulator::new();
        let first = accumulator.tool_name_delta(&pid("tool-0"), "add");
        let second = accumulator.tool_name_delta(&pid("tool-1"), "subtract");

        let published = accumulator.tool_call(
            &pid("call_late"),
            call_named("call_late", "add"),
            "freshly-minted".to_owned(),
        );
        assert_eq!(published, "freshly-minted");
        assert_ne!(published, first);
        assert_ne!(published, second);
    }

    /// A runaway wire cannot grow a fragment buffer without bound: past the
    /// accumulation cap the input truncates and the call finalizes through
    /// the wire's unparseable-input policy (dropped under `Drop`).
    #[test]
    fn oversized_tool_input_truncates_and_finalizes_through_policy() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "bulk");
        let chunk = "x".repeat(1024 * 1024);
        accumulator.tool_args_delta(&pid("call_1"), "{\"data\":\"");
        for _ in 0..33 {
            accumulator.tool_args_delta(&pid("call_1"), &chunk);
        }
        accumulator.tool_args_delta(&pid("call_1"), "\"}");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none(),
            "the truncated input must not fabricate a call"
        );
    }

    /// The single-shape wires (every in-tree provider that emits a whole
    /// call): with nothing to adopt, the emitter's minted id is published
    /// unchanged.
    #[test]
    fn a_full_call_with_no_open_assembly_keeps_its_minted_id() {
        let mut accumulator = PartsAccumulator::new();
        let published =
            accumulator.tool_call(&pid("tc1"), call_named("tc1", "add"), "minted-1".to_owned());
        assert_eq!(published, "minted-1");

        // A different id's assembly is untouched by an unrelated full call.
        let other = accumulator.tool_name_delta(&pid("tc2"), "subtract");
        let published =
            accumulator.tool_call(&pid("tc3"), call_named("tc3", "add"), "minted-2".to_owned());
        assert_eq!(published, "minted-2");
        assert_ne!(other, published);
        let (_, internal) = accumulator
            .tool_input_end(end("tc2", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("the untouched assembly still finalizes");
        assert_eq!(internal, other);
    }
}

/// The aggregation laws, as properties (#2258 A5).
///
/// The aggregate is a pure function of the fragment sequence: the parts list
/// is the single accumulated state and [`PartsAccumulator::finish`] derives
/// the choice from it — there is no separate projection to desync. These
/// properties pin the algebra that purity rests on:
///
/// - **fragment associativity/identity**: how a payload is split into
///   deltas (including empty fragments) cannot change the aggregate;
/// - **stale-end idempotence**: finalizing an already-finalized call again
///   is a no-op, never a duplicate part;
/// - **boundary idempotence**: repeated boundaries with nothing open change
///   nothing.
///
/// The reference precedents don't have these laws (langchain's merge is not
/// a monoid — `"stop" + "length"` concatenates; semantic-kernel raises on
/// conflicting scalars but has no property tests); Rust makes stating them
/// cheap, so rig does.
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn wire(id: &str) -> PartId {
        PartId::wire(id)
    }

    /// Split `payload` at the given fractional points into consecutive
    /// fragments (possibly empty at the edges), on char boundaries.
    fn split_fragments(payload: &str, points: &[usize]) -> Vec<String> {
        let chars: Vec<char> = payload.chars().collect();
        let mut cuts: Vec<usize> = points
            .iter()
            .map(|point| point % (chars.len() + 1))
            .collect();
        cuts.sort_unstable();
        let mut fragments = Vec::new();
        let mut start = 0usize;
        for cut in cuts {
            fragments.push(chars[start..cut.max(start)].iter().collect());
            start = cut.max(start);
        }
        fragments.push(chars[start..].iter().collect());
        fragments
    }

    proptest! {
        /// Text aggregation is invariant under fragmentation: any split of
        /// the payload into deltas produces the same single text part.
        #[test]
        fn text_aggregate_is_fragmentation_invariant(
            payload in ".{0,60}",
            points in proptest::collection::vec(0usize..1000, 0..6),
        ) {
            let mut whole = PartsAccumulator::new();
            whole.text_delta(&payload);
            let mut split = PartsAccumulator::new();
            for fragment in split_fragments(&payload, &points) {
                split.text_delta(&fragment);
            }
            prop_assert_eq!(whole.finish(), split.finish());
        }

        /// Reasoning-delta aggregation is invariant under fragmentation for
        /// a fixed item id.
        #[test]
        fn reasoning_aggregate_is_fragmentation_invariant(
            payload in ".{1,60}",
            points in proptest::collection::vec(0usize..1000, 0..6),
        ) {
            let id = wire("rs_1");
            let mut whole = PartsAccumulator::new();
            whole.reasoning_delta(&id, &payload);
            let mut split = PartsAccumulator::new();
            let mut pushed = false;
            for fragment in split_fragments(&payload, &points) {
                if !fragment.is_empty() {
                    split.reasoning_delta(&id, &fragment);
                    pushed = true;
                }
            }
            prop_assume!(pushed);
            prop_assert_eq!(whole.finish(), split.finish());
        }

        /// Tool-argument assembly is invariant under fragmentation: the
        /// finalized call's arguments do not depend on how the JSON was cut.
        #[test]
        fn tool_arguments_are_fragmentation_invariant(
            value in "[a-z]{0,20}",
            points in proptest::collection::vec(0usize..1000, 0..6),
        ) {
            let payload = format!("{{\"q\":\"{value}\"}}");
            let finalize = |fragments: &[String]| {
                let mut accumulator = PartsAccumulator::new();
                let id = wire("call_1");
                accumulator.tool_name_delta(&id, "probe");
                for fragment in fragments {
                    accumulator.tool_args_delta(&id, fragment);
                }
                accumulator
                    .tool_input_end(ToolInputEnd::new(id, UnparseableToolInput::Drop))
                    .expect("no error")
                    .map(|(call, _)| call.function.arguments)
            };
            let whole = finalize(std::slice::from_ref(&payload));
            let split = finalize(&split_fragments(&payload, &points));
            prop_assert_eq!(whole, split);
        }

        /// Stale-end idempotence: ending an already-finalized id any number
        /// of extra times adds nothing.
        #[test]
        fn stale_tool_input_ends_are_idempotent(extra_ends in 1usize..5) {
            let mut accumulator = PartsAccumulator::new();
            let id = wire("call_1");
            accumulator.tool_name_delta(&id, "probe");
            accumulator.tool_args_delta(&id, "{}");
            accumulator
                .tool_input_end(ToolInputEnd::new(id.clone(), UnparseableToolInput::Drop))
                .expect("no error")
                .expect("finalizes");
            for _ in 0..extra_ends {
                prop_assert!(
                    accumulator
                        .tool_input_end(ToolInputEnd::new(
                            id.clone(),
                            UnparseableToolInput::Drop
                        ))
                        .expect("no error")
                        .is_none()
                );
            }
            let calls = accumulator
                .finish()
                .into_iter()
                .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
                .count();
            prop_assert_eq!(calls, 1);
        }

        /// Boundary idempotence: any number of boundary events with no open
        /// minted reasoning leaves the aggregate untouched.
        #[test]
        fn boundaries_with_nothing_open_are_idempotent(boundaries in 0usize..5) {
            let minted = PartId::Minted {
                kind: crate::streaming::MintKind::Reasoning,
                index: 0,
            };
            let mut accumulator = PartsAccumulator::new();
            accumulator.reasoning_delta(&minted, "thought");
            accumulator.text_delta("answer");
            for _ in 0..boundaries {
                accumulator.text_delta("");
            }
            accumulator.reasoning_delta(&minted, "more");
            let parts = accumulator.finish();
            prop_assert_eq!(
                parts
                    .iter()
                    .filter(|part| matches!(part, AssistantContent::Reasoning(_)))
                    .count(),
                2
            );
        }
    }
}
