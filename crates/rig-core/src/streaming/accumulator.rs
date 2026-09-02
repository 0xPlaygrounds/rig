//! Accumulation of stream events into the ordered parts of one assistant
//! choice — as **entities with a lifecycle**, not id-keyed event sequences.
//!
//! Every part runs open → mutate → close (review 84a43e9e root cause A):
//!
//! - **open** registers the part in `parts` once, fixing arrival order, and
//!   tracks it in an open-set keyed by its [`BlockId`];
//! - **deltas** mutate the open part in place through that handle;
//! - **close** applies the end event's authoritative payload (a whole-block
//!   restatement supersedes the delta accumulation; a trailing signature
//!   attaches), then moves the key to a finished-set.
//!
//! The finished-set is the entity's `hasFinished` (vercel's
//! `streaming-tool-call-tracker`): **every** finalization route consults and
//! populates it, so a repeated end event — whatever payload it carries — is
//! a no-op rather than a duplicate part, and the guard cannot be forgotten
//! on one route. A key genuinely reused after finishing (new fragments, a
//! same-key whole block) opens a **new** part: the lenient reuse rule that
//! replaces the old ordinal machinery.
//!
//! The accumulator owns no transport: it is fed [`StreamEvent`]s one at a
//! time through [`BlockAccumulator::apply`] — by
//! [`StreamingCompletionResponse`](super::StreamingCompletionResponse) on the
//! provider path, by a bus client wrapping an effect stream, or by a test
//! over `futures::stream::iter` — and [`BlockAccumulator::snapshot`] is
//! non-destructive: two snapshots mid-stream are equal, and neither changes
//! what [`BlockAccumulator::finish`] later returns.
//!
//! Reference designs: vercel-ai-sdk's `stream-text` accumulator (`-start`
//! registers and pushes the handle once, `-delta` mutates through it,
//! `-end` deletes it) and pydantic-ai's `_stop_tracking_vendor_id`.

use std::collections::{HashMap, HashSet};

use crate::completion::CompletionError;
use crate::message::{AssistantContent, Reasoning, ReasoningContent, ToolCall, ToolFunction};
use crate::streaming::UnparseableToolInput;
use crate::streaming::block_id::BlockId;
use crate::streaming::event::{BlockClose, BlockKind, Delta, StreamEvent, ToolCallEnd};

/// Accumulates the streamed parts of one assistant choice, in arrival order.
///
/// Owns every aggregation decision the streaming surfaces make. Consumers
/// feed events through [`BlockAccumulator::apply`] and read the choice with
/// [`BlockAccumulator::snapshot`] or [`BlockAccumulator::finish`].
#[derive(Default)]
pub struct BlockAccumulator {
    /// The choice's parts, in arrival order — the single accumulated state;
    /// [`BlockAccumulator::finish`] derives the choice from it directly.
    parts: Vec<AssistantContent>,
    /// Open reasoning entities: key → index in `parts`. Invariant: every
    /// mapped index holds an `AssistantContent::Reasoning` part.
    open_reasoning: HashMap<BlockId, usize>,
    /// Finished reasoning entities: key → index of the latest finished part
    /// under that key. Repeated ends are no-ops; a trailing signature
    /// (`reasoning_end` with no restatement) attaches here — the block that
    /// holds the chain-of-thought, wherever it sits.
    finished_reasoning: HashMap<BlockId, usize>,
    /// Text-block identity → index in `parts`: a delta whose key was
    /// already seen extends that block, so a wire item's text keeps
    /// collapsing across interleaved output. Invariant: every mapped index
    /// holds an `AssistantContent::Text` part.
    text_ids: HashMap<BlockId, usize>,
    /// Tool calls under delta assembly, keyed by the fragment key, in start
    /// order.
    open_tool_inputs: Vec<OpenToolInput>,
    /// Finished tool entities — populated by **every** finalization route
    /// (fragment-assembled ends, authoritative-payload ends, whole-call
    /// adoption), so a repeated end cannot duplicate a call whatever payload
    /// it carries (84a43e9e #1). Cleared for a key when fragments reopen it.
    finished_tools: HashSet<BlockId>,
    /// Whether any completed tool call was recorded; the streaming
    /// counterpart of the unary path's finish-reason reconciliation input.
    saw_tool_call: bool,
}

/// A tool call whose input is still being assembled from streamed fragments.
///
/// Reference designs: pydantic-ai `handle_tool_call_delta` and vercel's
/// shared `streaming-tool-call-tracker` — fragment buffering, keying, and
/// the delta-to-part transition live in the one shared component, never in
/// a provider.
struct OpenToolInput {
    /// Assembly key: every fragment of one call carries this key.
    id: BlockId,
    /// Tool name; a later non-empty name fragment replaces it.
    name: String,
    /// Concatenated raw argument fragments. `None` until a fragment arrives —
    /// a call that streamed no arguments is a parameterless invocation.
    buffer: Option<String>,
    /// Whether the buffer hit [`MAX_TOOL_INPUT_BYTES`] and stopped
    /// accumulating. The truncated JSON then finalizes through the wire's
    /// `UnparseableToolInput` policy — no runaway wire can grow a fragment
    /// buffer without bound.
    overflowed: bool,
}

/// Upper bound on one streamed tool call's accumulated argument bytes.
///
/// Far beyond any real tool input; a wire that exceeds it is defective, and
/// its call finalizes through the unparseable-input policy instead of
/// growing memory without bound.
const MAX_TOOL_INPUT_BYTES: usize = 32 * 1024 * 1024;

impl BlockAccumulator {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one event into the accumulated choice.
    ///
    /// Returns the block a `BlockEnd` finalized for consumers, keyed by the
    /// block id it must be published under (the assembly key a whole call
    /// adopted, which can differ from the end event's own id), or `None`
    /// when the event finalized nothing to publish. An `Err` is a malformed
    /// complete tool input under [`UnparseableToolInput::Error`]; the
    /// accumulator stays consistent and the stream keeps consuming.
    pub fn apply(
        &mut self,
        event: &StreamEvent,
    ) -> Result<Option<(BlockId, AssistantContent)>, CompletionError> {
        match event {
            StreamEvent::BlockStart { id, kind } => {
                match kind {
                    BlockKind::Message => {}
                    BlockKind::Text { additional_params } => {
                        self.text_start(id, additional_params.clone());
                    }
                    BlockKind::Reasoning { provider_id } => {
                        self.reasoning_start(id, provider_id.as_deref());
                    }
                    BlockKind::ToolCall => {
                        self.ensure_open_tool_input(id);
                    }
                }
                Ok(None)
            }
            StreamEvent::BlockDelta { id, delta } => {
                match delta {
                    Delta::Text { text } => self.text_delta(id, text),
                    Delta::TextMeta { additional_params } => {
                        self.text_additional_params(id, additional_params.clone());
                    }
                    Delta::Reasoning { text } => self.reasoning_delta(id, None, text),
                    Delta::ToolName { name } => self.tool_name_delta(id, name),
                    Delta::ToolArguments { arguments } => self.tool_args_delta(id, arguments),
                }
                Ok(None)
            }
            StreamEvent::BlockEnd { id, end, .. } => match end {
                BlockClose::Text => Ok(None),
                BlockClose::Reasoning {
                    reasoning,
                    signature,
                    wire_sent,
                } => {
                    // The completed block is published when the wire said
                    // something at the boundary: an end payload (a
                    // restatement or a signature) or a bare end frame the
                    // wire actually sent. Only a bare end an adapter
                    // *synthesized* stays silent — the consumer already
                    // received every delta, and fabricating a "completed
                    // block" event the wire never sent would change what
                    // downstream history builders observe.
                    let authoritative = reasoning.is_some() || signature.is_some() || *wire_sent;
                    let completed = self.reasoning_end(id, reasoning.clone(), signature.clone());
                    Ok(completed
                        .filter(|_| authoritative)
                        .map(|reasoning| (id.clone(), AssistantContent::Reasoning(reasoning))))
                }
                BlockClose::ToolCall(end) => Ok(self
                    .tool_end(id, end.clone())?
                    .map(|(id, call)| (id, AssistantContent::ToolCall(call)))),
            },
            StreamEvent::Final(_) | StreamEvent::Unknown(_) => Ok(None),
        }
    }

    // --- text lifecycle -------------------------------------------------

    /// Open the text block identified by `id` — with its provider metadata
    /// when the wire announced some. A previously seen key reactivates its
    /// block (a wire item's text keeps collapsing across interleaved
    /// output); an unseen key opens lazily, on its first delta or metadata,
    /// so a content-less start never leaves an empty text part behind.
    fn text_start(
        &mut self,
        id: &BlockId,
        additional_params: Option<crate::message::AdditionalParams>,
    ) {
        if let Some(additional_params) = additional_params {
            self.text_additional_params(id, additional_params);
        }
    }

    /// Append streamed text to the block identified by `id`, opening it if
    /// unseen.
    fn text_delta(&mut self, id: &BlockId, text: &str) {
        let index = self.ensure_text_block(id);
        if let Some(AssistantContent::Text(existing)) = self.parts.get_mut(index) {
            existing.text.push_str(text);
        }
    }

    /// Merge provider metadata into the block identified by `id`, opening an
    /// empty block if unseen.
    ///
    /// [`crate::message::AdditionalParams`] is non-empty by construction, so
    /// there is no empty-carrier case to filter: an arriving params value is
    /// always data, and the stored-params invariant (`None` or data) holds by
    /// type.
    fn text_additional_params(
        &mut self,
        id: &BlockId,
        additional_params: crate::message::AdditionalParams,
    ) {
        let index = self.ensure_text_block(id);
        let Some(AssistantContent::Text(text)) = self.parts.get_mut(index) else {
            return;
        };
        match text.additional_params.as_mut() {
            Some(existing) => existing.merge(additional_params),
            None => text.additional_params = Some(additional_params),
        }
    }

    /// Index of the text block for `id`, opening one if unseen.
    fn ensure_text_block(&mut self, id: &BlockId) -> usize {
        if let Some(&index) = self.text_ids.get(id) {
            return index;
        }
        self.parts.push(AssistantContent::text(""));
        let index = self.parts.len() - 1;
        self.text_ids.insert(id.clone(), index);
        index
    }

    // --- reasoning lifecycle --------------------------------------------

    /// Open the reasoning entity for `id`, registering an empty part at the
    /// current arrival position. A start for an already-open key is a no-op
    /// (returns `false`); a start for a finished key opens a **new** part
    /// (key reuse) — the `true` return tells the stream handler to mint a
    /// fresh public correlator, so the new part never inherits the finished
    /// part's identity.
    fn reasoning_start(&mut self, id: &BlockId, provider_id: Option<&str>) -> bool {
        if self.open_reasoning.contains_key(id) {
            return false;
        }
        self.open_fresh_reasoning(id, provider_id, Vec::new());
        true
    }

    /// Append reasoning delta text to the entity's open part, opening one if
    /// none is open — the lenient bare-delta rule. A delta after the
    /// entity finished opens a new part (key reuse), never resurrects the
    /// finished one.
    fn reasoning_delta(&mut self, id: &BlockId, provider_id: Option<&str>, text: &str) {
        if let Some(&index) = self.open_reasoning.get(id) {
            if let Some(AssistantContent::Reasoning(existing)) = self.parts.get_mut(index) {
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
            }
            return;
        }
        self.open_fresh_reasoning(
            id,
            provider_id,
            vec![ReasoningContent::Text {
                text: text.to_owned(),
                signature: None,
            }],
        );
    }

    /// Close the reasoning entity for `id`, returning the completed part
    /// for the public completion event.
    ///
    /// - `restatement`: the wire's authoritative whole-block payload; it
    ///   **supersedes** the delta accumulation (pydantic-ai `_replace_part`
    ///   semantics — the deltas were a fallback and their buffer is
    ///   discarded).
    /// - `signature`: a provider signature closing the block; attaches to
    ///   the part's last text content.
    ///
    /// Idempotence is the entity's: an end for an already-finished key with
    /// **no new restatement** is a no-op. An end with a restatement for a
    /// finished key opens a new sibling part (the Responses
    /// multi-part-item shape, spelled by the adapter as a same- or
    /// composite-key whole block). An end for a never-seen key with a
    /// restatement creates the part whole (the replay path); with only a
    /// signature it records a signature-only part, so replay-required
    /// provider state still reaches history.
    fn reasoning_end(
        &mut self,
        id: &BlockId,
        restatement: Option<Reasoning>,
        signature: Option<String>,
    ) -> Option<Reasoning> {
        if let Some(index) = self.open_reasoning.remove(id) {
            if let Some(mut restatement) = restatement
                && let Some(part) = self.parts.get_mut(index)
            {
                // The restatement supersedes accumulated content, but an
                // absent field must not erase established identity: the
                // durable handle set at part-open survives a restatement
                // that doesn't restate it (the `?? existing` merge every
                // reference SDK applies to end payloads).
                if restatement.id.is_none()
                    && let AssistantContent::Reasoning(open) = &*part
                {
                    restatement.id.clone_from(&open.id);
                }
                *part = AssistantContent::Reasoning(restatement);
            }
            if let Some(signature) = signature
                && let Some(part) = self.parts.get_mut(index)
            {
                attach_signature(part, signature);
            }
            self.finished_reasoning.insert(id.clone(), index);
            return self.reasoning_at(index);
        }

        if let Some(&index) = self.finished_reasoning.get(id) {
            match (restatement, signature) {
                // Trailing lifecycle metadata for the finished block: the
                // signature lands on the part that holds the chain-of-
                // thought, wherever its arrival position was (#2258 B4) —
                // unless that part already carries a signature. Signatures
                // cannot merge ("Don't combine two Parts that both contain
                // signatures" — Google's own doctrine), so a second
                // signature under a per-stream constant key records a
                // DISTINCT sibling part and repoints the key, instead of
                // overwriting the first signature and losing it from the
                // replayed history.
                (None, Some(signature)) => {
                    let part_already_signed = matches!(
                        self.parts.get(index),
                        Some(AssistantContent::Reasoning(reasoning))
                            if reasoning.content.iter().any(|content| matches!(
                                content,
                                ReasoningContent::Text { signature: Some(_), .. }
                            ))
                    );
                    if part_already_signed {
                        return self.finish_signature_only(id, signature);
                    }
                    if let Some(part) = self.parts.get_mut(index) {
                        attach_signature(part, signature);
                    }
                    return self.reasoning_at(index);
                }
                // A repeated end with no new payload is a no-op — the
                // entity already finished (84a43e9e #1, for reasoning).
                (None, None) => return None,
                // A whole block under a finished key is a NEW sibling part
                // reusing the key.
                (Some(restatement), signature) => {
                    return self.finish_restated(id, restatement, signature);
                }
            }
        }

        // Never-seen key: create the part whole from the end payload.
        match (restatement, signature) {
            (Some(restatement), signature) => self.finish_restated(id, restatement, signature),
            // Signature-only stream: replay-required provider state with
            // nothing streamed to sign. Record it alone.
            (None, Some(signature)) => self.finish_signature_only(id, signature),
            (None, None) => None,
        }
    }

    /// Record a whole reasoning block as a finished part under `id`.
    fn finish_restated(
        &mut self,
        id: &BlockId,
        mut restatement: Reasoning,
        signature: Option<String>,
    ) -> Option<Reasoning> {
        if let Some(signature) = signature {
            attach_reasoning_signature(&mut restatement, signature);
        }
        let index = self.push_reasoning_part(restatement);
        self.finished_reasoning.insert(id.clone(), index);
        self.reasoning_at(index)
    }

    /// Record a signature with no chain-of-thought as its own finished part
    /// under `id`.
    fn finish_signature_only(&mut self, id: &BlockId, signature: String) -> Option<Reasoning> {
        let index = self.push_reasoning_part(Reasoning {
            id: None,
            content: vec![ReasoningContent::Text {
                text: String::new(),
                signature: Some(signature),
            }],
        });
        self.finished_reasoning.insert(id.clone(), index);
        self.reasoning_at(index)
    }

    fn open_fresh_reasoning(
        &mut self,
        id: &BlockId,
        provider_id: Option<&str>,
        content: Vec<ReasoningContent>,
    ) {
        let index = self.push_reasoning_part(Reasoning {
            id: provider_id.map(str::to_owned),
            content,
        });
        self.open_reasoning.insert(id.clone(), index);
    }

    /// Register a new reasoning part at the current arrival position.
    fn push_reasoning_part(&mut self, reasoning: Reasoning) -> usize {
        self.parts.push(AssistantContent::Reasoning(reasoning));
        self.parts.len() - 1
    }

    fn reasoning_at(&self, index: usize) -> Option<Reasoning> {
        match self.parts.get(index) {
            Some(AssistantContent::Reasoning(reasoning)) => Some(reasoning.clone()),
            _ => None,
        }
    }

    // --- tool-call lifecycle --------------------------------------------

    /// The single open minted assembly a wire-keyed whole call restates, if
    /// exactly one does.
    ///
    /// A wire that fragmented under a minted key and restates the call under
    /// its late-arriving wire key: minted-vs-wire is not a veto — with
    /// exactly one minted assembly open, the restatement CAN be that
    /// assembly completed. More than one is ambiguous; don't guess. And
    /// cardinality alone is not evidence: adoption also demands the whole
    /// call restate the assembly — same tool name, arguments covering
    /// whatever fragments streamed. An unrelated id-bearing call adopting
    /// the assembly marked it finished and silently dropped its streamed
    /// arguments when its own end arrived.
    fn adoptable_assembly(
        &self,
        id: &BlockId,
        name: &str,
        arguments: &serde_json::Value,
    ) -> Option<usize> {
        if id.is_minted() {
            return None;
        }
        let mut minted = self
            .open_tool_inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| input.id.is_minted());
        let (Some(candidate), None) = (minted.next(), minted.next()) else {
            return None;
        };
        let (index, input) = candidate;
        // An assembly opened by args-only deltas never saw a name
        // (`ensure_open_tool_input` records ""), so an empty name is no
        // evidence against the restatement — only a *different* established
        // name vetoes.
        let restates = (input.name.is_empty() || input.name == name)
            && fragments_covered_by(input.buffer.as_deref(), arguments);
        restates.then_some(index)
    }

    /// Append a completed tool call as the next part.
    ///
    /// A completed call is a part boundary for the ACTIVE text block: text
    /// around it must stay two blocks in arrival order. (A keyed block still
    /// reactivates through its own `TextStart` — the keyed collapse is
    /// explicit; only the anonymous active-block cursor resets.)
    fn push_tool_call(&mut self, tool_call: ToolCall) {
        self.saw_tool_call = true;
        self.parts.push(AssistantContent::ToolCall(tool_call));
    }

    /// Whether any completed tool call was recorded on this stream.
    pub fn saw_tool_call(&self) -> bool {
        self.saw_tool_call
    }

    /// Record a streamed tool name fragment, opening the call if `id` has no
    /// open call.
    ///
    /// A later non-empty name replaces the recorded one (OpenAI-compatible
    /// wire semantics: the established name is the last non-empty value).
    fn tool_name_delta(&mut self, id: &BlockId, name: &str) {
        let index = self.ensure_open_tool_input(id);
        // Last-*non-empty* semantics: an empty fragment must not erase an
        // established name, or finalization would drop the call as nameless.
        if let Some(input) = self.open_tool_inputs.get_mut(index)
            && !name.is_empty()
        {
            name.clone_into(&mut input.name);
        }
    }

    /// Append a streamed argument fragment to the call's buffer, opening the
    /// call if `id` has no open call.
    fn tool_args_delta(&mut self, id: &BlockId, fragment: &str) {
        let index = self.ensure_open_tool_input(id);
        // `ensure` returns a live index.
        if let Some(input) = self.open_tool_inputs.get_mut(index) {
            // The first fragment takes the same guarded path as every
            // later one — an oversized single fragment must trip the
            // bound, not bypass it.
            let buffer = input.buffer.get_or_insert_with(String::new);
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
    }

    /// Close a streamed tool call's input and finalize it into a completed
    /// call.
    ///
    /// Returns the completed call, `None` when the call
    /// is dropped (nameless, unparseable under [`UnparseableToolInput::Drop`],
    /// or a repeated end for an already-finished entity — whatever payload
    /// it carries), or an error item under [`UnparseableToolInput::Error`].
    /// Authoritative fields on the end event supersede the assembled state.
    /// An end with no open call and no finished entity opens and completes
    /// one from the event alone (the replay path).
    fn tool_end(
        &mut self,
        id: &BlockId,
        end: ToolCallEnd,
    ) -> Result<Option<(BlockId, ToolCall)>, CompletionError> {
        let position = self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == *id);
        // A whole call under a wire key restating a single open minted
        // assembly (fragments streamed under a minted key, the wire's own
        // id arriving late with the completed item) adopts that assembly:
        // the call is published under the key its deltas carried.
        let adopted = match (position, end.name.as_deref(), end.arguments.as_ref()) {
            (None, Some(name), Some(arguments)) => self.adoptable_assembly(id, name, arguments),
            _ => None,
        };
        let (position, published) = match adopted.and_then(|index| {
            self.open_tool_inputs
                .get(index)
                .map(|input| (index, input.id.clone()))
        }) {
            Some((index, key)) => (Some(index), key),
            None => (position, id.clone()),
        };
        // The entity already finished — by its own end, or by whole-call
        // adoption. A repeated end must finalize nothing, INCLUDING one
        // carrying an authoritative name/arguments payload: pre-84a43e9e-#1
        // that payload route bypassed the guard and duplicated the call.
        if position.is_none() && self.finished_tools.contains(id) {
            if end.name.is_some() || end.arguments.is_some() {
                tracing::debug!(
                    carries_name = end.name.is_some(),
                    carries_arguments = end.arguments.is_some(),
                    "ignoring a payload-bearing repeated end for a finished tool call"
                );
            }
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

        let (opened_id, mut name, buffer) = match open.as_ref() {
            Some(input) => (input.id.clone(), input.name.clone(), input.buffer.clone()),
            None => (id.clone(), String::new(), None),
        };
        let overflowed = open.as_ref().is_some_and(|input| input.overflowed);
        // The assembly key is opaque; a wire-derived key doubles as the
        // durable fallback when the end event carries no authoritative tool
        // id — the wire issued that key, so it is a provider handle.
        let opened_wire_id = opened_id.wire_str().map(str::to_owned);
        // An authoritative end-event name supersedes assembly, but an
        // *empty* one is filtered like the fragment path: it must not erase
        // an established name and turn a real call into a nameless drop.
        if let Some(final_name) = end.name.clone().filter(|final_name| !final_name.is_empty()) {
            name = final_name;
        }
        // A call whose name never arrived is not a call the model made
        // (OpenAI-compatible flush semantics: nameless entries drop).
        if name.is_empty() {
            if matches!(end.on_unparseable, UnparseableToolInput::Keep) {
                keep_open(self, open);
            } else {
                // The drop finalizes the entity: a later payload-bearing
                // end for this key must not resurrect it as a phantom call.
                self.finished_tools.insert(id.clone());
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
                // and fabricate a silently corrupted call, so overflow
                // forces the unparseable path.
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
                                // The drop finalizes the entity, exactly like
                                // a successful completion.
                                self.finished_tools.insert(id.clone());
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

        // Provider identifiers: a dual wire carries (call_id, item id); a
        // single wire's id arrives as `tool_id` (or as the wire-derived
        // assembly key). With none, the correlation handle is minted and
        // `provider` stays `None` — the empty-string sentinel is
        // unrepresentable here.
        let wire_tool_id = end.tool_id.or(opened_wire_id);
        let provider =
            crate::message::ProviderCallId::from_optional_wire(end.call_id, wire_tool_id);
        let durable_id = crate::message::ToolCallId::for_provider(provider.as_ref());
        let tool_call = ToolCall {
            id: durable_id,
            provider,
            function: ToolFunction { name, arguments },
            signature: end.signature,
            additional_params: end.additional_params,
        };
        // Every finalization route records the finished entity — the end's
        // own key and, on adoption, the assembly key it completed.
        self.finished_tools.insert(id.clone());
        self.finished_tools.insert(published.clone());
        self.push_tool_call(tool_call.clone());
        Ok(Some((published, tool_call)))
    }

    /// Index of the open call for `id`, opening one if none exists.
    fn ensure_open_tool_input(&mut self, id: &BlockId) -> usize {
        match self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == *id)
        {
            Some(index) => index,
            None => {
                // Fragments for a finished key are a *new* call reusing the
                // key, not a continuation of the finalized one: drop the
                // mark so its end event finalizes normally.
                self.finished_tools.remove(id);
                self.open_tool_inputs.push(OpenToolInput {
                    id: id.clone(),
                    name: String::new(),
                    buffer: None,
                    overflowed: false,
                });
                self.open_tool_inputs.len() - 1
            }
        }
    }

    // --- lifecycle end --------------------------------------------------

    /// The accumulated choice so far, without consuming it.
    ///
    /// Empty text parts that never received content are omitted (a lazily
    /// opened block that got content survives; the never-fed placeholder
    /// does not). Calls still open never fully arrived and are not part of
    /// the choice; reasoning still open is kept as-is (its deltas are real
    /// content). Two snapshots mid-stream are equal, and neither changes
    /// what [`BlockAccumulator::finish`] returns.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.parts
            .iter()
            .filter(|part| Self::survives(part))
            .cloned()
            .collect()
    }

    /// Consume the accumulated state into the ordered choice parts — the
    /// value [`BlockAccumulator::snapshot`] would return — and reset.
    ///
    /// Never padded: a stream that produced no content yields an empty
    /// choice. The fabricated empty-text part that used to be pushed here
    /// existed only to satisfy the non-empty content type, and it reached
    /// history and the wire as if the model had emitted it.
    pub fn finish(&mut self) -> Vec<AssistantContent> {
        let parts: Vec<AssistantContent> = std::mem::take(&mut self.parts)
            .into_iter()
            .filter(Self::survives)
            .collect();
        self.open_reasoning.clear();
        self.finished_reasoning.clear();
        self.text_ids.clear();
        self.open_tool_inputs.clear();
        self.finished_tools.clear();
        self.saw_tool_call = false;
        parts
    }

    fn survives(part: &AssistantContent) -> bool {
        match part {
            AssistantContent::Text(text) => {
                !(text.text.is_empty() && text.additional_params.is_none())
            }
            _ => true,
        }
    }
}

fn fragments_covered_by(buffer: Option<&str>, arguments: &serde_json::Value) -> bool {
    let Some(buffer) = buffer else {
        return true;
    };
    if buffer.trim().is_empty() {
        return true;
    }
    let Ok(partial) = crate::json_utils::parse_tool_arguments(buffer) else {
        return false;
    };
    // A buffer still holding the literal `null` placeholder (the gateway
    // shape `tool_args_delta` documents) streamed no real arguments yet:
    // any restatement covers it vacuously.
    if partial.is_null() {
        return true;
    }
    json_subsumes(arguments, &partial)
}

fn json_subsumes(outer: &serde_json::Value, inner: &serde_json::Value) -> bool {
    match (outer, inner) {
        (serde_json::Value::Object(outer), serde_json::Value::Object(inner)) => {
            inner.iter().all(|(key, value)| {
                outer
                    .get(key)
                    .is_some_and(|outer| json_subsumes(outer, value))
            })
        }
        (outer, inner) => outer == inner,
    }
}

fn attach_signature(part: &mut AssistantContent, signature: String) {
    if let AssistantContent::Reasoning(reasoning) = part {
        attach_reasoning_signature(reasoning, signature);
    }
}

fn attach_reasoning_signature(reasoning: &mut Reasoning, signature: String) {
    // Target the last UNSIGNED text slot: a signature never overwrites one
    // already recorded (signature strings cannot be merged, and replay
    // needs every one). With no unsigned slot — all signed, or no text at
    // all — the signature records on its own empty-text slot.
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

#[cfg(test)]
mod tests;

/// The aggregation laws, as properties (#2258 A5, rewritten to the
/// lifecycle vocabulary — not weakened).
#[cfg(test)]
mod property_tests;
