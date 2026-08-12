//! Accumulation of streamed events into the ordered parts of one assistant
//! choice — as **entities with a lifecycle**, not id-keyed event sequences.
//!
//! Every part runs open → mutate → close (review 84a43e9e root cause A):
//!
//! - **open** registers the part in `parts` once, fixing arrival order, and
//!   tracks it in an open-set keyed by its opaque [`StreamPartId`];
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
//! There is no interleaving-boundary machinery here. Wires that never
//! announce block boundaries have their adapters *synthesize* the end
//! events at the boundaries they already detect (openai-agents' "the
//! adapter synthesizes canonical events" pattern) — one grammar, stated
//! once, instead of per-wire lifecycle re-derivation.
//!
//! Reference designs: vercel-ai-sdk's `stream-text` accumulator (`-start`
//! registers and pushes the handle once, `-delta` mutates through it,
//! `-end` deletes it) and pydantic-ai's `_stop_tracking_vendor_id`.

use std::collections::{HashMap, HashSet};

use crate::completion::CompletionError;
use crate::message::{AssistantContent, Reasoning, ReasoningContent, ToolCall, ToolFunction};
use crate::streaming::identity::{StreamPartId, SyntheticIds, WireId};
use crate::streaming::{ToolInputEnd, UnparseableToolInput};

/// Accumulates the streamed parts of one assistant choice, in arrival order.
///
/// Owns every aggregation decision `StreamingCompletionResponse` makes.
/// Consumers feed normalized events and call [`PartsAccumulator::finish`]
/// once the stream ends.
pub(crate) struct PartsAccumulator {
    /// The choice's parts, in arrival order — the single accumulated state;
    /// [`PartsAccumulator::finish`] derives the choice from it directly.
    parts: Vec<AssistantContent>,
    /// Open reasoning entities: key → index in `parts`. Invariant: every
    /// mapped index holds an `AssistantContent::Reasoning` part.
    open_reasoning: HashMap<StreamPartId, usize>,
    /// Finished reasoning entities: key → index of the latest finished part
    /// under that key. Repeated ends are no-ops; a trailing signature
    /// (`reasoning_end` with no restatement) attaches here — the block that
    /// holds the chain-of-thought, wherever it sits.
    finished_reasoning: HashMap<StreamPartId, usize>,
    /// Text-block identity → index in `parts`: a `TextStart` whose key was
    /// already seen reactivates that block, so a wire item's text keeps
    /// collapsing across interleaved output. Invariant: every mapped index
    /// holds an `AssistantContent::Text` part.
    text_ids: HashMap<StreamPartId, usize>,
    /// The block receiving bare text deltas and metadata, until a text
    /// start/end switches it.
    active_text: Option<StreamPartId>,
    /// Identity announced by the latest `TextStart` whose block has not yet
    /// opened; blocks open lazily on the first delta or metadata so a
    /// content-less `TextStart` never leaves an empty text part behind.
    pending_text_id: Option<StreamPartId>,
    /// Mints keys for blocks a bare `Message` opens on wires that never
    /// announce text boundaries. Purely internal bookkeeping: text parts
    /// carry no id in the aggregated choice.
    minted_text_ids: SyntheticIds,
    /// Tool calls under delta assembly, keyed by the fragment key, in start
    /// order.
    open_tool_inputs: Vec<OpenToolInput>,
    /// Finished tool entities — populated by **every** finalization route
    /// (fragment-assembled ends, authoritative-payload ends, whole-call
    /// adoption), so a repeated end cannot duplicate a call whatever payload
    /// it carries (84a43e9e #1). Cleared for a key when fragments reopen it.
    finished_tools: HashSet<StreamPartId>,
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
    id: StreamPartId,
    /// Rig-minted correlation id, created when the call opens.
    internal_call_id: String,
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

impl Default for PartsAccumulator {
    fn default() -> Self {
        Self {
            parts: Vec::new(),
            open_reasoning: HashMap::new(),
            finished_reasoning: HashMap::new(),
            text_ids: HashMap::new(),
            active_text: None,
            pending_text_id: None,
            minted_text_ids: SyntheticIds::text(),
            open_tool_inputs: Vec::new(),
            finished_tools: HashSet::new(),
            saw_tool_call: false,
        }
    }
}

impl PartsAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    // --- text lifecycle -------------------------------------------------

    /// Append streamed text to the active text block, opening one (under a
    /// minted key) if none is active — the lenient bare-delta rule.
    pub(crate) fn text_delta(&mut self, text: &str) {
        let index = self.ensure_text_block();
        if let Some(AssistantContent::Text(existing)) = self.parts.get_mut(index) {
            existing.text.push_str(text);
        }
    }

    /// Open (or reactivate) the text block identified by `id`.
    ///
    /// A previously seen key reactivates its block — a wire item's text
    /// keeps collapsing across interleaved output. An unseen key closes the
    /// active block and opens the new one lazily, on its first delta or
    /// metadata.
    pub(crate) fn text_start(
        &mut self,
        id: &StreamPartId,
        additional_params: Option<crate::message::AdditionalParams>,
    ) {
        match self.text_ids.get(id) {
            Some(_) => {
                self.active_text = Some(id.clone());
                self.pending_text_id = None;
            }
            None => {
                self.active_text = None;
                self.pending_text_id = Some(id.clone());
            }
        }
        if let Some(additional_params) = additional_params {
            self.text_additional_params(additional_params);
        }
    }

    /// Close the text block identified by `id`: later bare deltas open a
    /// fresh block instead of extending this one. (A later `TextStart` with
    /// the same key still reactivates it — reactivation is the keyed
    /// collapse, and it is explicit.)
    pub(crate) fn text_end(&mut self, id: &StreamPartId) {
        if self.active_text.as_ref() == Some(id) {
            self.active_text = None;
        }
        if self.pending_text_id.as_ref() == Some(id) {
            self.pending_text_id = None;
        }
    }

    /// Merge provider metadata into the active text block, opening an empty
    /// block if none is active.
    ///
    /// [`crate::message::AdditionalParams`] is non-empty by construction, so
    /// there is no empty-carrier case to filter: an arriving params value is
    /// always data, and the stored-params invariant (`None` or data) holds by
    /// type.
    pub(crate) fn text_additional_params(
        &mut self,
        additional_params: crate::message::AdditionalParams,
    ) {
        let index = self.ensure_text_block();
        let Some(AssistantContent::Text(text)) = self.parts.get_mut(index) else {
            return;
        };
        match text.additional_params.as_mut() {
            Some(existing) => existing.merge(additional_params),
            None => text.additional_params = Some(additional_params),
        }
    }

    /// Index of the active text block, opening one if none is active.
    fn ensure_text_block(&mut self) -> usize {
        if let Some(active) = &self.active_text
            && let Some(&index) = self.text_ids.get(active)
        {
            return index;
        }
        let id = self
            .pending_text_id
            .take()
            .unwrap_or_else(|| self.minted_text_ids.mint());
        self.parts.push(AssistantContent::text(""));
        let index = self.parts.len() - 1;
        self.text_ids.insert(id.clone(), index);
        self.active_text = Some(id);
        index
    }

    // --- reasoning lifecycle --------------------------------------------

    /// Open the reasoning entity for `id`, registering an empty part at the
    /// current arrival position. A start for an already-open key is a no-op
    /// (returns `false`); a start for a finished key opens a **new** part
    /// (key reuse) — the `true` return tells the stream handler to mint a
    /// fresh public correlator, so the new part never inherits the finished
    /// part's identity.
    pub(crate) fn reasoning_start(
        &mut self,
        id: &StreamPartId,
        provider_id: Option<&WireId>,
    ) -> bool {
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
    pub(crate) fn reasoning_delta(
        &mut self,
        id: &StreamPartId,
        provider_id: Option<&WireId>,
        text: &str,
    ) {
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
    pub(crate) fn reasoning_end(
        &mut self,
        id: &StreamPartId,
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
                    restatement.id = open.id.clone();
                }
                *part = AssistantContent::Reasoning(restatement);
            }
            if let Some(signature) = signature {
                attach_signature(self.parts.get_mut(index), signature);
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
                    attach_signature(self.parts.get_mut(index), signature);
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
        id: &StreamPartId,
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
    fn finish_signature_only(&mut self, id: &StreamPartId, signature: String) -> Option<Reasoning> {
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
        id: &StreamPartId,
        provider_id: Option<&WireId>,
        content: Vec<ReasoningContent>,
    ) {
        let index = self.push_reasoning_part(Reasoning {
            id: provider_id.map(|id| id.as_str().to_owned()),
            content,
        });
        self.open_reasoning.insert(id.clone(), index);
    }

    /// Register a new reasoning part at the current arrival position.
    ///
    /// Like a completed tool call, a NEW reasoning part is a boundary for
    /// the anonymous active text block (deltas to an already-open reasoning
    /// part are not — they don't change the part order).
    fn push_reasoning_part(&mut self, reasoning: Reasoning) -> usize {
        self.active_text = None;
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

    /// Append a tool call the wire delivered whole, reconciling it with an
    /// open delta assembly of the same key. Returns the internal correlation
    /// id the completed call must be published under.
    ///
    /// The assembly's internal id is **adopted**, never replaced
    /// ([`StreamedAssistantContent::ToolCall`](crate::streaming::StreamedAssistantContent::ToolCall)
    /// promises delta/completion correlation); `minted_internal_call_id` is
    /// returned only when there was nothing to adopt. Adoption finishes the
    /// entity, so a trailing end event for the key finalizes nothing.
    pub(crate) fn tool_call(
        &mut self,
        id: &StreamPartId,
        tool_call: ToolCall,
        minted_internal_call_id: String,
    ) -> String {
        let position = self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == *id)
            .or_else(|| {
                // A wire that fragmented under a minted key and restates the
                // call under its late-arriving wire key: minted-vs-wire is
                // not a veto — with exactly one minted assembly open, the
                // restatement CAN be that assembly completed. More than one
                // is ambiguous; don't guess. And cardinality alone is not
                // evidence: adoption also demands the whole call restate the
                // assembly — same tool name, arguments covering whatever
                // fragments streamed. An unrelated id-bearing call adopting
                // the assembly marked it finished and silently dropped its
                // streamed arguments when its own end arrived.
                if id.is_minted() {
                    return None;
                }
                let mut minted = self
                    .open_tool_inputs
                    .iter()
                    .enumerate()
                    .filter(|(_, input)| input.id.is_minted());
                let candidate = match (minted.next(), minted.next()) {
                    (Some(candidate), None) => candidate,
                    _ => return None,
                };
                let (index, input) = candidate;
                // An assembly opened by args-only deltas never saw a name
                // (`ensure_open_tool_input` records ""), so an empty name
                // is no evidence against the restatement — only a
                // *different* established name vetoes.
                let restates = (input.name.is_empty() || input.name == tool_call.function.name)
                    && fragments_covered_by(input.buffer.as_deref(), &tool_call.function.arguments);
                restates.then_some(index)
            });
        let adopted = position.map(|index| {
            let input = self.open_tool_inputs.remove(index);
            self.finished_tools.insert(input.id);
            input.internal_call_id
        });
        self.finished_tools.insert(id.clone());
        self.push_tool_call(tool_call);
        adopted.unwrap_or(minted_internal_call_id)
    }

    /// Append a completed tool call as the next part.
    ///
    /// A completed call is a part boundary for the ACTIVE text block: text
    /// around it must stay two blocks in arrival order. (A keyed block still
    /// reactivates through its own `TextStart` — the keyed collapse is
    /// explicit; only the anonymous active-block cursor resets.)
    fn push_tool_call(&mut self, tool_call: ToolCall) {
        self.saw_tool_call = true;
        self.active_text = None;
        self.parts.push(AssistantContent::ToolCall(tool_call));
    }

    /// Whether any completed tool call was recorded on this stream.
    pub(crate) fn saw_tool_call(&self) -> bool {
        self.saw_tool_call
    }

    /// Record a streamed tool name fragment, opening the call if `id` has no
    /// open call. Returns the call's minted internal id.
    ///
    /// A later non-empty name replaces the recorded one (OpenAI-compatible
    /// wire semantics: the established name is the last non-empty value).
    pub(crate) fn tool_name_delta(&mut self, id: &StreamPartId, name: &str) -> String {
        let index = self.ensure_open_tool_input(id);
        match self.open_tool_inputs.get_mut(index) {
            Some(input) => {
                // Last-*non-empty* semantics: an empty fragment must not
                // erase an established name, or finalization would drop the
                // call as nameless.
                if !name.is_empty() {
                    name.clone_into(&mut input.name);
                }
                input.internal_call_id.clone()
            }
            // Unreachable (`ensure` returns a live index); degrade to a
            // fresh id rather than panic.
            None => crate::id::generate(),
        }
    }

    /// Append a streamed argument fragment to the call's buffer, opening the
    /// call if `id` has no open call. Returns the call's minted internal id.
    pub(crate) fn tool_args_delta(&mut self, id: &StreamPartId, fragment: &str) -> String {
        let index = self.ensure_open_tool_input(id);
        match self.open_tool_inputs.get_mut(index) {
            Some(input) => {
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
                input.internal_call_id.clone()
            }
            // Unreachable (`ensure` returns a live index); degrade to a
            // fresh id rather than panic.
            None => crate::id::generate(),
        }
    }

    /// Close a streamed tool call's input and finalize it into a completed
    /// call.
    ///
    /// Returns the completed call and its internal id, `None` when the call
    /// is dropped (nameless, unparseable under [`UnparseableToolInput::Drop`],
    /// or a repeated end for an already-finished entity — whatever payload
    /// it carries), or an error item under [`UnparseableToolInput::Error`].
    /// Authoritative fields on the end event supersede the assembled state.
    /// An end with no open call and no finished entity opens and completes
    /// one from the event alone (the replay path).
    pub(crate) fn tool_input_end(
        &mut self,
        end: ToolInputEnd,
    ) -> Result<Option<(ToolCall, String)>, CompletionError> {
        let position = self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == end.id);
        // The entity already finished — by its own end, or by whole-call
        // adoption. A repeated end must finalize nothing, INCLUDING one
        // carrying an authoritative name/arguments payload: pre-84a43e9e-#1
        // that payload route bypassed the guard and duplicated the call.
        if position.is_none() && self.finished_tools.contains(&end.id) {
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
                self.finished_tools.insert(end.id);
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
                                self.finished_tools.insert(end.id);
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
        let wire_tool_id = end.tool_id.map(WireId::into_string).or(opened_wire_id);
        let provider =
            crate::message::ProviderCallId::from_optional_wire(end.call_id, wire_tool_id);
        let id = crate::message::ToolCallId::for_provider(provider.as_ref());
        let tool_call = ToolCall {
            id,
            provider,
            function: ToolFunction { name, arguments },
            signature: end.signature,
            additional_params: end.additional_params,
        };
        // Every finalization route records the finished entity.
        self.finished_tools.insert(end.id);
        self.push_tool_call(tool_call.clone());
        Ok(Some((tool_call, internal_call_id)))
    }

    /// Index of the open call for `id`, opening one if none exists.
    fn ensure_open_tool_input(&mut self, id: &StreamPartId) -> usize {
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
                    internal_call_id: crate::id::generate(),
                    name: String::new(),
                    buffer: None,
                    overflowed: false,
                });
                self.open_tool_inputs.len() - 1
            }
        }
    }

    // --- lifecycle end --------------------------------------------------

    /// Consume the accumulated state into the ordered choice parts.
    ///
    /// Never empty: a stream that produced no content yields the single
    /// empty text part the aggregated choice has always defaulted to. Empty
    /// text parts that never received content are dropped. Calls still open
    /// at stream end never fully arrived and drop, per the settled
    /// truncation contract; reasoning still open is kept as-is (its deltas
    /// are real content).
    pub(crate) fn finish(&mut self) -> Vec<AssistantContent> {
        let parts: Vec<AssistantContent> = std::mem::take(&mut self.parts)
            .into_iter()
            .filter(|part| match part {
                // A lazily opened block that got content survives; the
                // never-fed placeholder does not.
                AssistantContent::Text(text) => {
                    !(text.text.is_empty() && text.additional_params.is_none())
                }
                _ => true,
            })
            .collect();
        self.open_reasoning.clear();
        self.finished_reasoning.clear();
        self.text_ids.clear();
        self.active_text = None;
        self.pending_text_id = None;
        self.minted_text_ids = SyntheticIds::text();
        self.open_tool_inputs.clear();
        self.finished_tools.clear();
        self.saw_tool_call = false;
        // A turn that streamed no parts finishes empty. The fabricated
        // empty-text part that used to be pushed here existed only to satisfy
        // the non-empty content type, and it reached history and the wire as if
        // the model had emitted it.
        parts
    }
}

/// Attach a signature to a reasoning part's last text content (pushing a
/// signature-only content item when the part has no text slot).
/// Whether an assembly's streamed argument fragments are covered by a whole
/// call's parsed arguments: no fragments at all, or a lenient parse of the
/// buffer that the arguments subsume (equal, or an object whose entries all
/// appear). Unparseable partial fragments are not evidence of restatement.
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

fn attach_signature(part: Option<&mut AssistantContent>, signature: String) {
    if let Some(AssistantContent::Reasoning(reasoning)) = part {
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
mod tests {
    use super::*;

    /// Test-side key syntax: legacy minted renderings decode to minted
    /// keys; anything else is wire-derived.
    fn pid(id: &str) -> StreamPartId {
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
        StreamPartId::wire(id)
    }

    /// Provider handle matching the key syntax: wire-shaped ids carry
    /// themselves; minted renderings carry none.
    fn wid(id: &str) -> Option<WireId> {
        pid(id).wire_str().and_then(|_| WireId::new(id))
    }

    fn full(id: &str, content: ReasoningContent) -> Reasoning {
        Reasoning {
            id: wid(id).map(|id| id.into_string()),
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

    fn end(id: &str, mode: UnparseableToolInput) -> ToolInputEnd {
        ToolInputEnd::new(pid(id), mode)
    }

    fn call_named(id: &str, name: &str) -> ToolCall {
        ToolCall::from_wire(
            id,
            crate::message::ToolFunction {
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            },
        )
    }

    // --- reasoning lifecycle ---

    /// An end's authoritative restatement supersedes the open part's delta
    /// accumulation (pydantic-ai `_replace_part` semantics).
    #[test]
    fn an_end_restatement_replaces_its_delta_accumulation() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "partial ");
        accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "thought");
        accumulator.reasoning_end(
            &pid("rs_1"),
            Some(full("rs_1", reasoning_text("the complete chain"))),
            None,
        );

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["the complete chain"]);
        assert_eq!(parts.len(), 1);
    }

    /// A restatement that doesn't restate the durable handle must not erase
    /// the one established at part-open; a restatement that does carries
    /// its own.
    #[test]
    fn an_id_less_restatement_keeps_the_open_parts_provider_handle() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "partial");
        let completed = accumulator
            .reasoning_end(
                &pid("rs_1"),
                Some(Reasoning {
                    id: None,
                    content: vec![reasoning_text("the complete chain")],
                }),
                None,
            )
            .expect("open part completes");
        assert_eq!(
            completed.id.as_deref(),
            Some("rs_1"),
            "an absent restatement id falls back to the handle set at open"
        );

        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_2"), wid("rs_2").as_ref(), "partial");
        let completed = accumulator
            .reasoning_end(
                &pid("rs_2"),
                Some(full("rs_other", reasoning_text("restated"))),
                None,
            )
            .expect("open part completes");
        assert_eq!(
            completed.id.as_deref(),
            Some("rs_other"),
            "a restated handle wins over the one set at open"
        );
    }

    /// An open part keeps collapsing across interleaved output: with no end
    /// synthesized, later deltas and the restatement extend/replace the
    /// SAME part (the wire-key Responses shape).
    #[test]
    fn an_open_part_collapses_across_interleaved_output() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "thinking");
        accumulator.tool_call(
            &pid("call_1"),
            call_named("call_1", "probe"),
            "internal-probe".to_owned(),
        );
        accumulator.reasoning_end(
            &pid("rs_1"),
            Some(full("rs_1", reasoning_text("full reasoning"))),
            None,
        );

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["full reasoning"]);
        assert_eq!(
            parts.len(),
            2,
            "reasoning replaced in place beside the call"
        );
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(_))
        ));
    }

    /// Two signature-bearing segments under a per-stream constant key
    /// (gemini REST/Interactions, cohere, ollama all key reasoning by one
    /// minted constant): the second signature must land on a DISTINCT part.
    /// Signatures cannot merge — overwriting the first would replay with
    /// only the last signature and fail `MISSING_THOUGHT_SIGNATURE`.
    #[test]
    fn a_second_signature_under_a_constant_key_records_a_distinct_part() {
        let mut accumulator = PartsAccumulator::new();
        let key = pid("reasoning-0");
        accumulator.reasoning_delta(&key, None, "thought A");
        accumulator.reasoning_end(&key, None, Some("sig_A".to_string()));
        // A signature-only end for the already-finished constant key: the
        // wire signed a second segment with nothing new streamed to sign.
        accumulator.reasoning_end(&key, None, Some("sig_B".to_string()));

        let parts = accumulator.finish();
        let signed: Vec<(String, Option<String>)> = parts
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .flat_map(|reasoning| {
                reasoning.content.iter().map(|content| match content {
                    ReasoningContent::Text { text, signature } => (text.clone(), signature.clone()),
                    other => (format!("{other:?}"), None),
                })
            })
            .collect();
        assert_eq!(
            signed,
            vec![
                ("thought A".to_string(), Some("sig_A".to_string())),
                (String::new(), Some("sig_B".to_string())),
            ],
            "both signatures survive, each on its own slot in its own part"
        );
        let reasoning_parts = parts
            .iter()
            .filter(|part| matches!(part, AssistantContent::Reasoning(_)))
            .count();
        assert_eq!(reasoning_parts, 2, "the second signature is a sibling part");
    }

    /// A synthesized end splits a constant-key wire's blocks: a delta after
    /// the end opens a NEW part (key reuse), never merging backwards.
    #[test]
    fn a_delta_after_the_end_opens_a_fresh_part() {
        let mut accumulator = PartsAccumulator::new();
        let key = pid("reasoning-0");
        accumulator.reasoning_delta(&key, None, "A");
        accumulator.reasoning_end(&key, None, None);
        accumulator.text_delta("visible");
        accumulator.reasoning_delta(&key, None, "B");

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
        assert!(matches!(
            parts.get(1),
            Some(AssistantContent::Text(text)) if text.text == "visible"
        ));
    }

    /// Same-key whole blocks after the entity finished are siblings: every
    /// part survives, in arrival order (the Responses multi-part item).
    #[test]
    fn same_key_whole_blocks_are_siblings_and_all_survive() {
        let mut accumulator = PartsAccumulator::new();
        for content in [
            summary("s1"),
            summary("s2"),
            reasoning_text("visible"),
            ReasoningContent::Encrypted("enc".to_owned()),
        ] {
            accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", content)), None);
        }

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "visible", "enc"]);
    }

    /// Deltas then the item's multi-part done block: the first restatement
    /// supersedes the delta accumulation, the rest append as siblings.
    #[test]
    fn deltas_then_sibling_whole_blocks_keep_each_part_once() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "s1");
        for content in [
            summary("s1"),
            summary("s2"),
            ReasoningContent::Encrypted("enc".to_owned()),
        ] {
            accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", content)), None);
        }

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "enc"]);
    }

    #[test]
    fn distinct_keys_never_interact() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "first item deltas");
        accumulator.reasoning_end(
            &pid("rs_2"),
            Some(full("rs_2", reasoning_text("a different item"))),
            None,
        );

        let parts = accumulator.finish();
        assert_eq!(
            reasoning_texts(&parts),
            vec!["first item deltas", "a different item"]
        );
    }

    /// A trailing signature-only end signs the block that HOLDS the
    /// chain-of-thought — wherever its arrival position was — never an
    /// empty sibling (#2258 B4).
    #[test]
    fn a_trailing_signature_signs_the_finished_block() {
        let mut accumulator = PartsAccumulator::new();
        let key = pid("reasoning-0");
        accumulator.reasoning_delta(&key, None, "the chain");
        accumulator.reasoning_end(&key, None, None);
        accumulator.text_delta("answer");
        accumulator.reasoning_end(&key, None, Some("sig".to_owned()));

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 2);
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(reasoning))
                if matches!(reasoning.content.first(), Some(ReasoningContent::Text { text, signature })
                    if text == "the chain" && signature.as_deref() == Some("sig"))
        ));
    }

    /// A signature closing an open block signs the accumulated deltas.
    #[test]
    fn a_signature_end_signs_the_open_block() {
        let mut accumulator = PartsAccumulator::new();
        let key = pid("reasoning-0");
        accumulator.reasoning_delta(&key, None, "the chain");
        accumulator.reasoning_end(&key, None, Some("sig".to_owned()));

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 1);
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(reasoning))
                if matches!(reasoning.content.first(), Some(ReasoningContent::Text { signature, .. })
                    if signature.as_deref() == Some("sig"))
        ));
    }

    /// A signature with nothing streamed records a signature-only part —
    /// replay-required provider state survives.
    #[test]
    fn a_signature_only_stream_records_a_signature_only_part() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_end(&pid("reasoning-0"), None, Some("sig".to_owned()));
        let parts = accumulator.finish();
        assert_eq!(parts.len(), 1);
    }

    /// A bare repeated end is a no-op: idempotence belongs to the entity.
    #[test]
    fn repeated_bare_ends_are_no_ops() {
        let mut accumulator = PartsAccumulator::new();
        let key = pid("reasoning-0");
        accumulator.reasoning_delta(&key, None, "A");
        assert!(accumulator.reasoning_end(&key, None, None).is_some());
        assert!(accumulator.reasoning_end(&key, None, None).is_none());
        assert!(accumulator.reasoning_end(&key, None, None).is_none());
        assert_eq!(accumulator.finish().len(), 1);
    }

    // --- text lifecycle ---

    #[test]
    fn text_and_reasoning_interleave_in_arrival_order() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_delta("intro");
        accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", summary("thinking"))), None);
        accumulator.text_start(&pid("msg_2"), None);
        accumulator.text_delta("out");
        accumulator.text_delta("ro");

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 3);
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Text(text)) if text.text == "intro"
        ));
        assert!(matches!(parts.get(1), Some(AssistantContent::Reasoning(_))));
        assert!(matches!(
            parts.get(2),
            Some(AssistantContent::Text(text)) if text.text == "outro"
        ));
    }

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

    /// A `TextStart` whose key was already seen reactivates that block
    /// across interleaved output instead of opening a duplicate part.
    #[test]
    fn a_seen_text_start_id_reactivates_its_block() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.text_delta("collapsing ");
        accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", summary("thinking"))), None);
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.text_delta("text");

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 2, "one text part, one reasoning part");
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Text(text)) if text.text == "collapsing text"
        ));
    }

    /// A `TextStart` that never receives content leaves no empty part.
    #[test]
    fn a_content_less_text_start_leaves_no_empty_part() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", summary("thinking"))), None);

        let parts = accumulator.finish();
        assert_eq!(parts.len(), 1);
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(_))
        ));
    }

    /// An explicit `TextEnd` closes the block: later bare deltas open a
    /// fresh part.
    #[test]
    fn text_end_closes_the_block_for_bare_deltas() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.text_start(&pid("msg_1"), None);
        accumulator.text_delta("first");
        accumulator.text_end(&pid("msg_1"));
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

    #[test]
    fn finish_on_an_empty_stream_yields_no_parts() {
        let mut accumulator = PartsAccumulator::new();
        let parts = accumulator.finish();
        // Was one fabricated empty-text part, which the old non-empty content
        // type forced and which was indistinguishable on the wire from a real
        // empty text block.
        assert!(parts.is_empty());
    }

    // --- tool-call lifecycle (the settled semantics) ---

    #[test]
    fn fragments_assemble_into_a_completed_tool_call_with_a_stable_internal_id() {
        let mut accumulator = PartsAccumulator::new();
        let first = accumulator.tool_name_delta(&pid("call_1"), "get_weather");
        let second = accumulator.tool_args_delta(&pid("call_1"), "{\"location\":");
        accumulator.tool_args_delta(&pid("call_1"), "\"Paris\"}");
        assert_eq!(first, second);

        let (tool_call, internal_call_id) = accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("call must finalize");
        assert_eq!(internal_call_id, first, "internal id is minted at start");
        assert_eq!(tool_call.id, "call_1");
        assert_eq!(tool_call.function.name, "get_weather");
        assert!(accumulator.saw_tool_call());
    }

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

    #[test]
    fn a_call_with_no_streamed_arguments_is_a_parameterless_invocation() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "ping");
        let (tool_call, _) = accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("parameterless calls are preserved");
        assert_eq!(tool_call.function.arguments, serde_json::json!({}));
    }

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
        // The authoritative correlator drives rig's id; the wire-derived
        // assembly key rides along as the provider item id.
        assert_eq!(tool_call.id, "call_abc");
        let provider = tool_call.provider.as_ref().expect("provider ids are kept");
        assert_eq!(provider.call_id, "call_abc");
        assert_eq!(provider.item_id.as_deref(), Some("fc_1"));
        assert_eq!(tool_call.function.name, "final_name");
    }

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

    /// 84a43e9e #1, closed structurally: a REPEATED end for a finished
    /// entity finalizes nothing — even when it carries the authoritative
    /// name/arguments payload that used to bypass the guard and duplicate
    /// the call. The finished-set is populated by every route.
    #[test]
    fn a_repeated_end_with_an_authoritative_payload_is_a_no_op() {
        let mut accumulator = PartsAccumulator::new();
        let mut done = end("fc_1", UnparseableToolInput::Drop);
        done.name = Some("add".to_owned());
        done.arguments = Some(serde_json::json!({"x": 2}));
        accumulator
            .tool_input_end(done.clone())
            .expect("no error")
            .expect("finalizes once");
        assert!(
            accumulator
                .tool_input_end(done)
                .expect("no error")
                .is_none(),
            "a repeated authoritative end must not duplicate the call"
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

    /// A drop is a finalization: a truncated call dismissed under `Drop`
    /// must not be resurrected by a later payload-bearing end for the same
    /// key — that end is the same repeated-end defect the finished-set
    /// guard exists for, arriving after a drop instead of a success.
    #[test]
    fn a_dropped_truncated_call_is_not_resurrected_by_a_payload_bearing_end() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "get_weather");
        accumulator.tool_args_delta(&pid("call_1"), "{\"loc\":");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none(),
            "truncated arguments drop"
        );
        let mut retry = end("call_1", UnparseableToolInput::Drop);
        retry.name = Some("get_weather".to_owned());
        retry.arguments = Some(serde_json::json!({"loc": "Paris"}));
        assert!(
            accumulator
                .tool_input_end(retry)
                .expect("no error")
                .is_none(),
            "the dropped entity is finished; a later payload must not fabricate a phantom call"
        );
        assert!(!accumulator.saw_tool_call());
    }

    /// Same for the nameless drop: the entity finished when it was
    /// dismissed, so a later end supplying the missing name is repeated,
    /// not completing.
    #[test]
    fn a_dropped_nameless_call_is_not_resurrected_by_a_named_end() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_args_delta(&pid("call_1"), "{\"y\":1}");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none(),
            "nameless call drops"
        );
        let mut retry = end("call_1", UnparseableToolInput::Drop);
        retry.name = Some("late_name".to_owned());
        retry.arguments = Some(serde_json::json!({"y": 1}));
        assert!(
            accumulator
                .tool_input_end(retry)
                .expect("no error")
                .is_none(),
            "the dropped entity is finished; a late name must not fabricate a phantom call"
        );
        assert!(!accumulator.saw_tool_call());
    }

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
    }

    /// The accumulation bound applies to the FIRST fragment too: a single
    /// oversized fragment trips overflow (and thereby the
    /// unparseable-input policy at finalization) exactly like the same
    /// payload split across fragments.
    #[test]
    fn an_oversized_first_fragment_trips_the_accumulation_bound() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "big");
        let oversized = "x".repeat(MAX_TOOL_INPUT_BYTES + 1);
        accumulator.tool_args_delta(&pid("call_1"), &oversized);
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none(),
            "overflow forces the unparseable path; the call must not finalize"
        );
        assert!(!accumulator.saw_tool_call());
    }

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

    #[test]
    fn minted_keys_keep_id_less_parallel_calls_distinct() {
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
        // Id-less calls mint distinct correlation handles and record that
        // the provider issued nothing — never a shared empty sentinel.
        assert!(!first.id.as_str().is_empty());
        assert!(!second.id.as_str().is_empty());
        assert_ne!(first.id, second.id);
        assert!(first.provider.is_none());
        assert!(second.provider.is_none());
        assert_eq!(
            first.function.arguments,
            serde_json::json!({"city": "Tokyo"})
        );
        assert_eq!(
            second.function.arguments,
            serde_json::json!({"zone": "UTC"})
        );
    }

    #[test]
    fn finish_discards_calls_still_open_at_stream_end() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("call_1"), "ping");
        accumulator.tool_args_delta(&pid("call_1"), "{\"x\":");
        let parts = accumulator.finish();
        // Discarding the only (incomplete) call leaves the turn with nothing,
        // which is now representable instead of being padded with an empty
        // text part.
        assert!(parts.is_empty());
        assert!(!accumulator.saw_tool_call());
    }

    #[test]
    fn the_tool_id_override_supersedes_the_assembly_key() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("tool-0"), "ping");
        let mut done = end("tool-0", UnparseableToolInput::Drop);
        done.tool_id = WireId::new("call_late");
        let (tool_call, _) = accumulator
            .tool_input_end(done)
            .expect("no error")
            .expect("finalizes");
        assert_eq!(tool_call.id, "call_late");
    }

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
        assert_eq!(adopted, published);
        assert_eq!(
            accumulator
                .finish()
                .iter()
                .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
                .count(),
            1
        );
    }

    #[test]
    fn an_end_after_a_full_call_for_the_same_key_is_a_no_op() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("tc1"), "add");
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

    #[test]
    fn fragments_reusing_a_finished_key_open_a_fresh_call() {
        let mut accumulator = PartsAccumulator::new();
        let first = accumulator.tool_name_delta(&pid("tc1"), "add");
        accumulator.tool_call(
            &pid("tc1"),
            call_named("tc1", "add"),
            "freshly-minted".to_owned(),
        );
        let second = accumulator.tool_name_delta(&pid("tc1"), "subtract");
        assert_ne!(second, first);
        accumulator.tool_args_delta(&pid("tc1"), "{\"y\":2}");
        let (tool_call, internal) = accumulator
            .tool_input_end(end("tc1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("the reused key's call finalizes");
        assert_eq!(internal, second);
        assert_eq!(tool_call.function.name, "subtract");
    }

    #[test]
    fn a_wire_restatement_adopts_the_single_open_minted_assembly() {
        let mut accumulator = PartsAccumulator::new();
        let published = accumulator.tool_name_delta(&pid("tool-0"), "add");
        accumulator.tool_args_delta(&pid("tool-0"), "{\"x\":1}");
        // A genuine restatement: same tool, arguments covering the
        // streamed fragments.
        let restated = ToolCall::from_wire(
            "call_late",
            crate::message::ToolFunction {
                name: "add".to_owned(),
                arguments: serde_json::json!({"x": 1}),
            },
        );
        let adopted =
            accumulator.tool_call(&pid("call_late"), restated, "freshly-minted".to_owned());
        assert_eq!(adopted, published);
        assert!(
            accumulator
                .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none()
        );
    }

    /// A gateway that fragments arguments WITHOUT ever sending a name
    /// delta leaves the assembly nameless (`ensure_open_tool_input`
    /// records `""`). An empty name is no evidence against the
    /// restatement — demanding name equality published the whole call
    /// under a fresh minted id (breaking delta correlation) and left the
    /// assembly to finalize as a duplicate.
    #[test]
    fn a_wire_restatement_adopts_a_nameless_args_only_assembly() {
        let mut accumulator = PartsAccumulator::new();
        let published = accumulator.tool_args_delta(&pid("tool-0"), "{\"x\":1}");
        let restated = ToolCall::from_wire(
            "call_late",
            crate::message::ToolFunction {
                name: "add".to_owned(),
                arguments: serde_json::json!({"x": 1}),
            },
        );
        let adopted =
            accumulator.tool_call(&pid("call_late"), restated, "freshly-minted".to_owned());
        assert_eq!(adopted, published);
        assert!(
            accumulator
                .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none(),
            "the adopted assembly must not finalize a second time"
        );
    }

    /// A buffer still holding the literal `null` placeholder (the gateway
    /// shape `tool_args_delta` documents) streamed no real arguments:
    /// any restatement covers it vacuously. Strict subset comparison
    /// rejected it (`{...} != null`) and published a duplicate.
    #[test]
    fn a_wire_restatement_adopts_an_assembly_holding_the_null_placeholder() {
        let mut accumulator = PartsAccumulator::new();
        let published = accumulator.tool_name_delta(&pid("tool-0"), "add");
        accumulator.tool_args_delta(&pid("tool-0"), "null");
        let restated = ToolCall::from_wire(
            "call_late",
            crate::message::ToolFunction {
                name: "add".to_owned(),
                arguments: serde_json::json!({"x": 1}),
            },
        );
        let adopted =
            accumulator.tool_call(&pid("call_late"), restated, "freshly-minted".to_owned());
        assert_eq!(adopted, published);
        assert!(
            accumulator
                .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none()
        );
    }

    /// A wire-keyed whole call must not steal an open minted assembly it
    /// doesn't evidently restate: a different tool name (or arguments that
    /// don't cover the streamed fragments) is an UNRELATED call. The
    /// assembly stays open for its own end — its streamed arguments were
    /// silently lost when adoption keyed on cardinality alone.
    #[test]
    fn an_unrelated_wire_keyed_call_does_not_steal_the_open_assembly() {
        let mut accumulator = PartsAccumulator::new();
        let published = accumulator.tool_name_delta(&pid("tool-0"), "get_weather");
        accumulator.tool_args_delta(&pid("tool-0"), "{\"city\":\"Paris\"}");

        let unrelated = ToolCall::from_wire(
            "call_other",
            crate::message::ToolFunction {
                name: "get_time".to_owned(),
                arguments: serde_json::json!({"zone": "UTC"}),
            },
        );
        let adopted =
            accumulator.tool_call(&pid("call_other"), unrelated, "fresh-internal".to_owned());
        assert_eq!(
            adopted, "fresh-internal",
            "an unrelated call has nothing to adopt"
        );

        // The assembly still finalizes from its own end, streamed args intact.
        let (weather, internal) = accumulator
            .tool_input_end(end("tool-0", UnparseableToolInput::Error))
            .expect("no error")
            .expect("the open assembly must finalize");
        assert_eq!(internal, published);
        assert_eq!(weather.function.name, "get_weather");
        assert_eq!(
            weather.function.arguments,
            serde_json::json!({"city": "Paris"})
        );

        let calls = accumulator
            .finish()
            .iter()
            .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
            .count();
        assert_eq!(calls, 2, "both calls survive as distinct parts");
    }

    /// Same-name adoption still demands the arguments cover the fragments:
    /// a same-tool call whose args contradict what streamed is a second
    /// invocation of that tool, not a restatement.
    #[test]
    fn a_same_name_call_with_foreign_arguments_does_not_adopt() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta(&pid("tool-0"), "add");
        accumulator.tool_args_delta(&pid("tool-0"), "{\"x\":1}");
        let second_invocation = ToolCall::from_wire(
            "call_other",
            crate::message::ToolFunction {
                name: "add".to_owned(),
                arguments: serde_json::json!({"x": 99}),
            },
        );
        let adopted = accumulator.tool_call(
            &pid("call_other"),
            second_invocation,
            "fresh-internal".to_owned(),
        );
        assert_eq!(adopted, "fresh-internal");
        assert!(
            accumulator
                .tool_input_end(end("tool-0", UnparseableToolInput::Error))
                .expect("no error")
                .is_some(),
            "the assembly still finalizes separately"
        );
    }

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
}

/// The aggregation laws, as properties (#2258 A5, rewritten to the
/// lifecycle vocabulary — not weakened).
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

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
        /// Text aggregation is invariant under fragmentation.
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

        /// Reasoning delta accumulation + end is invariant under
        /// fragmentation: accumulated delta content equals the completed
        /// part's payload (the langchain stream-lifecycle conservation law).
        #[test]
        fn reasoning_aggregate_is_fragmentation_invariant(
            payload in ".{1,60}",
            points in proptest::collection::vec(0usize..1000, 0..6),
        ) {
            let key = StreamPartId::wire("rs_1");
            let mut whole = PartsAccumulator::new();
            whole.reasoning_delta(&key, None, &payload);
            let completed_whole = whole.reasoning_end(&key, None, None);
            let mut split = PartsAccumulator::new();
            let mut pushed = false;
            for fragment in split_fragments(&payload, &points) {
                if !fragment.is_empty() {
                    split.reasoning_delta(&key, None, &fragment);
                    pushed = true;
                }
            }
            prop_assume!(pushed);
            let completed_split = split.reasoning_end(&key, None, None);
            prop_assert_eq!(completed_whole, completed_split);
            prop_assert_eq!(whole.finish(), split.finish());
        }

        /// Tool-argument assembly is invariant under fragmentation.
        #[test]
        fn tool_arguments_are_fragmentation_invariant(
            value in "[a-z]{0,20}",
            points in proptest::collection::vec(0usize..1000, 0..6),
        ) {
            let payload = format!("{{\"q\":\"{value}\"}}");
            let finalize = |fragments: &[String]| {
                let mut accumulator = PartsAccumulator::new();
                let key = StreamPartId::wire("call_1");
                accumulator.tool_name_delta(&key, "probe");
                for fragment in fragments {
                    accumulator.tool_args_delta(&key, fragment);
                }
                accumulator
                    .tool_input_end(ToolInputEnd::new(key, UnparseableToolInput::Drop))
                    .expect("no error")
                    .map(|(call, _)| call.function.arguments)
            };
            let whole = finalize(std::slice::from_ref(&payload));
            let split = finalize(&split_fragments(&payload, &points));
            prop_assert_eq!(whole, split);
        }

        /// Stale-end idempotence, on EVERY route: repeated ends — bare or
        /// carrying an authoritative payload — add nothing after the entity
        /// finished.
        #[test]
        fn stale_tool_input_ends_are_idempotent(
            extra_ends in 1usize..5,
            authoritative in proptest::bool::ANY,
        ) {
            let mut accumulator = PartsAccumulator::new();
            let key = StreamPartId::wire("call_1");
            accumulator.tool_name_delta(&key, "probe");
            accumulator.tool_args_delta(&key, "{}");
            accumulator
                .tool_input_end(ToolInputEnd::new(key.clone(), UnparseableToolInput::Drop))
                .expect("no error")
                .expect("finalizes");
            for _ in 0..extra_ends {
                let mut stale = ToolInputEnd::new(key.clone(), UnparseableToolInput::Drop);
                if authoritative {
                    stale.name = Some("probe".to_owned());
                    stale.arguments = Some(serde_json::json!({}));
                }
                prop_assert!(
                    accumulator
                        .tool_input_end(stale)
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

        /// Reasoning-end idempotence: bare repeated ends add nothing.
        #[test]
        fn stale_reasoning_ends_are_idempotent(extra_ends in 1usize..5) {
            let mut accumulator = PartsAccumulator::new();
            let key = StreamPartId::wire("rs_1");
            accumulator.reasoning_delta(&key, None, "thought");
            accumulator.reasoning_end(&key, None, None);
            for _ in 0..extra_ends {
                prop_assert!(accumulator.reasoning_end(&key, None, None).is_none());
            }
            prop_assert_eq!(accumulator.finish().len(), 1);
        }
    }
}
