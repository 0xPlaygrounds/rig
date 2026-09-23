//! Accumulates stream events into ordered assistant content without owning a
//! transport. Authoritative end payloads replace assembled fragments; repeated
//! tool ends are ignored until a new start or delta reopens the key.
//!
//! ```
//! use rig_core::streaming::{BlockAccumulator, BlockId, MintKind, StreamEvent};
//!
//! # fn example() -> Result<(), rig_core::error::ErrorReport> {
//! let mut accumulator = BlockAccumulator::new();
//! accumulator.apply(&StreamEvent::text(BlockId::minted(MintKind::Text, 0), "Hi"))?;
//! assert_eq!(accumulator.snapshot(), accumulator.finish());
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};

use crate::error::{ErrorDetail, ErrorKind, ErrorReport, MalformedToolInput};
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
    /// Accumulated parts in insertion order.
    parts: Vec<AssistantContent>,
    /// Open reasoning entities: key → index in `parts`. Invariant: every
    /// mapped index holds an `AssistantContent::Reasoning` part.
    open_reasoning: HashMap<BlockId, usize>,
    /// Index of the latest finished reasoning part for each key.
    /// Trailing signatures attach here unless the part is already signed.
    finished_reasoning: HashMap<BlockId, usize>,
    /// Text-block identity → index in `parts`: a delta whose key was
    /// already seen extends that block, so a wire item's text keeps
    /// collapsing across interleaved output. Invariant: every mapped index
    /// holds an `AssistantContent::Text` part.
    text_ids: HashMap<BlockId, usize>,
    /// Tool calls under delta assembly, keyed by the fragment key, in start
    /// order.
    open_tool_inputs: Vec<OpenToolInput>,
    /// Finalized tool keys, including adopted keys. Repeated ends cannot
    /// duplicate calls; new starts or deltas clear the corresponding key.
    finished_tools: HashSet<BlockId>,
    /// Whether any completed tool call was recorded; the streaming
    /// counterpart of the unary path's finish-reason reconciliation input.
    saw_tool_call: bool,
}

/// A tool call under fragment assembly.
struct OpenToolInput {
    /// Assembly key: every fragment of one call carries this key.
    id: BlockId,
    /// Tool name; a later non-empty name fragment replaces it.
    name: String,
    /// Concatenated arguments. `None` denotes a parameterless invocation.
    buffer: Option<String>,
    /// Whether a fragment exceeded the buffer limit. Finalization treats the
    /// assembled input as unparseable even if the retained bytes parse.
    overflowed: bool,
}

/// Maximum accumulated argument bytes per tool call.
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
    ) -> Result<Option<(BlockId, AssistantContent)>, ErrorReport> {
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
                    // Synthesized bare ends must not add completed-block events
                    // that the provider never emitted.
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

    /// Merges start metadata into the keyed text block.
    /// A start without metadata creates no part.
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

    /// Merges nonempty metadata into the keyed text block, creating it if absent.
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

    /// Opens an empty reasoning part unless the key is already open.
    /// Returns whether a new part was created, including reuse of finished keys.
    fn reasoning_start(&mut self, id: &BlockId, provider_id: Option<&str>) -> bool {
        if self.open_reasoning.contains_key(id) {
            return false;
        }
        self.open_fresh_reasoning(id, provider_id, Vec::new());
        true
    }

    /// Appends reasoning text, opening a new part if the key is not open.
    /// Finished parts are not reused.
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

    /// Finalizes reasoning, replacing open content with any restatement while
    /// retaining an omitted provider ID. Signatures attach to unsigned text.
    /// Finished keys accept new restatements as sibling parts; bare repeated
    /// ends do nothing. A signature creates a sibling if the finished part is
    /// already signed, or a signature-only part if the key is unseen.
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
                // An omitted restatement ID must not erase established identity.
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
                // Signatures cannot be merged or overwritten: replay needs each
                // one, including multiple signatures under a reused key.
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
                (None, None) => return None,
                (Some(restatement), signature) => {
                    return self.finish_restated(id, restatement, signature);
                }
            }
        }

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
            provider: None,
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
            provider: None,
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

    /// Finds an adoptable minted assembly for a provider-keyed completed call.
    /// Requires exactly one open minted assembly, a matching or absent name,
    /// and arguments covering the buffered fragments.
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
        // Missing names do not contradict a restatement; different names do.
        let restates = (input.name.is_empty() || input.name == name)
            && fragments_covered_by(input.buffer.as_deref(), arguments);
        restates.then_some(index)
    }

    /// Appends a completed tool call and records that the turn used tools.
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
        if let Some(input) = self.open_tool_inputs.get_mut(index) {
            // Enforce the bound even for the first fragment.
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

    /// Finalizes a tool call using authoritative end fields or assembled input.
    /// Returns the publication key and call, or `None` for dropped calls,
    /// incomplete probes, and repeated ends. Malformed complete input under
    /// [`UnparseableToolInput::Error`] returns an error with correlation metadata.
    /// An unseen key may finalize directly from the end payload.
    fn tool_end(
        &mut self,
        id: &BlockId,
        end: ToolCallEnd,
    ) -> Result<Option<(BlockId, ToolCall)>, ErrorReport> {
        let position = self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == *id);
        // An adopted call must retain its delta key for consumer correlation.
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
        // Payload-bearing repeated ends must not duplicate finalized calls.
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
        // Only provider-issued assembly keys can supply a replayable tool ID.
        let opened_wire_id = opened_id.wire_str().map(str::to_owned);
        // Empty end names must not erase names established by deltas.
        if let Some(final_name) = end.name.clone().filter(|final_name| !final_name.is_empty()) {
            name = final_name;
        }
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

        // Derive identity before parsing so malformed-input reports retain the
        // same correlation metadata as successful calls.
        let wire_tool_id = end.tool_id.or(opened_wire_id);
        let provider =
            crate::message::ProviderCallId::from_optional_wire(end.call_id, wire_tool_id);
        // No provider id: the block that assembled the call names it, so a
        // re-run of the same wire mints the same handle.
        let durable_id = end.durable_id.unwrap_or_else(|| {
            crate::message::ToolCallId::for_provider_or(
                provider.as_ref(),
                crate::message::ToolCallId::from_block(&published),
            )
        });

        let arguments = match end.arguments {
            // The wire's completed item is authoritative over assembly.
            Some(arguments) => arguments,
            None => match buffer {
                // No streamed arguments: a parameterless invocation.
                None => serde_json::Value::Object(serde_json::Map::new()),
                // Overflow must fail even if the retained prefix parses as JSON.
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
                            // Preserve input and identity for model-facing error
                            // feedback, while preventing repeated ends from retrying it.
                            UnparseableToolInput::Error => {
                                self.finished_tools.insert(id.clone());
                                self.finished_tools.insert(published.clone());
                                return Err(ErrorReport::new(
                                    ErrorKind::Response,
                                    format!(
                                        "tool call `{name}` arrived with malformed JSON input: {err}"
                                    ),
                                )
                                .with_detail(ErrorDetail::MalformedToolInput(
                                    MalformedToolInput {
                                        name,
                                        id: durable_id,
                                        provider,
                                        raw: buffer,
                                        error: err.to_string(),
                                    },
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
            id: durable_id,
            provider,
            function: ToolFunction { name, arguments },
            signature: end.signature,
            additional_params: end.additional_params,
        };
        // Both keys must reject repeated ends after adoption.
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

    /// Clones the accumulated choice without changing state. Omits unfinished
    /// tool calls and text with neither content nor metadata; retains open
    /// reasoning. Repeated snapshots without new events are equal.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.parts
            .iter()
            .filter(|part| Self::survives(part))
            .cloned()
            .collect()
    }

    /// Returns the same parts as [`Self::snapshot`] and resets all state.
    /// A stream with no content produces an empty vector.
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
            AssistantContent::ToolCall(_)
            | AssistantContent::Reasoning(_)
            | AssistantContent::Image(_) => true,
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
    // Replay needs every signature, so use the last unsigned text slot or
    // create a signature-only slot rather than overwriting one.
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
