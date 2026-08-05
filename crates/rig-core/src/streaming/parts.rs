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

use std::collections::HashMap;

use crate::completion::CompletionError;
use crate::message::{AssistantContent, Reasoning, ReasoningContent, ToolCall, ToolFunction};
use crate::streaming::{ToolInputEnd, UnparseableToolInput};

/// Identity of a reasoning part: the provider-scoped item id plus a part
/// ordinal distinguishing sibling parts of one multi-part item.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct PartKey {
    item_id: String,
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
/// single-active-block text assembly, keyed reasoning delta/replace/sibling
/// semantics, and tool-call ordering. Consumers feed normalized events and
/// call [`PartsAccumulator::finish`] once the stream ends.
#[derive(Default)]
pub(crate) struct PartsAccumulator {
    parts: Vec<ManagedPart>,
    /// Reasoning part identity → index in `parts`. Invariant: every mapped
    /// index holds an `AssistantContent::Reasoning` part.
    reasoning_index: HashMap<PartKey, usize>,
    /// Item id → ordinal of that item's currently open (latest) part.
    open_ordinal: HashMap<String, u32>,
    /// Active text block; text deltas and metadata merge here until the block
    /// is closed by a text start, a reasoning event, or a tool call.
    text_index: Option<usize>,
    /// Tool calls under delta assembly, keyed by the fragment id, in start
    /// order. A completed call leaves this list, so a reused id after
    /// completion opens a fresh call (the ordinal collapse the reasoning
    /// keying does explicitly).
    open_tool_inputs: Vec<OpenToolInput>,
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
    id: String,
    /// Rig-minted correlation id, created when the call opens.
    internal_call_id: String,
    /// Tool name; a later non-empty name fragment replaces it.
    name: String,
    /// Concatenated raw argument fragments. `None` until a fragment arrives —
    /// a call that streamed no arguments is a parameterless invocation.
    buffer: Option<String>,
}

impl PartsAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append streamed text to the active text block, opening one if none is
    /// active.
    pub(crate) fn text_delta(&mut self, text: &str) {
        self.close_minted_reasoning();
        if let Some(index) = self.text_index
            && let Some(ManagedPart::DeltaBuilt(AssistantContent::Text(existing))) =
                self.parts.get_mut(index)
        {
            existing.text.push_str(text);
            return;
        }

        self.parts
            .push(ManagedPart::DeltaBuilt(AssistantContent::text(
                text.to_owned(),
            )));
        self.text_index = Some(self.parts.len() - 1);
    }

    /// Close the active text block and, when metadata is provided, open the
    /// next block carrying it.
    pub(crate) fn text_start(&mut self, additional_params: Option<serde_json::Value>) {
        self.close_minted_reasoning();
        self.text_index = None;
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

        if self
            .text_index
            .and_then(|index| self.parts.get(index))
            .is_none_or(|part| !matches!(part, ManagedPart::DeltaBuilt(AssistantContent::Text(_))))
        {
            self.parts
                .push(ManagedPart::DeltaBuilt(AssistantContent::text("")));
            self.text_index = Some(self.parts.len() - 1);
        }

        let Some(ManagedPart::DeltaBuilt(AssistantContent::Text(text))) =
            self.text_index.and_then(|index| self.parts.get_mut(index))
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
    pub(crate) fn reasoning_delta(&mut self, id: &str, text: &str) {
        self.text_index = None;

        let ordinal = self.open_ordinal.get(id).copied().unwrap_or(0);
        let key = PartKey {
            item_id: id.to_owned(),
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
                    self.open_ordinal.insert(id.to_owned(), next);
                    self.push_reasoning(
                        PartKey {
                            item_id: id.to_owned(),
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
    pub(crate) fn reasoning_full(&mut self, reasoning: Reasoning) {
        self.text_index = None;

        // Ids are mandatory on the raw grammar; an absent id degrades to a
        // single shared "" identity rather than a panic.
        let id = reasoning.id.clone().unwrap_or_default();
        let ordinal = self.open_ordinal.get(&id).copied().unwrap_or(0);
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
                            item_id: id,
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

    /// Append a completed tool call.
    pub(crate) fn tool_call(&mut self, tool_call: ToolCall) {
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
    pub(crate) fn tool_name_delta(&mut self, id: &str, name: &str) -> String {
        let index = self.ensure_open_tool_input(id);
        match self.open_tool_inputs.get_mut(index) {
            Some(input) => {
                name.clone_into(&mut input.name);
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
    pub(crate) fn tool_args_delta(&mut self, id: &str, fragment: &str) -> String {
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
                        buffer.push_str(fragment);
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
    /// no open call opens and completes one from the event alone.
    pub(crate) fn tool_input_end(
        &mut self,
        end: ToolInputEnd,
    ) -> Result<Option<(ToolCall, String)>, CompletionError> {
        let position = self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == end.id);
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
        if let Some(final_name) = end.name.clone() {
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
                Some(buffer) => match crate::json_utils::parse_tool_arguments(&buffer) {
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
                },
            },
        };

        let tool_call = ToolCall {
            id: end.tool_id.unwrap_or(opened_id),
            call_id: end.call_id,
            function: ToolFunction { name, arguments },
            signature: end.signature,
            additional_params: end.additional_params,
        };
        self.tool_call(tool_call.clone());
        Ok(Some((tool_call, internal_call_id)))
    }

    /// Index of the open call for `id`, opening one if none exists.
    fn ensure_open_tool_input(&mut self, id: &str) -> usize {
        match self
            .open_tool_inputs
            .iter()
            .position(|input| input.id == id)
        {
            Some(index) => index,
            None => {
                self.open_tool_inputs.push(OpenToolInput {
                    id: id.to_owned(),
                    internal_call_id: crate::id::generate(),
                    name: String::new(),
                    buffer: None,
                });
                self.open_tool_inputs.len() - 1
            }
        }
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
        self.text_index = None;
        // Calls still open at stream end never fully arrived (no end event,
        // no adapter flush): truncated input drops, per the settled contract.
        self.open_tool_inputs.clear();
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
    fn close_minted_reasoning(&mut self) {
        let open_minted: Vec<(String, u32)> = self
            .reasoning_index
            .keys()
            .filter(|key| {
                key.ordinal == self.open_ordinal.get(&key.item_id).copied().unwrap_or(0)
                    && crate::streaming::is_boundary_minted_id(&key.item_id)
            })
            .map(|key| (key.item_id.clone(), key.ordinal))
            .collect();
        for (item_id, ordinal) in open_minted {
            self.open_ordinal.insert(item_id, ordinal + 1);
        }
    }

    fn push_reasoning(&mut self, key: PartKey, part: ManagedPart) {
        self.parts.push(part);
        self.reasoning_index.insert(key, self.parts.len() - 1);
    }
}

fn delta_reasoning(id: &str, text: &str) -> AssistantContent {
    AssistantContent::Reasoning(Reasoning {
        id: Some(id.to_owned()),
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
        accumulator.reasoning_delta("rs_1", "partial ");
        accumulator.reasoning_delta("rs_1", "thought");
        accumulator.reasoning_full(full("rs_1", reasoning_text("the complete chain")));

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["the complete chain"]);
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn replacement_is_keyed_across_interleaved_output() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta("rs_1", "partial");
        accumulator.tool_call(ToolCall {
            id: "call_1".to_owned(),
            call_id: None,
            function: crate::message::ToolFunction {
                name: "probe".to_owned(),
                arguments: serde_json::json!({}),
            },
            signature: None,
            additional_params: None,
        });
        accumulator.reasoning_full(full("rs_1", reasoning_text("full block")));

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
        accumulator.reasoning_full(full("rs_1", summary("s1")));
        accumulator.reasoning_full(full("rs_1", summary("s2")));
        accumulator.reasoning_full(full("rs_1", reasoning_text("visible")));
        accumulator.reasoning_full(full("rs_1", ReasoningContent::Encrypted("enc".to_owned())));

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "visible", "enc"]);
    }

    /// Deltas then the item's multi-part done block: the first full block
    /// supersedes the delta accumulation, the rest append as siblings.
    #[test]
    fn deltas_then_sibling_full_blocks_keep_each_part_once() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta("rs_1", "s1");
        accumulator.reasoning_full(full("rs_1", summary("s1")));
        accumulator.reasoning_full(full("rs_1", summary("s2")));
        accumulator.reasoning_full(full("rs_1", ReasoningContent::Encrypted("enc".to_owned())));

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "enc"]);
    }

    #[test]
    fn a_delta_after_replacement_opens_a_fresh_part() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta("rs_1", "first");
        accumulator.reasoning_full(full("rs_1", reasoning_text("first complete")));
        accumulator.reasoning_delta("rs_1", "second");

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["first complete", "second"]);
    }

    #[test]
    fn distinct_item_ids_never_interact() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta("rs_1", "first item deltas");
        accumulator.reasoning_full(full("rs_2", reasoning_text("a different item")));

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
        accumulator.reasoning_full(full("rs_1", summary("thinking")));
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
        accumulator.reasoning_delta("reasoning-0", "A");
        accumulator.tool_call(probe_tool_call());
        accumulator.reasoning_delta("reasoning-0", "B");

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
        accumulator.reasoning_delta("reasoning-0", "A");
        accumulator.tool_call(probe_tool_call());
        accumulator.reasoning_full(full("reasoning-0", reasoning_text("B")));

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
        assert!(matches!(parts.get(1), Some(AssistantContent::ToolCall(_))));
    }

    /// Text output is a boundary too: minted-id reasoning around streamed text
    /// stays two parts in arrival order.
    #[test]
    fn text_output_closes_a_minted_id_reasoning_item() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta("reasoning-0", "A");
        accumulator.text_delta("visible");
        accumulator.reasoning_delta("reasoning-0", "B");

        let parts = accumulator.finish();
        assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
        assert!(matches!(
            parts.get(1),
            Some(AssistantContent::Text(text)) if text.text == "visible"
        ));
    }

    /// Wire-supplied ids are exact item identities: the Responses replay case
    /// must keep collapsing a done block onto its deltas across interleaved
    /// output (the boundary bump applies to minted namespaces only).
    #[test]
    fn wire_id_full_blocks_still_collapse_across_interleaved_output() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.reasoning_delta("rs_1", "thinking");
        accumulator.tool_call(probe_tool_call());
        accumulator.reasoning_full(full("rs_1", reasoning_text("full reasoning")));

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
        ToolInputEnd::new(id, mode)
    }

    #[test]
    fn fragments_assemble_into_a_completed_tool_call_with_a_stable_internal_id() {
        let mut accumulator = PartsAccumulator::new();
        let first = accumulator.tool_name_delta("call_1", "get_weather");
        let second = accumulator.tool_args_delta("call_1", "{\"location\":");
        let third = accumulator.tool_args_delta("call_1", "\"Paris\",");
        accumulator.tool_args_delta("call_1", "\"temp\":\"20C\"}");
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

    #[test]
    fn a_call_with_no_streamed_arguments_is_a_parameterless_invocation() {
        for fragments in [Vec::new(), vec![""]] {
            let mut accumulator = PartsAccumulator::new();
            accumulator.tool_name_delta("call_1", "ping");
            for fragment in fragments {
                accumulator.tool_args_delta("call_1", fragment);
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
        accumulator.tool_name_delta("call_1", "ping");
        accumulator.tool_args_delta("call_1", "{\"x\":");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Drop))
                .expect("no error")
                .is_none()
        );

        accumulator.tool_args_delta("call_2", "{\"y\":1}");
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
        accumulator.tool_name_delta("call_1", "memory_search");
        accumulator.tool_args_delta("call_1", "{\"query\":");
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
        accumulator.tool_name_delta("call_1", "get_weather");
        accumulator.tool_args_delta("call_1", "{\"location\": not-json");
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
        let internal = accumulator.tool_name_delta("call_1", "search");
        accumulator.tool_args_delta("call_1", "{\"q\":\"ru");
        assert!(
            accumulator
                .tool_input_end(end("call_1", UnparseableToolInput::Keep))
                .expect("no error")
                .is_none()
        );

        accumulator.tool_args_delta("call_1", "st\"}");
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
        let internal = accumulator.tool_name_delta("fc_1", "provisional");
        accumulator.tool_args_delta("fc_1", "{\"partial\":");

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
        accumulator.tool_name_delta("call_1", "ping");
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
        accumulator.tool_name_delta("call_123", "web_search");
        accumulator.tool_args_delta("call_123", "null");
        accumulator.tool_args_delta("call_123", "{\"query\": \"META");
        accumulator.tool_args_delta("call_123", " Platforms news\"}");
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
                accumulator.tool_name_delta("call_1", "tool");
                accumulator.tool_args_delta("call_1", encoded);
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
        accumulator.tool_name_delta("call_a", "get_weather");
        accumulator.tool_name_delta("call_b", "get_time");
        accumulator.tool_args_delta("call_a", "{\"location\":\"Paris\"}");
        accumulator.tool_args_delta("call_b", "{\"zone\":\"UTC\"}");
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
        accumulator.tool_name_delta("tool-0", "get_weather");
        accumulator.tool_name_delta("tool-1", "get_time");
        accumulator.tool_args_delta("tool-0", "{\"city\":");
        accumulator.tool_args_delta("tool-1", "{\"zone\":");
        accumulator.tool_args_delta("tool-0", "\"Tokyo\"}");
        accumulator.tool_args_delta("tool-1", "\"UTC\"}");
        let (first, _) = accumulator
            .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        let (second, _) = accumulator
            .tool_input_end(end("tool-1", UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        assert_eq!(first.id, "tool-0");
        assert_eq!(second.id, "tool-1");
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
        accumulator.tool_name_delta("call_1", "ping");
        accumulator.tool_args_delta("call_1", "{\"x\":");
        let parts = accumulator.finish();
        assert_eq!(parts, vec![AssistantContent::text("")]);
        assert!(!accumulator.saw_tool_call());
    }

    /// The tool-id override: a call opened under an id-less key still reports
    /// the provider id its wire established later.
    #[test]
    fn the_tool_id_override_supersedes_the_assembly_key() {
        let mut accumulator = PartsAccumulator::new();
        accumulator.tool_name_delta("", "ping");
        let mut done = end("", UnparseableToolInput::Drop);
        done.tool_id = Some("call_late".to_owned());
        let (tool_call, _) = accumulator
            .tool_input_end(done)
            .expect("no error")
            .expect("finalizes");
        assert_eq!(tool_call.id, "call_late");
    }
}
