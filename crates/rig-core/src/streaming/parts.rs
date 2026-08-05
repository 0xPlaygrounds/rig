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

use crate::message::{AssistantContent, Reasoning, ReasoningContent, ToolCall};

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
}

impl PartsAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append streamed text to the active text block, opening one if none is
    /// active.
    pub(crate) fn text_delta(&mut self, text: &str) {
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
        self.text_index = None;
        self.parts
            .push(ManagedPart::Complete(AssistantContent::ToolCall(tool_call)));
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
        if parts.is_empty() {
            parts.push(AssistantContent::text(""));
        }
        parts
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

    #[test]
    fn finish_on_an_empty_stream_yields_one_empty_text_part() {
        let mut accumulator = PartsAccumulator::new();
        let parts = accumulator.finish();
        assert_eq!(parts, vec![AssistantContent::text("")]);
    }
}
