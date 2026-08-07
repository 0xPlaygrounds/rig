//! Debug-mode sequence laws over raw adapter output.
//!
//! The lifecycle grammar's laws were previously checked only against
//! hand-written accumulator fixtures — never against what the real adapters
//! emit ("the lifecycle laws live as accumulator-level unit + proptest cases
//! rather than an extension of the public-item validator"). This validator
//! runs inside the shared drivers under `cfg(any(test, debug_assertions))`,
//! so every conformance fixture, cassette replay, and debug-build stream
//! exercises the laws against real adapter output; release builds compile it
//! out entirely.
//!
//! The laws are the two obligation classes the review rounds kept re-finding
//! one adapter at a time (vercel's shared-accumulator rejections and
//! pydantic-ai's `UnexpectedModelBehavior` are the precedent — the check
//! lives in the component every provider funnels through, not in each
//! provider's own tests):
//!
//! - **Boundary law**: a *minted*-key reasoning part left open must be closed
//!   (a synthesized [`ReasoningEnd`](crate::streaming::RawStreamingChoice::ReasoningEnd))
//!   before any other content class is emitted. Constant minted keys have no
//!   wire boundary of their own — interleaving output IS the boundary, and
//!   the adapter owns synthesizing it. Wire-keyed parts are exempt: wires
//!   with real per-part ids (OpenAI Responses) deliberately keep a part open
//!   across interleaving and collapse later events into it. Whole-block
//!   [`Reasoning`](crate::streaming::RawStreamingChoice::Reasoning) events
//!   are also exempt — they are reasoning-class content, and id-less
//!   encrypted blocks legally interleave a constant-key text accumulation
//!   (the mixed OpenRouter stream).
//! - **Intra-batch order law**: within one `interpret` call's output,
//!   content-bearing events follow canonical class order — reasoning, then
//!   text, then tool calls. Lifecycle bookkeeping (`*End` events, terminal
//!   records, ids, unknown passthrough) is exempt: closing an older entity
//!   after newer content is legitimate eviction, not disorder.

// This module IS a panic facility: it exists only under
// `cfg(any(test, debug_assertions))` as a debug assertion over adapter
// output, and a law violation must abort the test that exposed it.
#![expect(
    clippy::panic,
    reason = "debug-only sequence assertions; compiled out of release builds"
)]

use crate::streaming::RawStreamingChoice;

/// Content class of one raw event, for the intra-batch order law.
/// `None` for lifecycle bookkeeping, which the law exempts.
fn content_class<R>(choice: &RawStreamingChoice<R>) -> Option<u8> {
    match choice {
        RawStreamingChoice::Reasoning { .. }
        | RawStreamingChoice::ReasoningStart { .. }
        | RawStreamingChoice::ReasoningDelta { .. } => Some(0),
        RawStreamingChoice::Message(_) | RawStreamingChoice::TextStart { .. } => Some(1),
        RawStreamingChoice::ToolCall(_) | RawStreamingChoice::ToolCallDelta { .. } => Some(2),
        RawStreamingChoice::ReasoningEnd { .. }
        | RawStreamingChoice::TextEnd { .. }
        | RawStreamingChoice::ToolInputEnd(_)
        | RawStreamingChoice::TextAdditionalParams(_)
        | RawStreamingChoice::FinalResponse(_)
        | RawStreamingChoice::MessageId(_)
        | RawStreamingChoice::Unknown(_) => None,
    }
}

/// Cross-frame validator state: which minted reasoning keys are open.
#[derive(Default)]
pub(crate) struct SequenceLaws {
    open_minted_reasoning: std::collections::HashSet<crate::streaming::StreamPartId>,
}

impl SequenceLaws {
    /// Check one `interpret` batch (the `out` buffer for a single frame)
    /// against the laws, updating cross-frame state. Panics on violation —
    /// this is a test/debug assertion, compiled out of release builds by the
    /// caller's `cfg`.
    pub(crate) fn check_batch<R>(
        &mut self,
        batch: &[Result<RawStreamingChoice<R>, crate::completion::CompletionError>],
    ) {
        let mut max_class_seen: Option<u8> = None;
        for item in batch {
            let Ok(choice) = item else { continue };

            // Boundary law: while a minted reasoning key is open, the only
            // legal content is more reasoning; text or tool content means an
            // adapter forgot to synthesize the boundary end.
            if !self.open_minted_reasoning.is_empty()
                && matches!(content_class(choice), Some(1) | Some(2))
            {
                panic!(
                    "sequence-law violation: {} emitted while a minted-key reasoning \
                     part is open — a boundary-less wire's adapter must synthesize \
                     ReasoningEnd before any other content class",
                    variant_name(choice),
                );
            }

            match choice {
                RawStreamingChoice::ReasoningStart { id, .. }
                | RawStreamingChoice::ReasoningDelta { id, .. } => {
                    if id.is_minted() {
                        self.open_minted_reasoning.insert(id.clone());
                    }
                }
                RawStreamingChoice::ReasoningEnd { id, .. } => {
                    self.open_minted_reasoning.remove(id);
                }
                _ => {}
            }

            // Intra-batch order law over content-bearing events.
            if let Some(class) = content_class(choice) {
                if let Some(max_class) = max_class_seen
                    && class < max_class
                {
                    panic!(
                        "sequence-law violation: {} emitted after a later content \
                         class in the same frame batch — canonical intra-chunk \
                         order is reasoning, text, tool calls",
                        variant_name(choice),
                    );
                }
                max_class_seen = Some(max_class_seen.unwrap_or(class).max(class));
            }
        }
    }
}

/// Stable variant name for law-violation messages (no payload — raw events
/// can carry wire content that must not reach logs or panic text).
fn variant_name<R>(choice: &RawStreamingChoice<R>) -> &'static str {
    match choice {
        RawStreamingChoice::Message(_) => "Message",
        RawStreamingChoice::TextStart { .. } => "TextStart",
        RawStreamingChoice::TextEnd { .. } => "TextEnd",
        RawStreamingChoice::TextAdditionalParams(_) => "TextAdditionalParams",
        RawStreamingChoice::ToolCall(_) => "ToolCall",
        RawStreamingChoice::ToolCallDelta { .. } => "ToolCallDelta",
        RawStreamingChoice::ToolInputEnd(_) => "ToolInputEnd",
        RawStreamingChoice::Reasoning { .. } => "Reasoning",
        RawStreamingChoice::ReasoningStart { .. } => "ReasoningStart",
        RawStreamingChoice::ReasoningDelta { .. } => "ReasoningDelta",
        RawStreamingChoice::ReasoningEnd { .. } => "ReasoningEnd",
        RawStreamingChoice::FinalResponse(_) => "FinalResponse",
        RawStreamingChoice::MessageId(_) => "MessageId",
        RawStreamingChoice::Unknown(_) => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::super::adapter::{AdapterOutput, WireAdapter, run_wire_buffered};
    use super::super::wire::WireEvent;
    use crate::streaming::{MintKind, RawStreamingChoice, StreamPartId};

    /// A scripted adapter: each frame index replays its preloaded batch.
    struct Scripted {
        batches: Vec<Vec<RawStreamingChoice<()>>>,
    }

    impl WireAdapter for Scripted {
        type Frame = usize;
        type Event = usize;
        type Response = ();

        fn classify(&self, frame: usize) -> WireEvent<usize> {
            WireEvent::Known(frame)
        }

        fn interpret(&mut self, event: usize, out: &mut AdapterOutput<()>) {
            if let Some(batch) = self.batches.get_mut(event) {
                out.extend(std::mem::take(batch).into_iter().map(Ok));
            }
        }

        fn finish(&mut self, _out: &mut AdapterOutput<()>) {}
    }

    fn drive(batches: Vec<Vec<RawStreamingChoice<()>>>) {
        let frames = 0..batches.len();
        run_wire_buffered(frames, Scripted { batches }).expect("no data errors");
    }

    fn minted_delta() -> RawStreamingChoice<()> {
        RawStreamingChoice::ReasoningDelta {
            id: StreamPartId::minted(MintKind::Reasoning, 0),
            provider_id: None,
            reasoning: "thinking".to_owned(),
        }
    }

    fn minted_end() -> RawStreamingChoice<()> {
        RawStreamingChoice::ReasoningEnd {
            id: StreamPartId::minted(MintKind::Reasoning, 0),
            reasoning: None,
            signature: None,
            wire_sent: false,
        }
    }

    #[test]
    #[should_panic(expected = "sequence-law violation: Message emitted while a minted-key")]
    fn text_while_a_minted_reasoning_part_is_open_panics() {
        drive(vec![
            vec![minted_delta()],
            vec![RawStreamingChoice::Message("visible".to_owned())],
        ]);
    }

    #[test]
    #[should_panic(expected = "sequence-law violation: ToolCallDelta emitted while a minted-key")]
    fn tool_content_while_a_minted_reasoning_part_is_open_panics() {
        drive(vec![
            vec![minted_delta()],
            vec![RawStreamingChoice::ToolCallDelta {
                id: StreamPartId::minted(MintKind::Tool, 0),
                content: crate::streaming::ToolCallDeltaContent::Name("probe".to_owned()),
            }],
        ]);
    }

    #[test]
    fn a_synthesized_end_before_text_satisfies_the_boundary_law() {
        drive(vec![
            vec![minted_delta()],
            vec![
                minted_end(),
                RawStreamingChoice::Message("visible".to_owned()),
            ],
        ]);
    }

    #[test]
    fn a_wire_keyed_part_may_stay_open_across_interleaving() {
        drive(vec![
            vec![RawStreamingChoice::ReasoningDelta {
                id: StreamPartId::wire("rs_1"),
                provider_id: crate::streaming::WireId::new("rs_1"),
                reasoning: "thinking".to_owned(),
            }],
            vec![RawStreamingChoice::Message("visible".to_owned())],
        ]);
    }

    #[test]
    #[should_panic(expected = "canonical intra-chunk order")]
    fn reasoning_after_text_in_one_batch_panics() {
        drive(vec![vec![
            RawStreamingChoice::Message("visible".to_owned()),
            RawStreamingChoice::Reasoning {
                id: StreamPartId::minted(MintKind::EncryptedReasoning, 0),
                provider_id: None,
                content: crate::message::ReasoningContent::Text {
                    text: "late".to_owned(),
                    signature: None,
                },
            },
        ]]);
    }

    #[test]
    fn lifecycle_bookkeeping_is_exempt_from_the_order_law() {
        // Closing an older tool entity after newer text is legitimate
        // eviction, not disorder.
        drive(vec![vec![
            RawStreamingChoice::Message("visible".to_owned()),
            RawStreamingChoice::ToolInputEnd(crate::streaming::ToolInputEnd {
                id: StreamPartId::minted(MintKind::Tool, 0),
                tool_id: None,
                call_id: None,
                name: None,
                arguments: None,
                signature: None,
                additional_params: None,
                on_unparseable: crate::streaming::UnparseableToolInput::Drop,
            }),
        ]]);
    }
}
