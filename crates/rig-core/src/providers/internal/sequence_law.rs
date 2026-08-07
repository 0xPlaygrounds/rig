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
//!
//! There is deliberately no intra-batch ORDER law: pass-through adapters
//! forward provider parts in wire order, and no wire contracts an order
//! (every reference SDK iterates parts as delivered — vercel's google
//! provider even documents "preserve original order"). Canonical order is
//! a property of the `chunk_lifecycle` canonicalizer and is unit-tested
//! there, where it is actually produced.
//!
//! Violations always emit `tracing::error!`; they panic only in rig's own
//! test-harness builds (`cfg(test)` or the `test-utils` feature, which
//! rig's test targets enable via the self-dev-dependency). A downstream
//! application's debug build gets a grep-able error log, never a process
//! abort — a library must not take down a user's process over wire data,
//! and a wrong law must cost rig's suites, not rig's users.

// A law violation must abort the rig test that exposed it; outside rig's
// own harness builds the same violation is an error log (see `violation`).
#![cfg_attr(
    any(test, feature = "test-utils"),
    expect(
        clippy::panic,
        reason = "harness-only sequence assertions; log-only outside rig's own test builds"
    )
)]

use crate::streaming::RawStreamingChoice;

/// Whether one raw event is non-reasoning CONTENT — the classes whose
/// arrival closes a boundary-less wire's open reasoning block. Lifecycle
/// bookkeeping (`*End` events, terminals, ids, unknown passthrough) is
/// exempt: closing an older entity after newer content is legitimate
/// eviction, not a boundary violation.
fn is_boundary_content<R>(choice: &RawStreamingChoice<R>) -> bool {
    matches!(
        choice,
        RawStreamingChoice::Message(_)
            | RawStreamingChoice::TextStart { .. }
            | RawStreamingChoice::ToolCall(_)
            | RawStreamingChoice::ToolCallDelta { .. }
    )
}

/// Cross-frame validator state: which minted reasoning keys are open.
#[derive(Default)]
pub(crate) struct SequenceLaws {
    open_minted_reasoning: std::collections::HashSet<crate::streaming::StreamPartId>,
}

impl SequenceLaws {
    /// Check one `interpret` batch (the `out` buffer for a single frame)
    /// against the boundary law, updating cross-frame state. Violations log
    /// always and panic only in rig's own harness builds (see `violation`).
    pub(crate) fn check_batch<R>(
        &mut self,
        batch: &[Result<RawStreamingChoice<R>, crate::completion::CompletionError>],
    ) {
        for item in batch {
            let Ok(choice) = item else { continue };

            // Boundary law: while a minted reasoning key is open, the only
            // legal content is more reasoning; text or tool content means an
            // adapter forgot to synthesize the boundary end.
            if !self.open_minted_reasoning.is_empty() && is_boundary_content(choice) {
                violation(
                    "boundary",
                    variant_name(choice),
                    "emitted while a minted-key reasoning part is open — a \
                     boundary-less wire's adapter must synthesize ReasoningEnd \
                     before any other content class",
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
                // A whole block is open + restatement + close in ONE event
                // (pydantic-ai's replace-part semantics; the Bedrock adapter
                // spells its delta accumulation's close exactly this way):
                // a same-key whole block replaces AND closes the open part.
                // For a never-opened key the remove is a no-op, keeping the
                // id-less encrypted interleave exempt.
                RawStreamingChoice::Reasoning { id, .. } => {
                    self.open_minted_reasoning.remove(id);
                }
                _ => {}
            }
        }
    }
}

/// Surface one law violation: always an error log (variant names only —
/// raw events can carry wire content that must not reach logs), a panic
/// only in rig's own test-harness builds. `test-utils` is the harness
/// signal; rig-core's self-dev-dependency turns it on for every rig-core
/// test target, so the laws fail rig's suites loudly while a downstream
/// application's debug build only logs.
fn violation(law: &'static str, variant: &'static str, message: &'static str) {
    tracing::error!(
        target: "rig::sequence_law",
        law,
        variant,
        "sequence-law violation: {message}"
    );
    #[cfg(any(test, feature = "test-utils"))]
    panic!("sequence-law violation ({law}): {variant} {message}");
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
    #[should_panic(expected = "sequence-law violation (boundary): Message")]
    fn text_while_a_minted_reasoning_part_is_open_panics() {
        drive(vec![
            vec![minted_delta()],
            vec![RawStreamingChoice::Message("visible".to_owned())],
        ]);
    }

    #[test]
    #[should_panic(expected = "sequence-law violation (boundary): ToolCallDelta")]
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

    /// Pass-through wire order is legal: there is no intra-batch order law
    /// (no wire contracts part order; canonical order is the
    /// `chunk_lifecycle` canonicalizer's property, tested there).
    #[test]
    fn wire_order_within_a_batch_is_not_a_violation() {
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

    /// A same-key whole-block Reasoning event is open + restatement + close
    /// in one event: it closes the accumulation its deltas opened (the
    /// Bedrock shape — deltas under a minted Block key, then the full block
    /// under the same key), so following text is legal.
    #[test]
    fn a_same_key_whole_block_closes_the_open_reasoning_part() {
        let key = || StreamPartId::minted(MintKind::Block, 0);
        drive(vec![
            vec![RawStreamingChoice::ReasoningDelta {
                id: key(),
                provider_id: None,
                reasoning: "thinking".to_owned(),
            }],
            vec![RawStreamingChoice::Reasoning {
                id: key(),
                provider_id: None,
                content: crate::message::ReasoningContent::Text {
                    text: "thinking, complete".to_owned(),
                    signature: None,
                },
            }],
            vec![RawStreamingChoice::Message("visible".to_owned())],
        ]);
    }

    #[test]
    fn lifecycle_bookkeeping_is_not_boundary_content() {
        // Closing an older tool entity after newer text is legitimate
        // eviction.
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
