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
//!   (a synthesized reasoning [`BlockEnd`](crate::streaming::StreamEvent::BlockEnd))
//!   before any other content class is emitted. Constant minted keys have no
//!   wire boundary of their own — interleaving output IS the boundary, and
//!   the adapter owns synthesizing it. Wire-keyed parts are exempt: wires
//!   with real per-part ids (OpenAI Responses) deliberately keep a part open
//!   across interleaving and collapse later events into it. Whole reasoning
//!   blocks (a start immediately closed by a restatement) are also exempt — they are reasoning-class content, and id-less
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

use crate::streaming::{BlockClose, BlockKind, Delta, StreamEvent};

/// Whether one raw event is non-reasoning CONTENT — the classes whose
/// arrival closes a boundary-less wire's open reasoning block. Lifecycle
/// bookkeeping (`*End` events, terminals, ids, unknown passthrough) is
/// exempt: closing an older entity after newer content is legitimate
/// eviction, not a boundary violation.
fn is_boundary_content(event: &StreamEvent) -> bool {
    matches!(
        event,
        StreamEvent::BlockStart {
            kind: BlockKind::Text { .. } | BlockKind::ToolCall,
            ..
        } | StreamEvent::BlockDelta {
            delta: Delta::Text { .. }
                | Delta::TextMeta { .. }
                | Delta::ToolName { .. }
                | Delta::ToolArguments { .. },
            ..
        }
    )
}

/// Cross-frame validator state: which minted reasoning keys are open.
#[derive(Default)]
pub(crate) struct SequenceLaws {
    open_minted_reasoning: std::collections::HashSet<crate::streaming::BlockId>,
}

impl SequenceLaws {
    /// Check one `interpret` batch (the `out` buffer for a single frame)
    /// against the boundary law, updating cross-frame state. Violations log
    /// always and panic only in rig's own harness builds (see `violation`).
    pub(crate) fn check_batch(&mut self, batch: &super::adapter::AdapterOutput) {
        for item in batch.iter() {
            let Ok(choice) = item else { continue };

            // Boundary law: while a minted reasoning key is open, the only
            // legal content is more reasoning; text or tool content means an
            // adapter forgot to synthesize the boundary end.
            if !self.open_minted_reasoning.is_empty() && is_boundary_content(choice) {
                violation(
                    "boundary",
                    choice.name(),
                    "emitted while a minted-key reasoning part is open — a \
                     boundary-less wire's adapter must synthesize ReasoningEnd \
                     before any other content class",
                );
            }

            match choice {
                StreamEvent::BlockStart {
                    id,
                    kind: BlockKind::Reasoning { .. },
                }
                | StreamEvent::BlockDelta {
                    id,
                    delta: Delta::Reasoning { .. },
                } => {
                    if id.is_minted() {
                        self.open_minted_reasoning.insert(id.clone());
                    }
                }
                // A close — bare, signed, or a whole-block restatement —
                // ends the open part. For a never-opened key the remove is a
                // no-op, keeping the id-less encrypted interleave exempt.
                StreamEvent::BlockEnd {
                    id,
                    end: BlockClose::Reasoning { .. },
                    ..
                } => {
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

#[cfg(test)]
mod tests;
