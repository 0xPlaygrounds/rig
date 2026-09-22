//! Debug-mode boundary validation over raw adapter output.
//!
//! Minted reasoning blocks must close before text or tool content. Wire-keyed
//! reasoning and already-closed whole blocks are exempt. No intra-batch ordering
//! is enforced. Violations log event names without payloads and panic only under
//! `cfg(test)` or the `test-utils` feature.

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

/// Whether an event carries text or tool content requiring reasoning to close.
/// Lifecycle bookkeeping is exempt, allowing older entities to close later.
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
pub struct SequenceLaws {
    open_minted_reasoning: std::collections::HashSet<crate::streaming::BlockId>,
}

impl SequenceLaws {
    /// Check one `interpret` batch (the `out` buffer for a single frame)
    /// against the boundary law, updating cross-frame state. Violations log
    /// always and panic only in rig's own harness builds (see `violation`).
    pub fn check_batch(&mut self, batch: &crate::operation::AdapterOutput) {
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
                } if id.is_minted() => {
                    self.open_minted_reasoning.insert(id.clone());
                }
                // Whole encrypted blocks may close without opening a tracked key.
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

/// Log a violation without wire payloads, which may contain sensitive content.
/// Panics under `cfg(test)` or the `test-utils` feature.
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
