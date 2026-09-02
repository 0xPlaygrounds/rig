use super::super::adapter::{AdapterOutput, WireAdapter, run_wire_buffered};
use super::super::wire::WireEvent;
use crate::streaming::{BlockClose, BlockId, BlockKind, Delta, MintKind, StreamEvent, ToolCallEnd};

/// A scripted adapter: each frame index replays its preloaded batch.
struct Scripted {
    batches: Vec<Vec<StreamEvent>>,
}

impl WireAdapter for Scripted {
    type Frame = usize;
    type Event = usize;

    fn classify(&self, frame: usize) -> WireEvent<usize> {
        WireEvent::Known(frame)
    }

    fn interpret(&mut self, event: usize, out: &mut AdapterOutput) {
        if let Some(batch) = self.batches.get_mut(event) {
            for event in std::mem::take(batch) {
                out.push(Ok(event));
            }
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {}
}

fn drive(batches: Vec<Vec<StreamEvent>>) {
    let frames = 0..batches.len();
    run_wire_buffered(frames, Scripted { batches }).expect("no data errors");
}

fn text(text: &str) -> StreamEvent {
    StreamEvent::text(BlockId::minted(MintKind::Text, 0), text)
}

fn minted_delta() -> StreamEvent {
    StreamEvent::BlockDelta {
        id: BlockId::minted(MintKind::Reasoning, 0),
        delta: Delta::Reasoning {
            text: "thinking".to_owned(),
        },
    }
}

fn minted_end() -> StreamEvent {
    StreamEvent::BlockEnd {
        id: BlockId::minted(MintKind::Reasoning, 0),
        end: BlockClose::Reasoning {
            reasoning: None,
            signature: None,
            wire_sent: false,
        },
        block: None,
    }
}

fn whole_reasoning(id: BlockId, provider_id: Option<String>, text: &str) -> StreamEvent {
    StreamEvent::BlockEnd {
        id,
        end: BlockClose::Reasoning {
            reasoning: Some(crate::message::Reasoning {
                id: provider_id,
                content: vec![crate::message::ReasoningContent::Text {
                    text: text.to_owned(),
                    signature: None,
                }],
            }),
            signature: None,
            wire_sent: true,
        },
        block: None,
    }
}

#[test]
#[should_panic(expected = "sequence-law violation (boundary): BlockDelta")]
fn text_while_a_minted_reasoning_part_is_open_panics() {
    drive(vec![vec![minted_delta()], vec![text("visible")]]);
}

#[test]
#[should_panic(expected = "sequence-law violation (boundary): BlockDelta")]
fn tool_content_while_a_minted_reasoning_part_is_open_panics() {
    drive(vec![
        vec![minted_delta()],
        vec![StreamEvent::BlockDelta {
            id: BlockId::minted(MintKind::Tool, 0),
            delta: Delta::ToolName {
                name: "probe".to_owned(),
            },
        }],
    ]);
}

#[test]
fn a_synthesized_end_before_text_satisfies_the_boundary_law() {
    drive(vec![
        vec![minted_delta()],
        vec![minted_end(), text("visible")],
    ]);
}

#[test]
fn a_wire_keyed_part_may_stay_open_across_interleaving() {
    drive(vec![
        vec![
            StreamEvent::BlockStart {
                id: BlockId::wire("rs_1"),
                kind: BlockKind::Reasoning {
                    provider_id: crate::streaming::non_empty_id("rs_1"),
                },
            },
            StreamEvent::BlockDelta {
                id: BlockId::wire("rs_1"),
                delta: Delta::Reasoning {
                    text: "thinking".to_owned(),
                },
            },
        ],
        vec![text("visible")],
    ]);
}

/// Pass-through wire order is legal: there is no intra-batch order law
/// (no wire contracts part order; canonical order is the
/// `chunk_lifecycle` canonicalizer's property, tested there).
#[test]
fn wire_order_within_a_batch_is_not_a_violation() {
    drive(vec![vec![
        text("visible"),
        whole_reasoning(
            BlockId::minted(MintKind::EncryptedReasoning, 0),
            None,
            "late",
        ),
    ]]);
}

/// A same-key whole-block reasoning end is open + restatement + close in
/// one event: it closes the accumulation its deltas opened (the Bedrock
/// shape — deltas under a minted Block key, then the full block under the
/// same key), so following text is legal.
#[test]
fn a_same_key_whole_block_closes_the_open_reasoning_part() {
    let key = || BlockId::minted(MintKind::Block, 0);
    drive(vec![
        vec![StreamEvent::BlockDelta {
            id: key(),
            delta: Delta::Reasoning {
                text: "thinking".to_owned(),
            },
        }],
        vec![whole_reasoning(key(), None, "thinking, complete")],
        vec![text("visible")],
    ]);
}

#[test]
fn lifecycle_bookkeeping_is_not_boundary_content() {
    // Closing an older tool entity after newer text is legitimate
    // eviction.
    drive(vec![vec![
        text("visible"),
        StreamEvent::BlockEnd {
            id: BlockId::minted(MintKind::Tool, 0),
            end: BlockClose::ToolCall(ToolCallEnd::new(
                crate::streaming::UnparseableToolInput::Drop,
            )),
            block: None,
        },
    ]]);
}
