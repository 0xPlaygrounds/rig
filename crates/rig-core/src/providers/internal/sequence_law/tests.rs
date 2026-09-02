use super::super::adapter::{AdapterOutput, WireAdapter, run_wire_buffered};
use super::super::wire::WireEvent;
use crate::streaming::{BlockId, MintKind, RawStreamingChoice};

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
        id: BlockId::minted(MintKind::Reasoning, 0),
        provider_id: None,
        reasoning: "thinking".to_owned(),
    }
}

fn minted_end() -> RawStreamingChoice<()> {
    RawStreamingChoice::ReasoningEnd {
        id: BlockId::minted(MintKind::Reasoning, 0),
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
            id: BlockId::minted(MintKind::Tool, 0),
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
            id: BlockId::wire("rs_1"),
            provider_id: crate::streaming::non_empty_id("rs_1"),
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
            id: BlockId::minted(MintKind::EncryptedReasoning, 0),
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
    let key = || BlockId::minted(MintKind::Block, 0);
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
            id: BlockId::minted(MintKind::Tool, 0),
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
