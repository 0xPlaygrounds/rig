//! Matrix Q: causal dispatch.
//!
//! A handler's way back onto the bus is its sink's dispatcher: every
//! dispatch made through it carries the served dispatch's id as its
//! parent, and the record names it. The `lookup` tool of these cells
//! dispatches from inside its own service — a completion on the agent's
//! model key, a host note, its own key, the host's relay (which nests once
//! more), the host's never-answering handler — and both interpreters
//! register the same tool and host handlers (program, not record): the
//! replayers answer only the leaves. Two rules are read off the chain:
//! under serial serving a dispatch descending from one in flight on its
//! own key is refused, from a spawned thread exactly as inline; a cancel
//! reaches the chain, so a child in flight is cancelled with its parent
//! and a child still queued never begins.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | nesting depth | 1 · 2 |
//! | where the nested dispatch is made | inside a tool's `Serve` · inside a detached sink's resolver · from a spawned thread |
//! | target key | another key (the model · a host note · the host's relay · the host's `never`) · the same key |
//! | host serving | `serial_per_handler` true · false |
//! | parent fate | answered · cancelled while the child is in flight · cancelled while the child is queued |
//! | medium | unary · streamed with events |
//!
//! Full cross-product: 2 × 3 × 5 × 2 × 3 × 2 = 360. Recorded: the 12 cells
//! below. Pruned: a layer's `before` as the nesting site (Matrix P, after
//! L7); depth 2 under every axis but the answered relay (the chain is the
//! claim; one cell shows three ids); the same-key child from a thread under
//! concurrent serving (the thread would block on a driver that runs on the
//! same runtime — the refusal is decided at the send, so only the serial
//! case is a thread cell); the cancelled fates on the answered children
//! (only `never` can be reached and held); the streamed medium on every
//! cell but the completion child (the medium changes the run's
//! completions, not where the tool's dispatch lands); the model child on
//! the mock wire (recorded live: the wire is what the cell shows).
//!
//! # Cells
//!
//! | golden | producer | policy | shape |
//! |---|---|---|---|
//! | `anthropic_causal_completion_serial` | anthropic `corpus_causal.rs` `completion_serial_…` | serial | `[Completion, Tool, Completion←1, Completion]` |
//! | `anthropic_causal_completion_concurrent` | `completion_concurrent_…` | concurrent | the same |
//! | `anthropic_causal_completion_streamed` | `completion_streamed_…` | concurrent, events kept | the same; the nested completion unary |
//! | `mock_causal_note_serial` | `tests/core/golden_causal.rs` `note_serial_…` | serial | `[Completion, Tool, Custom←1, Completion]` |
//! | `mock_causal_note_concurrent` | `note_concurrent_…` | concurrent | the same |
//! | `mock_causal_depth_two` | `depth_two_…` | concurrent | `[Completion, Tool, Custom←1, Custom←2, Completion]` |
//! | `mock_causal_same_key_serial_refused` | `same_key_serial_refused_…` | serial | `[Completion, Tool, Completion]`: the child refused, no record |
//! | `mock_causal_same_key_concurrent_served` | `same_key_concurrent_served_…` | concurrent | `[Completion, Tool, Tool←1, Completion]` |
//! | `mock_causal_same_key_from_thread_refused` | `same_key_from_thread_refused_…` | serial | as the serial refusal, dispatched from a spawned thread |
//! | `mock_causal_parent_cancelled_child_in_flight` | `parent_cancelled_child_in_flight_…` | concurrent | `[Completion, Tool✗, Custom✗←1]` |
//! | `mock_causal_parent_cancelled_child_queued` | `parent_cancelled_child_queued_…` | serial | `[Completion, Tool✗, Custom✗←1]`: the second child never began |
//! | `mock_causal_detached_resolver` | `detached_resolver_…` | concurrent | `[Completion, Tool, Custom←1, Completion]`, the parent from the detached sink |
//!
//! `←n`: the record's parent is record `n`; `✗`: `Cancelled`. The
//! depth-two cell is also interpreted by the resumed engine (the parent
//! ids survive the checkpoint: Matrix R's row for a chain).
//!
//! # What the matrix found
//!
//! - A tool served over the bus had no way to reach the bus it was served
//!   on without holding a dispatcher of its own — and a dispatch made
//!   through such a dispatcher under serial serving queued behind the very
//!   call that waited on it whenever it came from another thread. The
//!   scope on the sink (`SinkDispatch::dispatcher`) and on the tool's
//!   context (`ToolContext::scope`) is the way back; the chain is the rule.
//! - A host-bus golden does not name the host's policy, and the serial
//!   cells' refusal depends on it: `Program::host_serial` tells the replay
//!   which policy the producer ran under (the log stays as it was).

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{NESTING, NestedChild, Nesting, Program};
use rig_core::effect::EffectFamily;
use rig_core::error::ErrorKind;

const TOOLS_PREAMBLE: &str = "You are a research assistant. Use the lookup tool to answer.";
const LIVE_PROMPT: &str = "Use the lookup tool with q set to exactly \"What is the capital of France?\" and reply with just the lookup result.";
const MOCK_PROMPT: &str = "Look up the capital of France and reply with just the lookup result.";

const fn nesting(child: NestedChild) -> Nesting {
    Nesting { child, ..NESTING }
}

const LIVE: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: LIVE_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    nesting: Some(nesting(NestedChild::Completion)),
    ..Program::DEFAULT
};
const MOCK: Program = Program {
    prompt: MOCK_PROMPT,
    ..LIVE
};

const COMPLETION_SERIAL: Program = Program {
    fixture: "anthropic_causal_completion_serial",
    host_serial: true,
    ..LIVE
};
const COMPLETION_CONCURRENT: Program = Program {
    fixture: "anthropic_causal_completion_concurrent",
    ..LIVE
};
const COMPLETION_STREAMED: Program = Program {
    fixture: "anthropic_causal_completion_streamed",
    streamed: true,
    ..LIVE
};
const NOTE_SERIAL: Program = Program {
    fixture: "mock_causal_note_serial",
    nesting: Some(nesting(NestedChild::Note)),
    host_serial: true,
    ..MOCK
};
const NOTE_CONCURRENT: Program = Program {
    fixture: "mock_causal_note_concurrent",
    nesting: Some(nesting(NestedChild::Note)),
    ..MOCK
};
const DEPTH_TWO: Program = Program {
    fixture: "mock_causal_depth_two",
    nesting: Some(nesting(NestedChild::Relay)),
    ..MOCK
};
const SAME_KEY_SERIAL_REFUSED: Program = Program {
    fixture: "mock_causal_same_key_serial_refused",
    nesting: Some(nesting(NestedChild::Same)),
    host_serial: true,
    ..MOCK
};
const SAME_KEY_CONCURRENT_SERVED: Program = Program {
    fixture: "mock_causal_same_key_concurrent_served",
    nesting: Some(nesting(NestedChild::Same)),
    ..MOCK
};
const SAME_KEY_FROM_THREAD_REFUSED: Program = Program {
    fixture: "mock_causal_same_key_from_thread_refused",
    nesting: Some(Nesting {
        child: NestedChild::Same,
        from_thread: true,
        detached: false,
    }),
    host_serial: true,
    ..MOCK
};
const PARENT_CANCELLED_CHILD_IN_FLIGHT: Program = Program {
    fixture: "mock_causal_parent_cancelled_child_in_flight",
    nesting: Some(nesting(NestedChild::Never)),
    cancel_when_reached: true,
    ..MOCK
};
const PARENT_CANCELLED_CHILD_QUEUED: Program = Program {
    fixture: "mock_causal_parent_cancelled_child_queued",
    nesting: Some(nesting(NestedChild::NeverTwice)),
    host_serial: true,
    cancel_when_reached: true,
    ..MOCK
};
const DETACHED_RESOLVER: Program = Program {
    fixture: "mock_causal_detached_resolver",
    nesting: Some(Nesting {
        child: NestedChild::Note,
        from_thread: false,
        detached: true,
    }),
    ..MOCK
};

both_interpreters! {
    completion_serial: COMPLETION_SERIAL,
    completion_concurrent: COMPLETION_CONCURRENT,
    completion_streamed: COMPLETION_STREAMED,
    note_serial: NOTE_SERIAL,
    note_concurrent: NOTE_CONCURRENT,
    depth_two: DEPTH_TWO,
    same_key_serial_refused: SAME_KEY_SERIAL_REFUSED,
    same_key_concurrent_served: SAME_KEY_CONCURRENT_SERVED,
    same_key_from_thread_refused: SAME_KEY_FROM_THREAD_REFUSED,
    parent_cancelled_child_in_flight: PARENT_CANCELLED_CHILD_IN_FLIGHT,
    parent_cancelled_child_queued: PARENT_CANCELLED_CHILD_QUEUED,
    detached_resolver: DETACHED_RESOLVER,
}

/// The chain survives a checkpoint: the depth-two cell resumed after its
/// tool turn — the head (the tool and its two children) by hand, the tail
/// by the engine — records the same parents.
#[tokio::test]
async fn depth_two_resumed() {
    corpus::resume_reproduces(&DEPTH_TWO).await;
}

/// The parent of every record, by position.
fn parents(fixture: &str) -> Vec<Option<usize>> {
    let log = corpus::golden(fixture);
    log.iter()
        .map(|record| {
            record.parent.map(|parent| {
                log.iter()
                    .position(|candidate| candidate.id == parent)
                    .expect("a parent in the log")
            })
        })
        .collect()
}

/// Every child names the record it was made from, and only children do:
/// the cells' chains, as data.
#[test]
fn every_child_record_names_its_parent() {
    let one_child = [None, None, Some(1), None];
    for cell in [
        &COMPLETION_SERIAL,
        &COMPLETION_CONCURRENT,
        &COMPLETION_STREAMED,
        &NOTE_SERIAL,
        &NOTE_CONCURRENT,
        &SAME_KEY_CONCURRENT_SERVED,
        &DETACHED_RESOLVER,
    ] {
        assert_eq!(parents(cell.fixture), one_child, "{}", cell.fixture);
    }
    assert_eq!(
        parents(DEPTH_TWO.fixture),
        [None, None, Some(1), Some(2), None]
    );
    for cell in [
        &PARENT_CANCELLED_CHILD_IN_FLIGHT,
        &PARENT_CANCELLED_CHILD_QUEUED,
    ] {
        assert_eq!(
            parents(cell.fixture),
            [None, None, Some(1)],
            "{}",
            cell.fixture
        );
    }
    // No record for a refused child.
    for cell in [&SAME_KEY_SERIAL_REFUSED, &SAME_KEY_FROM_THREAD_REFUSED] {
        assert_eq!(
            parents(cell.fixture),
            [None, None, None],
            "{}",
            cell.fixture
        );
    }
}

/// A nested completion is the model key's record like any other — its
/// request is the tool's, not the run's — and the chain is the only thing
/// that distinguishes it.
#[test]
fn the_nested_completion_is_a_model_record_made_by_the_tool() {
    for cell in [&COMPLETION_SERIAL, &COMPLETION_CONCURRENT] {
        let log = corpus::golden(cell.fixture);
        assert_eq!(log[2].key, log[0].key, "the agent's model key");
        assert_eq!(log[2].kind.family(), EffectFamily::Completion);
        let question = match &log[2].kind {
            rig_core::effect::EffectKind::Completion { request, .. } => request
                .chat_history
                .iter()
                .rev()
                .find_map(rig_core::message::Message::rag_text)
                .expect("a prompt"),
            other => panic!("a completion, not {other:?}"),
        };
        assert_eq!(question, "What is the capital of France?");
    }
    // The two policies record the same chain: the trace is independent of
    // the host's serving policy, as Matrix C found for the agent's own.
    let shape = |fixture: &str| {
        corpus::golden(fixture)
            .iter()
            .map(|record| (record.key.clone(), record.kind.family()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        shape(COMPLETION_SERIAL.fixture),
        shape(COMPLETION_CONCURRENT.fixture)
    );
}

/// A refused child leaves no record; the parent's outcome carries the
/// refusal as the tool's text, so the model saw it.
#[test]
fn a_refused_child_leaves_no_record() {
    for cell in [&SAME_KEY_SERIAL_REFUSED, &SAME_KEY_FROM_THREAD_REFUSED] {
        let log = corpus::golden(cell.fixture);
        assert_eq!(log.len(), 3, "{}", cell.fixture);
        let Ok(rig_core::effect::Outcome::ToolResult { result, .. }) = &log[1].outcome else {
            panic!("a tool result: {:?}", log[1].outcome);
        };
        assert_eq!(result.output().render(), "refused:Request");
    }
    // Under concurrent serving the same child is served, and recorded.
    let served = corpus::golden(SAME_KEY_CONCURRENT_SERVED.fixture);
    assert_eq!(served.len(), 4);
    assert_eq!(served[2].key, served[1].key, "the tool's own key");
}

/// A parent's cancel reaches its children: both records say `Cancelled`,
/// the child's after the parent's; a child still queued behind a busy
/// serial key never begins, so it has no record.
#[test]
fn a_cancelled_parent_cancels_its_children() {
    for cell in [
        &PARENT_CANCELLED_CHILD_IN_FLIGHT,
        &PARENT_CANCELLED_CHILD_QUEUED,
    ] {
        let log = corpus::golden(cell.fixture);
        assert_eq!(log.len(), 3, "{}", cell.fixture);
        for position in [1, 2] {
            assert!(
                matches!(&log[position].outcome, Err(report) if report.kind == ErrorKind::Cancelled),
                "{}: {:?}",
                cell.fixture,
                log[position].outcome
            );
        }
        assert_eq!(log[2].key.as_str(), corpus::NEVER_KEY);
    }
}
