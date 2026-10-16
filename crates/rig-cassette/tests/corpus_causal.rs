//! Matrix Q: causal dispatch.
//!
//! A handler's way back into its handler table is the table it is served
//! with (`Handlers::serving`): every dispatch made through it carries the served dispatch's id as its
//! parent, and the record names it. The `lookup` tool of these cells
//! dispatches from inside its own service — a completion on the agent's
//! model key, a host note, its own key, the host's relay (which nests once
//! more), the host's never-answering handler — and both interpreters
//! register the same tool and host handlers (program, not record): the
//! replayers answer only the leaves. The rule read off the chain: a cancel
//! reaches it, so a child in flight is cancelled with its parent.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | nesting depth | 1 · 2 |
//! | where the nested dispatch is made | inside a tool's `Serve` · inside a detached sink's resolver |
//! | target key | another key (the model · a host note · the host's relay · the host's `never`) · the same key |
//! | parent fate | answered · cancelled while the child is in flight |
//! | medium | unary · streamed with events |
//!
//! Full cross-product: 2 × 3 × 5 × 3 × 2 = 180. Recorded: the 9 cells
//! below; the `serial` and `concurrent` pairs were recorded under the
//! deleted bus's two serving policies and now replay the same program.
//! Pruned: a layer's `before` as the nesting site (Matrix P, after
//! L7); depth 2 under every axis but the answered relay (the chain is the
//! claim; one cell shows three ids); the cancelled fates on the answered
//! children (only `never` can be reached and held); the streamed medium on every
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
//! | `mock_causal_same_key_concurrent_served` | `same_key_concurrent_served_…` | concurrent | `[Completion, Tool, Tool←1, Completion]` |
//! | `mock_causal_parent_cancelled_child_in_flight` | `parent_cancelled_child_in_flight_…` | concurrent | `[Completion, Tool✗, Custom✗←1]` |
//! | `mock_causal_detached_resolver` | `detached_resolver_…` | concurrent | `[Completion, Tool, Custom←1, Completion]`, the parent from the detached sink |
//!
//! `←n`: the record's parent is record `n`; `✗`: `Cancelled`. The
//! depth-two cell is also interpreted by the resumed engine (the parent
//! ids survive the resumption).
//!
//! # What the matrix found
//!
//! - A tool had no way back to the handlers it was served by without
//!   holding a dispatcher of its own. The serving table
//!   (`Handlers::serving`) is the way back; the chain is the rule.

use crate::corpus;

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
const SAME_KEY_CONCURRENT_SERVED: Program = Program {
    fixture: "mock_causal_same_key_concurrent_served",
    nesting: Some(nesting(NestedChild::Same)),
    ..MOCK
};
const PARENT_CANCELLED_CHILD_IN_FLIGHT: Program = Program {
    fixture: "mock_causal_parent_cancelled_child_in_flight",
    nesting: Some(nesting(NestedChild::Never)),
    cancel_when_reached: true,
    ..MOCK
};
const DETACHED_RESOLVER: Program = Program {
    fixture: "mock_causal_detached_resolver",
    nesting: Some(Nesting {
        child: NestedChild::Note,
        detached: true,
        no_temperature: false,
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
    same_key_concurrent_served: SAME_KEY_CONCURRENT_SERVED,
    parent_cancelled_child_in_flight: PARENT_CANCELLED_CHILD_IN_FLIGHT,
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
    assert_eq!(
        parents(PARENT_CANCELLED_CHILD_IN_FLIGHT.fixture),
        [None, None, Some(1)]
    );
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
                .find_map(|message| message.rag_text())
                .expect("a prompt"),
            other => panic!("a completion, not {other:?}"),
        };
        assert_eq!(question, "What is the capital of France?");
    }
    // The two recordings (made under the bus's two serving policies)
    // record the same chain.
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

/// A child on the tool's own key is served, and recorded.
#[test]
fn a_child_on_the_tools_own_key_is_served() {
    let served = corpus::golden(SAME_KEY_CONCURRENT_SERVED.fixture);
    assert_eq!(served.len(), 4);
    assert_eq!(served[2].key, served[1].key, "the tool's own key");
}

/// A parent's cancel reaches its children: both records say `Cancelled`,
/// the child's after the parent's.
#[test]
fn a_cancelled_parent_cancels_its_children() {
    for cell in [&PARENT_CANCELLED_CHILD_IN_FLIGHT] {
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
