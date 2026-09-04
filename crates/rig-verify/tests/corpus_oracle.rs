//! Matrix O: the oracle and the header.
//!
//! What the corpus's own instruments say, pinned: the record oracle is a
//! total order (every golden's ids are strictly increasing in log order,
//! and the replay is compared position by position across every key,
//! which the earlier passes' risk statements said it was not); two tool
//! calls served concurrently with a note inside each dispatch replay in
//! the recorded order; a hook that carries state names itself in the
//! header, so two programs differing only in that state are two headers;
//! `Rerank` has a golden through a mock reranker on a host's bus; a
//! `Prompted` answer is returned unvalidated; the handler table, the
//! signature and the required row are three sets with stated inclusions;
//! and the Bevy host replays a golden (`tests/core/bevy_bus_host.rs`).
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | instrument | id order · concurrent cross-key order · hook identity · the three header sets · a third interpreter |
//! | family | `Rerank` (mock) · `Prompted` unvalidated |
//!
//! Recorded: 4 goldens and 3 corpus-wide tests below (the matrix is the
//! oracle's, so its rows are assertions over the corpus as much as
//! cells). Pruned: a keyed rerank wire (none has a cassette suite:
//! voyageai's is keyless here, llamacpp's local); a `Prompted` violation
//! reprompted (the run does not validate prompted answers: pinned, not
//! added).
//!
//! # Cells
//!
//! | golden | producer | shape |
//! |---|---|---|
//! | `anthropic_oracle_concurrent_notes` | anthropic `corpus_oracle.rs` `concurrent_notes_…` | `[Completion, (Tool, Custom) × 2 in dispatch order, Completion]` under `tool_concurrency: 2` |
//! | `anthropic_oracle_stop_after_turn_two` | `stop_after_turn_two_…` | `[Completion, Tool, Completion]`; hooks `["StopAfterTurn(2)"]` |
//! | `mock_oracle_rerank` | `tests/core/golden_oracle.rs` `oracle_rerank_…` | `[Rerank, Completion]` over a host's bus |
//! | `mock_oracle_prompted_unvalidated` | `oracle_prompted_unvalidated_…` | `[Completion]`; the prose answer returned |
//!
//! # What the matrix found
//!
//! - The oracle was already total: the recorder keeps records in id order
//!   and the replay is compared positionally across keys, so #2451's
//!   "per key only" risk was a misreading. The id assertion is now in
//!   `assert_same_records`, and no golden failed it.
//! - The concurrent cell's pin is the interleaving shape — both tool
//!   dispatches, then both notes (`[Tool, Tool, Custom, Custom]`), since
//!   the engine mints both dispatches before the driver serves either
//!   and each note follows its tool's answer — and not which tool's note
//!   comes first: both notes carry the same payload (`at: "outcome"`),
//!   so the two orders are one record sequence. The replay reproduces the
//!   shape under the test runtime (current-thread tokio); a runtime that
//!   answers the second dispatch before the first would move a note, and
//!   that is a re-record with the rule restated, not a defect.
//! - Hook identity by type name could not tell two programs apart whose
//!   hooks differ only in a value; `AgentHook::name` lets a hook name
//!   itself, the stack records it, and the stateful stop's header reads
//!   `StopAfterTurn(2)` (`rig-agent`, with a unit test).

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Ending, Hook, Output, Program, RERANK_KEY};
use rig_core::effect::{EffectFamily, HandlerKey};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const PROMPT: &str = "Reply with the single word: ready.";
const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

const CONCURRENT_NOTES: Program = Program {
    fixture: "anthropic_oracle_concurrent_notes",
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: TWO_TOOL_STREAM_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(8),
    tool_concurrency: Some(2),
    hooks: &[Hook::NoteAtOutcome],
    ..Program::DEFAULT
};
const STOP_AFTER_TURN_TWO: Program = Program {
    fixture: "anthropic_oracle_stop_after_turn_two",
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    hooks: &[Hook::StopAfterTurnN(2)],
    ending: Ending::Cancelled("stopped after turn 2"),
    ..Program::DEFAULT
};
const RERANK: Program = Program {
    fixture: "mock_oracle_rerank",
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    hooks: &[Hook::RerankDocs],
    ..Program::DEFAULT
};
const PROMPTED_UNVALIDATED: Program = Program {
    fixture: "mock_oracle_prompted_unvalidated",
    preamble: Some(BASIC_PREAMBLE),
    prompt: STRUCTURED_OUTPUT_PROMPT,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Prompted),
    ..Program::DEFAULT
};

both_interpreters! {
    concurrent_notes: CONCURRENT_NOTES,
    stop_after_turn_two: STOP_AFTER_TURN_TWO,
    rerank: RERANK,
    prompted_unvalidated: PROMPTED_UNVALIDATED,
}

fn every_golden() -> Vec<(String, rig_effect_log::EffectLog)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let mut goldens: Vec<_> = std::fs::read_dir(dir)
        .expect("the fixtures directory")
        .map(|entry| entry.expect("an entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".effects.json"))
        })
        .map(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("a name")
                .trim_end_matches(".effects.json")
                .to_owned();
            let log = corpus::golden(&name);
            (name, log)
        })
        .collect();
    goldens.sort_by(|left, right| left.0.cmp(&right.0));
    goldens
}

/// Every golden's records are in dispatch order: ids strictly increasing
/// in log order, across every key.
#[test]
fn every_golden_is_in_dispatch_order() {
    let goldens = every_golden();
    assert!(goldens.len() > 150, "the corpus: {}", goldens.len());
    for (name, log) in goldens {
        let ids: Vec<u64> = log.iter().map(|record| record.id.as_u64()).collect();
        assert!(
            ids.windows(2).all(|pair| pair[0] < pair[1]),
            "{name}: {ids:?}"
        );
    }
}

/// The header's three sets: every required key is in the handler table
/// (the builder registered it), every signature key is in the table (it
/// was served), and a host's key is in the signature and the table but
/// not the row.
#[test]
fn the_header_sets_nest_as_stated() {
    for (name, log) in every_golden() {
        let table: std::collections::BTreeSet<_> = log
            .header
            .handlers
            .iter()
            .map(|handler| handler.key.clone())
            .collect();
        for key in log.header.required.keys() {
            assert!(
                table.contains(key),
                "{name}: required `{key}` is in the table"
            );
        }
        for key in log.header.signature.keys() {
            assert!(
                table.contains(key),
                "{name}: signature `{key}` is in the table"
            );
        }
        for key in log.header.signature.keys() {
            if key.as_str().starts_with("host/") {
                assert!(!log.header.required.contains_key(key), "{name}: `{key}`");
            }
        }
    }
    let rerank = corpus::golden(RERANK.fixture);
    let key = HandlerKey::from(RERANK_KEY);
    assert_eq!(
        rerank.header.signature.get(&key),
        Some(&EffectFamily::Rerank)
    );
    assert!(!rerank.header.required.contains_key(&key));
}

/// A stateful hook is its own header: the stop-after-turn-two program's
/// stack names the state, and differs from the stateless stop's.
#[test]
fn a_stateful_hook_names_its_state_in_the_header() {
    let stateful = corpus::golden(STOP_AFTER_TURN_TWO.fixture);
    assert_eq!(stateful.header.hooks, ["StopAfterTurn(2)"]);
    let stateless = corpus::golden("anthropic_endings_turn_finished_stop");
    assert_eq!(stateless.header.hooks[0], "StopAfterTurn");
}
