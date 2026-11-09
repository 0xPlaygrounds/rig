//! Matrix O: the oracle and the header.
//!
//! What the corpus's own instruments say, pinned. The record oracle is a total order: every
//! golden's ids are strictly increasing in log order (`assert_same_records`), and the replay is
//! compared position by position across every key. Every recorded stream item reads back on its
//! own to the value it was recorded as. A hook that carries state names itself in the header
//! (`AgentHook::name`), so two programs differing only in that state are two headers. The
//! handler table, the signature and the required row are three sets with stated inclusions.
//!
//! Two tool calls served concurrently with a note inside each dispatch replay as
//! `[Tool, Tool, Custom, Custom]`: the engine mints both dispatches before the driver serves
//! either. The pin is that shape, not which note comes first, since both notes carry the same
//! payload; a runtime that answers the second dispatch first would need a re-record, not a fix.
//! `Rerank` has a golden through a mock reranker on a host's bus, because no keyed rerank wire
//! has a cassette suite, and a `Prompted` answer that misses the output schema fails the
//! one-turn run.

use crate::corpus;

use corpus::{Ending, Hook, Output, Program, RERANK_KEY};
use rig_core::effect::{EffectFamily, HandlerKey};
use rig_core::streaming::{Item, StreamEvent};

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
const PROMPTED_MISSES_SCHEMA: Program = Program {
    fixture: "mock_oracle_prompted_misses_schema",
    preamble: Some(BASIC_PREAMBLE),
    prompt: STRUCTURED_OUTPUT_PROMPT,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Prompted),
    ending: Ending::MissesSchema,
    ..Program::DEFAULT
};

both_interpreters! {
    concurrent_notes: CONCURRENT_NOTES,
    stop_after_turn_two: STOP_AFTER_TURN_TWO,
    rerank: RERANK,
    prompted_misses_schema: PROMPTED_MISSES_SCHEMA,
}

fn every_golden() -> Vec<(String, rig_cassette::effect_log::EffectLog)> {
    let dir = corpus::fixtures::effects_dir();
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
            if key.parts().owner.as_deref() == Some("host") {
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

/// Every stream item a golden recorded reads back on its own, without the
/// items around it, and writes the value it was recorded as: a client can
/// decode a relayed stream item by item.
#[test]
fn every_recorded_stream_item_reads_back_on_its_own() {
    let dir = corpus::fixtures::effects_dir();
    let mut read = 0;
    for (name, _) in every_golden() {
        let path = dir.join(format!("{name}.effects.json"));
        let text = std::fs::read_to_string(&path).expect("the golden reads");
        let log: serde_json::Value = serde_json::from_str(&text).expect("the golden is JSON");
        let records = log["records"].as_array().expect("records");
        for events in records
            .iter()
            .filter_map(|record| record["events"].as_array())
        {
            for value in events {
                let item: Item<StreamEvent> = serde_json::from_value(value.clone())
                    .unwrap_or_else(|error| panic!("{name}: {value} does not read back: {error}"));
                assert_eq!(
                    &serde_json::to_value(&item).expect("an item serializes"),
                    value,
                    "{name}: an item writes what was recorded"
                );
                read += 1;
            }
        }
    }
    assert!(read > 1000, "the corpus records streams: {read} items");
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
