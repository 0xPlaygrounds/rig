//! Matrix T: the leftovers of the #2443 review, and `Denied`.
//!
//! The six leftovers were fixed as found (L1–L4); the ones a golden can
//! show are here: a host effect that does not serialize never reaches the
//! bus and leaves no record (L3); a required host key of the embed, rerank
//! or custom family the program registers and never dispatches to is
//! described from the handler table on replay, and a key nothing describes
//! is refused by name (L4); the recorder finds its slot from the back — five
//! thousand kept events on one key beside two hundred other records land in
//! the last slot, in order (L2). And `ErrorKind::Denied` under every
//! consumer mapping (L7, ruling 10): on a tool the model sees the skipped
//! result; on a completion the run fails; on a memory `Load` the run fails
//! at the record (`MemoryError::Policy`); on a custom effect from a hook,
//! the hook sees `Denied`. None of the denials is a record: the layer
//! decided before any handler served the dispatch. L1 is a loom model and
//! L2's search order a unit test beside this cell.
//!
//! # Cells
//!
//! | golden | producer | shape |
//! |---|---|---|
//! | `mock_leftovers_denied_tool` | `tests/core/golden_leftovers.rs` `denied_tool_…` | `[Completion, Completion]`: the skipped result in the second request |
//! | `mock_leftovers_denied_tool_streamed` | `denied_tool_streamed_…` | the same, streamed with events |
//! | `mock_leftovers_denied_completion` | `denied_completion_…` | `[]`: the run fails with `Denied` before any record |
//! | `mock_leftovers_denied_memory_load` | `denied_memory_load_…` | `[]`: the run fails at the load, `MemoryError::Policy` |
//! | `mock_leftovers_denied_custom_from_hook` | `denied_custom_from_hook_…` | `[Completion]`: the hook saw `Denied` and went on |
//! | `mock_leftovers_denied_custom_from_hook_streamed` | `denied_custom_from_hook_streamed_…` | the same, streamed with events |
//! | `mock_leftovers_unserializable_from_hook` | `unserializable_from_hook_…` | `[Completion]`: the hook saw `Request`, the handler was never entered (L3) |
//! | `mock_leftovers_required_embed` | `required_embed_…` | `[Completion]`; `host/embed` in the table, in no record (L4) |
//! | `mock_leftovers_required_rerank` | `required_rerank_…` | `[Completion]`; `host/rerank` likewise |
//! | `mock_leftovers_required_custom` | `required_custom_…` | `[Completion]`; `host/note` likewise |
//! | `mock_leftovers_five_thousand_events` | `five_thousand_events_…` | 201 records: two hundred host notes from the run-start hook, then the answer as a stream of 5 000 deltas kept verbatim (L2; the one large golden — its size is in the PR) |
//!
//! Pruned: `Rerank` through a mock reranker as a dispatched effect is
//! Matrix O's `mock_oracle_rerank`; the L1 close race is
//! `loom_close_for_commands_never_strands_a_late_enqueue`; the required
//! custom key with no table entry is an in-memory row here
//! (`a_key_nothing_describes_is_refused_by_name`), not a golden.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Ending, Hook, LayerAt, LayerKind, LayerSpec, Program};
use rig_core::effect::{EffectKind, HandlerKey};
use rig_core::error::ErrorKind;
use rig_effect_log::EffectLogReplayer;

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const PROMPT: &str = "Reply with the single word: ready.";

const fn deny(at: LayerAt) -> LayerSpec {
    LayerSpec {
        at,
        layer: LayerKind::DenyAll,
    }
}

const BASIC: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    max_turns: Some(200),
    ..Program::DEFAULT
};
const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(200),
    ..Program::DEFAULT
};

const DENIED_TOOL: Program = Program {
    fixture: "mock_leftovers_denied_tool",
    max_turns: Some(3),
    layers: &[deny(LayerAt::Tool)],
    ..TOOLS
};
const DENIED_TOOL_STREAMED: Program = Program {
    fixture: "mock_leftovers_denied_tool_streamed",
    streamed: true,
    layers: &[deny(LayerAt::Tool)],
    ..TOOLS
};
const DENIED_COMPLETION: Program = Program {
    fixture: "mock_leftovers_denied_completion",
    max_turns: None,
    ending: Ending::Failed(ErrorKind::Denied),
    layers: &[deny(LayerAt::Model)],
    ..BASIC
};
const DENIED_MEMORY_LOAD: Program = Program {
    fixture: "mock_leftovers_denied_memory_load",
    max_turns: None,
    conversation: Some(corpus::CONVERSATION),
    ending: Ending::MemoryError,
    layers: &[deny(LayerAt::Memory)],
    ..BASIC
};
const DENIED_CUSTOM_FROM_HOOK: Program = Program {
    fixture: "mock_leftovers_denied_custom_from_hook",
    max_turns: None,
    hooks: &[Hook::NoteDeniedAtStart],
    layers: &[deny(LayerAt::Note)],
    ..BASIC
};
const DENIED_CUSTOM_FROM_HOOK_STREAMED: Program = Program {
    fixture: "mock_leftovers_denied_custom_from_hook_streamed",
    streamed: true,
    hooks: &[Hook::NoteDeniedAtStart],
    layers: &[deny(LayerAt::Note)],
    ..BASIC
};
const UNSERIALIZABLE_FROM_HOOK: Program = Program {
    fixture: "mock_leftovers_unserializable_from_hook",
    hooks: &[Hook::NoteUnserializableAtStart],
    ..BASIC
};
const REQUIRED_EMBED: Program = Program {
    fixture: "mock_leftovers_required_embed",
    ..BASIC
};
const REQUIRED_RERANK: Program = Program {
    fixture: "mock_leftovers_required_rerank",
    ..BASIC
};
const REQUIRED_CUSTOM: Program = Program {
    fixture: "mock_leftovers_required_custom",
    ..BASIC
};
const FIVE_THOUSAND_EVENTS: Program = Program {
    fixture: "mock_leftovers_five_thousand_events",
    streamed: true,
    hooks: &[Hook::NotesAtStart(200)],
    ..BASIC
};

both_interpreters! {
    denied_tool: DENIED_TOOL,
    denied_tool_streamed: DENIED_TOOL_STREAMED,
    denied_completion: DENIED_COMPLETION,
    denied_memory_load: DENIED_MEMORY_LOAD,
    denied_custom_from_hook: DENIED_CUSTOM_FROM_HOOK,
    denied_custom_from_hook_streamed: DENIED_CUSTOM_FROM_HOOK_STREAMED,
    unserializable_from_hook: UNSERIALIZABLE_FROM_HOOK,
    required_embed: REQUIRED_EMBED,
    required_rerank: REQUIRED_RERANK,
    required_custom: REQUIRED_CUSTOM,
    five_thousand_events: FIVE_THOUSAND_EVENTS,
}

/// A denial before the handler is never a record; the keys the layers sat
/// on are in the handler table with the layer named.
#[test]
fn a_denial_is_never_a_record() {
    assert_eq!(corpus::golden(DENIED_TOOL.fixture).len(), 2);
    assert_eq!(corpus::golden(DENIED_TOOL_STREAMED.fixture).len(), 2);
    assert_eq!(corpus::golden(DENIED_COMPLETION.fixture).len(), 0);
    assert_eq!(corpus::golden(DENIED_MEMORY_LOAD.fixture).len(), 0);
    assert_eq!(corpus::golden(DENIED_CUSTOM_FROM_HOOK.fixture).len(), 1);
    assert_eq!(
        corpus::golden(DENIED_CUSTOM_FROM_HOOK_STREAMED.fixture).len(),
        1
    );
    for (cell, key) in [
        (&DENIED_COMPLETION, "golden/model:default"),
        (&DENIED_MEMORY_LOAD, "golden/memory"),
        (&DENIED_CUSTOM_FROM_HOOK, corpus::NOTE_KEY),
    ] {
        let log = corpus::golden(cell.fixture);
        let handler = log
            .header
            .handlers
            .iter()
            .find(|handler| handler.key.as_str() == key)
            .unwrap_or_else(|| panic!("{}: `{key}` in the handler table", cell.fixture));
        assert_eq!(handler.layers, ["DenyAllLayer"]);
        assert!(
            log.header.signature.get(&HandlerKey::from(key)).is_none(),
            "{}: a key with no record is not in the signature",
            cell.fixture
        );
    }
}

/// L3: the hook's effect had no wire form — the record holds nothing for
/// it, and the host's handler is in the table with no record.
#[test]
fn an_unserializable_effect_leaves_no_record() {
    let log = corpus::golden(UNSERIALIZABLE_FROM_HOOK.fixture);
    assert_eq!(log.len(), 1);
    assert!(
        log.header
            .handlers
            .iter()
            .any(|handler| handler.key.as_str() == corpus::UNSERIALIZABLE_KEY)
    );
    assert!(
        log.header
            .signature
            .get(&HandlerKey::from(corpus::UNSERIALIZABLE_KEY))
            .is_none()
    );
}

/// L4: a required host key of any family, described from the handler
/// table on replay, answers a dispatch with a divergence — the log ran out
/// before it began.
#[tokio::test]
async fn a_required_key_of_any_family_is_described_from_the_table_and_answers_a_divergence() {
    let cells: [(&Program, &str, EffectKind); 3] = [
        (
            &REQUIRED_EMBED,
            "host/embed",
            EffectKind::Embed {
                inputs: rig_core::effect::EmbedInputs::Texts(vec!["a".to_owned()]),
            },
        ),
        (
            &REQUIRED_RERANK,
            "host/rerank",
            EffectKind::Rerank {
                request: corpus::rerank_request("q"),
            },
        ),
        (
            &REQUIRED_CUSTOM,
            corpus::NOTE_KEY,
            EffectKind::Custom {
                kind: std::sync::Arc::from("corpus:note"),
                payload: serde_json::json!({"at": "never"}),
            },
        ),
    ];
    for (cell, key, kind) in cells {
        let replay = corpus::Replay::open(cell);
        let key = HandlerKey::from(key);
        let descriptor = replay
            .dispatcher
            .descriptor(&key)
            .unwrap_or_else(|| panic!("{}: `{key}` served on replay", cell.fixture));
        assert_eq!(descriptor.family.family(), kind.family());
        let report = corpus::within(replay.dispatcher.dispatch(&key, kind))
            .await
            .expect_err("no record to answer from");
        assert_eq!(
            report.kind,
            ErrorKind::Divergence,
            "{}: {report:?}",
            cell.fixture
        );
        assert!(
            report.message.contains("after its log ran out"),
            "{}: {}",
            cell.fixture,
            report.message
        );
        replay.close().await;
    }
}

/// L4: a required custom key with no handler-table entry is refused by
/// name, never dropped.
#[test]
fn a_key_nothing_describes_is_refused_by_name() {
    let mut log = corpus::golden(REQUIRED_CUSTOM.fixture);
    let note = HandlerKey::from(corpus::NOTE_KEY);
    log.header.handlers.retain(|handler| handler.key != note);
    let refusal = match EffectLogReplayer::for_key(&log, &note) {
        Ok(_) => panic!("nothing describes the key"),
        Err(report) => report,
    };
    assert_eq!(refusal.kind, ErrorKind::HandlerUnavailable);
    assert!(
        refusal.message.starts_with("`host/note` has no records"),
        "{}",
        refusal.message
    );
    let every = match EffectLogReplayer::for_log(&log) {
        Ok(replayers) => replayers,
        Err(report) => panic!("the table names every other key: {report:?}"),
    };
    assert!(
        every.iter().all(|replayer| replayer.key() != &note),
        "a key nothing describes is not a replayer either"
    );
}

/// L2: five thousand kept events on one key beside two hundred other
/// records, every one in the last slot and in order.
#[test]
fn the_recorder_kept_five_thousand_events_in_order() {
    let log = corpus::golden(FIVE_THOUSAND_EVENTS.fixture);
    assert_eq!(log.len(), 201);
    let events = log[200].events.as_ref().expect("kept");
    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            rig_core::streaming::StreamEvent::BlockDelta {
                delta: rig_core::streaming::Delta::Text { text },
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas.len(), 5000);
    assert!(
        deltas
            .iter()
            .enumerate()
            .all(|(n, delta)| *delta == format!("{n} "))
    );
    // Every tool turn's stream was kept too, small as it is.
    assert!(log.iter().take(200).all(|record| {
        record.kind.family() != rig_core::effect::EffectFamily::Completion
            || record.events.is_some()
    }));
}
