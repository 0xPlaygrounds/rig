//! Matrix S: the header's new types.
//!
//! The header carries `required: EffectRow`, `signature: EffectRow` and
//! handler descriptors. Three readers check
//! them before the first dispatch: the agent (`check_replayable`), the
//! replayer (`check_header`, `for_key`) and a host checking a row against
//! the bus it runs (`EffectRow::is_subset_of` over
//! `Dispatcher::descriptors()`, a host's startup check). Every cell edits
//! one header field of an existing golden in memory and pins the *text*
//! of refusals or accepted compatibility cases. A header has no global
//! format number, and a key it does not know is refused. No new
//! recordings.
//!
//! # Dimensions
//!
//! | axis | values |
//! |---|---|
//! | field | `required` · `signature` · an unknown key · the handler table |
//! | mismatch | missing key · extra key · family change · unknown key refused · table entry missing |
//! | who checks | `check_replayable` (the agent) · `check_header` / `for_key` (the replayer) · `is_subset_of` over a bare bus (a host) |
//!
//! Full cross-product: 4 × 5 × 3 = 60. Recorded: the 9 cells below.
//! Pruned: a mismatch a field cannot have (an unknown key has no family);
//! the replayer's reading of the required row beyond the table cell (it
//! describes, it does not compare); the host's reading of the signature
//! (a host checks the row it needs, the bus it built).
//!
//! # Cells
//!
//! | cell | field · mismatch · reader |
//! |---|---|
//! | `the_agent_names_a_required_key_the_log_never_served` | required · missing key · agent |
//! | `the_agent_names_a_required_key_served_as_another_family` | required · family change · agent |
//! | `the_agent_names_a_key_the_log_required_and_it_does_not` | required · extra key · agent |
//! | `the_agent_names_a_required_family_that_differs_between_the_rows` | required · family change between rows · agent |
//! | `the_agent_names_a_signature_key_nothing_serves` | signature · missing key · agent |
//! | `the_agent_names_a_signature_family_the_bus_serves_otherwise` | signature · family change · replayer first (the agent's check begins with `check_header`) |
//! | `a_log_header_refuses_an_unknown_key` | an unknown key · refused · the decoder, before any reader |
//! | `the_replayer_refuses_a_key_nothing_describes_by_name` | table · entry missing · replayer |
//! | `a_host_checks_its_row_against_the_bus_it_built` | required · missing key and family change · host |

use crate::corpus;
use rig_cassette::agent::AgentReplayExt;

use corpus::Program;
use rig_cassette::effect_log::{EffectLog, EffectLogReplayer};
use rig_core::effect::{EffectFamily, EffectRow, FamilyDescriptor, HandlerKey};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const PROMPT: &str = "Reply with the single word: ready.";

/// An own-bus golden with a tool: the required row has the model and
/// `add`, the policy is the agent's.
const TOOLS: Program = Program {
    fixture: "anthropic_hooks_patch_tool_args",
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    hooks: &[corpus::Hook::PatchAddArgs],
    ..Program::DEFAULT
};

/// A host-bus golden with the host's custom key in the handler table.
const HOST: Program = Program {
    fixture: "anthropic_host_custom_at_start",
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    hooks: &[corpus::Hook::NoteAtStart],
    ..Program::DEFAULT
};

const MODEL: &str = "golden/model:default";
const ADD: &str = "golden/tool:add#0";
const GHOST: &str = "golden/tool:ghost#0";

/// The program's agent over the replay bus, and the refusal it gives
/// `edit`ed copy of the golden's header.
async fn refusal(program: &Program, edit: impl FnOnce(&mut EffectLog)) -> String {
    let replay = corpus::Replay::open(program);
    let server = replay.tool_server_for(program);
    let agent = corpus::build_agent_unchecked(&replay, program, server, &replay.log);
    let mut log = replay.log.clone();
    edit(&mut log);
    let refusal = agent
        .check_replayable(&log)
        .expect_err("the edited header is another program's")
        .to_string();
    drop(agent);
    replay.close().await;
    refusal
}

fn family_of(log: &EffectLog, key: &str) -> EffectFamily {
    *log.header
        .required
        .get(&HandlerKey::from(key))
        .expect("the row names the key")
}

#[tokio::test]
async fn the_agent_names_a_required_key_the_log_never_served() {
    let refusal = refusal(&TOOLS, |log| {
        log.header
            .handlers
            .retain(|handler| handler.key.as_str() != ADD);
    })
    .await;
    assert_eq!(
        refusal,
        format!(
            "replay refused: this agent needs `{ADD}` (tool_call), which the log never served: `{ADD}` (tool_call) is not served"
        )
    );
}

#[tokio::test]
async fn the_agent_names_a_required_key_served_as_another_family() {
    let refusal = refusal(&TOOLS, |log| {
        let entry = log
            .header
            .handlers
            .iter_mut()
            .find(|handler| handler.key.as_str() == ADD)
            .expect("the tool's entry");
        entry.family = FamilyDescriptor::Retrieve {};
    })
    .await;
    assert_eq!(
        refusal,
        format!("replay refused: conflicting families for `{ADD}`: tool_call and retrieve")
    );
}

#[tokio::test]
async fn the_agent_names_a_key_the_log_required_and_it_does_not() {
    let refusal = refusal(&TOOLS, |log| {
        log.header
            .required
            .insert(HandlerKey::from(GHOST), EffectFamily::Tool);
    })
    .await;
    assert!(
        refusal.starts_with("replay refused: the log was recorded by a program requiring "),
        "{refusal}"
    );
    assert!(
        refusal.contains(GHOST) && refusal.contains("this agent requires"),
        "{refusal}"
    );
    assert!(
        refusal.ends_with(&format!(": `{GHOST}` (tool_call) is missing")),
        "the diff is read from this agent's row: the key the log's row has is missing from it: {refusal}"
    );
}

#[tokio::test]
async fn the_agent_names_a_required_family_that_differs_between_the_rows() {
    let refusal = refusal(&TOOLS, |log| {
        assert_eq!(family_of(log, ADD), EffectFamily::Tool);
        log.header
            .required
            .insert(HandlerKey::from(ADD), EffectFamily::Completion);
    })
    .await;
    assert!(
        refusal
            == format!(
                "replay refused: conflicting families for `{ADD}`: tool_call and completion"
            ),
        "the inconsistent log is refused before comparing it with the agent: {refusal}"
    );
}

#[tokio::test]
async fn the_agent_names_a_signature_key_nothing_serves() {
    let refusal = refusal(&TOOLS, |log| {
        log.header
            .signature
            .insert(HandlerKey::from("host/ghost"), EffectFamily::Custom);
    })
    .await;
    assert_eq!(
        refusal,
        "replay refused: nothing serves `host/ghost`, which the log needs"
    );
}

#[tokio::test]
async fn the_agent_names_a_signature_family_the_bus_serves_otherwise() {
    let refusal = refusal(&TOOLS, |log| {
        log.header
            .signature
            .insert(HandlerKey::from(MODEL), EffectFamily::Tool);
    })
    .await;
    // The replayer reads the signature against the records first: a
    // family the records contradict is refused there, before the agent
    // compares the signature with its bus.
    assert_eq!(
        refusal,
        format!("replay refused: conflicting families for `{MODEL}`: tool_call and completion")
    );
}

#[tokio::test]
async fn a_log_header_refuses_an_unknown_key() {
    let log = corpus::golden(TOOLS.fixture);
    let mut json = serde_json::to_value(&log).expect("serializes");
    assert!(json["header"].get("format").is_none());
    json["header"]["format"] = serde_json::json!(999);
    let error = serde_json::from_value::<EffectLog>(json)
        .expect_err("a header key this rig does not know is refused");
    assert!(error.to_string().contains("format"), "{error}");
}

#[tokio::test]
async fn the_replayer_refuses_a_key_nothing_describes_by_name() {
    let mut log = corpus::golden(HOST.fixture);
    let note = HandlerKey::from(corpus::NOTE_KEY);
    // Nothing describes the host's key once the table forgets it and no
    // record dispatched to it.
    log.header.handlers.retain(|handler| handler.key != note);
    log.records.retain(|record| record.key != note);
    log.header.signature.remove(&note);
    let refusal = EffectLogReplayer::for_key(&log, &note)
        .err()
        .expect("nothing describes the key")
        .to_string();
    assert_eq!(
        refusal,
        format!(
            "`{}` has no records in the log, no entry in its handler table and no place in its required row: nothing describes it",
            corpus::NOTE_KEY
        )
    );
    // Named by the required row as a custom key, the table alone can
    // describe it; without an entry the refusal says so.
    log.header
        .required
        .insert(note.clone(), EffectFamily::Custom);
    let refusal = EffectLogReplayer::for_key(&log, &note)
        .err()
        .expect("only the table describes a custom key")
        .to_string();
    assert!(
        refusal.starts_with(&format!(
            "the required key `{}` (custom) cannot be described",
            corpus::NOTE_KEY
        )) && refusal.ends_with("the log's handler table has no entry for it"),
        "{refusal}"
    );
    // With the table's entry back, it is described from the table: no
    // records, and a dispatch answers a divergence.
    log.header
        .handlers
        .push(rig_core::effect::HandlerDescriptor {
            key: note.clone(),
            family: FamilyDescriptor::Custom {
                kind: "corpus:note".to_owned(),
            },
            layers: Vec::new(),
        });
    let replayer = EffectLogReplayer::for_key(&log, &note).expect("described from the table");
    assert_eq!(replayer.remaining(), 0);
}

#[tokio::test]
async fn a_host_checks_its_row_against_the_bus_it_built() {
    // A host's startup check: a host with a row (the program's required
    // row, from the log) and a bus it populated asks whether the bus
    // serves the row, before any dispatch.
    let replay = corpus::Replay::open(&TOOLS);
    let server = replay.tool_server_for(&TOOLS);
    server.attach(&replay.handlers);
    let served = replay.handlers.descriptors();
    replay
        .log
        .header
        .required
        .is_subset_of(&served)
        .expect("the bus serves the golden's row");
    let mut wider = replay.log.header.required.clone();
    wider.insert(HandlerKey::from(GHOST), EffectFamily::Tool);
    let gap = wider
        .is_subset_of(&served)
        .expect_err("a key the bus does not serve");
    assert_eq!(gap.key.as_str(), GHOST);
    assert_eq!(
        gap.to_string(),
        format!("`{GHOST}` (tool_call) is not served")
    );
    let mut wrong: EffectRow = replay.log.header.required.clone();
    wrong.insert(HandlerKey::from(MODEL), EffectFamily::Tool);
    let gap = wrong
        .is_subset_of(&served)
        .expect_err("a family the bus serves otherwise");
    assert_eq!(
        gap.to_string(),
        format!("`{MODEL}` is needed as tool_call but served as completion")
    );
    replay.close().await;
}
