//! Matrix S: the header's new types.
//!
//! The header carries `required: EffectRow`, `signature: EffectRow`,
//! `bus: Option<ServingPolicy>` and `format`, and three readers check
//! them before the first dispatch: the agent (`check_replayable`), the
//! replayer (`check_header`, `for_key`) and a host checking a row against
//! the bus it runs (`EffectRow::is_subset_of` over
//! `Dispatcher::descriptors()` — the Bevy startup check). Every cell edits
//! one header field of an existing golden in memory and pins the *text*
//! of the refusal: the key, and both families or both policies. No new
//! recordings.
//!
//! # Dimensions
//!
//! | axis | values |
//! |---|---|
//! | field | `required` · `signature` · `bus` · `format` · the handler table |
//! | mismatch | missing key · extra key · family change · policy differs · policy absent on one side · format · table entry missing |
//! | who checks | `check_replayable` (the agent) · `check_header` / `for_key` (the replayer) · `is_subset_of` over a bare bus (a host) |
//!
//! Full cross-product: 5 × 7 × 3 = 105. Recorded: the 12 cells below.
//! Pruned: a mismatch a field cannot have (a format has no key; a policy
//! has no family); the replayer's reading of the required row beyond the
//! table cell (it describes, it does not compare); the host's reading of
//! the signature and the policy (a host checks the row it needs, the bus
//! it built).
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
//! | `the_agent_names_both_policies` | bus · policy differs · agent |
//! | `a_policy_absent_on_one_side_is_accepted` | bus · absent on one side · agent |
//! | `a_log_of_another_format_is_refused_by_number` | format · format · replayer and agent |
//! | `the_replayer_refuses_a_key_nothing_describes_by_name` | table · entry missing · replayer |
//! | `a_host_checks_its_row_against_the_bus_it_built` | required · missing key and family change · host |
//! | `a_policy_round_trips_through_the_header` | bus · none · replayer and driver |

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::Program;
use rig_agent::AgentBuilder;
use rig_core::effect::{EffectFamily, EffectRow, FamilyDescriptor, HandlerKey};
use rig_core::serve::ServingPolicy;
use rig_effect_log::{EffectLog, EffectLogReplayer};

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
        entry.family = FamilyDescriptor::Memory {};
    })
    .await;
    assert_eq!(
        refusal,
        format!(
            "replay refused: this agent needs `{ADD}` (tool_call), which the log never served: `{ADD}` is needed as tool_call but served as memory"
        )
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
        refusal.ends_with(&format!(": `{ADD}` is completion here and tool_call there")),
        "the diff names both families, the log's first: {refusal}"
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
        format!(
            "replay refused: the signature says `{MODEL}` serves tool_call, its records are completion"
        )
    );
}

/// An own-bus agent of the golden's program, serving the golden's keys on
/// its own bus under `policy`.
fn own_bus_agent(replay: &corpus::Replay, policy: ServingPolicy) -> rig_agent::Agent {
    AgentBuilder::with_bus_config(
        policy,
        "default",
        rig_core::test_utils::MockCompletionModel::text("unused"),
    )
    .name("golden")
    .preamble(TOOLS_PREAMBLE)
    .temperature(0.0)
    .tool_server_handle(replay.tool_server_for(&TOOLS))
    .add_hook(corpus::PatchAddArgs)
    .build()
}

#[tokio::test]
async fn the_agent_names_both_policies() {
    let replay = corpus::Replay::open(&TOOLS);
    let recorded = replay.log.header.bus.expect("an own-bus golden");
    let mine = ServingPolicy {
        serial_per_handler: !recorded.serial_per_handler,
        ..recorded
    };
    let agent = own_bus_agent(&replay, mine);
    let refusal = agent
        .check_replayable(&replay.log)
        .expect_err("another policy")
        .to_string();
    assert_eq!(
        refusal,
        format!(
            "replay refused: the log was recorded under bus policy {recorded:?}, this agent runs under {mine:?}"
        )
    );
    drop(agent);
    replay.close().await;
}

#[tokio::test]
async fn a_policy_absent_on_one_side_is_accepted() {
    // The log names no policy (a host recorded it): an own-bus agent
    // accepts it. The agent names none (over a host's bus): a log that
    // names one is accepted.
    let replay = corpus::Replay::open(&TOOLS);
    let recorded = replay.log.header.bus.expect("an own-bus golden");
    let other = ServingPolicy {
        serial_per_handler: !recorded.serial_per_handler,
        ..recorded
    };
    let mut nameless = replay.log.clone();
    nameless.header.bus = None;
    let agent = own_bus_agent(&replay, other);
    agent
        .check_replayable(&nameless)
        .expect("a log without a policy is accepted by any agent");
    drop(agent);
    let server = replay.tool_server_for(&TOOLS);
    let hosted = corpus::build_agent_unchecked(&replay, &TOOLS, server, &replay.log);
    assert_eq!(hosted.bus_config(), None, "over a host's bus");
    hosted
        .check_replayable(&replay.log)
        .expect("an agent without a policy accepts a log that names one");
    drop(hosted);
    replay.close().await;
}

#[tokio::test]
async fn a_log_of_another_format_is_refused_by_number() {
    let mut log = corpus::golden(TOOLS.fixture);
    log.header.format = 3;
    let by_the_replayer = EffectLogReplayer::check_header(&log)
        .expect_err("format 3")
        .to_string();
    assert_eq!(
        by_the_replayer,
        "replay refused: the log is format 3, this rig reads format 5"
    );
    // The agent's check reads the header first, before its own fields.
    let by_the_agent = refusal(&TOOLS, |log| log.header.format = 3).await;
    assert_eq!(by_the_agent, by_the_replayer);
    // And the replayer's registration refuses before any key.
    let (_dispatcher, _registrar, mut driver) = rig_bus::Bus::channel();
    let by_registration = EffectLogReplayer::register_all(&log, &mut driver)
        .expect_err("format 3")
        .to_string();
    assert_eq!(by_registration, by_the_replayer);
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
    // The Bevy startup check: a host with a row (the program's required
    // row, from the log) and a bus it populated asks whether the bus
    // serves the row, before any dispatch.
    let replay = corpus::Replay::open(&TOOLS);
    let server = replay.tool_server_for(&TOOLS);
    server.attach(&replay.registrar);
    let served = replay.dispatcher.descriptors();
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

#[tokio::test]
async fn a_policy_round_trips_through_the_header() {
    let log = corpus::golden(TOOLS.fixture);
    let policy = log.header.bus.expect("an own-bus golden");
    let json = serde_json::to_value(&log.header).expect("serializes");
    assert_eq!(
        json["bus"],
        serde_json::json!({
            "command_capacity": policy.command_capacity,
            "stream_capacity": policy.stream_capacity,
            "serial_per_handler": policy.serial_per_handler
        })
    );
    let restored: rig_effect_log::LogHeader = serde_json::from_value(json).expect("restores");
    assert_eq!(restored.bus, Some(policy));
    // And back into a bus: the driver a host builds under it runs it.
    let (_dispatcher, _registrar, driver) = rig_bus::Bus::channel_with(policy);
    assert_eq!(*driver.config(), policy);
    // A header without one is the host's business.
    let mut nameless = restored;
    nameless.bus = None;
    let json = serde_json::to_value(&nameless).expect("serializes");
    assert!(json.get("bus").is_none(), "absent when none");
}
