use std::sync::mpsc::Receiver;

use rig_compaction::{Compact, CompactionDefault, CompactionPlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_memory::{Messages, rewrite};
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, start_call,
};
use serde_json::json;

use super::*;

/// Lines of the probe's output.
const LINES: usize = 300;

/// The probe's output: long, with the error near its top.
fn probe_output() -> String {
    let mut lines = vec![
        "probe started".to_owned(),
        "error ERR_CODE_7731: the socket closed".to_owned(),
    ];
    lines.extend((3..LINES).map(|n| format!("trace line {n} of the probe output")));
    lines.push("probe done".to_owned());
    lines.join("\n")
}

/// A `probe` tool that answers its output.
fn probe(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut decision) = calls.get_mut(called.call) {
        let result = ToolResult::success(probe_output().into());
        decision.decide(Decider::plugin("probe"), result);
    }
}

/// The kernel, rig-compaction and the archive on `store`, after the first
/// frame restored the session.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .add_open_tool("probe", "Run the probe.", ToolOptions::default(), probe);
    rig_harness::load::<CompactionPlugin>(
        &mut app,
        "rig-compaction",
        "path plugins/rig-compaction",
    );
    rig_harness::load::<CompactionArchivePlugin>(
        &mut app,
        "corpus-ct-compaction-archive",
        "path corpus/ct-compaction-archive",
    );
    // A compaction the user asks for keeps the newest 20 estimated tokens:
    // the last two turns of short messages.
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .keep_asked = 20;
    app.finish();
    app.update();
    (app, wakes)
}

fn agent_of(app: &mut App) -> Entity {
    first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER)
}

fn use_model(app: &mut App, agent: Entity, model: &MockCompletionModel) {
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
}

fn ended(world: &World) -> usize {
    world.resource::<Ended>().0.len()
}

/// Runs frames until `count` turns ended, then one more for the journal.
fn until_ended(app: &mut App, wakes: &Receiver<()>, count: usize) {
    run_until(app, wakes, |world| ended(world) >= count);
    app.update();
}

/// The user says `text`; the turn runs to its end.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let count = ended(app.world()) + 1;
    let deliver = Deliver::new(agent, text, DeliveryMode::Steer);
    app.world_mut().trigger(deliver);
    until_ended(app, wakes, count);
}

/// Compacts the idle agent.
fn compact(app: &mut App, wakes: &Receiver<()>, agent: Entity) {
    let count = ended(app.world()) + 1;
    let focus = String::new();
    app.world_mut().trigger(Compact {
        entity: agent,
        focus,
    });
    until_ended(app, wakes, count);
}

/// Calls `tool` with `args` as call `id`, and its output.
fn call(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    (tool, id): (&str, &str),
    args: serde_json::Value,
) -> String {
    start_call(app.world_mut(), agent, (tool, id), args);
    let done = |world: &World| world.resource::<ToolOutputs>().0.contains_key(id);
    run_until(app, wakes, done);
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

fn archive(app: &App, store: &MemoryStore, agent: Entity) -> Vec<Archived> {
    let world = app.world();
    let (Some(id), Some(conversation)) = (
        world.get::<AgentId>(agent),
        world.get::<Conversation>(agent),
    ) else {
        return Vec::new();
    };
    read_archive(store, &id.0, conversation).unwrap_or_default()
}

fn ids(app: &App, agent: Entity) -> Vec<MessageId> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |it| it.entries().map(|entry| entry.id).collect())
}

/// The entry of `agent`'s conversation whose text has `needle`.
fn find(app: &App, agent: Entity, needle: &str) -> Option<MessageId> {
    let conversation = app.world().get::<Conversation>(agent)?;
    let mut entries = conversation.entries();
    let found = entries.find(|entry| text_of(entry.message).contains(needle));
    found.map(|entry| entry.id)
}

/// The archive id of message `id`.
fn archive_id(id: MessageId) -> String {
    let number = serde_json::to_value(id)
        .ok()
        .and_then(|value| value.as_u64());
    format!("m{}", number.unwrap_or(u64::MAX))
}

/// A removal: the message that says `VAULT_PATH` is replaced everywhere
/// the session keeps it.
fn forget_vault(message: &mut message::Message) -> bool {
    let hit = text_of(message).contains("VAULT_PATH");
    if hit {
        *message = message::Message::user("[removed]");
    }
    hit
}

#[test]
fn compacted_messages_are_found_across_compactions_forks_restarts_and_removals() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let agent = agent_of(&mut app);
    let early = MockCompletionModel::from_stream_turns([
        reply("one"),
        calls("c2", "probe", json!({})),
        reply("the probe failed"),
        reply("noted"),
        reply("four"),
        reply("five"),
        reply("six"),
        reply("## Goal\nFirst summary: the probe failed early on."),
    ]);
    use_model(&mut app, agent, &early);
    for text in [
        "turn one",
        "run the probe",
        "the API key lives in VAULT_PATH",
        "turn four",
        "turn five",
        "turn six",
    ] {
        say(&mut app, &wakes, agent, text);
    }
    let error = find(&app, agent, "ERR_CODE_7731");
    assert!(error.is_some(), "the probe's result is in the conversation");
    let error = archive_id(error.unwrap_or_default());

    // The first compaction keeps turns 5 and 6: turns 1 to 4 (four
    // requests, four replies, a call and its result) are archived, and
    // the summary names the archive.
    compact(&mut app, &wakes, agent);
    let first = archive(&app, &store, agent);
    assert_eq!(first.len(), 10, "{first:#?}");
    let summary = find(&app, agent, "First summary");
    assert!(summary.is_some(), "the summary is in the conversation");
    assert!(find(&app, agent, ARCHIVED_LINE) == summary, "{summary:?}");

    // The next turn searches, then reads the whole result.
    let middle = MockCompletionModel::from_stream_turns([
        calls("s1", SEARCH, json!({ "query": "ERR_CODE_7731" })),
        calls("g1", GET, json!({ "id": error })),
        reply("found it"),
        reply("eight"),
        reply("## Goal\nSecond summary."),
    ]);
    use_model(&mut app, agent, &middle);
    say(&mut app, &wakes, agent, "what was that error code?");
    let sent = middle.requests();
    let sent = sent
        .first()
        .map(|request| serde_json::to_string(request).unwrap_or_default());
    assert!(
        sent.is_some_and(|it| it.contains(ARCHIVED_LINE)),
        "the model reads the line"
    );
    let searched = output(&app, "s1");
    assert!(
        searched.contains("1 archived message matches"),
        "{searched}"
    );
    assert!(
        searched.contains(&format!("[{error}] turn 2, tool")),
        "{searched}"
    );
    let got = output(&app, "g1");
    assert!(
        got.contains("error ERR_CODE_7731: the socket closed"),
        "{got}"
    );
    assert!(got.contains("[truncated: lines 1-"), "{got}");
    let rest = call(
        &mut app,
        &wakes,
        agent,
        (GET, "g2"),
        json!({ "id": error, "from_line": LINES - 10, "to_line": LINES }),
    );
    assert!(
        rest.ends_with(&format!(
            "probe done\n[lines {}-{LINES} of {LINES}]",
            LINES - 10
        )),
        "{rest}"
    );

    // A second compaction archives only what it condensed, the first
    // summary with it.
    say(&mut app, &wakes, agent, "turn eight");
    let fork_point = find(&app, agent, "turn eight");
    let before = ids(&app, agent);
    compact(&mut app, &wakes, agent);
    let after = ids(&app, agent);
    let condensed = before.iter().filter(|id| !after.contains(id)).copied();
    let mut expected: Vec<MessageId> = first.iter().map(|it| it.id).chain(condensed).collect();
    expected.sort();
    let second: Vec<MessageId> = archive(&app, &store, agent)
        .iter()
        .map(|it| it.id)
        .collect();
    assert!(second.len() > first.len(), "{second:?}");
    assert_eq!(second, expected);
    let summary = call(
        &mut app,
        &wakes,
        agent,
        (SEARCH, "s2"),
        json!({ "query": "first summary" }),
    );
    assert!(summary.contains("1 archived message matches"), "{summary}");

    // A fork from before turn 8 sees the archive as it was then.
    let log = app
        .world()
        .get::<AgentId>(agent)
        .map(|id| id.0.clone())
        .unwrap_or_default();
    let forked = MemoryStore::default();
    assert!(fork_point.is_some(), "turn 8 is in the conversation");
    let point = fork_point.map(|id| (log.as_str(), id));
    let copied = fork(&store, &forked, point, &NoRewrite);
    assert!(copied.is_ok(), "{copied:?}");
    drop(app);

    // Restarted, the session still searches; a removal takes the message
    // out of the archive too.
    let (mut app, wakes) = start(&store);
    let agent = agent_of(&mut app);
    let found = call(
        &mut app,
        &wakes,
        agent,
        (SEARCH, "s3"),
        json!({ "query": "ERR_CODE_7731" }),
    );
    assert!(found.contains("1 archived message matches"), "{found}");
    let vault = call(
        &mut app,
        &wakes,
        agent,
        (SEARCH, "s4"),
        json!({ "query": "VAULT_PATH", "kind": "user" }),
    );
    assert!(vault.contains("turn 3, user"), "{vault}");
    drop(app);
    let forgot = rewrite(&store, Author::plugin("forget"), &Messages(forget_vault));
    assert!(forgot.is_ok(), "{forgot:?}");
    let (mut app, wakes) = start(&store);
    let agent = agent_of(&mut app);
    let vault = call(
        &mut app,
        &wakes,
        agent,
        (SEARCH, "s5"),
        json!({ "query": "VAULT_PATH" }),
    );
    assert!(vault.starts_with("No archived message matches"), "{vault}");
    drop(app);

    let (mut fork_app, wakes) = start(&forked);
    let agent = agent_of(&mut fork_app);
    let summary = call(
        &mut fork_app,
        &wakes,
        agent,
        (SEARCH, "s6"),
        json!({ "query": "first summary" }),
    );
    assert!(
        summary.starts_with("No archived message matches"),
        "{summary}"
    );
    let error = call(
        &mut fork_app,
        &wakes,
        agent,
        (SEARCH, "s7"),
        json!({ "query": "ERR_CODE_7731" }),
    );
    assert!(error.contains("1 archived message matches"), "{error}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CompactionArchivePlugin);
}
