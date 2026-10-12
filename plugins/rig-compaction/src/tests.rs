use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use rig_cassette::journal::MemoryStore;
use rig_core::ProviderResponseError;
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, AssistantMessage, CallId, ToolName, UserContent};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_core::transcript::final_answer;
use rig_harness::commands::RunCommand;

use super::*;
use rig_harness_test_support::{
    Ended, app_on, commit_message, connect, log_ended, reply, run_until,
};
use rig_telemetry::{Spending, UsagePlugin};

/// What a second Prepare system adds to every turn's request.
const INJECTED: &str = "Context from another plugin.";

/// The kernel, the compaction and usage plugins on `store`, after the first frame
/// restored the session, with its loop's wakes. The compaction plugin is
/// loaded as `plugins.toml` loads it.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin, UsagePlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    rig_harness::load::<CompactionPlugin>(
        &mut app,
        "rig-compaction",
        "path plugins/rig-compaction",
    );
    app.finish();
    app.update();
    (app, wakes)
}

/// The app's first agent, answered by `model` as the catalog's DeepSeek
/// model with a `window`-token context window, with `messages` in its
/// conversation.
fn agent(
    app: &mut App,
    model: &MockCompletionModel,
    window: u32,
    messages: Vec<Message>,
) -> Option<Entity> {
    let mut connection = connect(model)?;
    connection.spec = Arc::new(connection.spec.as_ref().clone().with_context_window(window));
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connection);
    for message in messages {
        commit_message(world, agent, message).ok()?;
    }
    Some(agent)
}

/// A call of the `read` tool for `path` and its `output`.
fn read(id: &str, path: &str, output: &str) -> Vec<Message> {
    let Ok(name) = ToolName::new("read") else {
        return Vec::new();
    };
    let arguments = serde_json::json!({ "path": path });
    let call = AssistantContent::tool_call(id, name.clone(), arguments);
    vec![
        Message::Assistant(AssistantMessage::new(vec![call])),
        Message::tool_result(CallId::from_wire(id), name, output),
    ]
}

/// A refusal of the request as longer than the model's window.
fn too_long() -> Vec<MockStreamEvent> {
    let refusal = ProviderResponseError::without_status("prompt is too long: 210000 tokens");
    vec![MockStreamEvent::Error(MockError::ProviderResponse(refusal))]
}

/// Asks `agent` a question, then runs frames until a turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity) {
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

/// The answer the one turn of `app` ended with.
fn answer(app: &App) -> Option<String> {
    match app.world().resource::<Ended>().0.as_slice() {
        [TurnOutcome::Answered(message)] => final_answer(message),
        _ => None,
    }
}

/// Every message of `agent`'s conversation with its origins and id.
fn entries(world: &World, agent: Entity) -> Vec<Entry> {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return Vec::new();
    };
    let at = 0..conversation.messages().len();
    at.filter_map(|at| conversation.entry(at)).collect()
}

/// Whether `entry` holds a compaction's summary.
fn holds_summary(entry: &Entry) -> bool {
    entry
        .origins
        .iter()
        .any(|(_, origin)| *origin == summary_origin())
}

/// The single agent `app` restored.
fn restored_agent(app: &mut App) -> Option<Entity> {
    let world = app.world_mut();
    world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .ok()
}

/// Each request `model` got, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect()
}

#[test]
fn a_conversation_near_the_window_is_summarized_before_the_request_and_restored() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    // A second Prepare system on every turn's request, beside the plugin's.
    let inject = |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        for mut call in &mut calls {
            if let Some(Message::User { content }) = call.request.chat_history.last_mut() {
                content.push(UserContent::text(INJECTED));
            }
        }
    };
    app.add_systems(AgentPass, inject.in_set(Stage::Prepare));
    let mut messages = vec![Message::user("What does the crate do?")];
    messages.extend(read("1", "src/lib.rs", "pub fn answer() -> u32 { 42 }"));
    let long = "word ".repeat(4_000);
    for turn in 0..6 {
        messages.push(Message::assistant(format!("{turn}: {long}")));
        messages.push(Message::user(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the user's questions about the crate."),
        reply("done"),
        reply("## Goal\nAnswer the next question."),
    ]);
    let agent = agent(&mut app, &model, 40_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let before = entries(app.world(), agent);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let requests = sent(&model);
    assert_eq!(
        requests.len(),
        2,
        "a summary, then the request: {requests:?}"
    );
    let [summary, request] = requests.as_slice() else {
        return;
    };
    assert!(
        summary.contains("You summarize a conversation"),
        "{summary}"
    );
    assert!(!summary.contains(INJECTED), "the held request was not sent");
    assert!(request.contains("## Goal"), "{request}");
    assert!(request.contains("<read-files>\\nsrc/lib.rs"), "{request}");
    assert_eq!(request.matches(INJECTED).count(), 1, "{request}");
    // The summary stands first, merged into the first kept message when
    // that is the user's; the kept messages keep their ids.
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
    let alone = matches!(live.first(), Some(Entry { message: Message::User { content }, .. })
        if content.len() == 1);
    let kept = live.get(usize::from(alone));
    assert!(
        kept.is_some_and(|kept| before.iter().any(|old| old.id == kept.id)),
        "{kept:?}"
    );
    assert!(live.len() < before.len());
    // The usage plugin counts the summary's call with the turn's.
    let spent = app.world().get::<Spending>(agent).map(|spent| spent.total);
    assert_eq!(spent.map(|spent| spent.calls), Some(2));

    // A second compaction builds on the first summary without reading it
    // again from the conversation.
    let compact = Compact {
        entity: agent,
        focus: String::new(),
    };
    app.world_mut().trigger(compact);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let requests = sent(&model);
    let again = requests.get(2).map(String::as_str).unwrap_or_default();
    assert!(again.contains("<previous-summary>"), "{again}");
    assert_eq!(
        again.matches("Answer the user's questions").count(),
        1,
        "{again}"
    );
    let live = entries(app.world(), agent);
    drop(app);

    // A restart has the conversation as it was.
    let (mut restarted, _wakes) = start(&store);
    let agent = restored_agent(&mut restarted);
    assert!(agent.is_some(), "one agent was restored");
    let Some(agent) = agent else {
        return;
    };
    let world = restarted.world();
    assert_eq!(entries(world, agent), live);
    let tracked = world
        .get::<Summarized>(agent)
        .map(|summarized| &summarized.0.tracked);
    assert!(
        tracked.is_some_and(|tracked| tracked
            .iter()
            .any(|set| set.name == "read-files" && set.values.contains("src/lib.rs"))),
        "{tracked:?}"
    );
}

/// The summaries sent again, and whether each was the plugin's.
#[derive(Resource, Default)]
struct Resent(Vec<bool>);

/// Sends the first summary again once it is answered, before it settles.
fn resend_first_summary(
    answered: Query<(Entity, Has<ProvidedBy>), (With<Summarizing>, Unreviewed)>,
    mut resent: ResMut<Resent>,
    mut commands: Commands,
) {
    for (summary, provided) in &answered {
        if resent.0.is_empty() {
            resent.0.push(provided);
            commands.queue(rig_harness::successor::resend(summary));
        }
    }
}

#[test]
fn a_summary_sent_again_mid_flight_is_taken_from_its_successor() {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.init_resource::<Resent>()
        .add_systems(AgentPass, resend_first_summary.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe stale summary."),
        reply("## Goal\nThe fresh summary."),
        reply("done"),
    ]);
    let agent = agent(&mut app, &model, 40_000, long_conversation(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert_eq!(app.world().resource::<Resent>().0, [true], "the plugin's");
    let requests = sent(&model);
    let turn_requests = requests
        .iter()
        .filter(|request| !request.contains("You summarize a conversation"));
    assert_eq!(turn_requests.count(), 1, "{requests:?}");
    let live = entries(app.world(), agent);
    let summary = live.first().map(|entry| format!("{:?}", entry.message));
    assert!(
        summary
            .as_deref()
            .is_some_and(|summary| summary.contains("fresh") && !summary.contains("stale")),
        "{summary:?}"
    );
}

#[test]
fn cleared_outputs_are_logged_and_stay_cleared_after_a_restore() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    // Five reads of 15k tokens each near an 80k window: clearing the
    // oldest is enough, so no summary is asked for.
    let big = "x".repeat(60_000);
    let mut messages = vec![Message::user("Read the five files.")];
    for (id, path) in [("1", "a"), ("2", "b"), ("3", "c"), ("4", "d"), ("5", "e")] {
        messages.extend(read(id, path, &big));
    }
    messages.push(Message::assistant("Read them."));
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let agent = agent(&mut app, &model, 80_000, messages);
    let Some(agent) = agent else {
        return;
    };
    let ids = |entries: &[Entry]| entries.iter().map(|entry| entry.id).collect::<Vec<_>>();
    let before = ids(&entries(app.world(), agent));
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert_eq!(sent(&model).len(), 1, "no summary");
    let live = entries(app.world(), agent);
    let placeholder = "[output cleared to fit the context window";
    let cleared = serde_json::to_string(&live).unwrap_or_default();
    assert!(cleared.contains(placeholder));
    assert_eq!(ids(&live).get(..before.len()), Some(before.as_slice()));
    drop(app);

    let (mut restarted, _wakes) = start(&store);
    let Some(agent) = restored_agent(&mut restarted) else {
        return;
    };
    assert_eq!(entries(restarted.world(), agent), live);
}

#[test]
fn a_request_too_long_clears_old_outputs_then_summarizes() {
    let (mut app, wakes) = start(&MemoryStore::default());
    // Three reads of 15k tokens each: the oldest is past the 40k kept.
    let big = "x".repeat(60_000);
    let mut messages = vec![Message::user("Read the three files.")];
    for (id, path) in [("1", "a.rs"), ("2", "b.rs"), ("3", "c.rs")] {
        messages.extend(read(id, path, &big));
    }
    messages.push(Message::assistant("Read them."));
    let model = MockCompletionModel::from_stream_turns([
        too_long(),
        too_long(),
        reply("## Goal\nRead three files."),
        reply("done"),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert_eq!(
        sent.len(),
        4,
        "the request, cleared, a summary, summarized: {sent:?}"
    );
    let [first, cleared, summary, last] = sent.as_slice() else {
        return;
    };
    let placeholder = "[output cleared to fit the context window";
    assert!(!first.contains(placeholder));
    assert!(cleared.contains(placeholder), "{cleared}");
    assert!(
        summary.contains("You summarize a conversation"),
        "{summary}"
    );
    assert!(last.contains("## Goal"), "{last}");
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
}

#[test]
fn an_interrupt_during_the_summary_cancels_the_compaction() {
    let (mut app, _wakes) = start(&MemoryStore::default());
    let messages = vec![
        Message::user("What is 2 + 2?"),
        Message::assistant("4"),
        Message::user("And 3 + 3?"),
        Message::assistant("6"),
    ];
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nArithmetic.")]);
    let agent = agent(&mut app, &model, 1_000_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, "compact keep the sums".to_owned()));
    world.flush();
    let focus = world
        .query::<&ModelRequest>()
        .iter(world)
        .map(|call| serde_json::to_string(&call.request).unwrap_or_default())
        .collect::<Vec<_>>();
    assert!(
        matches!(focus.as_slice(), [request] if request.contains("Additional focus: keep the sums")),
        "{focus:?}"
    );
    world.trigger(Interrupt::new(agent));
    world.flush();
    for _ in 0..5 {
        app.update();
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!entries(app.world(), agent).iter().any(holds_summary));
    let world = app.world_mut();
    assert_eq!(world.query::<&Summarizing>().iter(world).count(), 0);
    assert!(
        matches!(
            world.resource::<Ended>().0.as_slice(),
            [TurnOutcome::Stopped(None)]
        ),
        "{:?}",
        world.resource::<Ended>().0
    );
}

/// `turns` exchanges of about 5,000 tokens each, after a first question.
fn long_conversation(turns: usize) -> Vec<Message> {
    let long = "word ".repeat(4_000);
    let mut messages = vec![Message::user("What does the crate do?")];
    for turn in 0..turns {
        messages.push(Message::assistant(format!("{turn}: {long}")));
        messages.push(Message::user(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    messages
}

/// The model every turn's call is routed to, by [`route`].
#[derive(Resource)]
struct Routed(Connection);

/// Routes every turn's call to the [`Routed`] model.
fn route(
    mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>,
    routed: Res<Routed>,
) {
    for mut call in &mut calls {
        call.connection = Some(routed.0.clone());
    }
}

#[test]
fn a_call_routed_to_a_smaller_model_is_measured_against_its_window() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nAnswer."), reply("done")]);
    let small = connect(&model).map(|mut small| {
        small.spec = Arc::new(small.spec.as_ref().clone().with_context_window(40_000));
        small
    });
    assert!(small.is_some(), "the catalog lists the model");
    let Some(small) = small else {
        return;
    };
    app.insert_resource(Routed(small))
        .add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    // The agent's own model has room for the conversation; the call's has not.
    let agent = agent(&mut app, &model, 1_000_000, long_conversation(6));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [summary, _] if summary.contains("You summarize a conversation")),
        "a summary, then the request: {sent:?}"
    );
}

#[test]
fn a_model_that_calls_no_tools_is_not_measured_with_their_rules() {
    let (mut app, wakes) = start(&MemoryStore::default());
    // Rules of about 8,000 tokens: with them, the 20,000-token conversation
    // would not leave the window's 16k reserve free.
    let Ok(name) = ToolName::new("notes") else {
        return;
    };
    let definition = rig_core::completion::ToolDefinition {
        name,
        description: "Takes notes.".to_owned(),
        parameters: serde_json::json!({ "type": "object" }),
        annotations: Default::default(),
    };
    app.world_mut()
        .spawn((ToolDef(definition), ToolRules(vec!["rule ".repeat(6_400)])));
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let agent = agent(&mut app, &model, 40_000, long_conversation(4));
    let Some(agent) = agent else {
        return;
    };
    let world = app.world_mut();
    if let Some(mut connection) = world.get_mut::<Connection>(agent) {
        let mut spec = connection.spec.as_ref().clone();
        spec.tools = false;
        connection.spec = Arc::new(spec);
    }
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [request] if !request.contains("rule rule")),
        "the request alone, without the rules: {sent:?}"
    );
}

#[test]
fn the_request_is_measured_after_the_default_prompt_went_in() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    // A long default prompt, composed as rig-basics composes it, after the
    // plugin's systems were added.
    let prompt = |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        let long = "word ".repeat(8_000);
        for mut call in &mut calls {
            if let Some(prompt) = &mut call.prompt {
                prompt.role.insert_str(0, &long);
            }
        }
    };
    app.add_systems(AgentPass, prompt.in_set(PrepareSystems::Compose));
    let mut messages = vec![Message::user("What does the crate do?")];
    let long = "word ".repeat(2_000);
    for turn in 0..6 {
        messages.push(Message::assistant(format!("{turn}: {long}")));
        messages.push(Message::user(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nAnswer."), reply("done")]);
    // The conversation alone fits; with the prompt it nears the window.
    let agent = agent(&mut app, &model, 40_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [summary, _] if summary.contains("You summarize a conversation")),
        "a summary, then the request: {sent:?}"
    );
}

/// The requests a turn near a 40k window makes when its compacted request
/// still overflows, with `per_turn` summaries a turn, and its answer.
fn overflow_after_a_summary(per_turn: u32) -> (Vec<String>, Option<String>) {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.world_mut().resource_mut::<Compaction>().0.per_turn = per_turn;
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe first summary."),
        too_long(),
        reply("## Goal\nThe second summary."),
        reply("done"),
    ]);
    let Some(agent) = agent(&mut app, &model, 40_000, long_conversation(6)) else {
        return (Vec::new(), None);
    };
    ask(&mut app, &wakes, agent);
    (sent(&model), answer(&app))
}

#[test]
fn a_turn_asks_for_at_most_per_turn_summaries() {
    let summaries = |sent: &[String]| {
        let summary = |request: &&String| request.contains("You summarize a conversation");
        sent.iter().filter(summary).count()
    };
    // By default a second summary rescues the turn.
    let (sent, answer) = overflow_after_a_summary(2);
    assert_eq!(
        (summaries(&sent), answer.as_deref()),
        (2, Some("done")),
        "{sent:?}"
    );
    // With one a turn, the overflow after the first ends the turn.
    let (sent, answer) = overflow_after_a_summary(1);
    assert_eq!(
        (summaries(&sent), sent.len(), answer),
        (1, 2, None),
        "{sent:?}"
    );
}
