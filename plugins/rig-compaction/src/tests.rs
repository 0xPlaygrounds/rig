use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use rig_core::ProviderResponseError;
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, AssistantMessage, CallId, ToolName, UserContent};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness::commands::RunCommand;

use super::*;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, commit_message, connect, log_ended, reply, run_until,
};
use rig_telemetry::{AgentUsage, UsagePlugin};

/// What a second Prepare system adds to every turn's request.
const INJECTED: &str = "Context from another plugin.";

/// The kernel, the compaction and usage plugins on `store`, after the first frame
/// restored the session, with its loop's wakes. The compaction plugin is
/// loaded as `plugins.toml` loads it.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin, UsagePlugin))
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
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    }
}

/// Every message of `agent`'s conversation with its author and id.
fn entries(world: &World, agent: Entity) -> Vec<Entry> {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return Vec::new();
    };
    let at = 0..conversation.messages().len();
    at.filter_map(|at| conversation.entry(at)).collect()
}

/// Whether `entry` holds a compaction's summary.
fn holds_summary(entry: &Entry) -> bool {
    entry.author == Some(summary_author())
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
            let Some(history) = &mut call.history else {
                continue;
            };
            let last = history.messages().len().saturating_sub(1);
            if let Some(mut entry) = history.entry(last)
                && let Message::User { content } = &mut entry.message
            {
                content.push(UserContent::text(INJECTED));
                history.splice(last.., [entry], Author::plugin("inject"));
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
    let spent = app
        .world()
        .get::<AgentUsage>(agent)
        .map(|spent| spent.total);
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
    // The state the next compaction builds on is the restored summary's.
    let state = world.get::<Conversation>(agent).map(|c| summary_state(c));
    let tracked = state.as_ref().map(|state| &state.tracked);
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
    answered: Query<(Entity, Has<ProvidedBy>), (With<Summarizing>, ReadyToSettle)>,
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
    world.trigger(StopTurn::new(agent));
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
/// With `own`, the agent's own `per_turn` over the default's.
fn overflow_after_a_summary(per_turn: u32, own: Option<u32>) -> (Vec<String>, Option<String>) {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .per_turn = per_turn;
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe first summary."),
        too_long(),
        reply("## Goal\nThe second summary."),
        reply("done"),
    ]);
    let Some(agent) = agent(&mut app, &model, 40_000, long_conversation(6)) else {
        return (Vec::new(), None);
    };
    if let Some(per_turn) = own {
        let own = CompactionOverrides {
            per_turn: Some(per_turn),
            ..CompactionOverrides::default()
        };
        app.world_mut()
            .entity_mut(agent)
            .insert(CompactionKnobs(own));
    }
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
    let (sent, answer) = overflow_after_a_summary(2, None);
    assert_eq!(
        (summaries(&sent), answer.as_deref()),
        (2, Some("done")),
        "{sent:?}"
    );
    // With one a turn, the overflow after the first ends the turn: by
    // default, or by the agent's own policy over a default of two.
    for (per_turn, own) in [(1, None), (2, Some(1))] {
        let (sent, answer) = overflow_after_a_summary(per_turn, own);
        assert_eq!(
            (summaries(&sent), sent.len(), answer),
            (1, 2, None),
            "{sent:?}"
        );
    }
}

/// A review of a summary call, holding it until the test decides.
#[derive(Component)]
struct Reviewing(Entity);

/// On a summary call the test decided.
#[derive(Component)]
struct Reviewed;

/// Holds each answered summary call for review, as a plugin asking the
/// user would.
fn hold_for_review(
    calls: Query<Entity, (With<Summarizing>, ReadyToSettle, Without<Reviewed>)>,
    asked: Query<&Reviewing>,
    mut commands: Commands,
) {
    for call in &calls {
        if asked.iter().all(|Reviewing(it)| *it != call) {
            commands.spawn((Reviewing(call), HoldOf(call)));
        }
    }
}

/// Decides the summary under review: sets it to `summary` and lets it go.
fn decide(app: &mut App, wakes: &Receiver<()>, summary: Option<&str>) {
    let held = |world: &mut World| world.query::<&Reviewing>().iter(world).next().map(|r| r.0);
    run_until(app, wakes, |world| {
        world.iter_entities().any(|e| e.contains::<Reviewing>())
    });
    let world = app.world_mut();
    let call = held(world);
    assert!(call.is_some(), "a summary is held for review");
    let Some(call) = call else {
        return;
    };
    let read = world
        .get::<Summarizing>(call)
        .and_then(|it| it.summary.clone());
    assert!(
        read.is_some_and(|read| read.contains("## Goal")),
        "the model's summary"
    );
    if let Some(mut summarizing) = world.get_mut::<Summarizing>(call) {
        summarizing.summary = summary.map(str::to_owned);
    }
    world.entity_mut(call).insert(Reviewed);
    let review: Vec<Entity> = world
        .query_filtered::<Entity, With<Reviewing>>()
        .iter(world)
        .collect();
    for entity in review {
        world.despawn(entity);
    }
}

#[test]
fn a_held_summary_is_applied_as_reviewed_or_cancelled_and_undone_with_its_ids() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe first draft."),
        reply("## Goal\nThe second draft."),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, long_conversation(3));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.update();
    let original = entries(app.world(), agent);
    let compact = |app: &mut App| {
        let focus = String::new();
        app.world_mut().trigger(Compact {
            entity: agent,
            focus,
        });
    };
    let ended = |count: usize| move |world: &World| world.resource::<Ended>().0.len() == count;

    // Cancelled: nothing replaced.
    compact(&mut app);
    decide(&mut app, &wakes, None);
    run_until(&mut app, &wakes, ended(1));
    assert_eq!(entries(app.world(), agent), original);

    // Edited: the reviewer's text is the summary, and the context is
    // measured again.
    if let Some(mut last) = app.world_mut().get_mut::<LastUsage>(agent) {
        *last = LastUsage(Some(Usage::new().total_tokens(900_000)));
    }
    compact(&mut app);
    decide(&mut app, &wakes, Some("## Goal\nThe reviewed summary."));
    run_until(&mut app, &wakes, ended(2));
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
    let summary = live.first().map(|entry| format!("{:?}", entry.message));
    assert!(summary.is_some_and(|it| it.contains("The reviewed summary.")));
    let context = app
        .world()
        .get::<LastUsage>(agent)
        .and_then(LastUsage::context);
    assert!(
        context.is_some_and(|tokens| tokens < 900_000),
        "{context:?}"
    );
    let state = |app: &App| {
        app.world()
            .get::<Conversation>(agent)
            .map(|c| summary_state(c))
    };
    assert!(state(&app).is_some_and(|it| it.summary.contains("reviewed")));

    // Undone: the messages are back with their ids, and the state before.
    app.update();
    app.world_mut().trigger(Uncompact { entity: agent });
    app.update();
    assert_eq!(entries(app.world(), agent), original);
    assert_eq!(state(&app), Some(SummaryState::default()));
}

#[test]
fn the_compaction_notice_says_it_or_they_by_how_many_messages_were_kept() {
    let one = compacted_text((1, 100), (1, 38), 500);
    assert!(
        one.starts_with("Compacted 1 message (about 100 tokens)")
            && one.contains("kept the newest message (about 38 tokens) as it was;"),
        "{one}"
    );
    let two = compacted_text((42, 17_000), (2, 38), 500);
    assert!(
        two.contains("kept the newest 2 messages (about 38 tokens) as they were;"),
        "{two}"
    );
}

/// The texts of the notices so far.
fn notices(app: &App) -> Vec<String> {
    let notices = app.world().resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    cursor
        .read(notices)
        .map(|notice| notice.text.clone())
        .collect()
}

/// An agent's own policy decides its compaction: a cut it names from the
/// plugins' rules, until its own knobs are taken out again.
#[test]
fn an_agents_own_policy_and_a_named_cut_decide_its_compaction() {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.add_compaction_cut("never", |_: &[Message], _, _, _| None);
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nThe summary.")]);
    let Some(agent) = agent(&mut app, &model, 1_000_000, long_conversation(3)) else {
        return;
    };
    let never = CompactionOverrides {
        cut: Some(rig_memory::Cut::Named("never".to_owned())),
        ..CompactionOverrides::default()
    };
    app.world_mut()
        .entity_mut(agent)
        .insert(CompactionKnobs(never));
    let compact = |app: &mut App| {
        let focus = String::new();
        app.world_mut().trigger(Compact {
            entity: agent,
            focus,
        });
    };
    compact(&mut app);
    app.update();
    assert!(
        notices(&app)
            .iter()
            .any(|text| text == "Nothing to compact yet.")
    );
    assert!(model.requests().is_empty());

    app.world_mut()
        .entity_mut(agent)
        .remove::<CompactionKnobs>();
    compact(&mut app);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
}

/// `[compaction]` in settings sets only the knobs it names over the
/// built-in default, which tracks the built-in file tools.
#[test]
fn settings_set_only_the_knobs_they_name() {
    let (mut app, _wakes) = app_on(&MemoryStore::default());
    app.insert_resource(rig_harness::settings::Settings::from_toml(
        "[compaction]\nkeep_recent = 1000\ncut = \"before-last-user\"",
    ));
    app.add_plugins(CompactionPlugin);
    let default = app.world().resource::<CompactionDefault>().0.clone();
    assert_eq!(default.keep_recent, 1000);
    assert_eq!(
        default.cut,
        rig_memory::Cut::Named("before-last-user".to_owned())
    );
    assert_eq!(default.tracked, CompactionDefault::default().0.tracked);
    assert_eq!(default.tracked.len(), 3);
}

/// `/compaction` sets the agent's own knobs at runtime: they are saved with
/// its line, so a restore and a split of the line keep them, while a knob it
/// never set follows the default.
#[test]
fn a_policy_set_by_command_is_restored_inherited_and_unset_knobs_follow_the_default() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let Some(agent) = agent(&mut app, &model, 1_000_000, long_conversation(1)) else {
        return;
    };
    let run = |app: &mut App, line: &str| {
        app.world_mut()
            .trigger(RunCommand::new(agent, line.to_owned()));
        app.world_mut().flush();
    };
    run(&mut app, "compaction per-turn=1 keep=500");
    app.update();
    let own = CompactionOverrides {
        per_turn: Some(1),
        keep_recent: Some(500),
        ..CompactionOverrides::default()
    };
    let has = |app: &App, agent| {
        app.world()
            .get::<CompactionKnobs>(agent)
            .map(|it| it.0.clone())
    };
    assert_eq!(has(&app, agent), Some(own.clone()));
    run(&mut app, "compaction bogus=1");
    // A refused command line comes back to the user with why.
    let recalled = app.world().resource::<Messages<Recalled>>();
    let mut cursor = recalled.get_cursor();
    let why: Vec<_> = cursor
        .read(recalled)
        .filter_map(|it| it.why.clone())
        .collect();
    assert!(
        why.iter()
            .any(|why| why.contains("there is no knob `bogus`")),
        "{why:?}"
    );
    // A split of the line at its end starts with the line's knobs.
    let by = Author::plugin("tree");
    let split = app.world_mut().get_mut::<Conversation>(agent).map(|mut c| {
        let at = c.messages().len();
        c.branch_off(at, "split", by).is_ok()
    });
    assert_eq!(split, Some(true));
    app.update();
    assert_eq!(has(&app, agent), Some(own.clone()));
    drop(app);

    let (mut again, _wakes) = start(&store);
    let Some(agent) = restored_agent(&mut again) else {
        return;
    };
    assert_eq!(has(&again, agent), Some(own));
    again
        .world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .keep_asked = 7;
    let policy = |app: &App| {
        let own = app.world().get::<CompactionKnobs>(agent);
        policy_of(own, app.world().resource::<CompactionDefault>())
    };
    let now = policy(&again);
    assert_eq!((now.per_turn, now.keep_recent, now.keep_asked), (1, 500, 7));
    again
        .world_mut()
        .trigger(RunCommand::new(agent, "compaction reset".to_owned()));
    again.world_mut().flush();
    assert_eq!(has(&again, agent), None);
    assert_eq!(
        policy(&again).per_turn,
        CompactionDefault::default().0.per_turn
    );
}
