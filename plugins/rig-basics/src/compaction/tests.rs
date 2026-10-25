mod anthropic;

use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_core::ProviderResponseError;
use rig_core::completion::{FinishReason, Message, Usage};
use rig_core::message::{AssistantContent, AssistantMessage, CallId, ToolName, UserContent};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness::commands::RunCommand;

use super::*;
use crate::telemetry::usage::{AgentUsage, UsagePlugin};
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, advance, app_on, collect, commit_message, connect, log_ended,
    reply, run_until,
};

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
    rig_harness::load::<CompactionPlugin>(&mut app, "rig-basics", "path plugins/rig-basics");
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

/// Whether `request`, as [`sent`] has it, asks for a summary: at the end
/// of the turn's own request near the window, or in a request of its own.
fn asks_summary(request: &str) -> bool {
    request.contains("Stop here: do not continue the work")
        || request.contains("You summarize a conversation")
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
    // A second Prepare system on every request the kernel builds, beside
    // the plugin's.
    let inject = |mut calls: Query<
        &mut ModelRequest,
        (Added<Admitted>, Or<(With<TurnCall>, With<SideCall>)>),
    >| {
        for mut call in &mut calls {
            let Some(history) = &mut call.history else {
                continue;
            };
            let last = history.messages().len().saturating_sub(1);
            if let Some(mut entry) = history.entry(last)
                && let Message::User { content } = std::sync::Arc::make_mut(&mut entry.message)
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
    // The summary is the held request cut before the messages it keeps,
    // the newest one with its injected context among them, with the
    // instruction last; the request after it is built anew.
    assert!(
        summary.contains("Stop here: do not continue the work"),
        "{summary}"
    );
    assert!(
        summary.contains("3: word") && !summary.contains("4: word"),
        "{summary}"
    );
    assert_eq!(summary.matches(INJECTED).count(), 0, "{summary}");
    assert!(request.contains("## Goal"), "{request}");
    assert!(request.contains("<read-files>\\nsrc/lib.rs"), "{request}");
    assert_eq!(request.matches(INJECTED).count(), 1, "{request}");
    // The summary stands first, merged into the first kept message when
    // that is the user's; the kept messages keep their ids.
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
    let alone = live.first().is_some_and(
        |entry| matches!(&*entry.message, Message::User { content } if content.len() == 1),
    );
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

    // `/compact` asks the agent's last request again with the instruction
    // appended to its newest message (which the wire merges with it, and
    // which this test's plugin edited only in the turn's request), so it
    // folds in the summary that request opens with.
    let compact = Compact {
        entity: agent,
        focus: String::new(),
    };
    app.world_mut().trigger(compact);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let requests = model.requests();
    let (last, again) = (requests.get(1), requests.get(2));
    let extends = last.zip(again).is_some_and(|(last, again)| {
        let (last, again) = (&last.chat_history, &again.chat_history);
        let before = last.len().saturating_sub(1);
        again.len() == last.len() && again.get(..before) == last.get(..before)
    });
    assert!(extends, "{requests:?}");
    let again = sent(&model).get(2).cloned().unwrap_or_default();
    assert!(again.contains("Stop here"), "{again}");
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
    let turn_requests = requests.iter().filter(|request| !asks_summary(request));
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
    // The files are not on disk, so their reads would be noted first.
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .near_window = NearWindow::Clear;
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

/// A conversation of `messages`, and a model that answers the request
/// with `replies`: what it was sent, and how the turn ended.
fn overflowing(
    messages: Vec<Message>,
    replies: Vec<Vec<MockStreamEvent>>,
) -> (Vec<String>, Vec<TurnOutcome>, App, Entity) {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns(replies);
    let agent = agent(&mut app, &model, 1_000_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    ask(&mut app, &wakes, agent);
    let ended = app.world().resource::<Ended>().0.clone();
    (sent(&model), ended, app, agent)
}

/// A conversation in which the agent read `count` files of 15k tokens
/// each, none of them on disk.
fn read_files(count: usize) -> Vec<Message> {
    let big = "x".repeat(60_000);
    let mut messages = vec![Message::user("Read the files.")];
    for n in 0..count {
        messages.extend(read(&n.to_string(), &format!("{n}.rs"), &big));
    }
    messages.push(Message::assistant("Read them."));
    messages
}

/// A failure the provider says is passing.
fn unavailable() -> Vec<MockStreamEvent> {
    let busy = ProviderResponseError::without_status("overloaded").with_transient(Some(true));
    vec![MockStreamEvent::Error(MockError::ProviderResponse(busy))]
}

/// Caps the input of `agent`'s model at `tokens`, below its window.
fn limit_input(app: &mut App, agent: Entity, tokens: u32) {
    if let Some(mut connection) = app.world_mut().get_mut::<Connection>(agent) {
        let spec = connection.spec.as_ref().clone();
        connection.spec = Arc::new(spec.with_max_input_tokens(tokens));
    }
}

/// A request the model refused as too long, with tool outputs it could
/// not read again, is summarized in a request of its own, then sent
/// again: no notes are asked of the refused request.
#[test]
fn a_request_too_long_is_summarized_in_a_request_of_its_own() {
    let replies = vec![
        too_long(),
        reply("## Goal\nRead three files."),
        reply("done"),
    ];
    let (sent, _, app, agent) = overflowing(read_files(3), replies);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert!(!sent.iter().any(|request| request.contains("Stop here")));
    assert!(
        matches!(sent.as_slice(), [_, summary, last]
            if summary.contains("You summarize a conversation") && last.contains("## Goal")),
        "the request, a summary of its own, summarized: {sent:?}"
    );
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
}

/// A summary call that fails for a while is sent again after the retry
/// wait, counted with the turn's retries, and its summary goes in.
#[test]
fn a_summary_call_that_fails_for_a_while_is_retried_with_the_turns_retries() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns(vec![
        too_long(),
        unavailable(),
        reply("## Goal\nRead six files."),
        reply("done"),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 120_000);
    let steer = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(steer);
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Backoff>())
    });
    let world = app.world_mut();
    let retries: Vec<u32> = world
        .query::<&Recovery>()
        .iter(world)
        .map(|recovery| recovery.retries)
        .collect();
    assert_eq!(retries, [1], "the turn counts the retry");
    advance(&mut app, std::time::Duration::from_secs(3));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    let summaries = sent.iter().filter(|request| asks_summary(request)).count();
    assert_eq!((sent.len(), summaries), (4, 2), "{sent:?}");
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
}

/// A reply that sent nothing for five minutes.
fn stalled() -> Vec<MockStreamEvent> {
    let idle = std::time::Duration::from_secs(300);
    vec![MockStreamEvent::Error(MockError::Stalled(idle))]
}

/// A summary call that times out is sent again once; timing out again, it
/// fails, and the turn's call goes on without the summary.
#[test]
fn a_summary_call_that_times_out_is_retried_once() {
    let (mut app, wakes) = start(&MemoryStore::default());
    collect::<Notice>(&mut app);
    let model = MockCompletionModel::from_stream_turns(vec![
        too_long(),
        stalled(),
        stalled(),
        reply("done"),
        reply("## Goal\nNever asked."),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 120_000);
    let steer = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(steer);
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Backoff>())
    });
    advance(&mut app, std::time::Duration::from_secs(3));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    let summaries = sent.iter().filter(|request| asks_summary(request)).count();
    assert_eq!((sent.len(), summaries), (4, 2), "{sent:?}");
    // Its notice promises the one retry, not the turn's four.
    let notices = notices(&app);
    assert!(
        notices
            .iter()
            .any(|it| it.contains("(once, as it timed out)")),
        "{notices:?}"
    );
    assert!(!notices.iter().any(|it| it.contains("1/4")), "{notices:?}");
}

/// A request fits when the input limit holds it and the window leaves
/// room for the least of its answer it needs.
#[test]
fn a_request_fits_only_with_room_for_its_answer() {
    let Some(openai) = rig_core::providers::registry::ProviderId::catalog("openai") else {
        return;
    };
    let spec = ModelSpec::new(openai, "m").with_context_window(128_000);
    assert!(fits(&spec, 100_000, (64_000, 12_000)));
    // Within the input limit, but under 4k left for a 64k answer.
    assert!(!fits(&spec, 120_000, (64_000, 12_000)));
    assert!(fits(&spec, 120_000, (64_000, 2_000)));
    assert!(!fits(&spec, 124_000, (64_000, 2_000)));
    assert!(!fits(
        &spec.with_max_input_tokens(110_000),
        120_000,
        (64_000, 2_000)
    ));
}

/// A turn asks for its own limit, or 64k plus any thinking budget within
/// the model's limit, within the room left. With no room, or a model whose
/// limits are unknown (`None`), it is left as it is.
#[test]
fn turn_output_uses_a_working_budget_and_preserves_explicit_limits() {
    let spec = rig_core::catalog::Catalog::builtin().resolve("openrouter/moonshotai/kimi-k3");
    assert!(spec.is_ok(), "the catalog lists Kimi K3");
    let Ok(spec) = spec else { return };
    for (limit, asked, thinking, used, expected) in [
        (Some(943_718), None, None, 100_000, Some(64_000)),
        (Some(943_718), Some(200_000), None, 100_000, Some(200_000)),
        (Some(943_718), None, Some(16_000), 100_000, Some(81_024)),
        (Some(64_000), None, Some(16_384), 100_000, Some(64_000)),
        (Some(32_000), None, None, 100_000, Some(32_000)),
        (Some(943_718), None, None, 1_000_000, Some(44_480)),
        (Some(943_718), None, None, 1_048_576, None),
        (Some(943_718), Some(200_000), None, 1_048_576, Some(200_000)),
        (None, None, None, 100_000, None),
    ] {
        let mut spec = spec.spec.clone();
        spec.max_output_tokens = limit;
        spec.context_window = spec.context_window.filter(|_| limit.is_some());
        let mut request = CompletionRequest::new("Answer.");
        request.max_tokens = asked;
        request.options.reasoning = thinking.map(|tokens| completion::Reasoning::Budget { tokens });
        let mut model = ModelRequest::new(request);
        fit_output(&mut model, &spec, used);
        assert_eq!(
            model.request.max_tokens, expected,
            "{limit:?}, {asked:?}, {thinking:?}, {used}"
        );
    }
}

/// The one request a turn on five 15k reads sends, clearing near the
/// window, to a model with a 32,000 output limit in a window of `window`,
/// with a thinking budget of `budget`; and whether it cleared any output.
fn thinking_turn(window: u32, budget: Option<u32>) -> (Option<CompletionRequest>, bool) {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .near_window = NearWindow::Clear;
    let model = MockCompletionModel::from_stream_turns(vec![reply("done")]);
    let Some(agent) = agent(&mut app, &model, window, read_files(5)) else {
        return (None, false);
    };
    let world = app.world_mut();
    if let Some(mut connection) = world.get_mut::<Connection>(agent) {
        let mut spec = connection.spec.as_ref().clone();
        spec.max_output_tokens = Some(32_000);
        spec.reasoning = rig_core::catalog::ReasoningSupport::Unknown { default: None };
        connection.spec = Arc::new(spec);
    }
    let budget = budget.map(|tokens| completion::Reasoning::Budget { tokens });
    world.entity_mut(agent).insert(ReasoningSetting(budget));
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let requests = model.requests();
    assert_eq!(requests.len(), 1, "{requests:?}");
    let request = requests.first().cloned();
    let sent = serde_json::to_string(&request).unwrap_or_default();
    assert!(!asks_summary(&sent), "{sent}");
    (request, sent.contains("[output cleared"))
}

/// A thinking budget within 1024 of the output limit leaves the turn
/// within the window: with room for all it asks, no old output is cleared
/// and nothing summarized, and the one request asks for the model's
/// output limit.
#[test]
fn a_thinking_budget_near_the_output_limit_is_not_over_the_window() {
    let (request, cleared) = thinking_turn(1_000_000, Some(31_000));
    assert!(!cleared);
    let asked = request.map(|it| (it.max_tokens, it.options.reasoning));
    let budget = completion::Reasoning::Budget { tokens: 31_000 };
    assert_eq!(asked, Some((Some(32_000), Some(budget))));
}

/// A window that cuts the room below the output budget, to no more than
/// the thinking budget, leaves no room to answer: old outputs are cleared.
/// The same room without thinking is room enough, and nothing is cleared.
#[test]
fn room_cut_to_the_thinking_budget_is_over_the_window() {
    // Five 15k reads leave about 30k of a 105k window.
    let window = 105_000;
    assert!(thinking_turn(window, Some(31_000)).1, "with thinking");
    assert!(!thinking_turn(window, None).1, "without");
}

/// A summary asked in place keeps the turn's thinking budget, so its
/// output limit holds that budget and 1024 more before its answer, within
/// the model's output limit; with less than the least answer left, it is
/// not asked in place (`None`).
#[test]
fn a_summary_in_place_has_room_for_the_thinking_budget() {
    let Some(openai) = rig_core::providers::registry::ProviderId::catalog("openai") else {
        return;
    };
    let summarizer = &CompactionPolicy::default().summarizer;
    for (limit, thinking, expected) in [
        (128_000, None, Some(32_000)),
        (128_000, Some(32_000), Some(65_024)),
        (48_000, Some(32_000), Some(48_000)),
        (40_000, Some(32_000), None),
    ] {
        let mut spec = ModelSpec::new(openai, "m").with_context_window(200_000);
        spec.max_output_tokens = Some(limit);
        let mut request = CompletionRequest::new("Answer.");
        request.options.reasoning = thinking.map(|tokens| completion::Reasoning::Budget { tokens });
        let mut model = ModelRequest::new(request);
        let history = [Message::user("Read the code."), Message::assistant("Read.")];
        model.history = Some(rig_core::conversation::Conversation::from_messages(history));
        let measure = Measure::of(&model, &spec, None, 1_000);
        let asked = ask_for(
            &mut model,
            Upto::Before(None),
            (summarizer, Asking::Summary("")),
            measure,
        );
        let max_tokens = asked.filter(|asked| *asked).and(model.request.max_tokens);
        assert_eq!(max_tokens, expected, "{limit}, {thinking:?}");
    }
    // A model whose limits are unknown is asked with no output limit.
    let spec = ModelSpec::new(openai, "m");
    let mut model = ModelRequest::new(CompletionRequest::new("Answer."));
    let history = [Message::user("Read the code."), Message::assistant("Read.")];
    model.history = Some(rig_core::conversation::Conversation::from_messages(history));
    let measure = Measure::of(&model, &spec, None, 1_000);
    let asked = ask_for(
        &mut model,
        Upto::Before(None),
        (summarizer, Asking::Summary("")),
        measure,
    );
    assert_eq!((asked, model.request.max_tokens), (Some(true), None));
}

/// A conversation summarized before, then one reply that read `count`
/// files of 15k tokens each at once, none of them on disk, and their
/// outputs: no message after the summary can start what a summary keeps,
/// so none would be summarized.
fn summarized_then_read(world: &mut World, agent: Entity, count: usize) -> Option<()> {
    let name = ToolName::new("read").ok()?;
    let calls = (0..count).map(|n| {
        let arguments = serde_json::json!({ "path": format!("{n}.rs") });
        AssistantContent::tool_call(n.to_string(), name.clone(), arguments)
    });
    let results = (0..count).map(|n| {
        let output = vec![rig_core::message::ToolResultContent::text(
            "x".repeat(60_000),
        )];
        UserContent::tool_result(CallId::from_wire(n.to_string()), name.clone(), output)
    });
    commit_message(world, agent, Message::user("Read the files.")).ok()?;
    let reads = Message::Assistant(AssistantMessage::new(calls.collect()));
    commit_message(world, agent, reads).ok()?;
    // The newest output comes apart, as clearing keeps the last message.
    let mut results: Vec<UserContent> = results.collect();
    let newest = results.pop()?;
    commit_message(world, agent, Message::User { content: results }).ok()?;
    let newest = Message::User {
        content: vec![newest],
    };
    commit_message(world, agent, newest).ok()?;
    let mut conversation = world.get_mut::<Conversation>(agent)?;
    let keep_from = conversation.id(1);
    let state = SummaryState {
        summary: "## Goal\nRead the files.".to_owned(),
        ..SummaryState::default()
    };
    apply_summary(&mut conversation, keep_from, &state).map(|_| ())
}

/// An overflow with nothing older to summarize clears the tool outputs
/// without notes and sends the request again.
#[test]
fn an_overflow_with_nothing_to_summarize_clears_without_notes() {
    let (mut app, wakes) = start(&MemoryStore::default());
    collect::<Notice>(&mut app);
    let model = MockCompletionModel::from_stream_turns(vec![too_long(), reply("done")]);
    let agent = agent(&mut app, &model, 1_000_000, Vec::new());
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    let read = summarized_then_read(app.world_mut(), agent, 6);
    assert!(read.is_some(), "the conversation was summarized");
    // Called as it stands, as a turn's call after its tool outputs is.
    let world = app.world_mut();
    let turn = world.spawn(TurnOf(agent)).id();
    world.spawn((TurnCall, CallOf(turn), ForAgent(agent)));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [_, cleared] if cleared.contains("[output cleared")),
        "the request, then cleared: {sent:?}"
    );
    let notices = notices(&app);
    assert!(
        notices.iter().any(|it| it.contains("and sending it again")),
        "{notices:?}"
    );
}

/// A `/compact` summary too big to ask in place, a request of its own
/// instead, that fails for a while is sent again as that same request, not
/// built again in place.
#[test]
fn a_summary_too_big_to_ask_in_place_is_retried_as_it_was() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns(vec![
        unavailable(),
        reply("## Goal\nRead six files."),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 60_000);
    app.world_mut()
        .trigger(RunCommand::new(agent, "compact".to_owned()));
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Backoff>())
    });
    advance(&mut app, std::time::Duration::from_secs(3));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [first, again]
            if first.contains("You summarize a conversation") && first == again),
        "the same summary twice: {sent:?}"
    );
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
}

/// Two models: the agent's own, `own`, and `routed`, to which each of its
/// calls goes, whose spec `spec` makes.
fn routed_agent(
    app: &mut App,
    (own, routed): (&MockCompletionModel, &MockCompletionModel),
    spec: impl FnOnce(ModelSpec) -> ModelSpec,
    messages: Vec<Message>,
) -> Option<Entity> {
    let mut to = connect(routed)?;
    to.spec = Arc::new(spec(to.spec.as_ref().clone()));
    app.insert_resource(Routed(to))
        .add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    agent(app, own, 1_000_000, messages)
}

/// The summary after an overflow goes to the model the refused call was
/// routed to.
#[test]
fn the_summary_after_an_overflow_keeps_the_route() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let own = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let routed = MockCompletionModel::from_stream_turns(vec![
        too_long(),
        reply("## Goal\nRead six files."),
        reply("done"),
    ]);
    let limited = |spec: ModelSpec| spec.with_max_input_tokens(120_000);
    let agent = routed_agent(&mut app, (&own, &routed), limited, read_files(6));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert_eq!(sent(&own), Vec::<String>::new());
    let sent = sent(&routed);
    assert!(
        matches!(sent.as_slice(), [_, summary, _] if summary.contains("You summarize a conversation")),
        "the request, a summary of its own, the request: {sent:?}"
    );
}

/// A summary near the window too big to ask in place goes, as a request of
/// its own, to the model the call it holds was routed to.
#[test]
fn a_summary_too_big_to_ask_in_place_keeps_the_route() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let own = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let routed = MockCompletionModel::from_stream_turns([reply("## Goal\nAnswer."), reply("done")]);
    let small = |spec: ModelSpec| spec.with_context_window(40_000);
    let agent = routed_agent(&mut app, (&own, &routed), small, long_conversation(14));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert_eq!(sent(&own), Vec::<String>::new());
    let sent = sent(&routed);
    assert!(
        matches!(sent.as_slice(), [summary, _] if summary.contains("You summarize a conversation")),
        "a summary of its own, then the request: {sent:?}"
    );
}

/// A reply of `text` for which the provider counted `tokens`.
fn counted(text: &str, tokens: u64) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_total_tokens(tokens),
    ]
}

/// `/compact` right after a reply the provider counted as more tokens than
/// the estimate measures its summary as many: one the estimate alone would
/// ask in place is asked as a request of its own.
#[test]
fn compact_is_measured_as_the_provider_counted_the_last_request() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns(vec![
        // About 3.5 provider tokens for each one estimated.
        counted("Read them.", 160_000),
        reply("## Goal\nRead three files."),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(3));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 150_000);
    ask(&mut app, &wakes, agent);
    app.world_mut().resource_mut::<Ended>().0.clear();
    app.world_mut()
        .trigger(RunCommand::new(agent, "compact".to_owned()));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [_, summary] if summary.contains("You summarize a conversation")),
        "the request, then a summary of its own: {sent:?}"
    );
}

/// A reply the provider counted past the input limit makes the next
/// request summarize first: the reply stands, and nothing is sent again.
#[test]
fn a_reply_counted_past_the_input_limit_compacts_before_the_next_request() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns(vec![
        counted("Answered.", 130_000),
        reply("## Goal\nAnswer."),
        reply("done"),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, long_conversation(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 120_000);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("Answered."));
    app.world_mut().resource_mut::<Ended>().0.clear();
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [_, summary, _] if asks_summary(summary)),
        "the request, a summary, then the next request: {sent:?}"
    );
}

/// Near the limit, notes too big to ask give way to a summary of its own
/// on the model the held call was routed to, sized for that model.
#[test]
fn notes_too_big_near_the_limit_give_way_to_a_summary_on_the_route() {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .near_window = NearWindow::Summarize;
    let own = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let routed =
        MockCompletionModel::from_stream_turns([reply("## Goal\nRead six files."), reply("done")]);
    let limited = |spec: ModelSpec| spec.with_max_input_tokens(60_000);
    let agent = routed_agent(&mut app, (&own, &routed), limited, read_files(6));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert_eq!(sent(&own), Vec::<String>::new());
    let sent = sent(&routed);
    assert!(!sent.iter().any(|request| request.contains("Stop here")));
    assert!(
        matches!(sent.as_slice(), [summary, _]
            if summary.contains("You summarize a conversation") && summary.len() < 60_000 * 3),
        "a summary of its own within the routed model's limit, then the request: {sent:?}"
    );
}

/// A summary call that times out, fails for another passing reason, then
/// times out again is not sent a fourth time: the timeout is remembered
/// across the failure in between.
#[test]
fn a_timeout_is_remembered_across_another_failure() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns(vec![
        too_long(),
        stalled(),
        unavailable(),
        stalled(),
        reply("done"),
        reply("## Goal\nNever asked."),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 120_000);
    let steer = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(steer);
    let waiting = |world: &World| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Backoff>())
    };
    for _ in 0..2 {
        run_until(&mut app, &wakes, waiting);
        advance(&mut app, std::time::Duration::from_secs(10));
    }
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    let summaries = sent.iter().filter(|request| asks_summary(request)).count();
    assert_eq!((sent.len(), summaries), (5, 3), "{sent:?}");
}

/// After an overflow the summary reads at most half the input limit: one
/// near the limit is nearly the refused request again.
#[test]
fn the_summary_after_an_overflow_reads_at_most_half_the_input_limit() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns(vec![
        too_long(),
        reply("## Goal\nRead the conversation."),
        reply("done"),
    ]);
    // Long exchanges the summary reads as they are, unlike tool outputs.
    let agent = agent(&mut app, &model, 1_000_000, long_conversation(40));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 240_000);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    let summary = sent.get(1).map(String::as_str).unwrap_or_default();
    assert!(
        summary.contains("You summarize a conversation"),
        "{summary}"
    );
    // Half of 240k tokens, at three characters a token.
    assert!(summary.len() < 120_000 * 3, "{}", summary.len());
}

/// `/compact` on a conversation too long for the model to read whole asks
/// for the summary in a request of its own.
#[test]
fn compact_on_a_conversation_past_the_input_limit_asks_a_request_of_its_own() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nRead six files.")]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 60_000);
    app.world_mut()
        .trigger(RunCommand::new(agent, "compact".to_owned()));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [summary] if summary.contains("You summarize a conversation")),
        "a summary of its own: {sent:?}"
    );
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
}

/// Stopping the turn while a failed summary call waits to be sent again
/// sends nothing more and changes nothing: the call ends with its turn.
#[test]
fn a_stop_during_the_retry_wait_cancels_the_compaction() {
    let (mut app, wakes) = start(&MemoryStore::default());
    collect::<Notice>(&mut app);
    let model = MockCompletionModel::from_stream_turns(vec![
        too_long(),
        unavailable(),
        reply("## Goal\nNever asked."),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    limit_input(&mut app, agent, 120_000);
    let before = entries(app.world(), agent);
    let steer = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(steer);
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Backoff>())
    });
    app.world_mut().trigger(StopTurn::new(agent));
    app.world_mut().flush();
    app.update();
    advance(&mut app, std::time::Duration::from_secs(3));
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(sent(&model).len(), 2, "the request and one summary call");
    assert!(
        matches!(
            app.world().resource::<Ended>().0.as_slice(),
            [TurnOutcome::Stopped(None)]
        ),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    let after = entries(app.world(), agent);
    assert!(!after.iter().any(holds_summary), "{after:?}");
    assert_eq!(after.len(), before.len() + 1, "only the question went in");
    let notices = notices(&app);
    assert!(
        !notices.iter().any(|it| it.contains("Compaction failed")),
        "{notices:?}"
    );
}

/// Near the input limit, notes whose own request would not fit give way
/// to a summary of its own: the model is never asked for them.
#[test]
fn notes_too_big_for_the_input_limit_give_way_to_a_summary() {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .near_window = NearWindow::Summarize;
    let model =
        MockCompletionModel::from_stream_turns([reply("## Goal\nRead six files."), reply("done")]);
    let agent = agent(&mut app, &model, 1_000_000, read_files(6));
    assert!(agent.is_some(), "the catalog lists the model");
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    // Clearing all but the newest output would leave room, but the notes
    // on them would read about 75k tokens.
    limit_input(&mut app, agent, 60_000);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(!sent.iter().any(|request| request.contains("Stop here")));
    assert!(
        matches!(sent.as_slice(), [summary, _] if summary.contains("You summarize a conversation")),
        "a summary of its own, then the request: {sent:?}"
    );
}

/// A request that overflows with no reply in between is compacted, or
/// cleared when its summary is cut off or filtered, then the turn fails:
/// one summary, never a loop. With nothing to clear it is not sent again.
/// The question leaves the conversation.
#[test]
fn overflows_without_a_reply_fail_after_one_summary() {
    use FinishReason::{ContentFilter, Length};
    for (messages, summary, requests, cut) in [
        (read_files(3), reply("## Goal\nRead three files."), 3, false),
        (read_files(3), stopped_summary(Length), 3, true),
        (read_files(3), stopped_summary(ContentFilter), 3, true),
        (long_conversation(6), stopped_summary(Length), 2, false),
    ] {
        let replies = vec![too_long(), summary, too_long(), reply("never asked")];
        let (sent, ended, mut app, agent) = overflowing(messages, replies);
        app.update();
        let asked = Message::user("And the next question?");
        let conversation = app.world().get::<Conversation>(agent);
        let kept = conversation.is_some_and(|it| it.messages().iter().any(|m| **m == asked));
        assert!(!kept, "the question is out of the conversation");
        let summaries = sent.iter().filter(|request| asks_summary(request)).count();
        assert_eq!(
            (sent.len(), summaries),
            (requests, 1),
            "one summary: {sent:?}"
        );
        let cleared = sent.last().is_some_and(|it| it.contains("[output cleared"));
        assert_eq!(cleared, cut, "{sent:?}");
        assert!(
            matches!(ended.as_slice(), [TurnOutcome::Failed(why)] if why.contains("context window")),
            "{ended:?}"
        );
    }
}

/// A model whose limits are unknown has no floor: after an overflow and a
/// cut summary the cleared request is sent, whatever its thinking budget.
#[test]
fn a_model_of_unknown_limits_sends_the_cleared_request_after_a_cut_summary() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let replies = vec![
        too_long(),
        stopped_summary(FinishReason::Length),
        reply("done"),
    ];
    let model = MockCompletionModel::from_stream_turns(replies);
    let agent = agent(&mut app, &model, 1_000_000, read_files(3));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else { return };
    let world = app.world_mut();
    if let Some(mut connection) = world.get_mut::<Connection>(agent) {
        let mut spec = connection.spec.as_ref().clone();
        (
            spec.context_window,
            spec.max_input_tokens,
            spec.max_output_tokens,
        ) = (None, None, None);
        spec.reasoning = rig_core::catalog::ReasoningSupport::Unknown { default: None };
        connection.spec = Arc::new(spec);
    }
    let budget = completion::Reasoning::Budget { tokens: 16_000 };
    world
        .entity_mut(agent)
        .insert(ReasoningSetting(Some(budget)));
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    let cleared = sent.last().is_some_and(|it| it.contains("[output cleared"));
    assert!(sent.len() == 3 && cleared, "{sent:?}");
}

/// A summary that stopped for `reason` after some text.
fn stopped_summary(reason: FinishReason) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text("## Goal\nRead the fi"),
        MockStreamEvent::FinalResponse(rig_core::operation::Finish {
            reason: Some(reason),
            ..rig_core::test_utils::mock_final(Usage::default())
        }),
    ]
}

/// A summary cut off near the window is not used or asked again: old tool
/// outputs are cleared instead and the request goes. After the model's
/// reply, the next request near the window summarizes afresh.
#[test]
fn a_cut_summary_near_the_window_clears_instead_until_the_next_reply() {
    let (mut app, wakes) = start(&MemoryStore::default());
    collect::<Notice>(&mut app);
    /// For each "Compaction failed" notice, whether the answer was in
    /// the conversation as the agent loop read it.
    #[derive(Resource, Default)]
    struct FailedAfterAnswer(Vec<bool>);
    app.init_resource::<FailedAfterAnswer>().add_systems(
        AgentPass,
        (|mut read: MessageReader<Notice>,
          conversations: Query<&Conversation>,
          mut seen: ResMut<FailedAfterAnswer>| {
            let answered = conversations.iter().any(|it| {
                it.messages()
                    .iter()
                    .any(|m| m.final_answer().as_deref() == Some("done"))
            });
            let failed = read
                .read()
                .filter(|it| it.text.starts_with("Compaction failed"));
            seen.0.extend(failed.map(|_| answered));
        })
        .in_set(Stage::Arrive),
    );
    let model = MockCompletionModel::from_stream_turns(vec![
        stopped_summary(FinishReason::Length),
        counted("done", 99_000),
        reply("## Goal\nRead three files."),
        reply("again"),
    ]);
    let mut messages = long_conversation(10);
    messages.extend(read_files(3));
    let Some(agent) = agent(&mut app, &model, 1_000_000, messages) else {
        return;
    };
    limit_input(&mut app, agent, 100_000);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert!(!entries(app.world(), agent).iter().any(holds_summary));
    let notices = notices(&app).join("\n");
    assert!(
        notices.contains("Compaction failed: the summary hit") && notices.contains("instead."),
        "{notices}"
    );
    // The failure is told before the answer comes.
    assert_eq!(app.world().resource::<FailedAfterAnswer>().0, [false]);
    app.world_mut().resource_mut::<Ended>().0.clear();
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("again"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [cut, cleared, summary, _]
            if asks_summary(cut) && cleared.contains("[output cleared") && !asks_summary(cleared)
                && asks_summary(summary)),
        "{sent:?}"
    );
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
}

/// A request whose summary is cut off is not sent when it then does not
/// fit: past the input limit with nothing to clear, also for a model that
/// knows only its input limit, or, with a small reserve, inside it after
/// clearing but with little room to answer.
#[test]
fn a_request_that_still_does_not_fit_after_a_cut_summary_is_not_sent() {
    let mut cleared = long_conversation(20);
    cleared.extend(read_files(2));
    for (messages, window, input, reserve, only_input) in [
        (long_conversation(30), 1_000_000, 120_000, 16_384, false),
        (long_conversation(30), 1_000_000, 120_000, 16_384, true),
        (cleared, 122_000, 122_000, 4_096, false),
    ] {
        let (mut app, wakes) = start(&MemoryStore::default());
        let limits = &mut app
            .world_mut()
            .resource_mut::<CompactionDefault>()
            .into_inner()
            .0;
        limits.summarizer.limits.reserve = reserve;
        let model =
            MockCompletionModel::from_stream_turns(vec![stopped_summary(FinishReason::Length)]);
        let Some(agent) = agent(&mut app, &model, window, messages) else {
            return;
        };
        limit_input(&mut app, agent, input);
        if let Some(mut connection) = app.world_mut().get_mut::<Connection>(agent)
            && only_input
        {
            let mut spec = connection.spec.as_ref().clone();
            (spec.context_window, spec.max_output_tokens) = (None, None);
            connection.spec = Arc::new(spec);
        }
        ask(&mut app, &wakes, agent);
        let sent = sent(&model);
        assert!(
            matches!(sent.as_slice(), [summary] if asks_summary(summary)),
            "{reserve}: {sent:?}"
        );
        let ended = &app.world().resource::<Ended>().0;
        assert!(
            matches!(ended.as_slice(), [TurnOutcome::Failed(why)] if why == NOT_SUMMARIZED),
            "{reserve}: {ended:?}"
        );
    }
}

/// A long turn whose requests overflow again and again, each time after
/// the model replied, compacts each time and ends with its answer.
#[test]
fn overflows_separated_by_replies_compact_each_time_and_the_turn_ends_answered() {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.add_tool(rig_tools::Read::default());
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let mut replies = Vec::new();
    for n in 0..7 {
        replies.push(too_long());
        replies.push(reply(&format!("## Goal\nSummary {n}.")));
        let call = format!("call_{n}");
        replies.push(rig_harness_test_support::calls(
            &call,
            "read",
            serde_json::json!({ "path": path }),
        ));
    }
    replies.push(reply("done"));
    let model = MockCompletionModel::from_stream_turns(replies);
    let Some(agent) = agent(&mut app, &model, 1_000_000, long_conversation(2)) else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    let summaries = sent.iter().filter(|request| asks_summary(request)).count();
    assert_eq!((sent.len(), summaries), (22, 7), "{sent:?}");
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
    let state = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| summary_state(c));
    assert!(state.is_some_and(|state| state.summary.contains("Summary 6")));
}

/// A reply cut off by the output limit in the middle of a tool call: the
/// call is not run, and the model reads that its reply reached the limit,
/// with only the start of what it sent, so it sends less next time.
#[test]
fn a_call_cut_off_by_the_output_limit_is_answered_so() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let content = "line\n".repeat(10_000);
    let cut_off = vec![
        MockStreamEvent::tool_call_name_delta("call_1", "write"),
        MockStreamEvent::tool_call_arguments_delta(
            "call_1",
            format!("{{\"path\": \"big.rs\", \"content\": \"{content}"),
        ),
        MockStreamEvent::FinalResponse(rig_core::operation::Finish {
            reason: Some(rig_core::completion::FinishReason::Length),
            ..rig_core::test_utils::mock_final(Usage::default())
        }),
    ];
    let model = MockCompletionModel::from_stream_turns([cut_off, reply("done")]);
    let Some(agent) = agent(&mut app, &model, 1_000_000, long_conversation(1)) else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let live = entries(app.world(), agent);
    let result = live.iter().rev().find_map(|entry| match &*entry.message {
        Message::User { content } => content.iter().find_map(|part| match part {
            UserContent::ToolResult(result) => Some(format!("{:?}", result.content)),
            _ => None,
        }),
        _ => None,
    });
    assert!(result.is_some(), "the call has a result: {live:?}");
    let result = result.unwrap_or_default();
    assert!(
        result.contains("Not run: your reply reached the output limit before `write`"),
        "{result}"
    );
    assert!(
        result.contains("big.rs") && result.len() < 1_000,
        "{result}"
    );
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [_, next] if next.contains("reached the output limit")),
        "{sent:?}"
    );
}

#[test]
fn an_interrupt_during_the_summary_cancels_the_compaction() {
    let (mut app, wakes) = start(&MemoryStore::default());
    // The answered summary waits for review, so the stop comes while it is
    // open: unheld, it can settle in the frame it was asked in.
    app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    // A tool output past the 2,000 characters a summary of its own reads.
    let output = format!("{}END", "0123456789".repeat(500));
    let mut messages = vec![Message::user("What is 2 + 2?")];
    messages.extend(read("1", "sums.txt", &output));
    messages.extend([
        Message::assistant("4"),
        Message::user("And 3 + 3?"),
        Message::assistant("6"),
    ]);
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nArithmetic.")]);
    let agent = agent(&mut app, &model, 1_000_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut()
        .trigger(RunCommand::new(agent, "compact keep the sums".to_owned()));
    run_until(&mut app, &wakes, |world| {
        world.iter_entities().any(|e| e.contains::<Reviewing>())
    });
    let world = app.world_mut();
    let asked = world
        .query::<&ModelRequest>()
        .iter(world)
        .map(|call| serde_json::to_string(&call.request).unwrap_or_default())
        .collect::<Vec<_>>();
    // The summary reads the output whole, and what to focus on.
    assert!(
        matches!(asked.as_slice(), [request] if request.contains("Additional focus: keep the sums")
            && request.contains(&output)),
        "{asked:?}"
    );
    world.trigger(StopTurn::new(agent));
    world.flush();
    app.update();
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

/// Routes every turn's call, and every call built like one, to the
/// [`Routed`] model.
fn route(
    mut calls: Query<&mut ModelRequest, (Added<Admitted>, Or<(With<TurnCall>, With<SideCall>)>)>,
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
        matches!(sent.as_slice(), [summary, _] if asks_summary(summary)),
        "a summary, then the request: {sent:?}"
    );
}

#[test]
fn a_model_whose_input_limit_is_below_its_window_compacts_before_that_limit() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nAnswer."), reply("done")]);
    // The window has room for the conversation; the input limit has not.
    let agent = agent(&mut app, &model, 1_000_000, long_conversation(6));
    let Some(agent) = agent else {
        return;
    };
    if let Some(mut connection) = app.world_mut().get_mut::<Connection>(agent) {
        let spec = connection
            .spec
            .as_ref()
            .clone()
            .with_max_input_tokens(40_000);
        connection.spec = Arc::new(spec);
    }
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [summary, _] if asks_summary(summary)),
        "a summary, then the request: {sent:?}"
    );
}

#[test]
fn a_model_that_calls_no_tools_is_not_measured_with_them() {
    let (mut app, wakes) = start(&MemoryStore::default());
    // A tool of about 8,000 tokens: with it, the 20,000-token conversation
    // would not leave the window's 16k reserve free.
    let Ok(name) = ToolName::new("notes") else {
        return;
    };
    let definition = rig_core::completion::ToolDefinition {
        name,
        description: "rule ".repeat(6_400),
        parameters: serde_json::json!({ "type": "object" }),
        annotations: Default::default(),
    };
    app.world_mut().spawn(ToolDef(definition.into()));
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
        "the request alone, without the tool: {sent:?}"
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
        matches!(sent.as_slice(), [summary, _] if asks_summary(summary)),
        "a summary, then the request: {sent:?}"
    );
}

/// A review of a summary call, holding it until the test decides.
#[derive(Component)]
struct Reviewing(Entity);

/// On a summary call the test decided.
#[derive(Component)]
struct Reviewed;

/// Holds each answered summary or notes call for review, as a plugin
/// asking the user would.
fn hold_for_review(
    calls: Query<
        Entity,
        (
            Or<(With<Summarizing>, With<Noting>)>,
            ReadyToSettle,
            Without<Reviewed>,
        ),
    >,
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

    // Edited: the reviewer's text is the summary.
    compact(&mut app);
    decide(&mut app, &wakes, Some("## Goal\nThe reviewed summary."));
    run_until(&mut app, &wakes, ended(2));
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
    let summary = live.first().map(|entry| format!("{:?}", entry.message));
    assert!(summary.is_some_and(|it| it.contains("The reviewed summary.")));
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

/// A summary near the window that a reviewer cancels lets the request go
/// as it was, its answer clamped to the room left, when that is a fair
/// share of its output budget.
#[test]
fn a_cancelled_summary_near_the_window_sends_the_request_as_it_was() {
    let (mut app, wakes) = start(&MemoryStore::default());
    app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nAnswer."), reply("done")]);
    // About 30k tokens, with about 8k of the window left for the answer.
    let Some(agent) = agent(&mut app, &model, 42_000, long_conversation(6)) else {
        return;
    };
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    decide(&mut app, &wakes, None);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert!(!entries(app.world(), agent).iter().any(holds_summary));
    let requests = model.requests();
    let limit = requests.get(1).and_then(|request| request.max_tokens);
    assert!(
        matches!(requests.as_slice(), [_, _]) && limit.is_some_and(|it| it < 12_000),
        "a summary, then the request: {limit:?}"
    );
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

/// `[compaction]` in settings sets only the fields it names over the
/// built-in default, which tracks the built-in file tools; there is no
/// other source of policy, and no command to change it.
#[test]
fn settings_set_only_what_they_name() {
    let (mut app, _wakes) = app_on(&MemoryStore::default());
    app.insert_resource(rig_harness::settings::Settings::from_toml(
        "[compaction]\nkeep_recent = 5000\nlimits = { reserve = 30000 }",
    ));
    app.add_plugins(CompactionPlugin);
    let default = app.world().resource::<CompactionDefault>().0.clone();
    assert_eq!(default.keep_recent, 5000);
    assert_eq!(default.summarizer.limits.reserve, 30_000);
    let built_in = CompactionDefault::default().0;
    assert_eq!(default.keep_asked, built_in.keep_asked);
    assert_eq!(
        default.summarizer.limits.summary_tokens,
        built_in.summarizer.limits.summary_tokens
    );
    assert_eq!(default.tracked, built_in.tracked);
    assert_eq!(default.tracked.len(), 3);

    let (mut app, _wakes) = app_on(&MemoryStore::default());
    app.insert_resource(rig_harness::settings::Settings::from_toml(
        "[compaction]\nper_turn = 1",
    ));
    app.add_plugins(CompactionPlugin);
    let problems = app
        .world()
        .resource::<rig_harness::settings::Settings>()
        .problems()
        .to_vec();
    assert!(
        problems.iter().any(|problem| problem.contains("per_turn")),
        "{problems:?}"
    );
}

#[test]
fn a_summary_that_would_free_little_is_not_asked_for() {
    let (mut app, wakes) = start(&MemoryStore::default());
    // One long reply holds nearly all of the conversation, so a cut keeping
    // the newest 20k tokens can summarize only the short messages before it.
    let messages = vec![
        Message::user("Read the formats."),
        Message::assistant("Reading them."),
        Message::user("Go on."),
        Message::assistant("word ".repeat(56_000)),
    ];
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let Some(agent) = agent(&mut app, &model, 80_000, messages) else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert!(
        matches!(sent.as_slice(), [request] if !asks_summary(request)),
        "the request alone: {sent:?}"
    );
    assert!(!entries(app.world(), agent).iter().any(holds_summary));
}

/// A Responses reply streamed with `items` as its output, reporting
/// `input` tokens read, or no usage with `None`, so the next request is
/// measured by the estimate alone.
fn streamed(
    items: Vec<serde_json::Value>,
    input: Option<u64>,
) -> rig_core::test_utils::MockHttpResponse {
    let mut body = String::new();
    for (index, item) in items.iter().enumerate() {
        let event = serde_json::json!({
            "type": "response.output_item.done", "output_index": index, "item": item,
        });
        body.push_str(&format!(
            "event: response.output_item.done\ndata: {event}\n\n"
        ));
    }
    let mut response = serde_json::json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.4",
        "output": items,
    });
    if let (Some(input), Some(response)) = (input, response.as_object_mut()) {
        let usage = serde_json::json!({
            "input_tokens": input, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input + 5,
        });
        response.insert("usage".to_owned(), usage);
    }
    let done = serde_json::json!({"type": "response.completed", "response": response});
    body.push_str(&format!("event: response.completed\ndata: {done}\n\n"));
    rig_core::test_utils::MockHttpResponse::success_typed(body, "text/event-stream")
}

/// A reply's reasoning item `n`, with its ciphertext, and its text.
fn reasoned(n: usize, text: &str) -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "reasoning", "id": format!("rs_{n}"), "summary": [], "content": [],
            "encrypted_content": format!("ciphertext-{n}"),
        }),
        serde_json::json!({
            "type": "message", "id": format!("msg_{n}"), "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }),
    ]
}

/// The first agent of `app` on the catalog's `model` with a `window`-token
/// context window, through the real Responses wire of `dialect` over
/// `http`, a scripted transport, offered `read` and `shell` (stand-ins
/// for those the app does not have).
fn on_responses(
    app: &mut App,
    (dialect, model): (&rig_core::providers::openai::wire::Dialect, &str),
    window: u32,
    http: &rig_core::test_utils::SequencedHttpClient,
) -> Option<Entity> {
    for name in ["read", "shell"] {
        let world = app.world_mut();
        if world
            .query::<&ToolDef>()
            .iter(world)
            .any(|def| def.0.name.as_str() == name)
        {
            continue;
        }
        let definition = rig_core::completion::ToolDefinition {
            name: ToolName::new(name).ok()?,
            description: "A tool.".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            annotations: Default::default(),
        };
        app.world_mut().spawn(ToolDef(definition.into()));
    }
    let served = rig_core::providers::openai::OpenAIConfig::with_key(dialect, "key")
        .connect(http.clone())
        .completion(model);
    let spec = rig_core::catalog::Catalog::builtin()
        .resolve(&format!("{}/{model}", dialect.name))
        .ok()?
        .shared();
    let spec = Arc::new(spec.as_ref().clone().with_context_window(window));
    let handler = rig_core::serve::adapters::ModelAdapter::<rig_core::operation::Completion>::new(
        spec.reference(),
        served,
    );
    let connection = Connection {
        spec,
        handler: rig_ecs::effects::Handler(rig_core::serve::ErasedHandler::new(handler)),
    };
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connection);
    Some(agent)
}

/// Asks `agent` `turns` questions, one turn each, every one answered.
fn answered_turns(app: &mut App, wakes: &Receiver<()>, agent: Entity, turns: usize) {
    for turn in 1..=turns {
        ask(app, wakes, agent);
        let ended = std::mem::take(&mut app.world_mut().resource_mut::<Ended>().0);
        assert!(
            matches!(ended.as_slice(), [TurnOutcome::Answered(_)]),
            "turn {turn}: {ended:?}"
        );
    }
}

/// The request bodies `http` got.
fn bodies(http: &rig_core::test_utils::SequencedHttpClient) -> Vec<String> {
    http.requests()
        .iter()
        .map(|request| String::from_utf8_lossy(&request.body).into_owned())
        .collect()
}

/// The bodies of the requests an agent on the catalog's `model`, through
/// the real Responses wire of `dialect` over a scripted transport, sends
/// in three turns near a 40k window, a summary held for the third: two
/// turns, the summary, then the third turn compacted.
fn responses_bodies(
    dialect: &rig_core::providers::openai::wire::Dialect,
    model: &str,
) -> Vec<String> {
    let (mut app, wakes) = start(&MemoryStore::default());
    let http = rig_core::test_utils::SequencedHttpClient::new([
        streamed(reasoned(1, "One."), Some(900)),
        // The second reply reports a request past the threshold.
        streamed(reasoned(2, "Two."), Some(30_000)),
        streamed(reasoned(3, "## Goal\nAnswer the questions."), Some(30_000)),
        streamed(reasoned(4, "done"), Some(9_000)),
    ]);
    let Some(agent) = on_responses(&mut app, (dialect, model), 40_000, &http) else {
        return Vec::new();
    };
    let mut messages = vec![Message::user("What does the crate do?")];
    messages.extend(read("1", "src/lib.rs", "pub fn answer() -> u32 { 42 }"));
    for turn in 0..3 {
        let long = "word ".repeat(4_000);
        messages.push(Message::assistant(format!("{turn}: {long}")));
        messages.push(Message::user(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    for message in messages {
        if commit_message(app.world_mut(), agent, message).is_err() {
            return Vec::new();
        }
    }
    answered_turns(&mut app, &wakes, agent, 3);
    bodies(&http)
}

/// `body` up to the closing bracket of its `input`, the field before the
/// output limit, which the Responses wire puts last as it changes from
/// call to call.
fn open_input(body: &str) -> Option<&str> {
    let body = body.strip_suffix('}')?;
    let body = match body.rsplit_once(r#","max_output_tokens":"#) {
        Some((before, limit)) if limit.bytes().all(|byte| byte.is_ascii_digit()) => before,
        _ => body,
    };
    body.strip_suffix(']')
}

/// Whether `next` is `previous` with items appended to its input: its
/// bytes up to that field's closing bracket, then more.
fn extends(previous: &str, next: &str) -> bool {
    open_input(previous).is_some_and(|open| {
        next.strip_prefix(open)
            .is_some_and(|more| more.starts_with(','))
    })
}

/// Whether `next` asks something in place of `previous`'s newest
/// messages: its bytes up to the start of its input item holding `ask` are
/// `previous`'s up to the end of one of its input items.
fn asks_after_a_prefix(previous: &str, next: &str, ask: &str) -> bool {
    let item = next
        .find(ask)
        .and_then(|at| next.get(..at))
        .and_then(|head| head.rfind(",{"))
        .and_then(|item| next.get(..item));
    item.is_some_and(|head| {
        let rest = previous.strip_prefix(head);
        rest.is_some_and(|rest| rest.starts_with(",{") || rest.starts_with(']'))
    })
}

#[test]
fn each_request_and_the_summary_start_with_the_bytes_of_the_request_before() {
    use rig_core::providers::{chatgpt, openai};
    for (dialect, model) in [
        (&openai::wire::OPENAI, "gpt-5.4"),
        (&chatgpt::DIALECT, "gpt-5.4"),
    ] {
        let bodies = responses_bodies(dialect, model);
        let name = dialect.name;
        let four = bodies.len() == 4;
        assert!(
            four,
            "{name}: two turns, the summary, the third: {bodies:#?}"
        );
        let [first, second, summary, after] = bodies.as_slice() else {
            return;
        };
        assert!(extends(first, second), "{name}:\n{first}\n{second}");
        // The reasoning comes back as the provider sent it.
        assert!(
            second.contains(r#""encrypted_content":"ciphertext-1""#),
            "{name}: {second}"
        );
        // The summary is what the second request started with, up to the
        // messages it keeps, and then the instruction.
        let ask = "Stop here: do not continue the work";
        assert!(
            asks_after_a_prefix(second, summary, ask),
            "{name}:\n{second}\n{summary}"
        );
        // Compacted, the conversation starts anew after the system prompt
        // and tools.
        assert!(!extends(summary, after), "{name}: {after}");
        assert!(after.contains("Answer the questions."), "{name}: {after}");
    }
}

#[test]
fn a_resumed_session_sends_the_bytes_it_sent_before() {
    use rig_core::providers::{chatgpt, openai};
    for (dialect, model) in [
        (&openai::wire::OPENAI, "gpt-5.4"),
        (&chatgpt::DIALECT, "gpt-5.4"),
    ] {
        let name = dialect.name;
        let store = MemoryStore::default();
        let http = rig_core::test_utils::SequencedHttpClient::new([
            streamed(reasoned(1, "One."), Some(900)),
            streamed(reasoned(2, "Two."), Some(1_000)),
            streamed(reasoned(3, "Three."), Some(1_100)),
        ]);
        let (mut app, wakes) = start(&store);
        let Some(agent) = on_responses(&mut app, (dialect, model), 1_000_000, &http) else {
            return;
        };
        answered_turns(&mut app, &wakes, agent, 2);
        drop(app);
        // The conversation comes back from the session's log.
        let (mut again, wakes) = start(&store);
        let Some(agent) = on_responses(&mut again, (dialect, model), 1_000_000, &http) else {
            return;
        };
        answered_turns(&mut again, &wakes, agent, 1);
        let bodies = bodies(&http);
        let [first, second, resumed] = bodies.as_slice() else {
            assert_eq!(bodies.len(), 3, "{name}: {bodies:#?}");
            return;
        };
        assert!(extends(first, second), "{name}:\n{first}\n{second}");
        assert!(extends(second, resumed), "{name}:\n{second}\n{resumed}");
    }
}

/// Whether `previous` is a byte prefix of `next` as [`extends`] says, or
/// how many bytes of `next` follow the longest prefix they share.
fn rewritten(previous: &str, next: &str) -> Option<usize> {
    if extends(previous, next) {
        return None;
    }
    let same = previous
        .bytes()
        .zip(next.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    Some(next.len() - same)
}

/// A reply that calls `read` for `path`, as call `n`, reporting no usage.
fn reads(n: usize, path: &str) -> rig_core::test_utils::MockHttpResponse {
    let call = serde_json::json!({
        "type": "function_call", "id": format!("fc_{n}"), "call_id": format!("call_{n}"),
        "name": "read", "arguments": serde_json::json!({ "path": path }).to_string(),
        "status": "completed",
    });
    streamed(vec![call], None)
}

#[test]
fn a_long_turn_of_reads_rewrites_the_cached_prefix_seldom_and_shallowly() {
    // Forty files of about 18k estimated tokens each, read one per request
    // in one turn, with the room below the reserve at 200k: the
    // conversation crosses it again and again.
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("compaction-long-turn");
    let line = "f([3, -1, 4, 1, -5, 9, 2, -6])  ->  [9, 4, 3, 2, 1, -1, -5, -6]\n";
    let files: Vec<String> = (0..40)
        .map(|n| dir.join(format!("k{n:02}.md")).display().to_string())
        .collect();
    let written = std::fs::create_dir_all(&dir).is_ok()
        && files
            .iter()
            .all(|file| std::fs::write(file, line.repeat(1_000)).is_ok());
    assert!(written, "the files in {}", dir.display());
    let mut replies: Vec<_> = files
        .iter()
        .enumerate()
        .map(|(n, file)| reads(n, file))
        .collect();
    replies.push(streamed(reasoned(99, "All read."), None));
    let http = rig_core::test_utils::SequencedHttpClient::new(replies);
    let (mut app, wakes) = start(&MemoryStore::default());
    app.add_tool(rig_tools::Read::default());
    let model = (&rig_core::providers::chatgpt::DIALECT, "gpt-5.4");
    let Some(agent) = on_responses(&mut app, model, 200_000 + 16_384, &http) else {
        return;
    };
    answered_turns(&mut app, &wakes, agent, 1);
    std::fs::remove_dir_all(&dir).ok();
    let bodies = bodies(&http);
    assert_eq!(bodies.len(), 41, "forty reads and the answer");
    let breaks: Vec<usize> = bodies
        .windows(2)
        .filter_map(|pair| match pair {
            [previous, next] => rewritten(previous, next),
            _ => None,
        })
        .collect();
    // Clearing keeps the newest 20k tokens of output, so each one frees
    // most of the room: twice, about 110 KB each. The files are still on
    // disk, so no notes are asked for.
    assert!(!breaks.is_empty() && breaks.len() <= 2, "{breaks:?}");
    let all: usize = breaks.iter().sum();
    assert!(all < 300_000, "{all} bytes rewritten: {breaks:?}");
    assert!(!bodies.iter().any(|body| asks_summary(body)));
    assert!(!entries(app.world(), agent).iter().any(holds_summary));
}

/// What the test `shell` tool prints among its log for `readings`, which
/// no model could guess.
const READING: &str = "QX7-4419-KM2";

/// A `shell` tool of the test: `readings` prints a log holding [`READING`]
/// past its first 2,000 characters, any other command `chars` characters
/// of log lines.
struct Shell;

impl Tool for Shell {
    const NAME: &'static str = "shell";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Run a command.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {
            "command": {"type": "string"}, "chars": {"type": "integer"},
        }})
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        let chars = args.get("chars").and_then(serde_json::Value::as_u64);
        let chars = usize::try_from(chars.unwrap_or(0)).unwrap_or(0);
        Ok(
            match args.get("command").and_then(serde_json::Value::as_str) {
                Some("readings") => format!(
                    "{}calibration code: {READING}\n{}",
                    "status ok\n".repeat(300),
                    "status ok\n".repeat(500)
                ),
                _ => "log line 0123456789\n".repeat(chars / 20),
            },
        )
    }
}

/// A reply that calls the test `shell` with `command`, printing `chars`
/// characters, as call `n`, reporting no usage.
fn shells(n: usize, command: &str, chars: usize) -> rig_core::test_utils::MockHttpResponse {
    let arguments = serde_json::json!({ "command": command, "chars": chars });
    let call = serde_json::json!({
        "type": "function_call", "id": format!("fc_{n}"), "call_id": format!("call_{n}"),
        "name": "shell", "arguments": arguments.to_string(), "status": "completed",
    });
    streamed(vec![call], None)
}

/// The kinds of the input items of the request `body` that hold `text`: a
/// message by its role, any other item by its type.
fn holding(body: &str, text: &str) -> Vec<String> {
    let body: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let items = body.get("input").and_then(serde_json::Value::as_array);
    let items = items.into_iter().flatten();
    items
        .filter(|item| item.to_string().contains(text))
        .filter_map(|item| item.get("role").or(item.get("type"))?.as_str())
        .map(str::to_owned)
        .collect()
}

/// The texts of the notices `app` wrote.
fn notices(app: &App) -> Vec<String> {
    let notices = app.world().resource::<Collected<Notice>>();
    notices.0.iter().map(|notice| notice.text.clone()).collect()
}

/// One turn on `store`, by `near_window`, near a 200k room on the real
/// Responses wire, in which the agent runs `shell` three times: the
/// readings, a 160k-token log, then a 40k-token log, which brings the
/// request past the room. `notes` answers a notes request, if one is
/// asked; then the agent answers. The request bodies, the notices, and the
/// app with its agent.
fn noted_turn(
    store: &MemoryStore,
    near_window: NearWindow,
    notes: Option<rig_core::test_utils::MockHttpResponse>,
) -> (Vec<String>, Vec<String>, App, Option<Entity>) {
    let mut replies = vec![
        shells(1, "readings", 0),
        shells(2, "log", 640_000),
        shells(3, "log", 160_000),
    ];
    replies.extend(notes);
    replies.push(streamed(reasoned(9, "Done."), None));
    let http = rig_core::test_utils::SequencedHttpClient::new(replies);
    let (mut app, wakes) = start(store);
    collect::<Notice>(&mut app).add_tool(Shell);
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .near_window = near_window;
    let model = (&rig_core::providers::chatgpt::DIALECT, "gpt-5.4");
    let agent = on_responses(&mut app, model, 200_000 + 16_384, &http);
    if let Some(agent) = agent {
        answered_turns(&mut app, &wakes, agent, 1);
    }
    (bodies(&http), notices(&app), app, agent)
}

/// Takes the notes out of each notes call, as a reviewer that cancels it.
fn cancel_notes(mut calls: Query<&mut Noting, ReadyToSettle>) {
    for mut noting in &mut calls {
        noting.notes = None;
    }
}

/// A notes call a reviewer cancels clears nothing, and a request that
/// still leaves too little room to answer is not sent: the turn fails.
#[test]
fn a_cancelled_notes_call_under_the_floor_fails_the_turn() {
    let notes = streamed(reasoned(8, "- shell `readings`: noted"), None);
    let http = rig_core::test_utils::SequencedHttpClient::new(vec![
        shells(1, "readings", 0),
        shells(2, "log", 640_000),
        shells(3, "log", 200_000),
        notes,
        streamed(reasoned(9, "never asked"), None),
    ]);
    let (mut app, wakes) = start(&MemoryStore::default());
    app.add_tool(Shell)
        .add_systems(AgentPass, cancel_notes.in_set(ReviewSystems::Act));
    let model = (&rig_core::providers::chatgpt::DIALECT, "gpt-5.4");
    let Some(agent) = on_responses(&mut app, model, 200_000 + 16_384, &http) else {
        return;
    };
    ask(&mut app, &wakes, agent);
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Failed(why)] if why == NOT_SUMMARIZED),
        "{ended:?}"
    );
    let bodies = bodies(&http);
    let noted = bodies.get(3).is_some_and(|body| asks_summary(body));
    assert!(bodies.len() == 4 && noted, "{bodies:#?}");
}

#[test]
fn near_the_window_notes_keep_a_fact_that_clearing_drops() {
    let store = MemoryStore::default();
    let notes = format!("- shell `readings`: calibration code {READING}");
    let notes = streamed(reasoned(8, &notes), None);
    let (bodies, notices, app, agent) = noted_turn(&store, NearWindow::Summarize, Some(notes));
    let asked: Vec<&String> = bodies.iter().filter(|body| asks_summary(body)).collect();
    let five = bodies.len() == 5;
    assert!(
        five,
        "three requests, the notes, the cleared one: {bodies:#?}"
    );
    let [_, _, before, notes, after] = bodies.as_slice() else {
        return;
    };
    assert_eq!(asked, [notes]);
    // The notes request is the request before it plus the instruction, so
    // the provider reads it from its cache.
    assert!(extends(before, notes), "{before}\n{notes}");
    // It states its output limit, as a summary asked in place does.
    assert!(notes.contains("Keep it under"), "{notes}");
    // The output is cleared, and the reading is in a user message, never in
    // a tool output.
    assert!(
        after.contains(rig_memory::ClearToolOutputs::PLACEHOLDER),
        "{after}"
    );
    assert_eq!(holding(after, READING), ["user"], "{after}");
    assert!(notices.iter().any(|it| it.contains("noted")), "{notices:?}");
    drop(app);
    let (mut restarted, _wakes) = start(&store);
    let agent = agent.and(restored_agent(&mut restarted));
    let live = agent.map(|agent| entries(restarted.world(), agent));
    let live = serde_json::to_string(&live).unwrap_or_default();
    assert!(live.contains(READING), "restored: {live}");

    // Cleared without notes, the reading is gone.
    let (bodies, ..) = noted_turn(&MemoryStore::default(), NearWindow::Clear, None);
    assert_eq!(bodies.len(), 4, "{bodies:#?}");
    assert!(!bodies.iter().any(|body| asks_summary(body)));
    let after = bodies.last().map(String::as_str).unwrap_or_default();
    assert!(!after.contains(READING), "{after}");
    // The estimate is read as the provider counts, between 1 and 4 tokens
    // for each one estimated.
    assert_eq!(provider_scale(Some(250_000), 100_000), 2.5);
    assert_eq!(provider_scale(Some(900_000), 100_000), 4.0);
    assert_eq!(provider_scale(None, 100_000), 1.0);
}

#[test]
fn a_stop_while_the_notes_are_held_clears_nothing() {
    let notes = format!("- shell `readings`: calibration code {READING}");
    let http = rig_core::test_utils::SequencedHttpClient::new([
        shells(1, "readings", 0),
        shells(2, "log", 640_000),
        shells(3, "log", 160_000),
        streamed(reasoned(8, &notes), None),
    ]);
    let (mut app, wakes) = start(&MemoryStore::default());
    collect::<Notice>(&mut app).add_tool(Shell);
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .near_window = NearWindow::Summarize;
    // The answered notes wait for review, so the stop comes while they are
    // open.
    app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    let model = (&rig_core::providers::chatgpt::DIALECT, "gpt-5.4");
    let agent = on_responses(&mut app, model, 200_000 + 16_384, &http);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let steer = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(steer);
    run_until(&mut app, &wakes, |world| {
        world.iter_entities().any(|e| e.contains::<Reviewing>())
    });
    let world = app.world_mut();
    let held = world.query::<&Reviewing>().iter(world).next().map(|r| r.0);
    let noting = held.and_then(|call| world.get::<Noting>(call));
    assert!(
        noting.is_some_and(|noting| noting.notes.is_some()),
        "answered notes are held"
    );
    world.trigger(StopTurn::new(agent));
    app.world_mut().flush();
    app.update();
    let live = serde_json::to_string(&entries(app.world(), agent)).unwrap_or_default();
    assert!(
        !live.contains(rig_memory::ClearToolOutputs::PLACEHOLDER),
        "{live}"
    );
    let notices = notices(&app);
    assert!(
        !notices.iter().any(|it| it.contains("cleared")),
        "{notices:?}"
    );
    assert!(
        matches!(
            app.world().resource::<Ended>().0.as_slice(),
            [TurnOutcome::Stopped(None)]
        ),
        "{:?}",
        app.world().resource::<Ended>().0
    );
}

#[test]
fn a_failed_note_clears_anyway() {
    // The model calls a tool instead of writing notes.
    let notes = Some(shells(8, "readings", 0));
    let (bodies, notices, ..) = noted_turn(&MemoryStore::default(), NearWindow::Summarize, notes);
    let asked = bodies.iter().filter(|body| asks_summary(body)).count();
    assert_eq!((bodies.len(), asked), (5, 1), "{bodies:#?}");
    let after = bodies.last().map(String::as_str).unwrap_or_default();
    assert!(
        after.contains(rig_memory::ClearToolOutputs::PLACEHOLDER),
        "{after}"
    );
    assert!(!after.contains(READING), "{after}");
    let without = notices.iter().any(|it| it.contains("without notes"));
    assert!(without, "{notices:?}");
}

#[test]
fn clearing_that_frees_little_checkpoints_once_from_a_cached_prefix() {
    // Most of the conversation is a report the user pasted, which clearing
    // cannot take out.
    let report = format!(
        "Here is the crash report:\n{}",
        "frame 0x0000 in handler\n".repeat(25_000)
    );
    let http = rig_core::test_utils::SequencedHttpClient::new([
        shells(1, "log", 120_000),
        shells(2, "log", 120_000),
        streamed(reasoned(3, "## Goal\nFix the crash."), None),
        streamed(reasoned(4, "Done."), None),
        streamed(reasoned(5, "Five."), None),
        streamed(reasoned(6, "Six."), None),
    ]);
    let (mut app, wakes) = start(&MemoryStore::default());
    collect::<Notice>(&mut app).add_tool(Shell);
    let model = (&rig_core::providers::chatgpt::DIALECT, "gpt-5.4");
    let Some(agent) = on_responses(&mut app, model, 200_000 + 16_384, &http) else {
        return;
    };
    for message in [Message::user(report), Message::assistant("I read it.")] {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    answered_turns(&mut app, &wakes, agent, 3);
    let bodies = bodies(&http);
    assert_eq!(bodies.len(), 6, "{bodies:#?}");
    let [first, second, summary, rest @ ..] = bodies.as_slice() else {
        return;
    };
    let asked = bodies.iter().filter(|body| asks_summary(body)).count();
    assert_eq!((asked, rest.len()), (1, 3), "{bodies:#?}");
    assert!(asks_summary(summary), "{summary}");
    assert!(extends(second, summary), "{second}\n{summary}");
    let notices = notices(&app);
    assert!(
        !notices.iter().any(|it| it.contains("cleared")),
        "{notices:?}"
    );
    // The prefix the provider caches is rewritten once, by the checkpoint.
    let main: Vec<&String> = [first, second].into_iter().chain(rest).collect();
    let breaks = main.windows(2).filter(|pair| match pair {
        [previous, next] => rewritten(previous, next).is_some(),
        _ => false,
    });
    assert_eq!(breaks.count(), 1);
    assert!(
        entries(app.world(), agent)
            .first()
            .is_some_and(holds_summary)
    );
}

/// The whole reason a turn fails for when its summary was not used and the
/// request still does not fit: said once.
const NOT_SUMMARIZED: &str =
    "The conversation does not fit the model's context window, and was not summarized";
