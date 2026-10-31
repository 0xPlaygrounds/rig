use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, first_agent, log_ended, reply,
    run_until, scripted_models,
};

use super::*;

/// What the runaway command prints.
const LOG_BYTES: usize = 200_000;

/// The arguments of `shell`.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Command {
    /// The command.
    command: String,
}

/// A `shell` that prints a huge log, and never ends for `hang`.
fn shell(called: On<ToolCalled<Command>>, mut commands: Commands) {
    if called.args.command == "hang" {
        return;
    }
    let output = Outcome::from(ToolResult::success("x".repeat(LOG_BYTES).into()));
    commands.entity(called.call).insert_if_new(output);
}

/// The kernel and the plugin on `store`, with a `shell` tool, the agent
/// answered by `model`.
fn start(store: &MemoryStore, model: &MockCompletionModel) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, ConversationSurgeryPlugin))
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app).finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    Some((app, wakes, agent))
}

/// Asks `agent` something and runs frames until its turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Types `line` to `agent` and returns the notices it wrote, and why the
/// line was refused, marked `refused: `.
fn typed(app: &mut App, agent: Entity, line: &str) -> String {
    let world = app.world_mut();
    world.resource_mut::<Collected<Notice>>().0.clear();
    world.resource_mut::<Collected<Recalled>>().0.clear();
    let line = Deliver::new(agent, line, DeliveryMode::Queue);
    world.trigger(line);
    app.update();
    let world = app.world();
    let notices = world.resource::<Collected<Notice>>().0.iter();
    let recalled = world.resource::<Collected<Recalled>>().0.iter();
    let texts: Vec<String> = notices
        .map(|notice| notice.text.clone())
        .chain(recalled.map(|line| format!("refused: {}", line.why.clone().unwrap_or_default())))
        .collect();
    texts.join("\n")
}

fn messages(app: &App, agent: Entity) -> Vec<Message> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| {
        conversation
            .messages()
            .iter()
            .map(|message| (**message).clone())
            .collect()
    })
}

fn json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

#[test]
fn stub_and_drop_edit_the_logged_conversation_and_wait_for_the_turn() {
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "c1",
            "shell",
            serde_json::json!({ "command": "cat build.log" }),
        ),
        reply("The build failed."),
        reply("Still failed."),
        reply("Yes."),
    ]);
    let store = MemoryStore::default();
    let started = start(&store, &model);
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    ask(&mut app, &wakes, agent, "Read the build log.");
    assert_eq!(messages(&app, agent).len(), 4, "user, call, result, answer");
    let listed = typed(&mut app, agent, "/history");
    assert!(listed.contains("result of shell (c1)"), "{listed}");

    let id = |app: &App| app.world().get::<Conversation>(agent).and_then(|c| c.id(2));
    let result_id = id(&app);
    let stubbed = typed(&mut app, agent, "/stub 3");
    assert_eq!(id(&app), result_id, "the stubbed result keeps its id");
    assert!(stubbed.contains("200 KB"), "{stubbed}");
    let result = messages(&app, agent).get(2).map(json).unwrap_or_default();
    assert!(
        result.contains("[removed by the user: 200 KB]") && !result.contains("xxxx"),
        "{result}"
    );
    ask(&mut app, &wakes, agent, "And now?");
    let sent = model.requests().last().map(json).unwrap_or_default();
    assert!(
        sent.contains("[removed by the user: 200 KB]") && !sent.contains("xxxx"),
        "the next request carries the stub: {sent}"
    );

    let before = messages(&app, agent);
    // Dropping the answer between the result and the next question joins
    // them into one user message.
    let joined = typed(&mut app, agent, "/drop 4");
    assert!(joined.contains("joined message 5 to message 3"), "{joined}");
    assert_eq!(messages(&app, agent).len(), 4, "{joined}");
    let undone = typed(&mut app, agent, "/drop undo");
    assert_eq!(json(&messages(&app, agent)), json(&before), "{undone}");
    typed(&mut app, agent, "/drop 2");
    let undone = typed(&mut app, agent, "/drop undo");
    assert_eq!(json(&messages(&app, agent)), json(&before), "{undone}");
    let dropped = typed(&mut app, agent, "/drop 2");
    assert!(dropped.contains("Dropped message 2"), "{dropped}");
    let left = json(&messages(&app, agent));
    assert!(
        !left.contains("c1"),
        "the call and its result are gone: {left}"
    );
    assert_eq!(messages(&app, agent).len(), 4, "{left}");
    ask(&mut app, &wakes, agent, "Did it build?");
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.last(), Some(TurnOutcome::Answered(_))),
        "{ended:?}"
    );
    let sent = model.requests().last().map(json).unwrap_or_default();
    // Quoted: the request's cache key is a random UUID that may hold "c1".
    assert!(!sent.contains("\"c1\""), "{sent}");
    let edited = messages(&app, agent);
    drop(app);

    // A restart brings the edited conversation back, and an edit waits
    // for the running turn.
    let hangs = MockCompletionModel::from_stream_turns([calls(
        "c2",
        "shell",
        serde_json::json!({ "command": "hang" }),
    )]);
    let restarted = start(&store, &hangs);
    assert!(restarted.is_some(), "the agent was restored");
    let Some((mut app, wakes, agent)) = restarted else {
        return;
    };
    assert_eq!(json(&messages(&app, agent)), json(&edited));
    app.world_mut()
        .trigger(Deliver::new(agent, "Build it.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_some() && hangs.request_count() > 0
    });
    let refused = typed(&mut app, agent, "/drop 1");
    assert!(
        refused.contains(
            "refused: /drop: the agent's turn is running; wait for the turn to end or interrupt it."
        ),
        "{refused}"
    );
    assert_eq!(
        json(&messages(&app, agent).get(..6)),
        json(&edited.get(..6))
    );
}
