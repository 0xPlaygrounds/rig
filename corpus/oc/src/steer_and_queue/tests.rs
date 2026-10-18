use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, first_agent, log_ended, press, reply,
    run_until, scripted_models, tui_view,
};

use bevy_ecs::system::SystemState;
use rig_tui::{Editor, MainEditor};

use super::*;

/// The `shell` calls the model made, which the test answers when it
/// likes: a command that runs as long as the test needs.
#[derive(Resource, Default)]
struct Shells(Vec<Entity>);

fn shell(called: On<ToolCalled<serde_json::Value>>, mut shells: ResMut<Shells>) {
    shells.0.push(called.call);
}

/// The kernel, the journal, the terminal view and both halves of the
/// plugin on `store`,
/// with the scripted `model`, after the first frame; and the agent, shown.
fn start(store: &MemoryStore, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, PickPlugin, SteerQueuePlugin))
        .add_plugins(SteerQueueView)
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Shells>()
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Recalled>(&mut app);
    tui_view(&mut app);
    // The app is its own view: the agent half's lists reach the strip.
    rig_harness_test_support::view::loopback(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    app.world_mut().resource_mut::<ShownAgent>().0 = Some(agent);
    app.update();
    (app, wakes, agent)
}

/// The messages of the model's request `at`, as text.
fn request(model: &MockCompletionModel, at: usize) -> Vec<String> {
    let requests = model.requests();
    let history = requests.get(at).map(|request| request.chat_history.clone());
    let messages = history.unwrap_or_default();
    messages
        .iter()
        .map(|message| format!("{message:?}"))
        .collect()
}

fn last(said: &[String]) -> &str {
    said.last().map(String::as_str).unwrap_or_default()
}

fn pending(app: &mut App, agent: Entity) -> Vec<(String, PendingKind)> {
    let mut state = SystemState::<PendingMessages>::new(app.world_mut());
    let items = state.get(app.world()).map(|p| p.of(agent));
    let items = items.unwrap_or_default();
    items
        .into_iter()
        .map(|item| (item.text, item.kind))
        .collect()
}

/// Answers the `shell` call `at` with "ok", as the command ending.
fn finish_shell(app: &mut App, at: usize) {
    let call = app.world().resource::<Shells>().0.get(at).copied();
    let decision = call.and_then(|call| app.world_mut().get_mut::<Decision>(call));
    if let Some(mut decision) = decision {
        decision.decide(Decider::plugin("shell"), ToolResult::success("ok".into()));
    }
}

/// Types `text` in the terminal view's input and presses `keys`.
fn type_and(app: &mut App, text: &str, keys: &str) {
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut Editor, With<MainEditor>>();
    let typed = main.single_mut(world).map(|mut input| input.set(text));
    assert!(typed.is_ok(), "the default screen has a main editor");
    press(app, keys);
}

/// Runs the frames that bring the agent half's list to the strip.
fn shown(app: &mut App) {
    app.update();
    app.update();
}

/// The text in the terminal view's input.
fn typed(app: &mut App) -> String {
    let world = app.world_mut();
    let mut main = world.query_filtered::<&Editor, With<MainEditor>>();
    main.single(world)
        .map(|input| input.text().to_owned())
        .unwrap_or_default()
}

/// Steering, queueing, editing, reordering, interrupting and restarting,
/// as the spec's scripted test runs them.
#[test]
fn steer_queue_edit_interrupt_and_restart() {
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "shell", serde_json::json!({})),
        reply("done 1"),
        calls("c2", "shell", serde_json::json!({})),
        reply("done 2"),
        reply("docs updated"),
        reply("version bumped"),
        calls("c3", "shell", serde_json::json!({})),
        reply("done 5"),
        calls("c4", "shell", serde_json::json!({})),
        reply("kept"),
        calls("c5", "shell", serde_json::json!({})),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &model);
    let shells = |n: usize| move |world: &World| world.resource::<Shells>().0.len() >= n;
    let ended = |n: usize| move |world: &World| world.resource::<Ended>().0.len() >= n;

    // Enter while the shell runs steers: the shell finishes, and the next
    // request ends with its result, then the user's message.
    type_and(&mut app, "start", "enter");
    run_until(&mut app, &wakes, shells(1));
    type_and(&mut app, "use the v2 API", "enter");
    app.update();
    assert_eq!(
        pending(&mut app, agent),
        [("use the v2 API".to_owned(), PendingKind::Steer)]
    );
    finish_shell(&mut app, 0);
    run_until(&mut app, &wakes, ended(1));
    let said = request(&model, 1);
    // The tool's own result ("ok", not stopped), then the user's message.
    let ends = last(&said);
    let result = ends.find("text: \"ok\"");
    let steered = ends.find("use the v2 API");
    assert!(result.zip(steered).is_some_and(|(r, s)| r < s), "{said:?}");
    assert!(!ends.contains("interrupted"), "{said:?}");
    assert!(pending(&mut app, agent).is_empty());

    // Alt+Enter queues: each queued message is a later turn of its own,
    // in order, after the turn's answer.
    type_and(&mut app, "next task", "enter");
    run_until(&mut app, &wakes, shells(2));
    type_and(&mut app, "then update docs", "alt+enter");
    type_and(&mut app, "and bump version", "alt+enter");
    app.update();
    assert_eq!(pending(&mut app, agent).len(), 2);
    finish_shell(&mut app, 1);
    run_until(&mut app, &wakes, ended(4));
    assert_eq!(model.request_count(), 6);
    assert!(
        last(&request(&model, 3)).contains("Tool"),
        "the turn ends first"
    );
    let third = request(&model, 4);
    assert!(last(&third).contains("then update docs"), "{third:?}");
    assert!(third.iter().all(|m| !m.contains("and bump version")));
    let fourth = request(&model, 5);
    assert!(last(&fourth).contains("and bump version"), "{fourth:?}");
    assert!(pending(&mut app, agent).is_empty());

    // A queued message selected in the strip, edited to "X" and turned to
    // steer reaches the running turn's next request.
    type_and(&mut app, "long job", "enter");
    run_until(&mut app, &wakes, shells(3));
    type_and(&mut app, "draft", "alt+enter");
    shown(&mut app);
    press(&mut app, "up e");
    assert_eq!(typed(&mut app), "draft");
    type_and(&mut app, "X", "enter");
    shown(&mut app);
    press(&mut app, "up t esc");
    app.update();
    assert_eq!(
        pending(&mut app, agent),
        [("X".to_owned(), PendingKind::Steer)]
    );
    // A second steering message moved before it: the kernel sends both,
    // in the new order.
    type_and(&mut app, "Y", "enter");
    shown(&mut app);
    press(&mut app, "up k esc");
    app.update();
    let steer = |text: &str| (text.to_owned(), PendingKind::Steer);
    assert_eq!(pending(&mut app, agent), [steer("Y"), steer("X")]);
    finish_shell(&mut app, 2);
    run_until(&mut app, &wakes, ended(5));
    let said = request(&model, 7);
    let (y, x) = (last(&said).find("\"Y\""), last(&said).find("\"X\""));
    assert!(y.zip(x).is_some_and(|(y, x)| y < x), "{said:?}");

    // Esc with a steering message pending: it stays pending, not handed
    // back, and the next turn starts with it.
    type_and(&mut app, "again", "enter");
    run_until(&mut app, &wakes, shells(4));
    type_and(&mut app, "keep this", "enter");
    app.update();
    press(&mut app, "esc");
    run_until(&mut app, &wakes, ended(6));
    app.update();
    assert_eq!(
        pending(&mut app, agent),
        [("keep this".to_owned(), PendingKind::Steer)]
    );
    assert!(app.world().resource::<Collected<Recalled>>().0.is_empty());
    assert!(typed(&mut app).is_empty());
    type_and(&mut app, "go on", "enter");
    run_until(&mut app, &wakes, ended(7));
    let said = request(&model, 9);
    assert!(last(&said).contains("keep this\\n\\ngo on"), "{said:?}");
    assert!(pending(&mut app, agent).is_empty());

    // Queued during a turn, then the process dies: after the restart the
    // message is pending, and was never sent.
    type_and(&mut app, "final", "enter");
    run_until(&mut app, &wakes, shells(5));
    type_and(&mut app, "later", "alt+enter");
    app.update();
    drop(app);
    let (mut app, wakes, agent) = start(&store, &model);
    run_until(&mut app, &wakes, |world| {
        world
            .get::<Kept>(agent)
            .is_some_and(|kept| !kept.0.is_empty())
    });
    assert_eq!(
        pending(&mut app, agent),
        [("later".to_owned(), PendingKind::Queued)]
    );
    let sent = model.requests();
    assert!(
        sent.iter()
            .all(|request| !format!("{request:?}").contains("later"))
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SteerQueuePlugin);
}
