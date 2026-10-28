use std::sync::mpsc::Receiver;

use rig_basics::sessions::SessionsPlugin;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_in, app_on, calls, collect, connect, first_agent, log_ended,
    reply, run_until, scratch_home, start_call,
};
use rig_tui::StatusItems;
use serde_json::json;

use super::*;

/// The texts and done marks of `agent`'s list.
fn items(world: &World, agent: Entity) -> Vec<(String, bool)> {
    let list = world.get::<Checklist>(agent);
    let items = list.map(|list| list.items.iter().map(|item| (item.text.clone(), item.done)));
    items.into_iter().flatten().collect()
}

/// The todo plugin with the kernel, on `app`, after the first frame
/// restored the session; its first agent.
fn start(mut app: App) -> (App, Option<Entity>) {
    app.add_plugins((AgentPlugin, SessionLogPlugin, TodoPlugin))
        // The terminal view gives every agent its status items.
        .register_required_components::<Agent, StatusItems>()
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    (app, agent)
}

/// Runs `call` of the todo tool for `agent` with `args`, as its model
/// asked, until it settles; its output.
fn todo(app: &mut App, wakes: &Receiver<()>, agent: Entity, call: &str, args: serde_json::Value) {
    let started = start_call(app.world_mut(), agent, (TODO_TOOL, call), args);
    let found = started;
    assert!(found.is_some(), "a todo call");
    let Some(started) = found else {
        return;
    };
    run_until(app, wakes, |world| world.get::<Settled>(started).is_some());
}

/// The todo calls of `ids`, in one reply.
fn adds(ids: &[(&str, &str)]) -> Vec<MockStreamEvent> {
    let calls = ids.iter().map(|(id, text)| {
        MockStreamEvent::tool_call(*id, TODO_TOOL, json!({ "action": "add", "text": text }))
    });
    calls
        .chain([MockStreamEvent::final_response_with_default_usage()])
        .collect()
}

/// The session `id` of `home` in an app of its own, as `rig --resume`
/// runs it, with the session listing.
fn resumed(home: &Home, id: &SessionId) -> (App, Receiver<()>, Option<Entity>) {
    let (mut app, wakes) = app_in(home, id);
    app.add_plugins(SessionsPlugin);
    let (app, agent) = start(app);
    (app, wakes, agent)
}

#[test]
fn the_checklist_and_goal_survive_a_restart_and_a_crash_and_are_read_without_an_app() {
    let home = scratch_home("rf-03-restart");
    let id = SessionId::generate();
    let (mut app, wakes, agent) = resumed(&home, &id);
    let model = MockCompletionModel::from_stream_turns([
        adds(&[("1", "parse"), ("2", "check"), ("3", "emit")]),
        calls("4", TODO_TOOL, json!({ "action": "done", "id": 1 })),
        reply("Parsing is done."),
    ]);
    let pair = agent.zip(connect(&model));
    let found = pair;
    assert!(found.is_some(), "an agent and the catalog's model");
    let Some((agent, connection)) = found else {
        return;
    };
    let world = app.world_mut();
    world.entity_mut(agent).insert(connection);
    world.resource_mut::<Goal>().0 = "Ship the parser".to_owned();
    world.trigger(Deliver::new(agent, "Plan the parser.", DeliveryMode::Steer));
    let idle = |world: &World| {
        !world.resource::<Ended>().0.is_empty() && world.get::<ActiveTurn>(agent).is_none()
    };
    run_until(&mut app, &wakes, idle);
    app.update();
    // A crash: the app goes without its exit path.
    drop(app);

    // A program without an app reads the stopped session, its goal from
    // the listing alone.
    let dir = home.session(&id);
    let store = StoreOpener::default().open(&dir);
    assert!(store.is_ok(), "the session's store opens");
    let Ok(store) = store else {
        return;
    };
    assert_eq!(listed_goal(&*store).as_deref(), Some("Ship the parser"));
    let stopped = read_stopped(&*store);
    let lists = stopped.as_ref().map(|stopped| {
        let texts = stopped.lists.iter().map(|(_, list)| list.render());
        (stopped.goal.clone(), texts.collect::<Vec<_>>())
    });
    let expected = "[x] 1. parse\n[ ] 2. check\n[ ] 3. emit".to_owned();
    assert_eq!(lists, Ok(("Ship the parser".to_owned(), vec![expected])));
    drop(store);

    // Resumed: the list, the goal, the status line and the reminder are
    // back before any turn runs.
    let (mut app, _wakes, agent) = resumed(&home, &id);
    let found = agent;
    assert!(found.is_some(), "the restored agent");
    let Some(agent) = found else {
        return;
    };
    app.update();
    let world = app.world_mut();
    let want = [("parse", true), ("check", false), ("emit", false)];
    let want: Vec<(String, bool)> = want
        .iter()
        .map(|(text, done)| ((*text).to_owned(), *done))
        .collect();
    assert_eq!(items(world, agent), want);
    assert_eq!(world.resource::<Goal>().0, "Ship the parser");
    let status = world.get::<StatusItems>(agent);
    let shown: Vec<String> = status
        .into_iter()
        .flat_map(StatusItems::iter)
        .map(|item| item.text.clone())
        .collect();
    assert!(shown.contains(&"1/3 done".to_owned()), "{shown:?}");
    let armed = world
        .query::<&Reminder>()
        .iter(world)
        .any(|reminder| reminder.agent == agent);
    assert!(armed, "the reminder is re-armed after the restore");
    drop(app);
    std::fs::remove_dir_all(home.root()).ok();
}

/// Holds every todo call's outcome, as a crash would leave it: applied,
/// not settled.
fn hold_todo_outcomes(
    calls: Query<(Entity, &ToolRequest), (ReadyToSettle, ToolCalls, Without<Holds>)>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        if request.call.function.name.as_str() == TODO_TOOL {
            commands.spawn((Name::new("crash"), HoldOf(call)));
        }
    }
}

#[test]
fn a_call_a_restart_cut_short_is_not_applied_again() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.add_systems(AgentPass, hold_todo_outcomes.in_set(Stage::Review));
    let (mut app, agent) = start(app);
    let model = MockCompletionModel::from_stream_turns([adds(&[("1", "parse")])]);
    let found = agent.zip(connect(&model));
    assert!(found.is_some(), "an agent and the catalog's model");
    let Some((agent, connection)) = found else {
        return;
    };
    let world = app.world_mut();
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Plan the parser.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| !items(world, agent).is_empty());
    app.update();
    drop(app);

    let (app, wakes) = app_on(&store);
    let (mut app, agent) = start(app);
    let found = agent;
    assert!(found.is_some(), "the restored agent");
    let Some(agent) = found else {
        return;
    };
    let settled = |world: &World| {
        let mut results = world
            .iter_entities()
            .filter(|entity| entity.contains::<ToolRequest>());
        results.any(|entity| entity.contains::<Settled>())
    };
    run_until(&mut app, &wakes, settled);
    assert_eq!(items(app.world(), agent), [("parse".to_owned(), false)]);
    // The restart stops the turn: the call is answered as cut short, not
    // run again.
    let world = app.world_mut();
    let mut outcomes = world.query_filtered::<&Outcome, (With<ToolRequest>, With<Settled>)>();
    let outputs: Vec<String> = outcomes
        .iter(world)
        .map(|outcome| outcome.tool_result().output().render())
        .collect();
    assert_eq!(
        outputs,
        ["Stopped: the session restarted. It may have partly run; check before running it again."]
    );
}

/// Each fork made, from and to.
#[derive(Resource, Default)]
struct Forks(Vec<Entity>);

#[test]
fn a_fork_copies_the_list_and_diverges_and_a_subagent_starts_empty() {
    let store = MemoryStore::default();
    let (app, wakes) = app_on(&store);
    let (mut app, agent) = start(app);
    let found = agent;
    assert!(found.is_some(), "the first agent");
    let Some(parent) = found else {
        return;
    };
    app.init_resource::<Forks>().add_observer(
        |forked: On<AgentForked>, mut forks: ResMut<Forks>| forks.0.push(forked.fork),
    );
    todo(
        &mut app,
        &wakes,
        parent,
        "1",
        json!({ "action": "add", "text": "one" }),
    );
    todo(
        &mut app,
        &wakes,
        parent,
        "2",
        json!({ "action": "add", "text": "two" }),
    );
    app.world_mut().trigger(ForkAgent {
        agent: parent,
        through: None,
    });
    app.update();
    let found = app.world().resource::<Forks>().0.first();
    assert!(found.is_some(), "a fork");
    let Some(&fork) = found else {
        return;
    };
    todo(
        &mut app,
        &wakes,
        parent,
        "3",
        json!({ "action": "add", "text": "three" }),
    );
    todo(
        &mut app,
        &wakes,
        fork,
        "4",
        json!({ "action": "add", "text": "four" }),
    );
    let subagent = app.world_mut().spawn((Agent, SpawnedBy(parent))).id();
    app.update();
    let texts = |world: &World, agent| -> Vec<String> {
        items(world, agent)
            .into_iter()
            .map(|(text, _)| text)
            .collect()
    };
    assert_eq!(texts(app.world(), parent), ["one", "two", "three"]);
    assert_eq!(texts(app.world(), fork), ["one", "two", "four"]);
    assert!(texts(app.world(), subagent).is_empty());
    drop(app);

    let stopped = read_stopped(&store);
    let lists: Vec<String> = stopped
        .into_iter()
        .flat_map(|it| it.lists)
        .map(|(_, list)| list.render())
        .collect();
    assert_eq!(
        lists,
        [
            "[ ] 1. one\n[ ] 2. two\n[ ] 3. three",
            "[ ] 1. one\n[ ] 2. two\n[ ] 3. four"
        ],
    );
}

/// A saved checklist of an earlier version of the plugin, as any JSON.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
struct EarlierList(serde_json::Value);

#[test]
fn an_earlier_saved_form_is_read_or_skipped_with_a_notice() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = app_on(&store);
    app.add_plugins((AgentPlugin, SessionLogPlugin));
    let _unlisted = app.saved_as::<EarlierList>(LIST_KEY);
    app.finish();
    app.update();
    let found = first_agent(app.world_mut());
    assert!(found.is_some(), "the first agent");
    let Some(first) = found else {
        return;
    };
    // The first agent's list is of a version before `Item::call`; the
    // second's of one whose items were plain strings.
    let before_call = json!({ "items": [{ "id": 1, "text": "parse", "done": true }] });
    let world = app.world_mut();
    let second = world.spawn(Agent).id();
    app.update();
    // Values set after the agents were spawned, so their logs are kept.
    let world = app.world_mut();
    world.entity_mut(first).insert(EarlierList(before_call));
    world
        .entity_mut(second)
        .insert(EarlierList(json!(["check"])));
    app.update();
    drop(app);

    let (app, wakes) = app_on(&store);
    let (mut app, _) = start(app);
    let world = app.world_mut();
    let mut agents = world.query_filtered::<(Entity, &Checklist), With<Agent>>();
    let lists: Vec<(Entity, String)> = agents
        .iter(world)
        .map(|(agent, list)| (agent, list.render()))
        .collect();
    let rendered: Vec<&str> = lists.iter().map(|(_, list)| list.as_str()).collect();
    assert_eq!(rendered, ["[x] 1. parse", "The checklist is empty."]);
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let skipped = notices.iter().any(|notice| notice.text.contains(LIST_KEY));
    assert!(skipped, "{notices:?}");
    let found = lists.get(1);
    assert!(found.is_some(), "two agents");
    let Some(&(second, _)) = found else {
        return;
    };
    todo(
        &mut app,
        &wakes,
        second,
        "1",
        json!({ "action": "add", "text": "check" }),
    );
    drop(app);

    let stopped = read_stopped(&store);
    let lists: Vec<String> = stopped
        .into_iter()
        .flat_map(|it| it.lists)
        .map(|(_, list)| list.render())
        .collect();
    assert_eq!(lists, ["[x] 1. parse", "[ ] 1. check"]);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(TodoPlugin);
}
