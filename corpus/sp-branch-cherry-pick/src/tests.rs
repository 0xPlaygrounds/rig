use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::SystemTime;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, first_agent, log_ended, press, reply,
    run_until, scripted_models, tui_view,
};
use rig_tui::TuiTestScreen;

use super::*;

/// Where the `read` tool reads, and how often it ran.
#[derive(Resource, Clone)]
struct Work {
    root: PathBuf,
}

#[derive(Resource, Default)]
struct Reads(usize);

/// A turn that called `hang` and waits on it.
#[derive(Resource, Default)]
struct Hanging(bool);

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Path {
    /// The file.
    path: String,
}

/// `read`: the file's text, from the work directory.
fn read(
    called: On<ToolCalled<Path>>,
    work: Res<Work>,
    mut reads: ResMut<Reads>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    reads.0 += 1;
    let text = fs::read_to_string(work.root.join(&called.args.path)).unwrap_or_default();
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("read"), ToolResult::success(text.into()));
    }
}

/// `hang`: never answers, so its turn keeps running.
fn hang(_called: On<ToolCalled<serde_json::Value>>, mut hanging: ResMut<Hanging>) {
    hanging.0 = true;
}

/// The kernel, the picker and the plugin on `store`, the agent answered by
/// `model`.
fn start(
    store: &MemoryStore,
    model: &MockCompletionModel,
    work: &Work,
    tui: bool,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(scripted_models([("test/model", model)]))
        .insert_resource(work.clone())
        .init_resource::<Reads>()
        .init_resource::<Hanging>()
        .add_plugins((AgentPlugin, SessionLogPlugin, PickPlugin, CherryPickPlugin))
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .add_open_tool("hang", "Wait.", ToolOptions::default(), hang)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    if tui {
        app.insert_resource(TuiTestScreen::new(100, 30));
        tui_view(&mut app);
    }
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.insert_resource(Shown(Some(agent)));
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    app.update();
    (app, wakes, agent)
}

fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Runs `/line` for `agent`; why it was refused, if it was.
fn run(app: &mut App, agent: Entity, line: &str) -> Option<String> {
    app.world_mut()
        .resource_mut::<Collected<Recalled>>()
        .0
        .clear();
    app.world_mut().trigger(RunCommand::new(agent, line));
    app.update();
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    recalled
        .last()
        .map(|line| line.why.clone().unwrap_or_default())
}

fn conversation(app: &App, agent: Entity) -> Option<&Conversation> {
    app.world().get::<Conversation>(agent)
}

/// The active line, each message as its text, tool calls by name.
fn path(app: &App, agent: Entity) -> Vec<String> {
    let messages = conversation(app, agent)
        .map(|c| c.messages().to_vec())
        .unwrap_or_default();
    messages.iter().map(said).collect()
}

fn said(message: &Said) -> String {
    match message {
        Said::User { content } => content
            .iter()
            .map(|item| match item {
                UserContent::Text(text) => text.text.clone(),
                UserContent::ToolResult(result) => format!("result of {}", result.name),
                _ => String::new(),
            })
            .collect(),
        Said::Assistant(reply) => reply
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(text) => text.text.clone(),
                AssistantContent::ToolCall(call) => format!("call {}", call.function.name),
                _ => String::new(),
            })
            .collect(),
        Said::System { content } => content.clone(),
    }
}

/// The kept branch `title`'s entries.
fn branch(app: &App, agent: Entity, title: &str) -> Option<Vec<Entry>> {
    let kept = conversation(app, agent)?.branches();
    kept.into_iter()
        .find(|kept| kept.branch.title == title)
        .map(|kept| kept.branch.entries)
}

/// The open picker and its entity.
fn picker(app: &mut App) -> Option<(Entity, Picker)> {
    let world = app.world_mut();
    let mut pickers = world.query::<(Entity, &Picker)>();
    pickers
        .iter(world)
        .map(|(entity, picker)| (entity, picker.clone()))
        .last()
}

fn modified(file: &PathBuf) -> Option<SystemTime> {
    fs::metadata(file).and_then(|meta| meta.modified()).ok()
}

#[test]
fn a_turn_from_another_branch_is_copied_to_the_end_and_taken_back() {
    let root = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("cherry-pick-work");
    fs::remove_dir_all(&root).ok();
    fs::create_dir_all(root.join("src")).ok();
    let lock = root.join("src/lock.rs");
    fs::write(&lock, "// one mutex per shard, taken in shard order\n").ok();
    let before = modified(&lock);
    let work = Work { root: root.clone() };
    let read_lock = serde_json::json!({ "path": "src/lock.rs" });
    let model = MockCompletionModel::from_stream_turns([
        reply("ONE"),
        calls("c1", "read", read_lock.clone()),
        reply("TWO"),
        calls("c1", "read", read_lock.clone()),
        reply("LOCK-EXPLAINED"),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &model, &work, true);

    // Turns 1 and 2 on the main line; a branch from the end of turn 1
    // with turn 2'; back to the main line's leaf.
    ask(&mut app, &wakes, agent, "Turn one");
    ask(&mut app, &wakes, agent, "Turn two");
    let main = app
        .world_mut()
        .get_mut::<Conversation>(agent)
        .and_then(|mut c| c.branch_off(2, "retry", Author::user()).ok());
    ask(&mut app, &wakes, agent, "Explain the locking scheme");
    let switched = app
        .world_mut()
        .get_mut::<Conversation>(agent)
        .zip(main)
        .map(|(mut c, main)| c.switch_branch(main, Author::user()).is_ok());
    assert_eq!(switched, Some(true));
    app.update();
    let main_line = [
        "Turn one",
        "ONE",
        "Turn two",
        "call read",
        "result of read",
        "TWO",
    ];
    assert_eq!(path(&app, agent), main_line);
    let retry = branch(&app, agent, "retry");
    assert_eq!(app.world().resource::<Reads>().0, 2);

    // The picker lists turn 2' only, on the screen too.
    assert_eq!(run(&mut app, agent, "cherry-pick"), None);
    app.update();
    let listed = picker(&mut app)
        .map(|(_, picker)| picker.items)
        .unwrap_or_default();
    let labels: Vec<&str> = listed.iter().map(|item| item.label.as_str()).collect();
    assert_eq!(labels, ["[retry] Explain the locking scheme"]);
    let rows = app.world().resource::<TuiTestScreen>().rows();
    assert!(
        rows.iter()
            .any(|row| row.contains("[retry] Explain the locking scheme")),
        "{rows:#?}"
    );

    // Picked with Enter: turn 1, turn 2, the note, the copy of 2'.
    press(&mut app, "enter");
    app.update();
    let picked = path(&app, agent);
    assert_eq!(picked.get(..6), Some(&main_line.map(str::to_owned)[..]));
    assert!(
        picked
            .get(6)
            .is_some_and(|note| note.contains("NOT applied")),
        "{picked:?}"
    );
    let copy = [
        "Explain the locking scheme",
        "call read",
        "result of read",
        "LOCK-EXPLAINED",
    ];
    assert_eq!(picked.get(7..), Some(&copy.map(str::to_owned)[..]));
    let marks: Vec<String> = conversation(&app, agent)
        .map(|c| {
            c.asides()
                .filter_map(|(_, aside)| Note::of(aside))
                .map(|note| note.title)
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(marks, ["picked from retry"]);
    // Nothing ran, no file changed, the source branch is as it was.
    assert_eq!(app.world().resource::<Reads>().0, 2);
    assert_eq!(modified(&lock), before);
    assert_eq!(branch(&app, agent, "retry"), retry);
    drop(app);

    // Restarted: the same path; undone, back to turns 1 and 2.
    let model = MockCompletionModel::from_stream_turns([
        reply("Applied."),
        calls("h1", "hang", serde_json::json!({})),
    ]);
    let (mut app, wakes, agent) = start(&store, &model, &work, false);
    assert_eq!(path(&app, agent), picked);
    assert_eq!(run(&mut app, agent, "cherry-pick undo"), None);
    assert_eq!(path(&app, agent), main_line);

    // Picked again, then asked on: the request holds the note, the call
    // with its result and the reply, every call id once.
    assert_eq!(run(&mut app, agent, "cherry-pick"), None);
    let open = picker(&mut app).map(|(entity, _)| entity);
    if let Some(open) = open {
        app.world_mut()
            .trigger(Pick::person(open, Some(Chosen::item(0))));
    }
    app.update();
    assert_eq!(path(&app, agent), picked);
    ask(&mut app, &wakes, agent, "Apply that plan here.");
    let request = model
        .requests()
        .last()
        .map(|request| request.chat_history.clone())
        .unwrap_or_default();
    let sent: Vec<String> = request.iter().map(said).collect();
    assert!(
        sent.iter().any(|text| text.contains("NOT applied")),
        "{sent:?}"
    );
    assert!(sent.iter().any(|text| text == "LOCK-EXPLAINED"), "{sent:?}");
    let call_ids: Vec<String> = request
        .iter()
        .flat_map(|message| match message {
            Said::Assistant(reply) => reply.content.clone(),
            _ => Vec::new(),
        })
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call.id.to_string()),
            _ => None,
        })
        .collect();
    let mut unique = call_ids.clone();
    unique.sort();
    unique.dedup();
    assert_eq!((call_ids.len(), unique.len()), (2, 2), "{call_ids:?}");
    let results = request.iter().flat_map(|message| match message {
        Said::User { content } => content.clone(),
        Said::Assistant(_) | Said::System { .. } => Vec::new(),
    });
    let read_back = results.filter_map(|item| match item {
        UserContent::ToolResult(result) => Some(format!("{:?}", result.content)),
        _ => None,
    });
    assert_eq!(
        read_back
            .filter(|text| text.contains("one mutex per shard"))
            .count(),
        2
    );
    let pairs = rig_core::conversation::pairing::validate(&request);
    assert!(pairs.is_ok(), "{pairs:?}");

    // Something came after the pick: undo refuses.
    let refused = run(&mut app, agent, "cherry-pick undo").unwrap_or_default();
    assert!(refused.contains("added after"), "{refused}");

    // While a turn runs, picking is refused.
    app.world_mut()
        .trigger(Deliver::new(agent, "Wait a while.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Hanging>().0);
    let refused = run(&mut app, agent, "cherry-pick").unwrap_or_default();
    assert!(refused.contains("running turn"), "{refused}");
    drop(app);
    fs::remove_dir_all(&root).ok();
}
