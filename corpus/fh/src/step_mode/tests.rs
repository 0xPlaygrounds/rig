use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{ReadTool, ShellTool, WriteTool};
use rig_core::completion::{CompletionRequest, Message as Chat};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, answer_open, app_on, connect, first_agent, log_ended, reply, run_until,
};
use serde_json::json;

use super::*;

/// Each tool call that started, in order: its id and arguments.
#[derive(Resource, Default)]
struct Ran(Vec<(String, Value)>);

fn log_started(started: On<Add<Started>>, calls: Query<&ToolRequest>, mut ran: ResMut<Ran>) {
    if let Ok(request) = calls.get(started.entity) {
        let arguments = Value::Object(request.call.function.arguments.clone());
        ran.0.push((request.call.id.to_string(), arguments));
    }
}

/// An interactive app with the real coding tools and this plugin, and its
/// first agent, working in `dir` and answered by `model`, with step mode on.
fn start(
    store: &MemoryStore,
    model: &MockCompletionModel,
    dir: &Path,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(StepEditor(None))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ShellTool,
            WriteTool,
            ReadTool,
            StepModePlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<Ran>()
        .add_observer(log_started);
    // A front someone sits at; the test answers its questions.
    app.world_mut().spawn(Answerer::default());
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let mut entity = app.world_mut().entity_mut(agent);
    entity.insert(WorkDir(dir.to_path_buf()));
    if let Some(connection) = connect(model) {
        entity.insert(connection);
    }
    app.world_mut()
        .trigger(Deliver::new(agent, "/step", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.get::<Stepping>(agent).is_some()
    });
    (app, wakes, agent)
}

/// An empty directory under the test binary's target directory.
fn workspace(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("step-mode-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Whether a question titled `title` whose body says `says` is open.
fn asking_that(world: &World, title: &str, says: &str) -> bool {
    let Some(mut asks) = world.try_query_filtered::<(&Ask, &Decision), Without<Outcome>>() else {
        return false;
    };
    asks.iter(world).any(|(ask, decision)| {
        ask.title == title && ask.body.contains(says) && !decision.is_decided()
    })
}

/// Whether the question titled `title` is open.
fn asking(world: &World, title: &str) -> bool {
    asking_that(world, title, "")
}

/// Whether the call `id` started.
fn started(world: &World, id: &str) -> bool {
    world.resource::<Ran>().0.iter().any(|(ran, _)| ran == id)
}

/// The tool results a request sends, in order, as JSON text.
fn results(request: Option<&CompletionRequest>) -> Vec<String> {
    let messages = request.map(|request| request.chat_history.iter());
    messages
        .into_iter()
        .flatten()
        .filter_map(|message| match &**message {
            Chat::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|content| match content {
            UserContent::ToolResult(result) => serde_json::to_string(result).ok(),
            _ => None,
        })
        .collect()
}

#[test]
fn calls_are_stepped_one_at_a_time_skipped_and_edited() {
    let dir = workspace("edit");
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "shell", json!({ "command": "echo one" })),
            MockStreamEvent::tool_call("c2", "shell", json!({ "command": "echo two" })),
            MockStreamEvent::tool_call(
                "c3",
                "write",
                json!({ "path": "notes.txt", "content": "x" }),
            ),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Done."),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &model, &dir);
    let send = |app: &mut App, text: &str| {
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    };
    send(&mut app, "take notes");

    let first = "Step 1 of 3: shell";
    run_until(&mut app, &wakes, |world| asking(world, first));
    assert!(asking(app.world(), first));
    // Nothing runs beside the call being stepped.
    assert!(app.world().resource::<Ran>().0.is_empty());
    assert!(answer_open(app.world_mut(), first, &Answer::choice("run")));

    let second = "Step 2 of 3: shell";
    run_until(&mut app, &wakes, |world| asking(world, second));
    assert!(started(app.world(), "c1") && !started(app.world(), "c2"));
    assert!(answer_open(
        app.world_mut(),
        second,
        &Answer::choice("skip")
    ));

    let third = "Step 3 of 3: write";
    run_until(&mut app, &wakes, |world| asking(world, third));
    assert!(answer_open(app.world_mut(), third, &Answer::choice("edit")));
    let edit = "Edit the arguments of `write`";
    run_until(&mut app, &wakes, |world| asking(world, edit));
    let answer = |app: &mut App, text: &str| {
        let text = Answer::text(text);
        assert!(answer_open(app.world_mut(), edit, &text));
    };

    // Arguments that are not JSON are refused, and asked for again.
    answer(&mut app, "{ oops");
    run_until(&mut app, &wakes, |world| {
        asking_that(world, edit, "not JSON")
    });
    assert!(asking_that(app.world(), edit, "not JSON"));
    assert!(!started(app.world(), "c3"));

    answer(&mut app, r#"{ "path": "notes.txt", "content": "y" }"#);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let requests = model.requests();
    let sent = results(requests.get(1));
    assert_eq!(sent.len(), 3, "{sent:?}");
    let [one, skipped, wrote] = [0, 1, 2].map(|at| sent.get(at).cloned().unwrap_or_default());
    assert!(one.contains("one"), "{one}");
    assert!(skipped.contains(SKIPPED), "{skipped}");
    assert!(!wrote.contains("rror"), "{wrote}");
    assert_eq!(
        std::fs::read_to_string(dir.join("notes.txt"))
            .ok()
            .as_deref(),
        Some("y")
    );
    // The edited call ran once, with the edited arguments.
    let writes: Vec<&Value> = (app.world().resource::<Ran>().0.iter())
        .filter(|(id, _)| id == "c3")
        .map(|(_, arguments)| arguments)
        .collect();
    assert_eq!(writes, [&json!({ "path": "notes.txt", "content": "y" })]);
    let noted = app
        .world()
        .get::<Conversation>(agent)
        .is_some_and(|conversation| {
            (conversation.asides())
                .filter_map(|(_, aside)| Note::of(aside))
                .any(|note| note.title.contains(EDITED))
        });
    assert!(noted);
    // The saved conversation says the model asked for the edited call.
    let saved = Session::read(&store).ok().unwrap_or_default();
    let calls: Vec<ToolCall> = (saved.agents.iter())
        .flat_map(|agent| agent.conversation.messages().to_vec())
        .filter_map(|message| match &*message {
            Chat::Assistant(reply) => Some(reply.tool_calls().cloned().collect::<Vec<_>>()),
            _ => None,
        })
        .flatten()
        .collect();
    let saved_write = calls.iter().find(|call| call.id.to_string() == "c3");
    let content = saved_write.and_then(|call| call.function.arguments.get("content"));
    assert_eq!(content, Some(&json!("y")), "{calls:?}");
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Answered(_)]),
        "{ended:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn read_only_calls_of_a_reply_do_not_run_side_by_side() {
    let dir = workspace("reads");
    std::fs::write(dir.join("a.txt"), "alpha").ok();
    std::fs::write(dir.join("b.txt"), "beta").ok();
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("r1", "read", json!({ "path": "a.txt" })),
            MockStreamEvent::tool_call("r2", "read", json!({ "path": "b.txt" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Read."),
    ]);
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &model, &dir);
    app.world_mut()
        .trigger(Deliver::new(agent, "read both", DeliveryMode::Queue));

    let first = "Step 1 of 2: read";
    run_until(&mut app, &wakes, |world| asking(world, first));
    assert!(asking(app.world(), first));
    assert!(!started(app.world(), "r2"));
    assert!(answer_open(app.world_mut(), first, &Answer::choice("run")));
    let second = "Step 2 of 2: read";
    run_until(&mut app, &wakes, |world| asking(world, second));
    assert!(started(app.world(), "r1") && !started(app.world(), "r2"));
    assert!(answer_open(app.world_mut(), second, &Answer::choice("run")));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let sent = results(model.requests().get(1));
    assert!(
        sent.first().is_some_and(|read| read.contains("alpha")),
        "{sent:?}"
    );
    assert!(
        sent.get(1).is_some_and(|read| read.contains("beta")),
        "{sent:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(StepModePlugin);
}
