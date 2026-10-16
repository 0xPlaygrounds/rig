use std::fs;
use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, MemoryStore, app_in, app_on, calls, collect,
    commit_message, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use rig_telemetry::EffectLogPlugin;

use super::*;

const SECRET: &str = "sk-test-123abc";

/// A `read` that finds the leaked `.env`.
fn read(called: On<ToolCalled<Value>>, mut commands: Commands) {
    let found = format!("OPENAI_KEY={SECRET}\nDEBUG=1\n");
    let output = Outcome::from(ToolResult::success(found.into()));
    commands.entity(called.call).insert_if_new(output);
}

/// A `shell` that says it ran.
fn shell(called: On<ToolCalled<Value>>, mut commands: Commands) {
    let output = Outcome::from(ToolResult::success("ok".into()));
    commands.entity(called.call).insert_if_new(output);
}

/// The kernel, the effect log, the plugin and a front answering
/// `answers`, on session `id` of `home`, the agent answered by `model`.
fn start(
    home: &Home,
    id: &SessionId,
    model: &MockCompletionModel,
    answers: Vec<Answer>,
) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, EffectLogPlugin))
        .add_plugins((RetroactiveRedactPlugin, AnswerAsks))
        .insert_resource(Answers::new(answers))
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<Redacted>(&mut app).finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    Some((app, wakes, agent))
}

/// Sends `text` to `agent` as the user and runs frames until `done`.
fn send(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    text: &str,
    done: impl Fn(&World) -> bool,
) {
    let line = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(line);
    run_until(app, wakes, done);
}

fn turns_ended(n: usize) -> impl Fn(&World) -> bool {
    move |world| world.resource::<Ended>().0.len() >= n
}

/// Every file of the session's directory, as bytes, by path.
fn files_of(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut found = Vec::new();
    let mut todo = vec![dir.to_owned()];
    while let Some(dir) = todo.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                todo.push(path);
            } else if let Ok(bytes) = fs::read(&path) {
                found.push((path.display().to_string(), bytes));
            }
        }
    }
    found
}

fn holds_secret(bytes: &[u8]) -> bool {
    bytes
        .windows(SECRET.len())
        .any(|window| window == SECRET.as_bytes())
}

/// What `agent`'s conversation says, as one string.
fn conversation_text(app: &App, agent: Entity) -> String {
    let conversation = app.world().get::<Conversation>(agent);
    conversation
        .map(|conversation| format!("{:?}", conversation.messages()))
        .unwrap_or_default()
}

#[test]
fn a_redacted_secret_is_gone_from_the_session_and_the_next_request() {
    let home = scratch_home("oc-retroactive-redact");
    let id = SessionId::generate();
    let shell_call = MockStreamEvent::tool_call(
        "c2",
        "shell",
        serde_json::json!({ "command": format!("export OPENAI_KEY={SECRET}") }),
    );
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "read", serde_json::json!({ "path": ".env" })),
        vec![
            MockStreamEvent::text(format!("Found {SECRET}; exporting it.")),
            shell_call,
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply(&format!("Your key is {SECRET}.")),
        reply("Noted."),
    ]);
    let answers = vec![Answer::choice("redact")];
    let started = start(&home, &id, &model, answers);
    assert!(started.is_some(), "an agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };

    // Turn 1 leaks the key three ways.
    send(&mut app, &wakes, agent, "Show me .env", turns_ended(1));
    assert!(conversation_text(&app, agent).contains(SECRET));
    // Before: the agent's log and the effect log leak.
    app.update();
    let leaked = files_of(home.session(&id).path());
    let leaked: Vec<&String> = leaked
        .iter()
        .filter(|(_, bytes)| holds_secret(bytes))
        .map(|(path, _)| path)
        .collect();
    assert_eq!(leaked.len(), 2, "{leaked:?}");

    // `/redact`, Redact.
    send(
        &mut app,
        &wakes,
        agent,
        &format!("/redact {SECRET}"),
        |world| !world.resource::<Collected<Redacted>>().0.is_empty(),
    );
    let done = app
        .world()
        .resource::<Collected<Redacted>>()
        .0
        .first()
        .cloned();
    let asked = app.world().resource::<Answers>().asked.clone();
    assert_eq!(asked, [PREVIEW]);
    let done = done.unwrap_or_default();
    assert_eq!(done.placeholders, ["[REDACTED-1]"]);
    assert!(done.records > 0, "{done:?}");

    // The next request is redacted and says so.
    send(&mut app, &wakes, agent, "Thanks", turns_ended(2));
    let requests = model.requests();
    let last = format!("{:?}", requests.last());
    assert!(last.contains("[REDACTED-1]"), "{last}");
    assert!(!last.contains(SECRET), "{last}");
    assert!(last.contains(NOTE), "{last}");

    // `/redact --check` finds nothing saved.
    let line = Deliver::new(
        agent,
        format!("/redact --check {SECRET}"),
        DeliveryMode::Queue,
    );
    app.world_mut().trigger(line);
    run_until(&mut app, &wakes, |world| {
        let notices = &world.resource::<Collected<Notice>>().0;
        notices
            .iter()
            .any(|notice| notice.text.contains("not found"))
    });
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("not found"))
    );
    for _ in 0..3 {
        app.update();
    }
    drop(app);

    // No byte of the session holds the secret.
    for (path, bytes) in files_of(home.session(&id).path()) {
        assert!(!holds_secret(&bytes), "{path} still holds the secret");
    }

    // It loads and resumes, with the placeholder.
    let model = MockCompletionModel::from_stream_turns([reply("Resumed.")]);
    let started = start(&home, &id, &model, Vec::new());
    assert!(started.is_some(), "the session restores an agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    assert!(conversation_text(&app, agent).contains("[REDACTED-1]"));
    let before = app.world().resource::<Ended>().0.len();
    send(&mut app, &wakes, agent, "Go on", turns_ended(before + 1));
    let sent = format!("{:?}", model.requests().last());
    assert!(
        sent.contains("[REDACTED-1]") && !sent.contains(SECRET),
        "{sent}"
    );
    assert!(conversation_text(&app, agent).contains("Resumed."));
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn redact_alone_proposes_the_env_files_key() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        RetroactiveRedactPlugin,
        AnswerAsks,
    ))
    .finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let leaked = Message::user(format!("cat .env\nOPENAI_KEY={SECRET}\nDEBUG=1"));
    assert!(commit_message(app.world_mut(), agent, leaked).is_ok());
    app.update();

    let line = Deliver::new(agent, "/redact", DeliveryMode::Queue);
    app.world_mut().trigger(line);
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<RedactCandidates>())
    });
    let world = app.world_mut();
    let picker = world
        .query::<(Entity, &Picker, &RedactCandidates)>()
        .iter(world)
        .next()
        .map(|(entity, picker, found)| (entity, picker.clone(), found.0.clone()));
    assert!(picker.is_some(), "a picker");
    let Some((picker, shown, found)) = picker else {
        return;
    };
    let key = found.iter().position(|candidate| candidate.value == SECRET);
    assert!(key.is_some(), "{found:?}");
    let Some(key) = key else {
        return;
    };
    let label = shown
        .items
        .get(key)
        .map(|item| item.label.clone())
        .unwrap_or_default();
    assert!(
        label.contains("OPENAI_KEY") && !label.contains(SECRET),
        "{label}"
    );

    // Picking it shows the preview.
    app.world_mut().trigger(Picked {
        entity: picker,
        chosen: Some(Chosen::item(key)),
    });
    run_until(&mut app, &wakes, |world| {
        world
            .resource::<Answers>()
            .asked
            .iter()
            .any(|asked| asked == PREVIEW)
    });
    assert_eq!(app.world().resource::<Answers>().asked, [PREVIEW]);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RetroactiveRedactPlugin);
}
