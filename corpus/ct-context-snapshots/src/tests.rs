use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_coding_tools::ReadTool;
use rig_core::completion::Message;
use rig_core::message::UserContent;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{
    Collected, Ended, calls, collect, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models,
};
use rig_models::DefaultsPlugin;
use serde_json::json;

use super::*;

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ContextSnapshotsPlugin);
}

/// The sessions each app asked to switch to.
#[derive(Resource, Default)]
struct Switches(Vec<SessionId>);

fn switched(exit: On<Exit>, mut switches: ResMut<Switches>) {
    if let ExitReason::Switch(Some(id)) = &exit.0 {
        switches.0.push(id.clone());
    }
}

/// The kernel, the read tool, the defaults and the plugin on session `id` of `home`,
/// working in `root`, with `model` as `vendor/name`, after its first frame.
fn start(
    home: &Home,
    id: &SessionId,
    root: &Path,
    model: (&str, &MockCompletionModel),
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = rig_harness_test_support::app_in(home, id);
    app.insert_resource(scripted_models([model]))
        .insert_resource(ProjectRoot(root.to_owned()))
        .add_plugins((AgentPlugin, JournalPlugin, ReadTool, DefaultsPlugin))
        .add_plugins(ContextSnapshotsPlugin)
        .init_resource::<Switches>()
        .add_observer(switched)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// Types `line` for `agent` and runs until `done`.
fn type_line(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    line: &str,
    done: impl Fn(&World) -> bool,
) {
    let typed = Deliver::new(agent, line, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    run_until(app, wakes, done);
}

fn turns_ended(n: usize) -> impl Fn(&World) -> bool {
    move |world| world.resource::<Ended>().0.len() >= n
}

fn notice_with(text: &'static str) -> impl Fn(&World) -> bool {
    move |world| {
        let notices = &world.resource::<Collected<Notice>>().0;
        notices
            .iter()
            .any(|notice| format!("{notice:?}").contains(text))
    }
}

fn switches(n: usize) -> impl Fn(&World) -> bool {
    move |world| world.resource::<Switches>().0.len() >= n
}

/// The texts the user sent in `request`, attachments included, in order.
fn user_texts(request: &rig_core::completion::CompletionRequest) -> Vec<String> {
    let mut texts = Vec::new();
    for message in &request.chat_history {
        if let Message::User { content } = message {
            for item in content.iter() {
                if let UserContent::Text(text) = item {
                    texts.push(text.text.clone());
                }
            }
        }
    }
    texts
}

/// The main agent's turns in the snapshot `name` of `home`, from its listing.
fn snapshot_turns(app: &App, home: &Home, name: &str) -> Option<usize> {
    let world = app.world();
    let (stores, root) = (world.get_resource()?, world.get_resource()?);
    let all = snapshots(home, stores, root);
    all.into_iter()
        .find(|(_, it)| it.name == name)
        .map(|(_, it)| it.turns)
}

#[test]
fn a_snapshot_starts_new_sessions_and_says_what_changed() {
    let home = scratch_home("ct-context-snapshots-home");
    let root = home.root().join("project");
    assert!(fs::create_dir_all(root.join("src")).is_ok());
    let root: PathBuf = fs::canonicalize(&root).unwrap_or(root);
    assert!(fs::write(root.join("src/render.rs"), "fn render() {}\n").is_ok());
    assert!(fs::write(root.join("src/scene.rs"), "struct Scene;\n").is_ok());

    // Three turns that read the renderer's files, then `/snapshot save renderer`.
    let warm = MockCompletionModel::from_stream_turns([
        calls("c1", "read", json!({ "path": "src/render.rs" })),
        reply("render graph"),
        calls("c2", "read", json!({ "path": "src/scene.rs" })),
        reply("scene graph"),
        reply("architecture"),
    ]);
    let first = SessionId::generate();
    let (mut app, wakes, agent) = start(&home, &first, &root, ("test/model", &warm));
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
    });
    for (n, text) in (1..).zip(["read render", "read scene", "explain"]) {
        type_line(&mut app, &wakes, agent, text, turns_ended(n));
    }
    type_line(
        &mut app,
        &wakes,
        agent,
        "/snapshot save renderer",
        notice_with("Saved snapshot"),
    );
    assert_eq!(snapshot_turns(&app, &home, "renderer"), Some(3));
    type_line(
        &mut app,
        &wakes,
        agent,
        "/snapshot save early --at 1",
        notice_with("1 turns"),
    );
    assert_eq!(snapshot_turns(&app, &home, "early"), Some(1));
    type_line(
        &mut app,
        &wakes,
        agent,
        "/snapshot save team --shared",
        notice_with("Shared snapshot"),
    );
    assert!(root.join(".rig/snapshots/team").is_dir());

    // `/new --from renderer` forks it into a new session and switches there.
    type_line(&mut app, &wakes, agent, "/new --from renderer", switches(1));
    let second = app.world().resource::<Switches>().0.first().cloned();
    drop(app);
    assert!(second.is_some(), "no switch");
    let Some(second) = second else {
        return;
    };
    let next = MockCompletionModel::from_stream_turns([reply("bloom pass added")]);
    let (mut app, wakes, agent) = start(&home, &second, &root, ("test/model", &next));
    type_line(&mut app, &wakes, agent, "add a bloom pass", turns_ended(1));
    let requests = next.requests();
    let sent = requests.first().map(user_texts).unwrap_or_default();
    assert_eq!(
        sent,
        ["read render", "read scene", "explain", "add a bloom pass"]
    );
    assert_eq!(snapshot_turns(&app, &home, "renderer"), Some(3));

    // A file the snapshot read changed: the next session's first request says so.
    assert!(fs::write(root.join("src/scene.rs"), "struct Scene { lights: u8 }\n").is_ok());
    type_line(&mut app, &wakes, agent, "/new --from renderer", switches(1));
    let third = app.world().resource::<Switches>().0.first().cloned();
    drop(app);
    assert!(third.is_some(), "no switch");
    let Some(third) = third else {
        return;
    };
    let after = MockCompletionModel::from_stream_turns([reply("noted")]);
    let (mut app, wakes, agent) = start(&home, &third, &root, ("test/model", &after));
    type_line(&mut app, &wakes, agent, "go on", turns_ended(1));
    let requests = after.requests();
    let sent = requests.first().map(user_texts).unwrap_or_default();
    let note = sent
        .iter()
        .find(|text| text.contains("changed since this context"));
    assert!(
        note.is_some_and(|note| note.contains("scene.rs") && !note.contains("render.rs")),
        "{sent:?}"
    );

    // On a build without the snapshot's model, the session starts on the default.
    drop(app);
    let defaults = json!({ "model": "test/other" }).to_string();
    assert!(fs::write(home.defaults(), defaults).is_ok());
    let other = MockCompletionModel::from_stream_turns([reply("on the default")]);
    let (mut app, wakes, agent) = start(&home, &third, &root, ("test/other", &other));
    type_line(&mut app, &wakes, agent, "/new --from early", switches(1));
    let fourth = app.world().resource::<Switches>().0.first().cloned();
    drop(app);
    assert!(fourth.is_some(), "no switch");
    let Some(fourth) = fourth else {
        return;
    };
    let (app, _, agent) = start(&home, &fourth, &root, ("test/other", &other));
    let model = app
        .world()
        .get::<ModelChoice>(agent)
        .map(|model| model.0.clone());
    assert_eq!(model.as_deref(), Some("test/other"));
    assert!(notice_with("unavailable")(app.world()));
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}
