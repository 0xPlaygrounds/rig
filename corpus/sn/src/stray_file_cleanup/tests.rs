use std::fs;
use std::path::{Path, PathBuf};

use rig_basics::coding_tools::{EditTool, ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::commands::CommandPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, collect, first_agent, log_ended, reply, run_until, scripted_models,
};

use super::*;

const MODEL: &str = "corpus-scripted/agent";

/// A repository with a committed `src/lib.rs` and the user's untracked
/// `notes.txt`, in a directory beside the test binary.
fn repository(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("sn-stray-{name}"));
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(dir.join("src")).ok();
    fs::write(dir.join("src/lib.rs"), "// one\n").ok();
    for args in [
        &["init", "-q"][..],
        &["config", "user.email", "test@example.com"],
        &["config", "user.name", "Test"],
        &["config", "commit.gpgsign", "false"],
        &["add", "src/lib.rs"],
        &["commit", "-q", "-m", "Start"],
    ] {
        let mut git = Processes::default().launcher().command("git");
        git.arg("-C").arg(&dir).args(args).output().ok();
    }
    fs::write(dir.join("notes.txt"), "the user's own notes\n").ok();
    fs::canonicalize(&dir).unwrap_or(dir)
}

/// A reply that writes `scratch.py`, edits `src/lib.rs` and makes
/// `out.log` with `shell`.
fn makes_files() -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(
            "w",
            "write",
            serde_json::json!({ "path": "scratch.py", "content": "print('hi')\n" }),
        ),
        MockStreamEvent::tool_call(
            "e",
            "edit",
            serde_json::json!({
                "path": "src/lib.rs",
                "edits": [{ "old_text": "one", "new_text": "two" }],
            }),
        ),
        MockStreamEvent::tool_call(
            "s",
            "shell",
            serde_json::json!({ "command": "echo hi > out.log" }),
        ),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// An app on `dir` whose agent ran one turn of `model`, with a front that
/// answers when `answered`.
fn after_a_turn(
    dir: &Path,
    model: &MockCompletionModel,
    answered: bool,
) -> (App, std::sync::mpsc::Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([(MODEL, model)]))
        .insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((AgentPlugin, SessionLogPlugin, CommandPlugin, PickPlugin))
        .add_plugins((WriteTool, EditTool, ShellTool, StrayFilesPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    if answered {
        app.world_mut().spawn(Answerer::default());
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    world.trigger(SetModel {
        entity: agent,
        model: MODEL.to_owned(),
    });
    world.trigger(Deliver::new(agent, "Find the bug.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    (app, wakes, agent)
}

/// Every open picker.
fn pickers(world: &World) -> Vec<(Entity, Picker)> {
    let Some(mut query) = world.try_query::<(Entity, &Picker)>() else {
        return Vec::new();
    };
    query
        .iter(world)
        .map(|(entity, picker)| (entity, picker.clone()))
        .collect()
}

/// The open picker titled `title`.
fn picker(world: &World, title: &str) -> Option<(Entity, Picker)> {
    pickers(world)
        .into_iter()
        .find(|(_, picker)| picker.title == title)
}

/// The open picker carrying `T`.
fn picker_with<T: Component>(world: &World) -> Option<(Entity, Picker)> {
    pickers(world)
        .into_iter()
        .find(|(entity, _)| world.get::<T>(*entity).is_some())
}

/// Answers the open picker carrying `T` with `chosen`.
fn answer<T: Component>(app: &mut App, chosen: Chosen) {
    let open = picker_with::<T>(app.world()).map(|(entity, _)| entity);
    let entity = open.unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(Pick::person(entity, Some(chosen)));
    app.update();
}

/// The file `name` in some batch of the trash.
fn trashed(dir: &Path, name: &str) -> Option<PathBuf> {
    let batches = fs::read_dir(dir.join(TRASH)).ok()?;
    batches
        .flatten()
        .map(|batch| batch.path().join(name))
        .find(|path| path.is_file())
}

#[test]
fn the_files_a_turn_created_are_offered_deleted_and_restored() {
    let dir = repository("picker");
    let model = MockCompletionModel::from_stream_turns([
        makes_files(),
        reply("Found it."),
        reply("Noted."),
    ]);
    let (mut app, wakes, agent) = after_a_turn(&dir, &model, true);

    let offer = picker_with::<StrayPicker>(app.world());
    let files: Vec<(String, bool)> = offer
        .iter()
        .filter(|(_, picker)| picker.multiple && picker.title == TITLE)
        .flat_map(|(_, picker)| &picker.items)
        .map(|item| (item.label.clone(), item.checked))
        .collect();
    let expected = [
        ("out.log".to_owned(), true),
        ("scratch.py".to_owned(), true),
    ];
    assert_eq!(files, expected, "only the agent's new files, preselected");
    assert_eq!(
        fs::read_to_string(dir.join("src/lib.rs")).ok().as_deref(),
        Some("// two\n")
    );

    // Tick scratch.py only, then delete it.
    answer::<StrayPicker>(&mut app, Chosen::items([1]));
    answer::<StrayActions>(&mut app, Chosen::item(0));
    let told = |world: &World| {
        world
            .get::<Strays>(agent)
            .is_some_and(|strays| !strays.deleted.is_empty())
    };
    run_until(&mut app, &wakes, told);
    assert!(!dir.join("scratch.py").exists(), "deleted from the project");
    let copy = trashed(&dir, "scratch.py").and_then(|path| fs::read_to_string(path).ok());
    assert_eq!(copy.as_deref(), Some("print('hi')\n"), "kept in the trash");
    assert!(dir.join("out.log").exists(), "the unselected file stays");

    app.world_mut()
        .trigger(Deliver::new(agent, "Thanks.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > 1
    });
    let last = model
        .requests()
        .last()
        .map(|request| format!("{request:?}"));
    let last = last.unwrap_or_default();
    assert!(
        last.contains("deleted these files you created") && last.contains("scratch.py"),
        "{last}"
    );

    app.world_mut()
        .trigger(Deliver::new(agent, "/restore-strays", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        picker(world, RESTORE_TITLE).is_some()
    });
    let batch = picker(app.world(), RESTORE_TITLE).map(|(entity, _)| entity);
    app.world_mut().trigger(Pick::person(
        batch.unwrap_or(Entity::PLACEHOLDER),
        Some(Chosen::item(0)),
    ));
    run_until(&mut app, &wakes, |_| dir.join("scratch.py").exists());
    let back = fs::read_to_string(dir.join("scratch.py")).ok();
    assert_eq!(back.as_deref(), Some("print('hi')\n"), "restored");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_run_nobody_answers_deletes_nothing_and_lists_the_files() {
    let dir = repository("headless");
    let model = MockCompletionModel::from_stream_turns([makes_files(), reply("Found it.")]);
    let (app, _wakes, _) = after_a_turn(&dir, &model, false);
    assert!(pickers(app.world()).is_empty(), "nobody to ask");
    let notices = app.world().resource::<Collected<Notice>>();
    let printed: Vec<&str> = notices.0.iter().map(|notice| notice.text.as_str()).collect();
    let printed = printed.join("\n");
    assert!(
        printed.contains("out.log")
            && printed.contains("scratch.py")
            && !printed.contains("notes.txt"),
        "{printed}"
    );
    assert!(dir.join("scratch.py").exists() && dir.join("out.log").exists());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(StrayFilesPlugin);
}
