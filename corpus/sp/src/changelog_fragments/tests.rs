use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use rig_basics::coding_tools::WriteTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::commands::CommandPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Collected, Ended, app_in, calls, collect, first_agent, log_ended, reply, run_until,
    scratch_home, scripted_models, tui_view,
};
use rig_tui::TuiTestScreen;
use serde_json::json;

use super::*;

/// Whether the test holds each drafting call before it starts.
#[derive(Resource)]
struct HoldDrafts(bool);

/// The test's hold on a drafting call, as a slow model.
#[derive(Component)]
struct SlowModel;

fn hold_drafts(
    drafts: Query<Entity, (With<Drafting>, ReadyToStart)>,
    holding: Res<HoldDrafts>,
    mut seen: Local<HashSet<Entity>>,
    mut commands: Commands,
) {
    for call in &drafts {
        if holding.0 && seen.insert(call) {
            commands.spawn((SlowModel, HoldOf(call)));
        }
    }
}

fn held(world: &World) -> bool {
    world
        .try_query_filtered::<(), With<SlowModel>>()
        .is_some_and(|mut holds| holds.iter(world).next().is_some())
}

fn release(app: &mut App) {
    app.insert_resource(HoldDrafts(false));
    let mut holds = app.world_mut().query_filtered::<Entity, With<SlowModel>>();
    let holds: Vec<Entity> = holds.iter(app.world()).collect();
    for hold in holds {
        app.world_mut().despawn(hold);
    }
}

fn ended(world: &World) -> usize {
    world.resource::<Ended>().0.len()
}

/// The fragment files in `dir`.
fn fragment_files(dir: &Path) -> Vec<PathBuf> {
    let entries = fs::read_dir(dir).into_iter().flatten().flatten();
    // A fragment being written is a hidden `.tmp` file until it is renamed.
    let written = |path: &PathBuf| !name(path).starts_with('.');
    let mut files: Vec<PathBuf> = entries.map(|entry| entry.path()).filter(written).collect();
    files.sort();
    files
}

/// How many notices said `text`.
fn told(world: &World, text: &str) -> usize {
    let notices = &world.resource::<Collected<Notice>>().0;
    notices
        .iter()
        .filter(|notice| notice.text.contains(text))
        .count()
}

fn screen(world: &World) -> String {
    world.resource::<TuiTestScreen>().rows().join("\n")
}

fn write(id: &str, path: &str, content: &str) -> Vec<MockStreamEvent> {
    calls(id, "write", json!({ "path": path, "content": content }))
}

#[test]
fn source_turns_get_fragments_in_the_background() {
    let main = MockCompletionModel::from_stream_turns([
        write("1", "src/parse.rs", "pub fn parse() {}\n"),
        reply("Fixed the parser."),
        write("2", "tests/parse.rs", "#[test] fn empty() {}\n"),
        reply("Added a test."),
        write("3", "src/lib.rs", "pub mod parse;\n"),
        reply("Tidied."),
        write("4", "src/extra.rs", "pub fn extra() {}\n"),
        reply("Added extra."),
    ]);
    let drafter = MockCompletionModel::from_stream_turns([
        reply("fixed: Parser no longer panics on empty input"),
        vec![MockStreamEvent::error("the drafting model is overloaded")],
        reply("added: An extra module"),
    ]);
    let project = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("sp-changelog-fragments-project");
    fs::remove_dir_all(&project).ok();
    assert!(fs::create_dir_all(project.join("src")).is_ok());
    assert!(fs::write(project.join("src/parse.rs"), "old\n").is_ok());
    let changes = project.join("changes");

    let home = scratch_home("sp-changelog-fragments");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(Settings::from_toml("[changelog]\nmodel = \"test/drafter\""))
        .insert_resource(scripted_models([
            ("test/main", &main),
            ("test/drafter", &drafter),
        ]))
        .insert_resource(ProjectRoot(project.clone()))
        .insert_resource(HoldDrafts(true))
        .insert_resource(TuiTestScreen::new(100, 40))
        .add_plugins((AgentPlugin, SessionLogPlugin, CommandPlugin, WriteTool))
        .add_plugins(ChangelogFragmentsPlugin)
        .add_systems(AgentPass, hold_drafts.in_set(Stage::Gate))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    tui_view(&mut app);
    collect::<Notice>(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    // The terminal view shows the main agent, as its front would.
    app.insert_resource(ShownAgent(Some(agent)));
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    let say = |app: &mut App, text: &str| {
        let message = Deliver::new(agent, text, DeliveryMode::Steer);
        app.world_mut().trigger(message);
    };

    // Turn 1 changes source: its draft is held, as a slow model would.
    say(&mut app, "Fix the panic on empty input");
    run_until(&mut app, &wakes, |world| held(world) && ended(world) == 1);
    assert!(held(app.world()));
    // Turn 2 starts and ends while the draft waits.
    say(&mut app, "Add a test for it");
    run_until(&mut app, &wakes, |world| ended(world) == 2);
    assert!(held(app.world()), "the draft still waits");
    assert_eq!(main.request_count(), 4);
    assert_eq!(drafter.request_count(), 0);
    assert!(fragment_files(&changes).is_empty());

    release(&mut app);
    run_until(&mut app, &wakes, |_| !fragment_files(&changes).is_empty());
    let files = fragment_files(&changes);
    assert_eq!(files.len(), 1, "{files:?}");
    let first = files.first().cloned().unwrap_or_default();
    assert!(name(&first).ends_with("-1.fixed.md"), "{first:?}");
    let text = fs::read_to_string(&first).unwrap_or_default();
    assert_eq!(text, "Parser no longer panics on empty input");
    let asked = format!("{:?}", drafter.requests().first());
    assert!(asked.contains("Fix the panic on empty input"), "{asked}");
    assert!(asked.contains("+pub fn parse() {}"), "{asked}");

    // Turn 3's draft fails: nothing written, one notice, no retry.
    say(&mut app, "Tidy the crate root");
    let failed = "drafting turn 3's fragment failed";
    run_until(&mut app, &wakes, |world| told(world, failed) > 0);
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(told(app.world(), failed), 1);
    assert!(
        screen(app.world()).contains(failed),
        "{}",
        screen(app.world())
    );
    assert_eq!(drafter.request_count(), 2);
    assert_eq!(fragment_files(&changes).len(), 1);

    // Turn 4's change is undone before its draft is written: no fragment.
    app.insert_resource(HoldDrafts(true));
    say(&mut app, "Add an extra module");
    run_until(&mut app, &wakes, |world| held(world) && ended(world) == 4);
    assert!(fs::remove_file(project.join("src/extra.rs")).is_ok());
    release(&mut app);
    run_until(&mut app, &wakes, |world| told(world, "were undone") > 0);
    assert_eq!(told(app.world(), "were undone"), 1);
    assert_eq!(fragment_files(&changes).len(), 1);

    // The main model never read anything of the fragments.
    for request in main.requests() {
        let request = format!("{request:?}");
        assert!(!request.contains("Parser no longer panics"), "{request}");
        assert!(!request.contains("changelog"), "{request}");
    }

    say(&mut app, "/changelog");
    run_until(&mut app, &wakes, |world| {
        told(world, "Changelog fragments:") > 0
    });
    let shown = screen(app.world());
    assert!(shown.contains("Changelog fragments:"), "{shown}");
    assert!(shown.contains(name(&first)), "{shown}");

    say(&mut app, "/changelog drop 1");
    run_until(&mut app, &wakes, |_| !first.exists());
    assert!(!first.exists());
    assert_eq!(main.request_count(), 8, "commands never reach the model");
}
