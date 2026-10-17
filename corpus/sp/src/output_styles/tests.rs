use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::pick::PickPlugin;
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{
    Collected, app_in, calls, collect, reply, run_until, scratch_home, scripted_models, tui_view,
};
use rig_subagents::{SubagentsPlugin, TASK};
use rig_tui::TuiTestScreen;
use serde_json::json;

use super::*;

const MARKER: &str = "PIRATE-MARKER";

/// The models every run of the test answers with: the main agent's and
/// its subagent's.
struct Scripts {
    main: MockCompletionModel,
    sub: MockCompletionModel,
}

/// The app on session `id` of `home`, working in `project`, with the
/// subagents, this plugin and the terminal view, after its first frame.
fn start(home: &Home, id: &SessionId, project: &Path, models: &Scripts) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    let served = [("test/main", &models.main), ("test/sub", &models.sub)];
    app.insert_resource(ProjectRoot(project.to_owned()))
        .insert_resource(scripted_models(served))
        .insert_resource(TuiTestScreen::new(100, 30))
        .add_plugins((AgentPlugin, SessionLogPlugin, PickPlugin, SubagentsPlugin));
    app.add_plugins(OutputStylesPlugin);
    collect::<Notice>(&mut app);
    tui_view(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    (app, wakes)
}

/// The agent the user talks to.
fn main_agent(app: &mut App) -> Entity {
    let world = app.world_mut();
    let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER)
}

/// No turn runs and no message waits.
fn idle(world: &World) -> bool {
    let busy = |entity: EntityRef| entity.contains::<ActiveTurn>() || entity.contains::<Incoming>();
    !world.iter_entities().any(busy)
}

/// Types `line` for `agent` and runs until everything it started is done.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    let typed = Deliver::new(agent, line, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    for _ in 0..3 {
        app.update();
        run_until(app, wakes, idle);
    }
}

/// How often the marker is in a request's system prompt.
fn markers(request: &CompletionRequest) -> usize {
    let system = request.system_instructions().unwrap_or_default();
    system.matches(MARKER).count()
}

/// How often the marker is in each of `model`'s requests from `from` on.
fn markers_since(model: &MockCompletionModel, from: usize) -> Vec<usize> {
    let requests = model.requests();
    requests.iter().skip(from).map(markers).collect()
}

fn screen(app: &App) -> String {
    app.world().resource::<TuiTestScreen>().rows().join("\n")
}

fn style_of(app: &App, agent: Entity) -> Option<String> {
    app.world()
        .get::<OutputStyle>(agent)
        .map(|style| style.0.clone())
}

/// The project, with `pirate.md` in its styles.
fn project(home: &Home) -> PathBuf {
    let project = home.root().join("project");
    let styles = project.join(".rig").join("styles");
    fs::create_dir_all(&styles).ok();
    let pirate = "---\ndescription: Talk like a pirate\n---\nPIRATE-MARKER\n";
    fs::write(styles.join("pirate.md"), pirate).ok();
    project
}

#[test]
fn a_style_is_in_each_request_once_stays_out_of_subagents_and_resumes() {
    let home = scratch_home("sp-output-styles");
    let project = project(&home);
    let task =
        json!({ "description": "look around", "prompt": "Check the readme.", "model": "test/sub" });
    let mut main = vec![reply("r1"), calls("t1", TASK, task)];
    main.extend((0..10).map(|_| reply("ok")));
    let models = Scripts {
        main: MockCompletionModel::from_stream_turns(main),
        sub: MockCompletionModel::from_stream_turns((0..3).map(|_| reply("sub done"))),
    };
    let id = SessionId::generate();
    let (mut app, wakes) = start(&home, &id, &project, &models);
    let agent = main_agent(&mut app);
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });

    // `/style` alone offers every style, the current one marked.
    say(&mut app, &wakes, agent, "/style");
    let world = app.world_mut();
    let picker = world.query::<(Entity, &Picker)>().iter(world).next();
    let picker = picker.map(|(entity, picker)| (entity, picker.clone()));
    let labels: Vec<String> = picker
        .iter()
        .flat_map(|(_, picker)| picker.items.iter().map(|item| item.label.clone()))
        .collect();
    assert_eq!(labels, ["● default", "  terse", "  teaching", "  pirate"]);
    if let Some((entity, _)) = picker {
        app.world_mut().entity_mut(entity).despawn();
    }

    // Two prompts in the pirate style; the second turn starts a subagent.
    say(&mut app, &wakes, agent, "/style pirate");
    say(&mut app, &wakes, agent, "one");
    assert!(screen(&app).contains("style: pirate"), "{}", screen(&app));
    say(&mut app, &wakes, agent, "two");
    let seen = markers_since(&models.main, 0);
    assert!(seen.len() >= 3 && seen.iter().all(|n| *n == 1), "{seen:?}");
    let sub = markers_since(&models.sub, 0);
    assert!(!sub.is_empty() && sub.iter().all(|n| *n == 0), "{sub:?}");

    // An unknown style is refused with the list; the style stays.
    say(&mut app, &wakes, agent, "/style nope");
    let shown = screen(&app);
    assert!(
        shown.contains("Styles: default, terse, teaching, pirate."),
        "{shown}"
    );
    let before = models.main.request_count();
    say(&mut app, &wakes, agent, "three");
    assert_eq!(markers_since(&models.main, before), [1]);

    // `/style default` takes the marker out of the next request.
    say(&mut app, &wakes, agent, "/style default");
    assert!(!screen(&app).contains("style: "), "{}", screen(&app));
    let before = models.main.request_count();
    say(&mut app, &wakes, agent, "four");
    assert_eq!(markers_since(&models.main, before), [0]);
    say(&mut app, &wakes, agent, "/style pirate");
    drop(app);

    // The session resumes as a pirate, and its next request says so.
    let (mut app, wakes) = start(&home, &id, &project, &models);
    let agent = main_agent(&mut app);
    assert_eq!(style_of(&app, agent).as_deref(), Some("pirate"));
    let before = models.main.request_count();
    say(&mut app, &wakes, agent, "five");
    assert_eq!(markers_since(&models.main, before), [1]);

    // Its style file gone, the agent falls back to `default`, said once.
    fs::remove_file(project.join(".rig/styles/pirate.md")).ok();
    run_until(&mut app, &wakes, |world| {
        world
            .get::<OutputStyle>(agent)
            .is_some_and(|style| style.0 == DEFAULT)
    });
    app.update();
    let gone = app.world().resource::<Collected<Notice>>().0.iter();
    let gone = gone
        .filter(|notice| notice.text.contains("is gone"))
        .count();
    assert_eq!((style_of(&app, agent).as_deref(), gone), (Some(DEFAULT), 1));
}
