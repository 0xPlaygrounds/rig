use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use rig_coding_tools::{EditTool, ReadTool, ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::clock::ClockPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    ToolOutputs, app_on, calls, first_agent, log_tool_outputs, reply, run_until, scripted_models,
};
use serde_json::json;

use super::*;

const QUESTION: &str = "HashMap or BTreeMap for the cache?";
const CACHE: &str = "pub struct Cache { entries: Vec<(String, String)> }\n";

/// The models whose calls a test holds, as a model that never answers.
#[derive(Resource, Default)]
struct Hold(Vec<String>);

/// On a call [`hold_calls`] held.
#[derive(Component)]
struct Held;

/// Holds each model call of an agent on a model in [`Hold`].
fn hold_calls(
    calls: Query<(Entity, &ForAgent), (Undecided, ModelCalls, Without<Held>)>,
    agents: Query<&ModelChoice>,
    hold: Res<Hold>,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent)) in &calls {
        if agents
            .get(agent)
            .is_ok_and(|model| hold.0.contains(&model.0))
        {
            commands.entity(call).insert(Held);
            commands.spawn((Name::new("never answers"), HoldOf(call)));
        }
    }
}

/// The scripted models.
struct Models3 {
    main: MockCompletionModel,
    a: MockCompletionModel,
    b: MockCompletionModel,
}

/// An app whose agent consults `scripted-a` and `scripted-b` within
/// `timeout` seconds, holding the calls of the models in `hold`, in a
/// project with `src/cache.rs`; its agent asked to consult.
fn consulting(
    name: &str,
    models: &Models3,
    timeout: u64,
    hold: &[&str],
) -> (App, Receiver<()>, Entity) {
    let dir: PathBuf = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name(name);
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(dir.join("src")).is_ok());
    assert!(std::fs::write(dir.join("src/cache.rs"), CACHE).is_ok());
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);

    let settings = format!(
        "[consult]\nmodels = [\"test/scripted-a\", \"test/scripted-b\"]\ntimeout = {timeout}\n"
    );
    let served = scripted_models([
        ("test/main", &models.main),
        ("test/scripted-a", &models.a),
        ("test/scripted-b", &models.b),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(served)
        .insert_resource(Settings::from_toml(&settings))
        .insert_resource(ProjectRoot(dir))
        .insert_resource(Hold(hold.iter().map(|&model| model.to_owned()).collect()))
        .add_plugins((AgentPlugin, JournalPlugin, ClockPlugin))
        .add_plugins((ReadTool, EditTool, WriteTool, ShellTool, ConsultPlugin))
        .add_systems(AgentPass, hold_calls.in_set(Stage::Gate))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    let ask = "Make the cache faster.";
    app.world_mut()
        .trigger(Deliver::new(agent, ask, DeliveryMode::Steer));
    (app, wakes, agent)
}

/// The main agent's script: `consult`, then an answer.
fn main_model() -> MockCompletionModel {
    let args = json!({ "question": QUESTION, "paths": ["src/cache.rs"] });
    MockCompletionModel::from_stream_turns([calls("m1", CONSULT, args), reply("done")])
}

/// What request `at` of `model` carried.
fn asked(model: &MockCompletionModel, at: usize) -> String {
    format!("{:?}", model.requests().get(at))
}

/// The tools request `at` of `model` offered.
fn offered(model: &MockCompletionModel, at: usize) -> Vec<String> {
    let request = model.requests().into_iter().nth(at);
    let tools = request.map(|request| request.tools).unwrap_or_default();
    tools
        .into_iter()
        .map(|tool| tool.name.to_string())
        .collect()
}

#[test]
fn consultants_read_only_answer_side_by_side_and_a_failure_shows() {
    let models = Models3 {
        main: main_model(),
        a: MockCompletionModel::from_stream_turns([
            calls("a1", "read", json!({ "path": "src/cache.rs" })),
            reply("Use a BTreeMap"),
        ]),
        b: MockCompletionModel::from_stream_turns([vec![MockStreamEvent::error("overloaded")]]),
    };
    let (mut app, wakes, _) = consulting("sn-consult-files", &models, 120, &[]);
    run_until(&mut app, &wakes, |_| models.main.request_count() >= 2);

    for consultant in [&models.a, &models.b] {
        let first = asked(consultant, 0);
        assert!(first.contains(QUESTION), "{first}");
        assert!(first.contains("pub struct Cache"), "{first}");
        let tools = offered(consultant, 0);
        assert!(tools.contains(&"read".to_owned()), "{tools:?}");
        for writer in ["shell", "edit", "write", CONSULT] {
            assert!(!tools.contains(&writer.to_owned()), "{tools:?}");
        }
    }
    let read = app.world().resource::<ToolOutputs>().0.get("a1").cloned();
    assert!(read.unwrap_or_default().contains("pub struct Cache"));
    let result = asked(&models.main, 1);
    assert!(result.contains("scripted-a"), "{result}");
    assert!(result.contains("Use a BTreeMap"), "{result}");
    assert!(result.contains("scripted-b: failed"), "{result}");
}

#[test]
fn a_consultant_that_never_answers_times_out() {
    let models = Models3 {
        main: main_model(),
        a: MockCompletionModel::from_stream_turns([reply("Use a BTreeMap")]),
        b: MockCompletionModel::from_stream_turns([reply("unused")]),
    };
    let started = Instant::now();
    let (mut app, wakes, _) = consulting("sn-consult-timeout", &models, 1, &["test/scripted-b"]);
    run_until(&mut app, &wakes, |_| models.main.request_count() >= 2);
    let waited = started.elapsed();

    let result = asked(&models.main, 1);
    assert!(result.contains("Use a BTreeMap"), "{result}");
    assert!(result.contains("scripted-b: failed (timeout)"), "{result}");
    assert!(waited >= Duration::from_millis(900), "{waited:?}");
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    assert_eq!(models.b.request_count(), 0);
}

#[test]
fn interrupting_the_agent_stops_its_consultants() {
    let models = Models3 {
        main: main_model(),
        a: MockCompletionModel::from_stream_turns([reply("unused")]),
        b: MockCompletionModel::from_stream_turns([reply("unused")]),
    };
    let both = ["test/scripted-a", "test/scripted-b"];
    let (mut app, wakes, agent) = consulting("sn-consult-interrupt", &models, 120, &both);
    let held = |world: &World| {
        world
            .try_query_filtered::<(), With<Held>>()
            .is_some_and(|mut held| held.iter(world).count() == 2)
    };
    run_until(&mut app, &wakes, held);
    assert!(held(app.world()));

    app.world_mut().trigger(Interrupt::new(agent));
    app.world_mut().resource_mut::<Hold>().0.clear();
    let gone = |world: &World| {
        world
            .try_query_filtered::<(), Or<(With<Consultant>, With<Consultation>)>>()
            .is_some_and(|mut left| left.iter(world).next().is_none())
    };
    run_until(&mut app, &wakes, gone);
    for _ in 0..5 {
        app.update();
    }
    assert!(
        gone(app.world()),
        "the consultants and the consult are gone"
    );
    assert_eq!(models.a.request_count(), 0);
    assert_eq!(models.b.request_count(), 0);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ConsultPlugin);
}
