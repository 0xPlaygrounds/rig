use rig_cassette::journal::MemoryStore;
use rig_core::catalog::Catalog;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;
use rig_ecs::effects::Handler;
use rig_harness_test_support::{app_on, calls, reply, run_until};

use super::*;

/// `model` as the catalog model `reference`.
fn connect(reference: &str, model: &MockCompletionModel) -> Option<Connection> {
    let spec = Catalog::builtin().resolve(reference).ok()?.shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    Some(Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    })
}

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

#[test]
fn a_plan_goes_to_the_strong_model_and_its_read_only_follow_up_to_the_cheap_one() {
    let cheap = MockCompletionModel::from_stream_turns([reply("It is a small crate.")]);
    let strong = MockCompletionModel::from_stream_turns([calls(
        "1",
        "read",
        serde_json::json!({ "path": "src/lib.rs" }),
    )]);
    let targets = [
        ("cheap", connect("deepseek/deepseek-flash", &cheap)),
        ("strong", connect("deepseek/deepseek-v4-pro", &strong)),
    ];
    let targets: Option<BTreeMap<String, Connection>> = targets
        .into_iter()
        .map(|(name, connection)| Some((name.to_owned(), connection?)))
        .collect();
    assert!(targets.is_some(), "the catalog lists both models");
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.add_plugins((AgentPlugin, JournalPlugin, rig_models::ModelsPlugin))
        .insert_resource(RouterTargets(targets.unwrap_or_default()))
        .add_plugins(ModelRouterPlugin)
        .add_open_tool(
            "read",
            "Reads a file.",
            ToolOptions {
                rules: &[],
                footprint: Some(Footprint::ReadOnly),
            },
            |called: On<ToolCalled<serde_json::Value>>, mut commands: Commands| {
                let output = ToolResult::success("pub fn answer() -> u32 { 42 }".into());
                commands.entity(called.call).insert(ToolOutput(output));
            },
        )
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    assert!(agent.is_some(), "the session has an agent");
    let Some(agent) = agent else {
        return;
    };

    // `/model router` is the wrapped `/model`.
    world.trigger(RunCommand {
        entity: agent,
        line: "model router".to_owned(),
    });
    app.update();
    assert_eq!(app.world().get::<Routing>(agent), Some(&Routing(true)));

    app.world_mut().trigger(Deliver::new(
        agent,
        "plan the refactor",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);

    assert_eq!(
        strong.request_count(),
        1,
        "the plan's request went to strong"
    );
    assert_eq!(cheap.request_count(), 1, "the follow-up went to cheap");
    let spent = app.world().get::<RoutedSpending>(agent);
    let models: Vec<&str> = spent
        .map(|spent| spent.0.keys().map(String::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        models,
        ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"],
        "usage lists both real models"
    );
    let last = app
        .world()
        .get::<LastRoute>(agent)
        .map(|last| last.0.as_str());
    assert_eq!(last, Some("cheap (deepseek-flash)"));

    // Any other model goes through `rig-models`' own `/model`, and leaves
    // the router.
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: "model deepseek/deepseek-flash".to_owned(),
    });
    app.update();
    assert_eq!(app.world().get::<Routing>(agent), Some(&Routing(false)));
    let chosen = app
        .world()
        .get::<ModelChoice>(agent)
        .map(|model| model.0.as_str());
    assert_eq!(chosen, Some("deepseek/deepseek-flash"));
}
