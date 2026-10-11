use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, ModelSpec, ReasoningSupport};
use rig_core::completion::{Effort as Level, Reasoning};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{Ended, app_on, calls, log_ended, reply, run_until};
use rig_telemetry::{Spending, UsagePlugin};

use super::*;

/// The settings: `cheap` and `strong` are the scripted models, `default`
/// is not configured.
const SETTINGS: &str = r#"
[router.targets]
cheap = "scripted/cheap"
strong = "scripted/strong"
"#;

/// `cheap` and `strong` in the catalog as the scripted vendor's models,
/// answered by the mock of the same name; `cheap` does not reason.
fn scripted(models: &mut Models, cheap: &MockCompletionModel, strong: &MockCompletionModel) {
    let Ok(vendor) = ProviderId::declare("scripted") else {
        return;
    };
    let catalog = models.0.catalog_mut();
    catalog.insert(ModelSpec::new(vendor, "cheap").with_tools(true));
    let reasons = ReasoningSupport::Unknown { default: None };
    catalog.insert(
        ModelSpec::new(vendor, "strong")
            .with_tools(true)
            .with_reasoning(reasons),
    );
    let (cheap, strong) = (cheap.clone(), strong.clone());
    models.0.provide(
        vendor,
        move |spec: &ModelSpec, _: &Catalog| -> Result<ErasedHandler, ConnectError> {
            let model = if spec.id == "cheap" { &cheap } else { &strong };
            let adapter = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
            Ok(ErasedHandler::new(adapter))
        },
    );
}

#[test]
fn a_plan_goes_to_the_strong_model_and_its_read_only_follow_up_to_the_cheap_one() {
    let cheap = MockCompletionModel::from_stream_turns([reply("It is a small crate.")]);
    let read = calls("1", "read", serde_json::json!({ "path": "src/lib.rs" }));
    let strong = MockCompletionModel::from_stream_turns([read]);
    let mut models = Models::default();
    scripted(&mut models, &cheap, &strong);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(Settings::from_toml(SETTINGS))
        .insert_resource(models)
        .add_plugins((AgentPlugin, JournalPlugin, UsagePlugin))
        .add_plugins((rig_models::ModelsPlugin, ModelRouterPlugin))
        .add_open_tool(
            "read",
            "Reads a file.",
            ToolOptions {
                rules: &[],
                footprint: Some(Footprint::ReadOnly),
            },
            |called: On<ToolCalled<serde_json::Value>>, mut commands: Commands| {
                let output = ToolResult::success("pub fn answer() -> u32 { 42 }".into());
                commands
                    .entity(called.call)
                    .insert_if_new(Outcome::from(output));
            },
        )
        .init_resource::<Ended>()
        .add_observer(log_ended);
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

    // The picker's `router` item runs `/model router`; the agent reasons
    // hard, which only `strong` takes.
    let high = Effort(Some(Reasoning::Effort(Level::High)));
    world.entity_mut(agent).insert(high);
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
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let (planned, followed) = (strong.requests(), cheap.requests());
    assert_eq!(planned.len(), 1, "the plan's request went to strong");
    assert_eq!(followed.len(), 1, "the read-only follow-up went to cheap");
    let reasoning = |requests: &[rig_core::completion::CompletionRequest]| {
        requests
            .first()
            .and_then(|request| request.options.reasoning)
    };
    assert_eq!(reasoning(&planned), high.0, "strong reasons as asked");
    assert_eq!(
        reasoning(&followed),
        None,
        "cheap falls back to its default"
    );
    let spent = app.world().get::<Spending>(agent);
    let models: Vec<&str> = spent
        .map(|spent| spent.models.keys().map(String::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        models,
        ["scripted/cheap", "scripted/strong"],
        "usage lists both"
    );
    let label = app
        .world()
        .get::<ModelLabel>(agent)
        .map(|label| label.0.as_str());
    assert_eq!(label, Some("router → cheap"));

    // Any other model leaves the router.
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: "model scripted/strong".to_owned(),
    });
    app.update();
    assert_eq!(app.world().get::<Routing>(agent), Some(&Routing(false)));
    assert!(app.world().get::<ModelLabel>(agent).is_none());
}
