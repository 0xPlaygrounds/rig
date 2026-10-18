use rig_basics::telemetry::usage::{AgentUsage, UsagePlugin};
use rig_core::catalog::ReasoningSupport;
use rig_core::completion::{CompletionRequest, Effort as Level, Reasoning};
use rig_core::effect::model_key;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, calls, collect, first_agent, log_ended, reply, run_until,
    scripted_models,
};

use super::*;

/// The settings: `cheap` and `strong` are the scripted models, `default`
/// is not configured.
const SETTINGS: &str = r#"
[router.targets]
cheap = "scripted/cheap"
strong = "scripted/strong"
"#;

#[test]
fn a_plan_goes_to_the_strong_model_and_its_read_only_follow_up_to_the_cheap_one() {
    let cheap = MockCompletionModel::from_stream_turns([reply("It is a small crate.")]);
    let read = calls("1", "read", serde_json::json!({ "path": "src/lib.rs" }));
    let strong = MockCompletionModel::from_stream_turns([read]);
    let mut models = scripted_models([("scripted/cheap", &cheap), ("scripted/strong", &strong)]);
    // `strong` reasons; `cheap` does not.
    let catalog = models.0.catalog_mut();
    let reasons = ReasoningSupport::Unknown { default: None };
    let spec = catalog
        .resolve("scripted/strong")
        .map(|found| found.spec.clone());
    if let Ok(spec) = spec {
        catalog.insert(spec.with_reasoning(reasons));
    }
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(Settings::from_toml(SETTINGS))
        .insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, UsagePlugin))
        .add_plugins((rig_basics::models::ModelsPlugin, ModelRouterPlugin))
        .add_open_tool(
            "read",
            "Reads a file.",
            ToolOptions {
                footprint: Some(Touches::Reads(Scope::Anything)),
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
    collect::<EffectRecorded>(&mut app);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world);
    assert!(agent.is_some(), "the session has an agent");
    let Some(agent) = agent else {
        return;
    };

    // The picker's `router` item runs `/model router`; the agent reasons
    // hard, which only `strong` takes.
    let high = ReasoningSetting(Some(Reasoning::Effort(Level::High)));
    world.entity_mut(agent).insert(high);
    world.trigger(RunCommand::new(agent, "model router".to_owned()));
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
    let reasoning = |requests: &[CompletionRequest]| {
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
    let spent = app.world().get::<AgentUsage>(agent);
    let models: Vec<&str> = spent
        .map(|spent| spent.models.keys().map(String::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        models,
        ["scripted/cheap", "scripted/strong"],
        "usage lists both"
    );
    let completions: Vec<_> = app
        .world()
        .resource::<Collected<EffectRecorded>>()
        .0
        .iter()
        .filter(|EffectRecorded(record)| record.kind.name() == "completion")
        .map(|EffectRecorded(record)| record.key.clone())
        .collect();
    assert_eq!(
        completions,
        [model_key("scripted/strong"), model_key("scripted/cheap")],
        "the effect log records each call under its real model"
    );
    let warned = app.world().resource::<Collected<Notice>>();
    let warned = warned.0.iter().any(|notice| {
        notice.text.contains("→ default\" is off")
            && notice.text.contains("not in [router.targets]")
    });
    assert!(
        warned,
        "the rule to the unconfigured `default` is off, with a warning"
    );
    let label = app
        .world()
        .get::<ModelLabel>(agent)
        .map(|label| label.0.as_str());
    assert_eq!(label, Some("router → cheap"));

    // Any other model leaves the router.
    app.world_mut()
        .trigger(RunCommand::new(agent, "model scripted/strong".to_owned()));
    app.update();
    assert_eq!(app.world().get::<Routing>(agent), Some(&Routing(false)));
    assert!(app.world().get::<ModelLabel>(agent).is_none());
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ModelRouterPlugin);
}
