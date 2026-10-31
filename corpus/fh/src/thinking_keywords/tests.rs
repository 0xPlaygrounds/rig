use rig_core::catalog::{ModelSpec, ReasoningSupport};
use rig_core::completion::CompletionRequest;
use rig_core::providers::registry::ProviderId;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, app_on, calls, collect, reply, run_until, scripted_models,
};
use serde_json::json;

use super::*;

/// An app with the plugin, the default config, a `read` tool that answers
/// at once, and one agent on `model` of `models`.
fn app(models: Models, model: &str) -> (App, std::sync::mpsc::Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(models)
        .insert_resource(ThinkConfig::default())
        .add_plugins((AgentPlugin, SessionLogPlugin, ThinkingKeywordsPlugin));
    let read = |called: On<ToolCalled<serde_json::Value>>,
                mut calls: Query<&mut Decision, ToolCalls>| {
        if let Ok(mut call) = calls.get_mut(called.call) {
            call.decide(
                Decider::plugin("fh-thinking-keywords"),
                ToolResult::success("fn main() {}".into()),
            );
        }
    };
    app.add_open_tool("read", "Read a file.", ToolOptions::default(), read);
    collect::<Notice>(&mut app);
    let agent = app.world_mut().spawn(Agent).id();
    app.finish();
    app.update();
    let model = model.to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    app.update();
    (app, wakes, agent)
}

/// Sends `text` and runs until the model got `requests` requests in all
/// and the turn ended.
fn turn(
    app: &mut App,
    wakes: &std::sync::mpsc::Receiver<()>,
    (agent, text): (Entity, &str),
    model: &MockCompletionModel,
    requests: usize,
) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        model.request_count() == requests && !world.entity(agent).contains::<ActiveTurn>()
    });
}

fn reasoning(request: &CompletionRequest) -> Option<Reasoning> {
    request.options.reasoning
}

#[test]
fn a_keyword_raises_every_request_of_its_turn_only() {
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": "src/main.rs" })),
        reply("It fails because of X."),
        reply("You're welcome."),
        reply("Use `think` in a sentence."),
    ]);
    let mut models = scripted_models([("test/thinker", &model)]);
    let levels = vec![Level::Low, Level::Medium, Level::High];
    let support = ReasoningSupport::Listed {
        levels,
        budget: None,
        can_disable: false,
        default: None,
    };
    if let Ok(vendor) = ProviderId::declare("test") {
        let spec = ModelSpec::new(vendor, "thinker")
            .with_tools(true)
            .with_reasoning(support);
        models.0.catalog_mut().insert(spec);
    }
    let (mut app, wakes, agent) = app(models, "test/thinker");

    turn(
        &mut app,
        &wakes,
        (agent, "ultrathink: why does this fail?"),
        &model,
        2,
    );
    turn(&mut app, &wakes, (agent, "ok thanks"), &model, 3);
    turn(
        &mut app,
        &wakes,
        (agent, "what does `think` do?"),
        &model,
        4,
    );

    let requests = model.requests();
    let efforts: Vec<_> = requests.iter().map(reasoning).collect();
    let high = Some(Reasoning::Effort(Level::High));
    assert_eq!(efforts, [high, high, None, None]);
    // The model reads the message as typed.
    let first = requests
        .first()
        .map(|request| format!("{:?}", request.chat_history));
    assert!(
        first
            .unwrap_or_default()
            .contains("ultrathink: why does this fail?")
    );
}

#[test]
fn a_model_without_reasoning_ignores_keywords_with_one_notice() {
    let model = MockCompletionModel::from_stream_turns([reply("A."), reply("B.")]);
    let (mut app, wakes, agent) = app(scripted_models([("test/plain", &model)]), "test/plain");

    turn(
        &mut app,
        &wakes,
        (agent, "Think harder about the race."),
        &model,
        1,
    );
    turn(&mut app, &wakes, (agent, "ultrathink, please"), &model, 2);

    let efforts: Vec<_> = model.requests().iter().map(reasoning).collect();
    assert_eq!(efforts, [None, None]);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let ignored = notices
        .iter()
        .filter(|notice| notice.text.contains("has no reasoning setting"));
    let ignored: Vec<_> = ignored.map(|notice| notice.text.clone()).collect();
    assert_eq!(ignored.len(), 1, "{ignored:?}");
    assert!(
        ignored
            .iter()
            .all(|text| text.ends_with("ignoring 'think harder'")),
        "{ignored:?}"
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ThinkingKeywordsPlugin);
}
