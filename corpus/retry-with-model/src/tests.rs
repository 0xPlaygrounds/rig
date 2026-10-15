use std::sync::mpsc::Receiver;

use rig_core::catalog::{Catalog, Connector, ModelSpec, Transport};
use rig_core::message::Message;
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderId;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{app_on, reply, run_until};

/// Models `test/a` and `test/b`, answered by `a` and `b`.
fn models(a: MockCompletionModel, b: MockCompletionModel) -> Models {
    let mut connector = Connector::new(Catalog::default());
    if let Ok(vendor) = ProviderId::declare("test") {
        for id in ["a", "b"] {
            connector
                .catalog_mut()
                .insert(ModelSpec::new(vendor, id).with_tools(true));
        }
        connector.provide(
            vendor,
            move |spec: &ModelSpec, _: &Catalog, _: &Transport| {
                let model = if spec.id == "a" { a.clone() } else { b.clone() };
                let adapter = ModelAdapter::<Completion>::new(spec.reference(), model);
                Ok(ErasedHandler::new(adapter))
            },
        );
    }
    Models(connector)
}

/// The kernel and the plugin on `store`, its session restored, and the
/// session's agent.
fn start(store: &MemoryStore, models: Models) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, RetryWithModelPlugin));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    (app, wakes, agent.unwrap_or(Entity::PLACEHOLDER))
}

/// The text of the conversation's last message, when it is an answer.
fn last_answer(world: &World, agent: Entity) -> String {
    let conversation = world.get::<Conversation>(agent);
    let last = conversation.and_then(|conversation| conversation.messages().last());
    last.and_then(|message| message.final_answer())
        .unwrap_or_default()
}

/// How many user messages the conversation holds.
fn asked(world: &World, agent: Entity) -> usize {
    world.get::<Conversation>(agent).map_or(0, |conversation| {
        let messages = conversation.messages().iter();
        messages
            .filter(|message| matches!(&***message, Message::User { .. }))
            .count()
    })
}

/// Runs `line` for `agent` and returns the notices it wrote, and why it
/// was refused.
fn command(app: &mut App, agent: Entity, line: &str) -> String {
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, line.to_owned()));
    world.flush();
    let mut notices = world.resource_mut::<Messages<Notice>>();
    let mut texts: Vec<String> = notices.drain().map(|notice| notice.text).collect();
    let mut recalled = world.resource_mut::<Messages<Recalled>>();
    texts.extend(recalled.drain().filter_map(|recalled| recalled.why));
    texts.join("\n")
}

#[test]
fn a_retried_turn_answers_on_another_model_and_the_old_answer_can_be_picked() {
    let store = MemoryStore::default();
    let a = MockCompletionModel::from_stream_turns([reply("answer A")]);
    let b = MockCompletionModel::from_stream_turns([reply("answer B")]);
    let (mut app, wakes, agent) = start(&store, models(a, b));
    app.world_mut()
        .entity_mut(agent)
        .insert(ModelChoice("test/a".to_owned()));
    let idle_with = |answer: &'static str| {
        move |world: &World| {
            !world.entity(agent).contains::<ActiveTurn>() && last_answer(world, agent) == answer
        }
    };

    app.world_mut()
        .trigger(Deliver::new(agent, "the question", DeliveryMode::Queue));
    run_until(&mut app, &wakes, idle_with("answer A"));
    assert_eq!(last_answer(app.world(), agent), "answer A");

    let world = app.world_mut();
    let mut completions = world.query::<&rig_tui::CompletionSource>();
    let offered: Vec<String> = completions
        .iter(world)
        .flat_map(|completion| completion.candidates.iter().map(|item| item.text.clone()))
        .collect();
    assert_eq!(offered, ["--force", "test/a", "test/b"]);
    let refused = command(&mut app, agent, "retry-with test/bb");
    assert!(refused.contains("`test/b`"), "{refused}");

    command(&mut app, agent, "retry-with test/b");
    run_until(&mut app, &wakes, idle_with("answer B"));
    let world = app.world();
    assert_eq!(last_answer(world, agent), "answer B");
    assert_eq!(asked(world, agent), 1, "the user message appears once");
    let choice = world
        .get::<ModelChoice>(agent)
        .map(|choice| choice.0.as_str());
    assert_eq!(choice, Some("test/a"));
    let listed = command(&mut app, agent, "alternatives");
    assert!(listed.contains("test/a: answer A"), "{listed}");

    command(&mut app, agent, "pick 1");
    assert_eq!(last_answer(app.world(), agent), "answer A");
    // The frame's end writes the session log.
    app.update();
    drop(app);

    // The swap and the alternatives are kept with the session.
    let idle = MockCompletionModel::from_stream_turns(Vec::<Vec<_>>::new());
    let (mut app, _wakes, agent) = start(&store, models(idle.clone(), idle));
    assert_eq!(last_answer(app.world(), agent), "answer A");
    assert_eq!(asked(app.world(), agent), 1);
    let listed = command(&mut app, agent, "alternatives");
    assert!(listed.contains("test/b: answer B"), "{listed}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RetryWithModelPlugin);
}
