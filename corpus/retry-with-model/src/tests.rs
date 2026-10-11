use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector, ModelSpec, SignIn};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};

/// Serves the one catalog model of the test, `moonshot/b`, by a scripted
/// model, as a sign-in would.
struct Scripted(MockCompletionModel);

impl SignIn for Scripted {
    fn handler(&self, spec: &ModelSpec) -> Option<Result<ErasedHandler, ConnectError>> {
        let adapter = ModelAdapter::<Completion>::new(spec.reference(), self.0.clone());
        Some(Ok(ErasedHandler::new(adapter)))
    }

    fn plan(&self, _spec: &ModelSpec) -> Option<&'static str> {
        None
    }
}

/// The text of the conversation's last message, when it is an answer.
fn last_answer_text(world: &World, agent: Entity) -> String {
    let conversation = world.get::<Conversation>(agent);
    let last = conversation.and_then(|conversation| conversation.messages().last());
    last.and_then(answer_text).unwrap_or_default()
}

/// Runs `line` for `agent` and returns the notices it wrote.
fn command(app: &mut App, agent: Entity, line: &str) -> Vec<String> {
    let world = app.world_mut();
    world.trigger(RunCommand {
        entity: agent,
        line: line.to_owned(),
    });
    world.flush();
    let mut notices = world.resource_mut::<Messages<Notice>>();
    notices.drain().map(|notice| notice.text).collect()
}

#[test]
fn a_retried_turn_answers_on_another_model_and_the_old_answer_can_be_picked() {
    let model_a = MockCompletionModel::from_stream_turns([reply("answer A")]);
    let model_b = MockCompletionModel::from_stream_turns([reply("answer B")]);
    let mut catalog = Catalog::default();
    if let Some(provider) = ProviderId::catalog("moonshot") {
        catalog.insert(ModelSpec::new(provider, "b"));
    }
    let mut models = Connector::new(catalog);
    models.add_sign_in(Scripted(model_b));

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(models)).add_plugins((
        AgentPlugin,
        JournalPlugin,
        RetryWithModelPlugin,
    ));
    let mut agent = app.world_mut().spawn(Agent);
    let mut model_a_choice = String::new();
    if let Some(connection) = connect(&model_a) {
        model_a_choice = connection.spec.reference();
        agent.insert((ModelChoice(model_a_choice.clone()), connection));
    }
    let agent = agent.id();
    app.finish();
    app.update();

    let idle_with = |answer: &'static str| {
        move |world: &World| {
            !world.entity(agent).contains::<ActiveTurn>()
                && last_answer_text(world, agent) == answer
        }
    };
    app.world_mut()
        .trigger(Deliver::new(agent, "the question", DeliveryMode::Queue));
    run_until(&mut app, &wakes, idle_with("answer A"));
    assert_eq!(last_answer_text(app.world(), agent), "answer A");

    command(&mut app, agent, "retry-with moonshot/b");
    run_until(&mut app, &wakes, idle_with("answer B"));

    let world = app.world();
    assert_eq!(last_answer_text(world, agent), "answer B");
    let asked = world.get::<Conversation>(agent).map_or(0, |conversation| {
        let users = conversation.messages().iter();
        users
            .filter(|message| matches!(message, Message::User { .. }))
            .count()
    });
    assert_eq!(asked, 1, "the user message appears once");
    let choice = world
        .get::<ModelChoice>(agent)
        .map(|choice| choice.0.clone());
    assert_eq!(choice, Some(model_a_choice));

    let listed = command(&mut app, agent, "alternatives").join("\n");
    assert!(listed.contains("answer A"), "{listed}");

    command(&mut app, agent, "pick 1");
    assert_eq!(last_answer_text(app.world(), agent), "answer A");
    let listed = command(&mut app, agent, "alternatives").join("\n");
    assert!(listed.contains("answer B"), "{listed}");
}
