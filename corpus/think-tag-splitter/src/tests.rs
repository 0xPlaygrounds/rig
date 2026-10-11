use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{app_on, connect, reply, run_until};

use super::*;

/// The last streamed text and reasoning of each model call, as fronts saw
/// them, and how many turns ended.
#[derive(Resource, Default)]
struct Seen {
    partials: Vec<(String, String)>,
    ended: usize,
}

fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, JournalPlugin, ThinkTagSplitterPlugin))
        .init_resource::<Seen>()
        .add_observer(
            |gone: On<Remove<Partial>>, partials: Query<&Partial>, mut seen: ResMut<Seen>| {
                if let Ok(partial) = partials.get(gone.entity) {
                    let streamed = (partial.text.clone(), partial.reasoning.clone());
                    seen.partials.push(streamed);
                }
            },
        )
        .add_observer(|_: On<TurnEnded>, mut seen: ResMut<Seen>| seen.ended += 1);
    app.finish();
    app.update();
    (app, wakes)
}

fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, "What is 2 + 2?", DeliveryMode::Queue));
    run_until(app, wakes, |world| world.resource::<Seen>().ended >= turns);
}

#[test]
fn think_tags_stream_as_reasoning_and_stay_out_of_the_answer() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("<thi"),
            MockStreamEvent::text("nk>plan it</think>The answer"),
            MockStreamEvent::text(" is 4."),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Yes."),
    ]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent and a model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    ask(&mut app, &wakes, agent, 1);
    let seen = app.world().resource::<Seen>();
    assert_eq!(
        seen.partials.first(),
        Some(&("The answer is 4.".to_owned(), "plan it".to_owned()))
    );
    let conversation = app
        .world()
        .get::<Conversation>(agent)
        .map(Conversation::messages);
    let answer = match conversation {
        Some([_, message::Message::Assistant(answer)]) => Some(&answer.content),
        _ => None,
    };
    assert_eq!(
        answer,
        Some(&vec![
            AssistantContent::Reasoning(Reasoning::new("plan it")),
            AssistantContent::Text(Text::new("The answer is 4.")),
        ]),
        "{conversation:?}"
    );

    ask(&mut app, &wakes, agent, 2);
    let requests = model.requests();
    let sent = requests
        .get(1)
        .and_then(|request| serde_json::to_string(request).ok());
    let sent = sent.unwrap_or_default();
    assert!(sent.contains("The answer is 4."), "{sent}");
    assert!(
        !sent.contains("think>") && !sent.contains("plan it"),
        "{sent}"
    );
}
