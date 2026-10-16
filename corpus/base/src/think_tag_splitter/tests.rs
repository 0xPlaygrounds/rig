use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};

use rig_harness::rig_core::streaming::StreamRecord;

use super::*;

/// The text and reasoning each turn's model call streamed, as fronts saw
/// them.
#[derive(Resource, Default)]
struct Seen {
    partials: Vec<(String, String)>,
}

fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, ThinkTagSplitterPlugin))
        .init_resource::<Seen>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(
            |gone: On<Remove<Partial>>, partials: Query<&Partial>, mut seen: ResMut<Seen>| {
                if let Ok(partial) = partials.get(gone.entity) {
                    let streamed = (partial.text.clone(), partial.reasoning.clone());
                    seen.partials.push(streamed);
                }
            },
        );
    app.finish();
    app.update();
    (app, wakes)
}

fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, "What is 2 + 2?", DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
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
        .map(|conversation| conversation.messages());
    let answer = match conversation {
        Some([_, last]) => match &**last {
            message::Message::Assistant(answer) => Some(&answer.content),
            _ => None,
        },
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

    // The split reply told its call so: the model is known to send tags.
    let sends = &app.world().resource::<SendsTags>().0;
    assert_eq!(sends.len(), 1, "{sends:?}");

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

    // The rewritten events, which the record keeps, are a transcript of
    // the rewritten response: each part ends with the block at its index.
    let text = Part::new(0);
    let chunks = ["<thi", "nk>plan it</think>The answer", " is 4."];
    let whole = AssistantContent::Text(Text::new(chunks.concat()));
    let events = std::iter::once(StreamEvent::Start {
        part: text,
        kind: PartKind::Text,
        name: None,
    })
    .chain(chunks.map(|chunk| StreamEvent::Text {
        part: text,
        text: chunk.to_owned(),
    }))
    .chain([StreamEvent::End {
        part: text,
        content: whole.clone(),
    }]);
    let mut parts = Parts::new(false, PublishedContext::new());
    let items: Vec<Relayed> = events
        .flat_map(|event| parts.relay(Relayed::Item(Item::Event(event))))
        .collect();
    let ends: Vec<AssistantContent> = items
        .iter()
        .filter_map(|item| match item {
            Relayed::Item(Item::Event(StreamEvent::End { content, .. })) => Some(content.clone()),
            _ => None,
        })
        .collect();
    let items: Vec<_> = items
        .into_iter()
        .filter_map(|item| match item {
            Relayed::Item(item) => Some(item),
            _ => None,
        })
        .collect();
    let transcript = serde_json::to_value(items).map(StreamRecord::parse);
    assert!(matches!(transcript, Ok(Ok(_))), "{transcript:?}");
    assert_eq!(ends, split_choice(&[whole], false));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ThinkTagSplitterPlugin);
}
