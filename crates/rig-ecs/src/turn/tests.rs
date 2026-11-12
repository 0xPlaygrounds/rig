use std::sync::Arc;

use bevy_ecs::message::Messages;
use bevy_ecs::prelude::*;
use rig_core::completion::message::Origin;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message, Usage};
use rig_core::effect::Answered;
use rig_core::message::AssistantContent;
use rig_core::streaming::{Part, PartKind, StreamEvent};

use super::{Fed, ModelRequest, StreamDelta, feed, stream_partials};
use crate::stage::{Outcome, SupersededBy};

/// A world `stream_partials` runs in.
fn world() -> World {
    let mut world = World::new();
    world.init_resource::<Messages<StreamDelta>>();
    world
}

/// A model call's request.
fn call() -> ModelRequest {
    ModelRequest::new(CompletionRequest::new(Message::user("hi")))
}

/// A reply of `text`, as its outcome.
fn replied(text: &str) -> Outcome {
    let choice = vec![AssistantContent::text(text)];
    let origin = Origin::new("test.api", "test", "model");
    let reply = CompletionResponse::new(choice, Usage::default(), origin, serde_json::Value::Null);
    Outcome(Ok(Answered::Completion(Arc::new(reply))))
}

/// Runs `stream_partials` `passes` times; the text each call's deltas
/// carried.
fn texts(world: &mut World, passes: usize) -> Vec<(Entity, String)> {
    for _ in 0..passes {
        assert!(world.run_system_cached(stream_partials).is_ok());
    }
    let deltas = world.resource::<Messages<StreamDelta>>();
    let mut cursor = deltas.get_cursor();
    let texts = cursor.read(deltas).filter_map(|delta| match &delta.event {
        StreamEvent::Text { text, .. } => Some((delta.effect, text.clone())),
        _ => None,
    });
    texts.collect()
}

fn text_event(text: &str) -> Fed {
    let text = StreamEvent::Text {
        part: Part::new(0),
        text: text.to_owned(),
    };
    Fed::Item(web_time::SystemTime::now(), Some(text))
}

#[test]
fn a_reply_that_did_not_stream_comes_as_deltas_once() {
    let mut world = world();
    let answered = world.spawn((call(), replied("in place"))).id();
    // A stream that ended with no event.
    let (sender, fed) = feed();
    let ended = world.spawn((call(), fed, replied("ended"))).id();
    assert!(
        sender
            .try_send(Fed::Item(web_time::SystemTime::now(), None))
            .is_ok()
    );
    let mut texts = texts(&mut world, 3);
    texts.sort();
    assert_eq!(
        texts,
        [
            (answered, "in place".to_owned()),
            (ended, "ended".to_owned())
        ]
    );
}

#[test]
fn a_reply_that_streamed_any_text_gets_no_copy() {
    let mut world = world();
    // Part of a reply came, then the call was answered otherwise.
    let (sender, fed) = feed();
    let call = world.spawn((call(), fed)).id();
    assert!(sender.try_send(text_event("Hel")).is_ok());
    assert!(world.run_system_cached(stream_partials).is_ok());
    world.entity_mut(call).insert(replied("Hello"));
    assert!(sender.try_send(text_event("lo")).is_ok());
    let texts = texts(&mut world, 2);
    assert_eq!(texts, [(call, "Hel".to_owned()), (call, "lo".to_owned())]);
}

#[test]
fn a_superseded_reply_writes_no_deltas() {
    let mut world = world();
    let successor = world.spawn(call()).id();
    world.spawn((call(), replied("replaced"), SupersededBy(successor)));
    assert_eq!(texts(&mut world, 2), []);
}

#[test]
fn each_part_of_a_reply_that_did_not_stream_starts_grows_and_ends() {
    let mut world = world();
    let choice = vec![
        AssistantContent::reasoning("hm"),
        AssistantContent::text(""),
    ];
    let origin = Origin::new("test.api", "test", "model");
    let reply = CompletionResponse::new(choice, Usage::default(), origin, serde_json::Value::Null);
    world.spawn((call(), Outcome(Ok(Answered::Completion(Arc::new(reply))))));
    assert!(world.run_system_cached(stream_partials).is_ok());
    let deltas = world.resource::<Messages<StreamDelta>>();
    let events: Vec<StreamEvent> = deltas
        .get_cursor()
        .read(deltas)
        .map(|delta| delta.event.clone())
        .collect();
    let (reasoning, text) = (Part::new(0), Part::new(1));
    assert_eq!(
        events,
        [
            StreamEvent::Start {
                part: reasoning,
                kind: PartKind::Reasoning,
                name: None,
            },
            StreamEvent::Reasoning {
                part: reasoning,
                text: "hm".to_owned(),
            },
            StreamEvent::End {
                part: reasoning,
                content: AssistantContent::reasoning("hm"),
            },
            // An empty text part grows by nothing.
            StreamEvent::Start {
                part: text,
                kind: PartKind::Text,
                name: None,
            },
            StreamEvent::End {
                part: text,
                content: AssistantContent::text(""),
            },
        ]
    );
}
