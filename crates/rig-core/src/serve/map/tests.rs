use std::sync::Arc;

use super::*;
use crate::{
    effect::{EffectId, FamilyDescriptor, HandlerKey},
    message::AssistantContent,
    operation::Finish,
    serve::{DispatchContext, RecordBuffer, Recorder},
    streaming::{Item, StreamEvent},
    tool::PublishedContext,
};

/// Streams two text deltas and a terminal, or answers them as one
/// completion when asked unary; answers a custom effect with its payload.
struct Streamer;

impl Serve for Streamer {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("streamer"),
            family: FamilyDescriptor::Custom {
                kind: "test".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Custom { payload, .. } => Reply::Outcome(Ok(Answered::Custom { payload })),
            EffectKind::Completion { stream: false, .. } => {
                Reply::Outcome(written().into_outcome().await)
            }
            _ => written(),
        }
    }
}

/// "hel", "lo", and the end.
fn written() -> Reply {
    Reply::written(
        crate::message::Origin::new("test", "test", "test"),
        |mut writer| async move {
            for text in ["hel", "lo"] {
                if writer.text(text).await.is_err() {
                    return;
                }
            }
            let _ = writer.finish(Finish::default()).await;
        },
    )
}

fn completion(stream: bool) -> EffectKind {
    EffectKind::Completion {
        request: crate::completion::CompletionRequest::new("hi").into(),
        stream,
    }
}

/// What a rewrite tells its driver about a reply.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Shouted(usize);

impl crate::tool::ContextValue for Shouted {
    const KEY: &'static str = "shouted";
}

/// Upper-cases the text deltas and the response's text, and publishes how
/// many deltas it changed once the reply is done.
fn shout(published: Arc<PublishedContext>) -> impl FnMut(Relayed) -> Vec<Relayed> {
    let mut deltas = 0;
    move |relayed| {
        if let Relayed::Item(Item::Event(StreamEvent::Text { .. })) = &relayed {
            deltas += 1;
        }
        if let Relayed::Done(_) = &relayed {
            published.insert(Shouted(deltas)).ok();
        }
        vec![match relayed {
            Relayed::Item(Item::Event(StreamEvent::Text { part, text })) => {
                Relayed::Item(Item::Event(StreamEvent::Text {
                    part,
                    text: text.to_uppercase(),
                }))
            }
            Relayed::Done(mut response) => {
                for block in &mut std::sync::Arc::make_mut(&mut response).choice {
                    if let AssistantContent::Text(text) = block {
                        text.text = text.text.to_uppercase();
                    }
                }
                Relayed::Done(response)
            }
            other => other,
        }]
    }
}

/// A rewrite of the events also rewrites the response the reply ends with,
/// and the record keeps the rewritten reply, events and answer, under the
/// rewriter's name, with what the rewrite published beside it.
#[tokio::test]
async fn a_rewrite_reaches_the_events_the_answer_and_the_record() {
    let handler = ErasedHandler::new(Streamer).map_stream("shout", shout);
    assert_eq!(handler.descriptor().layers, ["shout"]);
    let buffer = RecordBuffer::new().keeping_stream_events();
    let recorder: Arc<dyn Recorder + Send + Sync> = Arc::new(buffer.clone());
    let id = EffectId::from_raw(1);
    recorder.begin(
        id,
        HandlerKey::from("streamer"),
        completion(true),
        DispatchContext::default(),
    );
    let published = PublishedContext::new();
    let dispatch = Dispatch::new(id, true)
        .with_scope(published.clone())
        .recorded_by(recorder);
    let items: Vec<_> = handler
        .handle(completion(true), dispatch)
        .await
        .into_stream()
        .collect()
        .await;
    let texts: Vec<String> = items
        .iter()
        .filter_map(|item| match item {
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["HEL", "LO"]);
    assert!(
        matches!(items.last(), Some(Ok(Relayed::Done(response))) if response.text() == "HELLO"),
        "{items:?}"
    );
    let record = buffer.take(id).expect("recorded");
    assert!(
        matches!(&record.outcome, Ok(Answered::Completion(response)) if response.text() == "HELLO")
    );
    let events = serde_json::to_string(&record.events).expect("serializes");
    assert!(
        events.contains("\"HEL\"") && !events.contains("\"hel\""),
        "{events}"
    );
    // The fact comes back beside the reply, and in its record.
    let fact = |context: Option<crate::tool::ToolResultContext>| {
        context.and_then(|context| context.get::<Shouted>().ok().flatten())
    };
    assert_eq!(fact(record.published), Some(Shouted(2)));
    let back = published.take().map(|context| context.result_context());
    assert_eq!(fact(back), Some(Shouted(2)));
}

/// A unary completion is rewritten too; another outcome passes untouched.
#[tokio::test]
async fn a_unary_completion_is_rewritten_and_other_outcomes_pass() {
    let handler = ErasedHandler::new(Streamer).map_stream("shout", shout);
    let unary = crate::serve::serve_inline(&handler, completion(false)).await;
    assert!(
        matches!(&unary, Ok(Answered::Completion(response)) if response.text() == "HELLO"),
        "{unary:?}"
    );
    let custom = EffectKind::Custom {
        kind: Arc::from("test"),
        payload: serde_json::json!("as is"),
    };
    let outcome = crate::serve::serve_inline(&handler, custom).await;
    assert!(
        matches!(&outcome, Ok(Answered::Custom { payload }) if payload == "as is"),
        "{outcome:?}"
    );
}
