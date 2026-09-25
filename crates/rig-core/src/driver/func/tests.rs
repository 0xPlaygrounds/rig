//! Closure models run through the driver: both modes from either closure
//! shape, the request untouched, errors in band, and truncation when a
//! stream names no terminal.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::Value;

use crate::completion::ProviderCapabilities;
use crate::completion::{CompletionRequest, CompletionRequestBuilder, CompletionResponse, Usage};
use crate::driver::{BoxedModel, Model};
use crate::embeddings::Embedding as Vector;
use crate::embeddings::EmbeddingsBuilder;
use crate::error::ProviderError;
use crate::message::{AssistantContent, Message, Reasoning};
use crate::operation::{AdapterOutput, Completion, EmbeddingCapabilities};
use crate::streaming::{StreamEvent, StreamFinal};
use crate::wire::Wire;

fn request() -> CompletionRequest {
    CompletionRequestBuilder::new("hello")
        .preamble("be brief".to_owned())
        .max_tokens(12)
        .temperature(0.5)
        .build()
}

fn answer(text: &str) -> CompletionResponse {
    let mut response = CompletionResponse::new(
        vec![AssistantContent::text(text)],
        Usage::default(),
        "inner",
        Value::Null,
    );
    response.message_id = Some("msg_1".to_owned());
    response
}

fn text_of(response: &CompletionResponse) -> String {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text().to_owned()),
            _ => None,
        })
        .collect()
}

/// The events a streaming closure yields for `text`, with a terminal when
/// `terminal` is set: the same grammar every wire speaks.
fn events(text: &[&str], terminal: bool) -> Vec<Result<StreamEvent, ProviderError>> {
    let mut out = AdapterOutput::default();
    out.message_id("msg_1");
    for piece in text {
        out.text(*piece);
    }
    if terminal {
        out.final_record(StreamFinal::new("inner", Usage::default(), Value::Null));
    }
    out.drain().collect()
}

#[tokio::test]
async fn a_unary_closure_answers_a_call_and_a_stream() {
    let model = Model::completion_fn("closure", |_| async { Ok(answer("hi")) });

    let called = model.call(request()).await.expect("the call succeeds");
    assert_eq!(text_of(&called), "hi");
    assert_eq!(called.message_id.as_deref(), Some("msg_1"));

    let mut stream = model.stream(request()).expect("the stream opens");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.expect("every item is an event"));
    }
    assert!(
        items
            .iter()
            .any(|event| matches!(event, StreamEvent::Final(_))),
        "the whole reply re-emits its terminal: {items:?}"
    );
    let streamed = stream.finish().expect("the stream folds");
    assert_eq!(text_of(&streamed), "hi");
    assert_eq!(streamed.message_id.as_deref(), Some("msg_1"));
}

#[tokio::test]
async fn a_streaming_closure_answers_a_call_and_a_stream() {
    let model = Model::completion_stream_fn("closure", |_| async {
        Ok(futures::stream::iter(events(&["hel", "lo"], true)))
    });

    let called = model
        .call(request())
        .await
        .expect("the call folds the events");
    assert_eq!(text_of(&called), "hello");

    let mut stream = model.stream(request()).expect("the stream opens");
    let mut deltas = 0;
    while let Some(item) = stream.next().await {
        if matches!(
            item.expect("every item is an event"),
            StreamEvent::BlockDelta { .. }
        ) {
            deltas += 1;
        }
    }
    assert_eq!(deltas, 2, "each event passes through");
    assert_eq!(
        text_of(&stream.finish().expect("the stream folds")),
        "hello"
    );
}

#[tokio::test]
async fn the_request_reaches_the_closure_untouched() {
    let seen = Arc::new(Mutex::new(None));
    let recorder = Arc::clone(&seen);
    let model = Model::completion_fn("closure", move |request| {
        *recorder.lock().expect("seen") = Some(request);
        async { Ok(answer("hi")) }
    });
    // Reasoning another issuer signed: a wire that scoped the history to
    // its own issuers would drop it before the closure saw it.
    let mut reasoning =
        Reasoning::new_with_signature("weighed the options", Some("sig".to_owned()));
    reasoning.provider = Some("other".to_owned());
    let mut request = request();
    request.chat_history.insert(
        0,
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::Reasoning(reasoning)],
        },
    );
    model
        .call(request.clone())
        .await
        .expect("the call succeeds");

    let seen = seen.lock().expect("seen").clone().expect("the closure ran");
    assert_eq!(
        serde_json::to_value(&seen).expect("json"),
        serde_json::to_value(&request).expect("json"),
        "the closure sees the request the caller built"
    );
}

#[tokio::test]
async fn a_closure_wire_is_its_own_issuer_and_scopes_nothing() {
    let model = Model::completion_fn("closure", |_| async { Ok(answer("hi")) });
    assert_eq!(model.wire.name(), "closure");
    assert_eq!(model.wire.reasoning_issuer(None), Some("closure"));
    assert_eq!(model.wire.reasoning_issuer(Some("gpt")), Some("closure"));
    assert_eq!(model.wire.replay_issuers(None), None);
    assert_eq!(model.wire.model(), None);
}

#[tokio::test]
async fn a_unary_closures_error_is_the_calls_error_in_both_modes() {
    let model = Model::completion_fn("closure", |_| async {
        Err::<CompletionResponse, _>(ProviderError::Provider("boom".to_owned()))
    });

    let error = model.call(request()).await.expect_err("the call fails");
    assert!(
        matches!(&error, ProviderError::Provider(message) if message == "boom"),
        "{error:?}"
    );

    let mut stream = model.stream(request()).expect("the stream opens");
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            errors.push(error.message);
        }
    }
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("boom"), "{errors:?}");
    assert!(stream.finish().is_err(), "a failed stream is not an answer");
}

#[tokio::test]
async fn a_streaming_closures_errors_arrive_in_band_after_its_events() {
    let opening = Model::completion_stream_fn("closure", |_| async {
        Err::<futures::stream::Iter<std::vec::IntoIter<Result<StreamEvent, ProviderError>>>, _>(
            ProviderError::Provider("no stream".to_owned()),
        )
    });
    let error = opening
        .call(request())
        .await
        .expect_err("a closure that fails to open fails the call");
    assert!(
        matches!(&error, ProviderError::Provider(message) if message == "no stream"),
        "{error:?}"
    );

    let midway = Model::completion_stream_fn("closure", |_| async {
        let mut items = events(&["hel"], false);
        items.push(Err(ProviderError::Provider("cut".to_owned())));
        Ok(futures::stream::iter(items))
    });
    let mut stream = midway.stream(request()).expect("the stream opens");
    let mut deltas = 0;
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockDelta { .. }) => deltas += 1,
            Ok(_) => {}
            Err(error) => errors.push(error.message),
        }
    }
    assert_eq!(deltas, 1, "the event before the error is delivered");
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("cut"), "{errors:?}");
    let error = midway
        .call(request())
        .await
        .expect_err("the folded call carries the error");
    assert!(
        matches!(&error, ProviderError::Provider(message) if message == "cut"),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_stream_without_a_terminal_is_truncation() {
    let model = Model::completion_stream_fn("closure", |_| async {
        Ok(futures::stream::iter(events(&["hel", "lo"], false)))
    });
    let mut stream = model.stream(request()).expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("no error is fabricated");
    }
    let error = stream
        .finish()
        .expect_err("a stream that named no terminal is truncated");
    assert!(
        matches!(&error, ProviderError::Response(message) if message.contains("truncated")),
        "{error:?}"
    );

    let error = model
        .call(request())
        .await
        .expect_err("a call over the same reply is truncated too");
    assert!(
        matches!(&error, ProviderError::Response(message) if message.contains("truncated")),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_closure_runs_when_its_reply_is_polled() {
    let runs = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&runs);
    let model = Model::completion_fn("closure", move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        async { Ok(answer("hi")) }
    });

    let stream = model.stream(request()).expect("the stream opens");
    assert_eq!(runs.load(Ordering::SeqCst), 0, "opening sends nothing");
    let _items: Vec<_> = stream.collect().await;
    assert_eq!(runs.load(Ordering::SeqCst), 1);

    let call = model.call(request());
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1,
        "a call sends nothing until polled"
    );
    call.await.expect("the call succeeds");
    assert_eq!(runs.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_closure_model_states_capabilities_and_boxes() {
    let model = Model::completion_fn("closure", |_| async { Ok(answer("hi")) }).with_capabilities(
        ProviderCapabilities {
            composes_native_output_with_tools: true,
        },
    );
    assert!(model.wire.capabilities().composes_native_output_with_tools);

    let boxed: BoxedModel<Completion> = model.boxed();
    assert_eq!(boxed.name(), "closure");
    assert!(boxed.capabilities().composes_native_output_with_tools);
    let response = boxed
        .call(request())
        .await
        .expect("the boxed call succeeds");
    assert_eq!(text_of(&response), "hi");
}

#[tokio::test]
async fn an_embedding_closure_embeds_a_batch_at_its_width() {
    let model = Model::embedding_fn("embed", 3, |texts| async move {
        Ok(texts
            .into_iter()
            .map(|document| Vector {
                document,
                vec: vec![1.0, 2.0, 3.0],
            })
            .collect())
    });
    assert_eq!(model.wire.capabilities().ndims, 3);

    let response = model
        .call(vec!["a".to_owned(), "b".to_owned()])
        .await
        .expect("the batch embeds");
    assert_eq!(response.provider, "embed");
    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[1].document, "b");
    let embedding = model
        .clone()
        .boxed()
        .embed_text("a")
        .await
        .expect("the erased model embeds one text");
    assert_eq!(embedding.vec, vec![1.0, 2.0, 3.0]);

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(["a".to_owned(), "b".to_owned(), "c".to_owned()])
        .expect("documents are added")
        .build()
        .await
        .expect("the builder batches through the closure");
    assert_eq!(embeddings.len(), 3);

    let declared = model.with_capabilities(EmbeddingCapabilities::new(8, 3).declaring(Some(4)));
    let error = declared
        .call(vec!["a".to_owned()])
        .await
        .expect_err("the driver holds the closure to the declared width");
    assert!(
        matches!(
            error,
            ProviderError::MismatchedDimensions {
                requested: 4,
                returned: 3,
                ..
            }
        ),
        "{error:?}"
    );
}
