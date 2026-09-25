//! A boxed model and the model it was made from are one code path: the same
//! response, the same stream items (including `BlockEnd.block`), the same
//! span fields.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::{Value, json};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

use super::BoxedModel;
use crate::completion::{CompletionRequest, CompletionRequestBuilder, Usage};
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::streaming::StreamEvent;
use crate::test_utils::{MockCompletionModel, MockStreamEvent, MockTurn};

/// Every span opened while a body runs: name, target, declared fields and
/// recorded values.
#[derive(Clone, Default)]
struct Spans {
    spans: Arc<Mutex<Vec<Value>>>,
    index: Arc<Mutex<BTreeMap<u64, usize>>>,
}

struct Values<'a>(&'a mut serde_json::Map<String, Value>);

impl Visit for Values<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().into(), json!(format!("{value:?}")));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), json!(value));
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().into(), json!(value));
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().into(), json!(value));
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().into(), json!(value));
    }
}

impl<S> Layer<S> for Spans
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _: Context<'_, S>) {
        let metadata = attrs.metadata();
        let mut values = serde_json::Map::new();
        attrs.record(&mut Values(&mut values));
        let fields: Vec<&str> = metadata.fields().iter().map(|field| field.name()).collect();
        let mut spans = self.spans.lock().expect("spans");
        self.index
            .lock()
            .expect("index")
            .insert(id.into_u64(), spans.len());
        spans.push(json!({
            "name": metadata.name(),
            "target": metadata.target(),
            "fields": fields,
            "values": Value::Object(values),
        }));
    }

    fn on_record(&self, id: &Id, record: &Record<'_>, _: Context<'_, S>) {
        let Some(&index) = self.index.lock().expect("index").get(&id.into_u64()) else {
            return;
        };
        let mut spans = self.spans.lock().expect("spans");
        if let Some(values) = spans[index]["values"].as_object_mut() {
            record.record(&mut Values(values));
        }
    }
}

/// Run `body` under a fresh capturing subscriber and return what it opened.
fn spans_of(body: impl FnOnce()) -> Vec<Value> {
    let spans = Spans::default();
    tracing::subscriber::with_default(Registry::default().with(spans.clone()), body);
    spans.spans.lock().expect("spans").clone()
}

fn request() -> CompletionRequest {
    CompletionRequestBuilder::new("hello")
        .preamble("be brief".to_owned())
        .model("probe-model")
        .build()
}

fn usage() -> Usage {
    Usage {
        input_tokens: Some(7),
        output_tokens: Some(3),
        total_tokens: Some(10),
        ..Usage::default()
    }
}

fn unary_turn() -> MockTurn {
    MockTurn::tool_call("call_1", "lookup", json!({"q": 1}))
        .with_message_id("msg_1")
        .with_response_id("resp_1")
        .with_provider_request_id("req_1")
        .with_usage(usage())
}

fn stream_turn() -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::message_id("msg_1"),
        MockStreamEvent::text("hel"),
        MockStreamEvent::text("lo"),
        MockStreamEvent::tool_call("call_1", "lookup", json!({"q": 1})),
        MockStreamEvent::final_response(usage()),
    ]
}

/// The two models share one script with two identical turns: the direct
/// call takes the first, the boxed call the second.
#[test]
fn a_boxed_unary_call_matches_the_direct_call_and_its_span() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    let model = MockCompletionModel::from_turns([unary_turn(), unary_turn()]);
    let boxed: BoxedModel<Completion> = model.clone().boxed();

    let mut direct = None;
    let direct_spans = spans_of(|| {
        direct = Some(futures::executor::block_on(model.call(request())));
    });
    let mut erased = None;
    let boxed_spans = spans_of(|| {
        erased = Some(futures::executor::block_on(boxed.call(request())));
    });

    let direct = direct.expect("ran").expect("the direct call succeeds");
    let erased = erased.expect("ran").expect("the boxed call succeeds");
    assert_eq!(
        serde_json::to_value(&direct).expect("json"),
        serde_json::to_value(&erased).expect("json"),
        "the boxed call folds the same response"
    );
    assert_eq!(direct.provider_request_id.as_deref(), Some("req_1"));
    assert!(!direct_spans.is_empty(), "the direct call opened a span");
    assert_eq!(
        direct_spans, boxed_spans,
        "the boxed call records the same span"
    );
}

#[test]
fn a_boxed_stream_yields_the_direct_stream_item_for_item() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    let model = MockCompletionModel::from_stream_turns([stream_turn(), stream_turn()]);
    let boxed = BoxedModel::from(model.clone());

    let mut direct = Vec::new();
    let direct_spans = spans_of(|| {
        let stream = model.stream(request()).expect("the direct stream opens");
        direct = futures::executor::block_on(stream.collect::<Vec<_>>());
    });
    let mut erased = Vec::new();
    let boxed_spans = spans_of(|| {
        let stream = boxed.stream(request()).expect("the boxed stream opens");
        erased = futures::executor::block_on(stream.collect::<Vec<_>>());
    });

    assert!(
        direct.iter().any(|item| matches!(
            item,
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(_)),
                ..
            })
        )),
        "a tool call end carries its finalized block: {direct:?}"
    );
    let direct = serde_json::to_value(&direct).expect("json");
    let erased = serde_json::to_value(&erased).expect("json");
    assert_eq!(direct, erased, "the boxed stream yields the same items");
    assert_eq!(
        direct_spans, boxed_spans,
        "the boxed stream records the same span"
    );
}

#[test]
fn a_boxed_model_names_its_wire() {
    let boxed = MockCompletionModel::text("x").boxed();
    assert_eq!(boxed.name(), crate::test_utils::MOCK_PROVIDER);
    assert_eq!(boxed.model(), None);
    assert_eq!(
        format!("{boxed:?}"),
        r#"BoxedModel { name: "mock", model: None }"#
    );
    let clone = boxed.clone();
    assert_eq!(clone.name(), boxed.name());
}
