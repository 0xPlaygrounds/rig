//! An erased model and the model it was made from are one code path: the same
//! response, the same stream items (including each part's `End` content), the same
//! span fields.

use futures::StreamExt;
use serde_json::{Value, json};

use super::DynModel;
use crate::completion::{CompletionRequest, Usage};
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::streaming::StreamEvent;
use crate::test_utils::{
    CapturedSpan, MockCompletionModel, MockStreamEvent, MockTurn, TraceCapture,
};

/// Every span `body` opens, without ids.
fn spans_of(body: impl FnOnce()) -> Vec<Value> {
    let capture = TraceCapture::default();
    tracing::subscriber::with_default(capture.subscriber(), body);
    capture.spans().iter().map(CapturedSpan::summary).collect()
}

fn request() -> CompletionRequest {
    CompletionRequest::new("hello")
        .preamble("be brief")
        .model("probe-model")
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
        .with_response_id("resp_1")
        .with_provider_request_id("req_1")
        .with_usage(usage())
}

fn stream_turn() -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text("hel"),
        MockStreamEvent::text("lo"),
        MockStreamEvent::tool_call("call_1", "lookup", json!({"q": 1})),
        MockStreamEvent::final_response(usage()),
    ]
}

/// The two models share one script with two identical turns: the direct
/// call takes the first, the erased call the second.
#[test]
fn an_erased_unary_call_matches_the_direct_call_and_its_span() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    let model = MockCompletionModel::from_turns([unary_turn(), unary_turn()]);
    let dyn_model: DynModel<Completion> = model.clone().erase();

    let mut direct = None;
    let direct_spans = spans_of(|| {
        direct = Some(futures::executor::block_on(model.call(request())));
    });
    let mut erased = None;
    let erased_spans = spans_of(|| {
        erased = Some(futures::executor::block_on(dyn_model.call(request())));
    });

    let direct = direct.expect("ran").expect("the direct call succeeds");
    let erased = erased.expect("ran").expect("the erased call succeeds");
    assert_eq!(
        serde_json::to_value(&direct).expect("json"),
        serde_json::to_value(&erased).expect("json"),
        "the erased call folds the same response"
    );
    assert_eq!(direct.provider_request_id.as_deref(), Some("req_1"));
    assert!(!direct_spans.is_empty(), "the direct call opened a span");
    assert_eq!(
        direct_spans, erased_spans,
        "the erased call records the same span"
    );
}

#[test]
fn an_erased_stream_yields_the_direct_stream_item_for_item() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    let model = MockCompletionModel::from_stream_turns([stream_turn(), stream_turn()]);
    let dyn_model = DynModel::from(model.clone());

    let mut direct = Vec::new();
    let direct_spans = spans_of(|| {
        let stream = model.stream(request()).expect("the direct stream opens");
        direct = futures::executor::block_on(stream.collect::<Vec<_>>());
    });
    let mut erased = Vec::new();
    let erased_spans = spans_of(|| {
        let stream = dyn_model
            .stream(request())
            .expect("the erased stream opens");
        erased = futures::executor::block_on(stream.collect::<Vec<_>>());
    });

    assert!(
        direct.iter().any(|item| matches!(
            item,
            Ok(crate::streaming::Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(_),
                ..
            }))
        )),
        "a tool call end carries its finalized call: {direct:?}"
    );
    assert_eq!(
        format!("{direct:?}"),
        format!("{erased:?}"),
        "the erased stream yields the same items"
    );
    assert_eq!(
        direct_spans, erased_spans,
        "the erased stream records the same span"
    );
}

#[test]
fn an_erased_model_names_its_wire() {
    let erased = MockCompletionModel::text("x").erase();
    assert_eq!(erased.name(), crate::test_utils::MOCK_PROVIDER);
    assert_eq!(erased.id(), None);
    assert_eq!(
        format!("{erased:?}"),
        r#"DynModel { name: "mock", id: None }"#
    );
    let clone = erased.clone();
    assert_eq!(clone.name(), erased.name());
}

/// The erased call's future stays `'static` when the prompt is borrowed: it
/// converts the prompt before it returns, so the future outlives the
/// prompt and can be spawned.
#[test]
fn an_erased_call_on_a_borrowed_prompt_outlives_the_prompt() {
    fn spawnable<F: std::future::Future + Send + 'static>(future: F) -> F {
        future
    }
    let dyn_model: DynModel<Completion> =
        MockCompletionModel::from_turns([MockTurn::text("hi")]).erase();
    let future = {
        let prompt = String::from("Say hi.");
        spawnable(dyn_model.call(prompt.as_str()))
    };
    let response = futures::executor::block_on(future).expect("the call succeeds");
    assert_eq!(response.text(), "hi");
}

/// A model connected through the catalog, with a transport that records
/// what it sends.
fn connected(reference: &str) -> (DynModel<Completion>, crate::test_utils::RecordingHttpClient) {
    let http = crate::test_utils::RecordingHttpClient::new("{}");
    let model = crate::catalog::Catalog::builtin()
        .connect_with(
            reference,
            crate::providers::registry::ConnectOptions::new()
                .api_key("sk-test")
                .http(http.clone()),
        )
        .expect("connects");
    (model, http)
}

/// `check` lists every option the model refuses, the catalog's and the
/// wire's, sends nothing, and leaves a request it accepts alone.
#[test]
fn check_lists_every_refusal_and_sends_nothing() {
    use crate::completion::{CacheRetention, CheckError, Effort, Reasoning};
    let (model, http) = connected("openai/gpt-6-sol");
    let request = CompletionRequest::new("hi").temperature(0.2).options(
        crate::completion::GenerationOptions::default()
            .reasoning(Reasoning::Budget { tokens: 2048 })
            .top_p(0.9)
            .cache(CacheRetention::Long),
    );
    let Err(CheckError::Unsupported(refused)) = model.check(&request) else {
        panic!("GPT-6 Sol refuses a budget and sampling while it reasons");
    };
    let options: Vec<&str> = refused
        .iter()
        .map(|refusal| refusal.option.as_ref())
        .collect();
    assert_eq!(
        options,
        ["reasoning", "top_p", "temperature"],
        "{refused:?}"
    );
    assert!(
        refused
            .iter()
            .all(|refusal| refusal.provider == "openai" && refusal.model == "gpt-6-sol")
    );
    assert!(http.requests().is_empty(), "a check sends nothing");

    let fine = CompletionRequest::new("hi").reasoning(Effort::High);
    assert!(model.check(&fine).is_ok());
    let unchecked = CompletionRequest::new("hi").temperature(0.2);
    assert!(
        model.check(&unchecked).is_ok(),
        "a request with default options is not checked against the catalog"
    );
}

/// A request that cannot be built for another reason is `Invalid`, whatever
/// it refuses.
#[test]
fn check_reports_a_request_that_cannot_be_built() {
    use crate::completion::CheckError;
    let (model, _) = connected("openai/gpt-6-sol");
    let mut request = CompletionRequest::new("hi").temperature(0.2).top_p(0.9);
    request.chat_history.clear();
    assert!(matches!(model.check(&request), Err(CheckError::Invalid(_))));
}

/// The refusals `check` lists are what sending reports one at a time: under
/// `Error` the call fails with the first.
#[tokio::test]
async fn check_lists_first_what_the_call_refuses() {
    use crate::completion::CheckError;
    let (model, _) = connected("anthropic/claude-fable-5");
    let request = CompletionRequest::new("hi").temperature(0.2).top_p(0.9);
    let Err(CheckError::Unsupported(refused)) = model.check(&request) else {
        panic!("Claude Fable 5 takes no sampling parameters");
    };
    let Err(crate::error::ProviderError::UnsupportedOption(first)) = model.call(request).await
    else {
        panic!("the call is refused");
    };
    assert_eq!(refused.first(), Some(&first));
    assert_eq!(refused.len(), 2, "{refused:?}");
}
