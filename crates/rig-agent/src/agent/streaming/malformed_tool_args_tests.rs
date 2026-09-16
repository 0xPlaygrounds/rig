//! rig#2447: a streamed tool call that closes with input that is not JSON
//! must not end the run by itself. The provider stream reports it as a
//! typed `ErrorReport`; the engine routes that into the same invalid-tool
//! recovery it offers for an unknown name. One test per action, each
//! proving what the *model* sees on the next request — the observable
//! contract — and that the default is still fail-fast.

use super::{MultiTurnStreamItem, StreamedUserContent, StreamingError};
use crate::agent::AgentBuilder;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::{InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason};
use crate::completion::PromptError;
use crate::test_utils::{MockAddTool, MockCompletionModel, MockStreamEvent};
use futures::StreamExt;
use rig_core::error::ErrorKind;
use rig_core::message::{Message, ToolResultContent, UserContent};

const RAW: &str = "{\"x\": 2, \"y\": \x01";

/// A stream whose tool call closes with malformed arguments, followed by a
/// healthy second turn the model would produce after feedback.
fn model_with_malformed_call() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_call_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_call_1", RAW),
            // Closes under `UnparseableToolInput::Error`: the wire promised
            // a complete block.
            MockStreamEvent::tool_call_end("tool_call_1"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("recovered"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ])
}

#[derive(Clone)]
struct DecideHook(InvalidToolCallAction);

impl AgentHook for DecideHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        // The hook sees the typed reason and the raw text, not a string.
        assert_eq!(context.tool_name, "add");
        assert!(
            matches!(&context.reason, InvalidToolCallReason::MalformedArguments { error } if !error.is_empty()),
            "hook must see MalformedArguments, got {:?}",
            context.reason
        );
        assert_eq!(context.args.as_deref(), Some(RAW));
        assert!(context.is_streaming);
        Some(self.0.clone())
    }
}

/// Everything the consumer observed from one streamed run.
struct Observed {
    items: Vec<MultiTurnStreamItem>,
    error: Option<StreamingError>,
}

async fn run_with(action: Option<InvalidToolCallAction>) -> (Observed, MockCompletionModel) {
    let model = model_with_malformed_call();
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let prompt = agent
        .prompt("add 2 and something")
        .max_turns(3)
        .max_invalid_tool_call_retries(1);
    let mut stream = match action {
        Some(action) => prompt.add_hook(DecideHook(action)).stream(),
        None => prompt.stream(),
    };
    let mut items = Vec::new();
    let mut error = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(item) => items.push(item),
            Err(err) => {
                error = Some(err);
                break;
            }
        }
    }
    (Observed { items, error }, recorded)
}

fn assert_original_report(error: StreamingError) {
    match error {
        StreamingError::Prompt(err) => match err {
            PromptError::Report(report) => {
                assert_eq!(report.kind, ErrorKind::Response);
                assert!(
                    report
                        .message
                        .contains("tool call `add` arrived with malformed JSON input"),
                    "{}",
                    report.message
                );
            }
            other => panic!("expected the provider report, got {other:?}"),
        },
        StreamingError::Report(report) => {
            assert_eq!(report.kind, ErrorKind::Response);
            assert!(report.message.contains("malformed JSON input"));
        }
        other => panic!("expected the provider report, got {other:?}"),
    }
}

/// No hook, default policy: identical to before — the run ends with the
/// provider's own report and no second request is made.
#[tokio::test]
async fn malformed_arguments_fail_fast_by_default() {
    let (observed, recorded) = run_with(None).await;
    assert!(
        !observed
            .items
            .iter()
            .any(|item| matches!(item, MultiTurnStreamItem::ToolCall { .. })),
        "a malformed call must never be executed"
    );
    assert_original_report(observed.error.expect("run must fail"));
    assert_eq!(recorded.request_count(), 1);
}

/// `Fail` from a hook is the same outcome as no hook.
#[tokio::test]
async fn malformed_arguments_fail_action_reproduces_the_report() {
    let (observed, recorded) = run_with(Some(InvalidToolCallAction::fail())).await;
    assert_original_report(observed.error.expect("run must fail"));
    assert_eq!(recorded.request_count(), 1);
}

/// `Retry`: the partial turn is rolled back with feedback and a **second
/// model request** happens instead of run termination.
#[tokio::test]
async fn malformed_arguments_retry_reissues_the_model_request() {
    let (observed, recorded) = run_with(Some(InvalidToolCallAction::retry(
        "arguments were not JSON; try again",
    )))
    .await;
    assert!(observed.error.is_none(), "{:?}", observed.error);
    assert!(observed.items.iter().any(|item| matches!(
        item,
        MultiTurnStreamItem::FinalResponse(response) if response.output() == "recovered"
    )));
    let requests = recorded.requests();
    assert_eq!(requests.len(), 2, "retry must re-issue the model request");
    let feedback_present = requests[1].chat_history.iter().any(|message| {
        matches!(message, Message::User { content } if content.iter().any(|item| matches!(
            item,
            UserContent::ToolResult(result)
                if result.name == "add"
                    && result.content.iter().any(|content| matches!(
                        content,
                        ToolResultContent::Text(text) if text.text.contains("not JSON")
                    ))
        )))
    });
    assert!(
        feedback_present,
        "the retry request must carry the feedback as a tool result for the call: {:?}",
        requests[1].chat_history
    );
}

/// `Skip`: the model receives a tool result naming the parse failure (never
/// the raw bytes) so it can re-emit the call.
#[tokio::test]
async fn malformed_arguments_skip_feeds_the_parse_failure_back() {
    let (observed, recorded) = run_with(Some(InvalidToolCallAction::skip(
        "add: arguments were not valid JSON",
    )))
    .await;
    assert!(observed.error.is_none(), "{:?}", observed.error);
    let skipped = observed
        .items
        .iter()
        .find_map(|item| match item {
            MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            }) => Some(tool_result.clone()),
            _ => None,
        })
        .expect("skip must emit a synthetic tool result");
    assert_eq!(skipped.call.explicit(), Some("tool_call_1"));
    assert!(skipped.content.iter().any(|content| matches!(
        content,
        ToolResultContent::Text(text) if text.text.contains("not valid JSON") && !text.text.contains(RAW)
    )));

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].chat_history.iter().any(|message| {
        matches!(message, Message::User { content } if content.iter().any(|item| matches!(
            item,
            UserContent::ToolResult(result) if result.call.explicit() == Some("tool_call_1")
        )))
    }));
}

/// `Repair` replaces a *name*; it cannot rewrite argument bytes. It fails
/// closed with the same report `Fail` produces, and makes no second request.
#[tokio::test]
async fn malformed_arguments_repair_fails_closed() {
    let (observed, recorded) = run_with(Some(InvalidToolCallAction::repair("add"))).await;
    assert_original_report(observed.error.expect("repair of malformed input must fail"));
    assert_eq!(recorded.request_count(), 1);
}

/// `Stop`: the run ends cleanly with the hook's reason, not with an error
/// report, and no second request.
#[tokio::test]
async fn malformed_arguments_stop_ends_the_run_cleanly() {
    let (observed, recorded) = run_with(Some(InvalidToolCallAction::stop("operator halted"))).await;
    match observed.error.expect("stop is surfaced as a cancellation") {
        StreamingError::Prompt(err) => match err {
            PromptError::PromptCancelled { reason, .. } => {
                assert_eq!(reason, "operator halted");
            }
            other => panic!("expected PromptCancelled, got {other:?}"),
        },
        other => panic!("expected a prompt cancellation, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

/// Under the default retry budget (zero) a `Retry` is rejected exactly as
/// it is for an unknown name — with the original provider report — so the
/// new reason does not widen the budget.
#[tokio::test]
async fn malformed_arguments_retry_respects_the_retry_budget() {
    let model = model_with_malformed_call();
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let mut stream = agent
        .prompt("add 2 and something")
        .add_hook(DecideHook(InvalidToolCallAction::retry("try again")))
        .max_turns(3)
        .stream();
    let mut error = None;
    while let Some(item) = stream.next().await {
        if let Err(err) = item {
            error = Some(err);
            break;
        }
    }
    assert_original_report(error.expect("a retry past the budget must fail"));
    assert_eq!(recorded.request_count(), 1);
}
