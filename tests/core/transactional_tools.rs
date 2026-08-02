//! Cross-driver regression coverage for positional tool invocation identity.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use rig::OneOrMany;
use rig::agent::AgentConfig;
use rig::completion::{CompletionResponse, FinishReason, Usage};
use rig::message::{AssistantContent, ToolCall, ToolFunction};
use rig::provider::{MockScript, ProviderConfig, Runtime};
use rig::session::{AgentSession, SessionPolicy};
use rig::stream::AgentStream;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use rig_agent::executor::ToolExecutor;
use rig_agent::hooks::{HookDecision, HookEntry, HookEvent, Hooks};
use rig_agent::tool::ToolOutput;
use rig_core::tool::PortableDynamicTool;

#[derive(Clone, Debug, PartialEq)]
struct Observation {
    internal_call_id: String,
    args: serde_json::Value,
    raw_result: Option<String>,
}

#[derive(Clone, Default)]
struct InvocationProbe {
    calls: Arc<Mutex<Vec<Observation>>>,
    results: Arc<Mutex<Vec<Observation>>>,
}

impl InvocationProbe {
    fn entry(&self) -> HookEntry {
        let probe = self.clone();
        HookEntry::sync("invocation-identity", move |event| {
            match event {
                HookEvent::ToolCall {
                    call,
                    internal_call_id,
                } => probe.calls.lock().expect("calls").push(Observation {
                    internal_call_id,
                    args: call.function.arguments,
                    raw_result: None,
                }),
                HookEvent::ToolResult {
                    call,
                    internal_call_id,
                    result,
                    ..
                } => probe.results.lock().expect("results").push(Observation {
                    internal_call_id,
                    args: call.function.arguments,
                    raw_result: Some(result.output().render()),
                }),
                _ => {}
            }
            HookDecision::Continue
        })
    }

    fn observations(&self) -> (Vec<Observation>, Vec<Observation>) {
        (
            self.calls.lock().expect("calls").clone(),
            self.results.lock().expect("results").clone(),
        )
    }
}

fn usage(total: u64) -> Usage {
    let mut usage = Usage::new();
    usage.total_tokens = total;
    usage
}

fn duplicate_tool_turn() -> CompletionResponse {
    CompletionResponse::new(
        OneOrMany::many([
            AssistantContent::ToolCall(ToolCall::new(
                "duplicate".to_string(),
                ToolFunction::new("echo_slot".to_string(), serde_json::json!({"slot": 0})),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                "duplicate".to_string(),
                ToolFunction::new("echo_slot".to_string(), serde_json::json!({"slot": 1})),
            )),
        ])
        .expect("two calls"),
        usage(3),
        "mock",
    )
    .with_finish_reason(FinishReason::ToolCalls)
}

fn final_turn() -> CompletionResponse {
    CompletionResponse::new(
        OneOrMany::one(AssistantContent::text("done")),
        usage(5),
        "mock",
    )
    .with_finish_reason(FinishReason::Stop)
}

fn executor() -> ToolExecutor {
    let tool = PortableDynamicTool::new(
        "echo_slot",
        "echoes its input slot",
        serde_json::json!({"type": "object"}),
        |args| async move {
            // Finish slot 1 first to prove that completion order is not used
            // as the association key.
            if args.get("slot").and_then(serde_json::Value::as_u64) == Some(0) {
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
            Ok(ToolOutput::json(args))
        },
    );
    ToolExecutor::new().register(tool).tool_concurrency(2)
}

fn hooks(probe: &InvocationProbe) -> Hooks {
    Hooks::new().with(probe.entry())
}

fn policy() -> SessionPolicy {
    SessionPolicy {
        surface_tool_calls: true,
        surface_tool_results: true,
        ..SessionPolicy::default()
    }
}

fn assert_probe(probe: &InvocationProbe, expected_ids: Option<[&str; 2]>) {
    let (calls, results) = probe.observations();
    assert_eq!(calls.len(), 2);
    assert_eq!(results.len(), 2);
    assert_ne!(calls[0].internal_call_id, calls[1].internal_call_id);
    if let Some(expected_ids) = expected_ids {
        assert_eq!(calls[0].internal_call_id, expected_ids[0]);
        assert_eq!(calls[1].internal_call_id, expected_ids[1]);
    }
    for index in 0..2 {
        assert_eq!(
            calls[index].internal_call_id,
            results[index].internal_call_id
        );
        assert_eq!(calls[index].args, serde_json::json!({"slot": index}));
        assert_eq!(results[index].args, serde_json::json!({"slot": index}));
        assert!(
            results[index]
                .raw_result
                .as_deref()
                .is_some_and(|raw| raw.contains(&index.to_string())),
            "raw result was paired with the wrong duplicate-id invocation: {:?}",
            results[index]
        );
    }
}

#[tokio::test]
async fn duplicate_provider_ids_stay_positionally_paired_in_both_drivers() {
    let blocking_script = MockScript::from_responses(vec![duplicate_tool_turn(), final_turn()]);
    let blocking_executor = executor();
    let blocking_probe = InvocationProbe::default();
    let mut config = AgentConfig::new();
    config.max_turns = Some(2);
    let mut session = AgentSession::new(
        config.clone(),
        ProviderConfig::Mock(blocking_script),
        Arc::new(Runtime::new()),
        "go",
    )
    .with_tools(blocking_executor.catalog())
    .with_policy(policy());
    let response = session
        .drive(&hooks(&blocking_probe), Some(&blocking_executor))
        .await
        .expect("blocking driver");
    assert_eq!(response.output, "done");
    assert_probe(&blocking_probe, None);

    let streaming_script = MockScript::from_responses(vec![duplicate_tool_turn(), final_turn()])
        .with_streams(vec![vec![
            StreamedAssistantContent::ToolCall {
                tool_call: ToolCall::new(
                    "duplicate".to_string(),
                    ToolFunction::new("echo_slot".to_string(), serde_json::json!({"slot": 0})),
                ),
                internal_call_id: "stream-internal-0".to_string(),
            },
            StreamedAssistantContent::ToolCall {
                tool_call: ToolCall::new(
                    "duplicate".to_string(),
                    ToolFunction::new("echo_slot".to_string(), serde_json::json!({"slot": 1})),
                ),
                internal_call_id: "stream-internal-1".to_string(),
            },
            StreamedAssistantContent::Final(StreamFinal::new("mock", usage(3))),
        ]]);
    let streaming_executor = executor();
    let streaming_probe = InvocationProbe::default();
    let stream = AgentStream::new(
        config,
        ProviderConfig::Mock(streaming_script),
        Arc::new(Runtime::new()),
        "go",
    )
    .with_tools(streaming_executor.catalog())
    .with_policy(policy())
    .drive(hooks(&streaming_probe), Some(streaming_executor));
    futures::pin_mut!(stream);
    let mut final_output = None;
    while let Some(item) = stream.next().await {
        if let rig::stream::AgentStreamItem::Final(response) = item.expect("streaming driver") {
            final_output = Some(response.output);
        }
    }
    assert_eq!(final_output.as_deref(), Some("done"));
    assert_probe(
        &streaming_probe,
        Some(["stream-internal-0", "stream-internal-1"]),
    );
}
