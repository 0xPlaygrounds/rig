use anyhow::{Context, Result, ensure};
use futures::TryStreamExt;
use serde_json::{Value, json};

use crate::agent::{
    AgentBuilder, AgentHook, CompletionCallAction, CompletionCallEvent, DispatchAction,
    DispatchEvent, HookContext, RequestPatch,
};
use crate::completion::{CompletionRequest, Usage};
use crate::test_utils::{
    CapturedSpan, MockAddTool, MockCompletionModel, MockStreamEvent, MockSubtractTool, MockTurn,
    TraceCapture,
};
use rig_core::effect::EffectKind;
use std::sync::Arc;

/// Offers `add` on the first turn and `subtract` after it, and rewrites every
/// description at dispatch, so only the request actually sent matches.
struct SelectAndRewriteTools;

impl AgentHook for SelectAndRewriteTools {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let tool = if ctx.turn() == 1 { "add" } else { "subtract" };
        CompletionCallAction::patch(RequestPatch::new().active_tools([tool]))
    }

    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let EffectKind::Completion { request, stream } = event.kind else {
            return DispatchAction::Proceed;
        };
        let mut request = (**request).clone();
        for tool in &mut request.tools {
            let tool = Arc::make_mut(tool);
            tool.description = "rewritten at dispatch".to_owned();
            tool.parameters = json!({"type": "object"});
        }
        DispatchAction::Patch(EffectKind::Completion {
            request: request.into(),
            stream: *stream,
        })
    }
}

/// Run a two-turn agent with a hosted tool and return its completion spans
/// and the requests the model received.
async fn capture(
    streaming: bool,
    agent_content: bool,
    run_content: Option<bool>,
) -> Result<(Vec<CapturedSpan>, Vec<CompletionRequest>)> {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let capture = TraceCapture::default();
    let _default = tracing::subscriber::set_default(capture.subscriber());
    AgentBuilder::new(MockCompletionModel::text("warmup"))
        .build()
        .prompt("warmup")
        .await?;
    tracing::callsite::rebuild_interest_cache();
    capture.clear();

    let model = if streaming {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("call-add", "add", json!({"x": 2, "y": 3})),
                MockStreamEvent::final_response(Usage::default()),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(Usage::default()),
            ],
        ])
    } else {
        MockCompletionModel::from_turns([
            MockTurn::tool_call("call-add", "add", json!({"x": 2, "y": 3})),
            MockTurn::text("done"),
        ])
    };
    let probe = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .additional_params(json!({
            "tools": [
                {"type": "mcp", "headers": {"Authorization": "Bearer secret"}},
                {"googleSearch": {}, "codeExecution": {}},
                {"type": "custom", "custom": {"name": "grep", "description": "search"}},
            ],
        }))
        .add_hook(SelectAndRewriteTools)
        .record_content_telemetry(agent_content)
        .build();
    let mut runner = agent.prompt("go").max_turns(3);
    if let Some(enabled) = run_content {
        runner = runner.record_content_telemetry(enabled);
    }
    if streaming {
        let mut stream = runner.stream();
        while stream.try_next().await?.is_some() {}
    } else {
        runner.run().await?;
    }

    let spans: Vec<_> = capture
        .spans()
        .into_iter()
        .filter(|span| span.target == "rig::agent_chat")
        .collect();
    let requests = probe.requests();
    ensure!(spans.len() == 2 && requests.len() == 2, "two completions");
    Ok((spans, requests))
}

fn ensure_recorded(spans: &[CapturedSpan], requests: &[CompletionRequest]) -> Result<()> {
    for ((span, request), name) in spans.iter().zip(requests).zip(["add", "subtract"]) {
        let function = json!({
            "type": "function",
            "name": name,
            "description": "rewritten at dispatch",
            "parameters": {"type": "object"},
        });
        let sent: Vec<_> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name.as_str(),
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect();
        ensure!(sent == [function.clone()], "the model received {sent:?}");
        let expected = json!([
            function,
            {"type": "mcp", "name": "mcp"},
            {"type": "googleSearch", "name": "googleSearch"},
            {"type": "codeExecution", "name": "codeExecution"},
            {"type": "custom", "name": "grep"},
        ]);
        let recorded = span
            .text("gen_ai.tool.definitions")
            .context("tool definitions recorded")?;
        ensure!(
            serde_json::from_str::<Value>(&recorded)? == expected,
            "{recorded}"
        );
    }
    Ok(())
}

fn ensure_absent(spans: &[CapturedSpan]) -> Result<()> {
    ensure!(
        spans
            .iter()
            .all(|span| span.value("gen_ai.tool.definitions").is_none()),
        "tool definitions recorded without content telemetry"
    );
    Ok(())
}

#[tokio::test]
async fn tool_definitions_record_each_dispatched_request() -> Result<()> {
    for streaming in [false, true] {
        let (spans, requests) = capture(streaming, true, None).await?;
        ensure_recorded(&spans, &requests)?;
    }
    Ok(())
}

#[tokio::test]
async fn tool_definitions_follow_content_telemetry() -> Result<()> {
    for streaming in [false, true] {
        ensure_absent(&capture(streaming, false, None).await?.0)?;
        ensure_absent(&capture(streaming, true, Some(false)).await?.0)?;
        let (spans, requests) = capture(streaming, false, Some(true)).await?;
        ensure_recorded(&spans, &requests)?;
    }
    Ok(())
}
