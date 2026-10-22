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

struct SelectTurnTools;

impl AgentHook for SelectTurnTools {
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
            tool.description = "Effective description from the dispatch hook".to_owned();
            tool.parameters = json!({
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/number"}, "y": {"type": "number"}},
                "$defs": {"number": {"type": "number", "description": "Complete nested schema"}},
                "required": ["x", "y"],
                "additionalProperties": false,
            });
            tool.annotations =
                rig_core::completion::ToolAnnotations::default().with_read_only(true);
        }
        DispatchAction::Patch(EffectKind::Completion {
            request: request.into(),
            stream: *stream,
        })
    }
}

// These tests exercise local span recording against the requests captured at
// dispatch. They do not change provider wire formats or require a cassette.
async fn capture_tool_definitions(
    streaming: bool,
    agent_opt_in: bool,
    run_opt_in: Option<bool>,
) -> Result<(Vec<CapturedSpan>, Vec<CompletionRequest>)> {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let capture = TraceCapture::default();
    let _default = tracing::subscriber::set_default(capture.subscriber());

    let warmup = if streaming {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("warmup"),
            MockStreamEvent::final_response(Usage::default()),
        ]])
    } else {
        MockCompletionModel::text("warmup")
    };
    let runner = AgentBuilder::new(warmup).build().prompt("warmup");
    if streaming {
        let mut stream = runner.stream();
        while stream.try_next().await?.is_some() {}
    } else {
        runner.run().await?;
    }
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
    let builder = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .add_hook(SelectTurnTools);
    let agent = if agent_opt_in {
        builder.record_content_telemetry(true).build()
    } else {
        builder.build()
    };
    let mut runner = agent.prompt("go").max_turns(3);
    if let Some(enabled) = run_opt_in {
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
    ensure!(spans.len() == 2, "both completion spans must be captured");
    let requests = probe.requests();
    ensure!(
        requests.len() == 2,
        "both model requests must be dispatched"
    );
    let selected: Vec<_> = requests
        .iter()
        .flat_map(|request| request.tools.iter().map(|tool| tool.name.as_str()))
        .collect();
    ensure!(
        selected == ["add", "subtract"],
        "selection must change between turns"
    );
    Ok((spans, requests))
}

fn check_definitions(spans: &[CapturedSpan], requests: &[CompletionRequest]) -> Result<()> {
    for (span, request) in spans.iter().zip(requests) {
        let encoded = span
            .text("gen_ai.tool.definitions")
            .context("opted-in completion must record tool definitions")?;
        let actual: Value = serde_json::from_str(&encoded)?;
        let expected: Vec<_> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect();
        ensure!(
            actual == json!(expected),
            "must record this turn's effective tools"
        );
        ensure!(
            request.tools.len() == 1,
            "hook must narrow the registered tools"
        );
        ensure!(
            span.record_count("gen_ai.tool.definitions") == 1,
            "record once per turn"
        );
        ensure!(
            !request.record_telemetry_content,
            "provider must not duplicate telemetry"
        );
    }
    Ok(())
}

#[tokio::test]
async fn tool_definitions_follow_effective_tools_for_both_drivers() -> Result<()> {
    for streaming in [false, true] {
        let (spans, requests) = capture_tool_definitions(streaming, true, None).await?;
        check_definitions(&spans, &requests)?;
    }
    Ok(())
}

#[tokio::test]
async fn tool_definitions_are_absent_without_content_opt_in() -> Result<()> {
    for streaming in [false, true] {
        let (spans, _) = capture_tool_definitions(streaming, false, None).await?;
        ensure!(
            spans
                .iter()
                .all(|span| span.value("gen_ai.tool.definitions").is_none())
        );
    }
    Ok(())
}

#[tokio::test]
async fn tool_definitions_honor_per_run_content_overrides() -> Result<()> {
    for streaming in [false, true] {
        let (spans, requests) = capture_tool_definitions(streaming, false, Some(true)).await?;
        check_definitions(&spans, &requests)?;
        let (spans, _) = capture_tool_definitions(streaming, true, Some(false)).await?;
        ensure!(
            spans
                .iter()
                .all(|span| span.value("gen_ai.tool.definitions").is_none())
        );
    }
    Ok(())
}
