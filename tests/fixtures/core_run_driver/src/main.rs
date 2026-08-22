//! A complete agent turn driven with `rig-core` + `rig-run` only — no
//! `rig-agent`, no async runtime.
//!
//! Exercises every seam a systems driver needs: an erased [`ModelHandle`] over a
//! local model, a [`ToolSet`] of [`PortableDynamicTool`]s pinned into a
//! [`ToolCatalog`], a run built from a [`RunSpec`], [`prepare_request`] for
//! each `CallModel` step, and tool dispatch by name for each `CallTools` step.
//! Exits non-zero on any deviation; `tests/core/core_run_driver.rs` runs it.

use std::{
    future::Future,
    pin::pin,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
};

use rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    ModelHandle, ModelRef, Usage,
};
use rig_core::message::{Message, ToolCall, ToolFunction};
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::tool::{PortableDynamicTool, ToolCatalog, ToolContext, ToolOutput, ToolSet};
use rig_core::wasm_compat::WasmCompatSend;
use rig_run::{AgentRun, AgentRunStep, ModelTurn, RunSpec, prepare_request, transcript};

/// Every future here is ready on first poll (a scripted model, in-process
/// tools); a no-op waker is all the "runtime" this driver needs.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
}

/// Calls `add(2, 3)` on its first turn, answers "done" on its second.
#[derive(Default)]
struct ScriptedModel {
    calls: AtomicUsize,
}

impl CompletionModel for ScriptedModel {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, CompletionError>> + WasmCompatSend {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let choice = if call == 0 {
            // The request must carry the tool the run will call back into.
            assert!(
                request.tools.iter().any(|tool| tool.name == "add"),
                "prepared request advertises the catalog's tools"
            );
            assert!(
                matches!(
                    request.chat_history.first(),
                    Some(Message::System { content }) if content == "be brief"
                ),
                "the spec's preamble leads the prepared history"
            );
            vec![AssistantContent::ToolCall(ToolCall::from_wire(
                "call-1",
                ToolFunction::new("add".to_string(), serde_json::json!({"x": 2, "y": 3})),
            ))]
        } else {
            vec![AssistantContent::text("done")]
        };
        std::future::ready(Ok(CompletionResponse::new(choice, Usage::new(), "fixture")))
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>> + WasmCompatSend
    {
        std::future::ready(Err(CompletionError::ProviderError(
            "fixture drives unary completions only".to_string(),
        )))
    }
}

fn add_tool() -> PortableDynamicTool {
    PortableDynamicTool::new(
        "add",
        "Add x and y",
        serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"]
        }),
        |arguments| {
            Box::pin(async move {
                let x = arguments["x"].as_i64().unwrap_or_default();
                let y = arguments["y"].as_i64().unwrap_or_default();
                Ok(ToolOutput::json(serde_json::json!(x + y)))
            })
        },
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. An erased model with a serializable identity.
    let model = ModelHandle::named(ModelRef::new("fixture"), ScriptedModel::default());
    assert_eq!(model.label(), Some("fixture"));

    // 2. An erased tool set, pinned into a catalog for the turn.
    let mut tools = ToolSet::default();
    tools.add_portable_dynamic_tool(add_tool());
    let catalog: ToolCatalog = tools.catalog();
    assert_eq!(catalog.names().collect::<Vec<_>>(), ["add"]);

    // 3. A run from a spec.
    let spec = RunSpec {
        preamble: Some("be brief".into()),
        max_turns: Some(3),
        ..RunSpec::new()
    };
    let mut run = AgentRun::from_spec(&spec, "add 2 and 3", None);

    // 4. Drive it: prepare → model; dispatch by name → run.
    let mut model_calls = 0;
    let mut tool_calls = 0;
    let output = loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                let prepared = prepare_request(
                    &spec,
                    &model.capabilities(),
                    &history,
                    catalog.definitions().to_vec(),
                    run.output_tool_name(),
                    None,
                )?;
                run.advertise_tools(turn, prepared.tools.clone());
                let executable = prepared.executable_tool_names.clone();
                let allowed = prepared.allowed_tool_names.clone();
                let request = prepared.apply(model.completion_request(prompt)).build();
                let response = block_on(model.completion(request))?;
                model_calls += 1;
                run.model_response(ModelTurn::new(
                    None,
                    response.choice,
                    response.usage,
                    executable,
                    allowed,
                ))?;
            }
            AgentRunStep::CallTools { calls } => {
                let mut results = Vec::with_capacity(calls.len());
                for call in calls {
                    let name = call.tool_call.function.name.clone();
                    let arguments = call.tool_call.function.arguments.to_string();
                    let result =
                        block_on(catalog.execute(&name, &arguments, &mut ToolContext::new()));
                    assert!(result.is_success(), "dispatch by name through the catalog");
                    tool_calls += 1;
                    results.push(transcript::tool_result_output(
                        call.tool_call.id.clone(),
                        call.tool_call.provider.clone(),
                        name,
                        result.output().clone(),
                    ));
                }
                run.tool_results(results)?;
            }
            AgentRunStep::Done(response) => break response.output,
        }
    };

    assert_eq!(output, "done");
    assert_eq!(model_calls, 2, "a tool turn then the final answer");
    assert_eq!(tool_calls, 1, "the catalog dispatched `add` once");
    assert_eq!(
        run.advertised_tools()
            .map(|advertised| advertised.definitions.len()),
        Some(1),
        "the run recorded the advertised tools"
    );
    println!("core-run-driver: ok");
    Ok(())
}
