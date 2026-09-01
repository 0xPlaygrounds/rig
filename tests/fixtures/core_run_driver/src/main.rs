//! A complete agent turn driven with `rig-core` only — no `rig-agent`, no
//! `AgentRun`, no async runtime.
//!
//! Proves that the run *vocabulary* alone supports a hand-rolled driver: an
//! erased [`ModelHandle`] over a local model, a [`ToolSet`] of
//! [`PortableDynamicTool`]s pinned into a [`ToolCatalog`], a [`RunSpec`],
//! [`prepare_request`] for each model call, tool dispatch by name, and the
//! transcript helpers to thread the tool result back and validate the result.
//! Exits non-zero on any deviation; `tests/core/core_run_driver.rs` runs it.

use std::{
    future::Future,
    pin::pin,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
};

use rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    ModelHandle, ModelRef, Usage, prepare::prepare_request, spec::RunSpec,
};
use rig_core::message::{Message, ToolCall, ToolFunction};
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::tool::{PortableDynamicTool, ToolCatalog, ToolContext, ToolOutput, ToolSet};
use rig_core::transcript::{
    assistant_text_from_choice, build_full_history, build_history_for_request,
    tool_result_output, validate_canonical,
};
use rig_core::wasm_compat::WasmCompatSend;

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
            // The second call must see the whole threaded transcript.
            assert_eq!(
                request.chat_history.len(),
                4,
                "preamble, user prompt, assistant tool call, then the tool-result prompt"
            );
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

/// One model call: prepare from the spec and the history so far, send, and
/// return the assistant turn as a message plus its content.
fn call_model(
    model: &ModelHandle,
    spec: &RunSpec,
    catalog: &ToolCatalog,
    history: &[Message],
    prompt: Message,
) -> Result<(Message, Vec<AssistantContent>), Box<dyn std::error::Error>> {
    let prepared = prepare_request(
        spec,
        &model.capabilities(),
        history,
        catalog.definitions().to_vec(),
        None,
        None,
    )?;
    let request = prepared.apply(model.completion_request(prompt)).build();
    let response = block_on(model.completion(request))?;
    let choice = response.choice;
    Ok((
        Message::Assistant {
            id: None,
            content: choice.clone(),
        },
        choice,
    ))
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

    // 3. A spec — the protocol-facing configuration, as data.
    let spec = RunSpec {
        preamble: Some("be brief".into()),
        max_turns: Some(3),
        ..RunSpec::new()
    };

    // 4. First model call: the prompt alone.
    let prompt = Message::user("add 2 and 3");
    let (assistant_turn, choice) = call_model(&model, &spec, &catalog, &[], prompt.clone())?;
    let calls: Vec<&ToolCall> = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1, "the scripted model calls one tool");

    // 5. Dispatch by name through the catalog and thread the result back as
    //    the canonical tool-result user message.
    let mut results = Vec::with_capacity(calls.len());
    for call in calls {
        let name = call.function.name.clone();
        let arguments = call.function.arguments.to_string();
        let result = block_on(catalog.execute(&name, &arguments, &mut ToolContext::new()));
        assert!(result.is_success(), "dispatch by name through the catalog");
        results.push(tool_result_output(
            call.id.clone(),
            call.provider.clone(),
            name,
            result.output().clone(),
        ));
    }
    let tool_result_turn = Message::User { content: results };

    // 6. Second model call: history is everything before the tool result.
    let history = build_history_for_request(None, &[prompt, assistant_turn]);
    let (final_turn, final_choice) =
        call_model(&model, &spec, &catalog, &history, tool_result_turn.clone())?;

    // 7. The full transcript is canonical and yields the answer.
    let transcript = build_full_history(Some(&history), vec![tool_result_turn, final_turn]);
    validate_canonical(&transcript)?;
    assert_eq!(transcript.len(), 4, "prompt, tool call, tool result, answer");
    assert_eq!(assistant_text_from_choice(&final_choice), "done");
    println!("core-run-driver: ok");
    Ok(())
}
