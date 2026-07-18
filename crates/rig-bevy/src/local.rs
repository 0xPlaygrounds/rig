//! Local typed model adapter. Provider response types remain concrete and non-persisted.

use crate::{
    CompletionIngress, RunSnapshot, Runtime, RuntimeError, RuntimeHandle, StreamingIngress,
    TenantId,
};
use futures::StreamExt;
use rig_core::completion::{CompletionError, CompletionModel, CompletionRequest};
use rig_core::streaming::StreamedAssistantContent;
use rig_core::{
    completion::{AssistantContent, Message, ToolDefinition},
    message::ToolChoice,
    message::ToolResult as MessageToolResult,
    schemars,
};
use serde::de::DeserializeOwned;
use std::sync::Arc;
use thiserror::Error;

/// Successful local blocking run with its concrete provider final.
#[derive(Debug)]
pub struct LocalBlockingResult<R> {
    pub handle: RuntimeHandle,
    pub snapshot: RunSnapshot,
    pub raw_response: R,
}

/// Successful local streaming run. Deltas are observations only; `snapshot`
/// contains the canonical state committed after the provider final arrived.
#[derive(Debug)]
pub struct LocalStreamingResult<R> {
    pub handle: RuntimeHandle,
    pub snapshot: RunSnapshot,
    pub provisional: Vec<StreamedAssistantContent<R>>,
    pub raw_response: R,
}

/// Successful structured output finalized by a synthetic tool call.
#[derive(Debug)]
pub struct LocalToolOutputResult<T, R> {
    pub handle: RuntimeHandle,
    pub snapshot: RunSnapshot,
    pub output: T,
    pub raw_response: R,
}

/// Local adapter failure with provider and runtime errors kept distinct.
#[derive(Debug, Error)]
pub enum LocalRunError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Completion(#[from] CompletionError),
    #[error("structured output failed: {0}")]
    StructuredOutput(String),
}

/// Typed local runtime construction. This deliberately does not mimic classic
/// agent construction or share its orchestration engine.
pub struct LocalRuntime<M> {
    runtime: Runtime,
    model: M,
    tenant_id: TenantId,
}

impl<M> LocalRuntime<M>
where
    M: CompletionModel,
{
    fn fail_effect(
        &mut self,
        effect: &crate::EffectRequest,
        message: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        self.runtime
            .submit_stream(StreamingIngress::ProviderFailure {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                message: message.into(),
            })
    }

    pub fn new(model: M, tenant_id: TenantId) -> Self {
        Self {
            runtime: Runtime::new(),
            model,
            tenant_id,
        }
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    /// Execute one owned model effect and atomically commit its final response.
    pub async fn run(
        &mut self,
        request: CompletionRequest,
        max_calls: usize,
    ) -> Result<LocalBlockingResult<M::Response>, LocalRunError> {
        let handle = self.runtime.spawn_run(self.tenant_id, request, max_calls);
        let effect = self.runtime.dispatch(handle)?;
        let response = match self.model.completion(effect.request).await {
            Ok(response) => response,
            Err(error) => {
                self.runtime
                    .submit_stream(StreamingIngress::ProviderFailure {
                        world_id: effect.world_id,
                        tenant_id: effect.tenant_id,
                        run_id: effect.run_id,
                        operation_id: effect.operation_id,
                        correlation_id: effect.correlation_id,
                        generation: effect.generation,
                        message: error.to_string(),
                    })?;
                return Err(error.into());
            }
        };
        let raw_response = response.raw_response;
        self.runtime.submit(CompletionIngress {
            world_id: effect.world_id,
            tenant_id: effect.tenant_id,
            run_id: effect.run_id,
            operation_id: effect.operation_id,
            correlation_id: effect.correlation_id,
            generation: effect.generation,
            choice: response.choice.into_iter().collect(),
            usage: response.usage,
        })?;
        let snapshot = self.runtime.snapshot(handle)?;
        Ok(LocalBlockingResult {
            handle,
            snapshot,
            raw_response,
        })
    }

    /// Execute a complete blocking model/tool loop using ECS capability
    /// snapshots and the total model-call budget.
    pub async fn run_with_tools(
        &mut self,
        request: CompletionRequest,
        max_calls: usize,
        tools: Vec<Arc<dyn crate::ToolImplementation>>,
    ) -> Result<LocalBlockingResult<M::Response>, LocalRunError> {
        self.run_with_tools_policy(request, max_calls, tools, crate::InvalidToolPolicy::Fail)
            .await
    }

    /// Execute the tool loop with an explicit ECS invalid-call policy.
    pub async fn run_with_tools_policy(
        &mut self,
        mut request: CompletionRequest,
        max_calls: usize,
        tools: Vec<Arc<dyn crate::ToolImplementation>>,
        invalid_policy: crate::InvalidToolPolicy,
    ) -> Result<LocalBlockingResult<M::Response>, LocalRunError> {
        request.tools = tools.iter().map(|tool| tool.definition()).collect();
        let handle = self.runtime.spawn_run(self.tenant_id, request, max_calls);
        let mut capabilities = Vec::new();
        for tool in tools {
            let capability = self.runtime.register_tool(self.tenant_id, tool);
            self.runtime.grant_tool(handle, capability)?;
            capabilities.push(capability);
        }
        loop {
            let effect = self.runtime.dispatch(handle)?;
            let response = match self.model.completion(effect.request.clone()).await {
                Ok(response) => response,
                Err(error) => {
                    self.fail_effect(&effect, error.to_string())?;
                    return Err(error.into());
                }
            };
            let raw_response = response.raw_response;
            let choice = response.choice.into_iter().collect::<Vec<_>>();
            let ingress = CompletionIngress {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                choice: choice.clone(),
                usage: response.usage,
            };
            let calls = choice
                .iter()
                .filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if calls.is_empty() {
                self.runtime.submit(ingress)?;
                return Ok(LocalBlockingResult {
                    handle,
                    snapshot: self.runtime.snapshot(handle)?,
                    raw_response,
                });
            }
            let request = self.runtime.commit_tool_turn(ingress)?;
            let snapshot = self
                .runtime
                .snapshot_tools(handle, capabilities.iter().copied())?;
            let mut results = std::collections::BTreeMap::new();
            let mut retry_requested = false;
            'chunks: for (chunk_index, chunk) in calls.chunks(8).enumerate() {
                let mut dispatched = Vec::with_capacity(chunk.len());
                for (offset, call) in chunk.iter().cloned().enumerate() {
                    let index = chunk_index * 8 + offset;
                    let arguments = serde_json::to_string(&call.function.arguments)
                        .map_err(CompletionError::JsonError)?;
                    let effect = match self.runtime.dispatch_tool(
                        handle,
                        &snapshot,
                        index,
                        &call.function.name,
                        arguments,
                    ) {
                        Ok(effect) => effect,
                        Err(RuntimeError::ToolNotAdvertised) => {
                            match self.runtime.apply_invalid_tool_policy(
                                handle,
                                index,
                                &call.function.name,
                                invalid_policy.clone(),
                            )? {
                                crate::InvalidToolResolution::Repair { name, arguments } => self
                                    .runtime
                                    .dispatch_tool(handle, &snapshot, index, &name, arguments)?,
                                crate::InvalidToolResolution::Skip { result } => {
                                    results.insert(
                                        index,
                                        Message::from(MessageToolResult {
                                            id: call.id,
                                            call_id: call.call_id,
                                            content: result.output().as_content().clone(),
                                        }),
                                    );
                                    continue;
                                }
                                crate::InvalidToolResolution::Retry { .. } => {
                                    retry_requested = true;
                                    break 'chunks;
                                }
                                crate::InvalidToolResolution::Failed
                                | crate::InvalidToolResolution::Stopped => {
                                    return Err(RuntimeError::Terminal.into());
                                }
                            }
                        }
                        Err(error) => return Err(error.into()),
                    };
                    dispatched.push((call, effect));
                }
                let completed = futures::future::join_all(
                    dispatched
                        .into_iter()
                        .map(|(call, effect)| async move { (call, effect.execute().await) }),
                )
                .await;
                for (call, ingress) in completed {
                    let index = ingress.call_index;
                    let message = Message::from(MessageToolResult {
                        id: call.id,
                        call_id: call.call_id,
                        content: ingress.result.output().as_content().clone(),
                    });
                    self.runtime.submit_tool(ingress)?;
                    results.insert(index, message);
                }
            }
            if retry_requested {
                continue;
            }
            self.runtime
                .continue_after_tools(handle, request, results.into_values().collect())?;
        }
    }

    /// Finalize structured output from one collision-safe synthetic tool call.
    /// The output tool is never executed as an ordinary side effect.
    pub async fn run_tool_output<T>(
        &mut self,
        mut request: CompletionRequest,
    ) -> Result<LocalToolOutputResult<T, M::Response>, LocalRunError>
    where
        T: DeserializeOwned + schemars::JsonSchema,
    {
        let name = crate::synthetic_output_tool_name(
            request
                .tools
                .iter()
                .map(|definition| definition.name.as_str()),
        );
        let parameters = serde_json::to_value(schemars::schema_for!(T))
            .map_err(|error| LocalRunError::StructuredOutput(error.to_string()))?;
        request.tools.push(ToolDefinition {
            name: name.clone(),
            description: "Return the final structured response.".to_string(),
            parameters,
        });
        request.tool_choice = Some(ToolChoice::Specific {
            function_names: vec![name.clone()],
        });
        let policy = crate::RecoveryPolicy::default();
        let handle = self.runtime.spawn_run(
            self.tenant_id,
            request,
            policy.max_response_retries.saturating_add(1),
        );
        let mut effect = self.runtime.dispatch(handle)?;
        let mut retries = 0usize;
        loop {
            let response = match self.model.completion(effect.request.clone()).await {
                Ok(response) => response,
                Err(error) => {
                    self.fail_effect(&effect, error.to_string())?;
                    return Err(error.into());
                }
            };
            let raw_response = response.raw_response;
            let choice = response.choice.into_iter().collect::<Vec<_>>();
            let output_result: Result<T, LocalRunError> = choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(call) if call.function.name == name => {
                        Some(serde_json::from_value(call.function.arguments.clone()))
                    }
                    _ => None,
                })
                .ok_or_else(|| {
                    LocalRunError::StructuredOutput(
                        "provider did not call the output tool".to_string(),
                    )
                })
                .and_then(|value| {
                    value.map_err(|error| LocalRunError::StructuredOutput(error.to_string()))
                });
            let completion = CompletionIngress {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                choice,
                usage: response.usage,
            };
            match output_result {
                Ok(output) => {
                    self.runtime.submit(completion)?;
                    return Ok(LocalToolOutputResult {
                        handle,
                        snapshot: self.runtime.snapshot(handle)?,
                        output,
                        raw_response,
                    });
                }
                Err(error) if retries < policy.max_response_retries => {
                    retries += 1;
                    self.runtime
                        .retry_tool_output(completion, &name, error.to_string())?;
                    effect = self.runtime.dispatch(handle)?;
                }
                Err(error) => {
                    self.runtime
                        .exhaust_response(completion, error.to_string())?;
                    return Err(error);
                }
            }
        }
    }

    /// Execute the selected structured-output policy with the same bounded,
    /// total-budget recovery semantics for native and prompted JSON modes.
    pub async fn run_structured<T>(
        &mut self,
        mut request: CompletionRequest,
        requested: crate::OutputMode,
        native_supported: bool,
        native_composes_with_tools: bool,
    ) -> Result<LocalToolOutputResult<T, M::Response>, LocalRunError>
    where
        T: DeserializeOwned + schemars::JsonSchema,
    {
        let mode = crate::select_output_mode(
            requested,
            native_supported,
            !request.tools.is_empty(),
            native_composes_with_tools,
        );
        if mode == crate::OutputMode::Tool {
            return self.run_tool_output(request).await;
        }
        let schema = schemars::schema_for!(T);
        if mode == crate::OutputMode::Native {
            request.output_schema = Some(schema.clone());
        } else if request.output_schema.is_none() {
            let instruction = format!(
                "Return only JSON matching this schema: {}",
                serde_json::to_string(&schema)
                    .map_err(|error| LocalRunError::StructuredOutput(error.to_string()))?
            );
            request.preamble = Some(match request.preamble.take() {
                Some(existing) => format!("{existing}\n\n{instruction}"),
                None => instruction,
            });
        }
        let policy = crate::RecoveryPolicy::default();
        let handle = self.runtime.spawn_run(
            self.tenant_id,
            request,
            policy.max_response_retries.saturating_add(1),
        );
        let mut effect = self.runtime.dispatch(handle)?;
        let mut retries = 0usize;
        loop {
            let response = match self.model.completion(effect.request.clone()).await {
                Ok(response) => response,
                Err(error) => {
                    self.fail_effect(&effect, error.to_string())?;
                    return Err(error.into());
                }
            };
            let raw_response = response.raw_response;
            let choice = response.choice.into_iter().collect::<Vec<_>>();
            let parsed = choice.iter().find_map(|content| match content {
                AssistantContent::Text(text) => Some(serde_json::from_str::<T>(&text.text)),
                _ => None,
            });
            let output = parsed
                .ok_or_else(|| {
                    LocalRunError::StructuredOutput(
                        "provider returned no structured text".to_string(),
                    )
                })
                .and_then(|value| {
                    value.map_err(|error| LocalRunError::StructuredOutput(error.to_string()))
                });
            let completion = CompletionIngress {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                choice,
                usage: response.usage,
            };
            match output {
                Ok(output) => {
                    self.runtime.submit(completion)?;
                    return Ok(LocalToolOutputResult {
                        handle,
                        snapshot: self.runtime.snapshot(handle)?,
                        output,
                        raw_response,
                    });
                }
                Err(error) if retries < policy.max_response_retries => {
                    retries += 1;
                    self.runtime.retry_response(completion, error.to_string())?;
                    effect = self.runtime.dispatch(handle)?;
                }
                Err(error) => {
                    self.runtime
                        .exhaust_response(completion, error.to_string())?;
                    return Err(error);
                }
            }
        }
    }

    /// Execute one streamed model effect through the same runtime state and
    /// commit only after the stream ends successfully with a typed final.
    pub async fn stream(
        &mut self,
        request: CompletionRequest,
        max_calls: usize,
    ) -> Result<LocalStreamingResult<M::StreamingResponse>, LocalRunError> {
        let handle = self.runtime.spawn_run(self.tenant_id, request, max_calls);
        let effect = self.runtime.dispatch(handle)?;
        let mut response = match self.model.stream(effect.request.clone()).await {
            Ok(response) => response,
            Err(error) => {
                self.fail_effect(&effect, error.to_string())?;
                return Err(error.into());
            }
        };
        let mut provisional = Vec::new();
        let mut raw_response = None;
        let mut sequence = 0u64;
        while let Some(item) = response.next().await {
            match item {
                Ok(StreamedAssistantContent::Final(final_response)) => {
                    raw_response = Some(final_response);
                }
                Ok(item) => {
                    let observable = match &item {
                        StreamedAssistantContent::Text(text) => {
                            Some(AssistantContent::Text(text.clone()))
                        }
                        StreamedAssistantContent::ToolCall { tool_call, .. } => {
                            Some(AssistantContent::ToolCall(tool_call.clone()))
                        }
                        StreamedAssistantContent::Reasoning(reasoning) => {
                            Some(AssistantContent::Reasoning(reasoning.clone()))
                        }
                        StreamedAssistantContent::ToolCallDelta { .. }
                        | StreamedAssistantContent::ReasoningDelta { .. }
                        | StreamedAssistantContent::Unknown(_)
                        | StreamedAssistantContent::Final(_) => None,
                    };
                    if let Some(content) = observable {
                        self.runtime.submit_stream(StreamingIngress::Delta {
                            world_id: effect.world_id,
                            tenant_id: effect.tenant_id,
                            run_id: effect.run_id,
                            operation_id: effect.operation_id,
                            correlation_id: effect.correlation_id,
                            generation: effect.generation,
                            sequence,
                            content,
                        })?;
                        sequence = sequence.saturating_add(1);
                    }
                    provisional.push(item);
                }
                Err(error) => {
                    self.runtime
                        .submit_stream(StreamingIngress::ProviderFailure {
                            world_id: effect.world_id,
                            tenant_id: effect.tenant_id,
                            run_id: effect.run_id,
                            operation_id: effect.operation_id,
                            correlation_id: effect.correlation_id,
                            generation: effect.generation,
                            message: error.to_string(),
                        })?;
                    return Err(error.into());
                }
            }
        }
        let raw_response = match raw_response {
            Some(response) => response,
            None => {
                let error = CompletionError::ProviderError(
                    "provider stream ended without a final response".to_string(),
                );
                self.fail_effect(&effect, error.to_string())?;
                return Err(error.into());
            }
        };
        let choice = response.choice.iter().cloned().collect();
        let usage = response.usage();
        self.runtime
            .submit_stream(StreamingIngress::ProviderFinal {
                completion: CompletionIngress {
                    world_id: effect.world_id,
                    tenant_id: effect.tenant_id,
                    run_id: effect.run_id,
                    operation_id: effect.operation_id,
                    correlation_id: effect.correlation_id,
                    generation: effect.generation,
                    choice,
                    usage,
                },
                final_response: crate::ProviderFinalEnvelope {
                    provider: std::any::type_name::<M>().to_string(),
                    type_name: std::any::type_name::<M::StreamingResponse>().to_string(),
                    // The local result above carries the concrete value. The
                    // hosted/erased event deliberately records metadata only so
                    // provider payloads and credentials cannot leak into logs.
                    diagnostic_json: serde_json::json!({ "typed_final_available": true }),
                },
            })?;
        let snapshot = self.runtime.snapshot(handle)?;
        Ok(LocalStreamingResult {
            handle,
            snapshot,
            provisional,
            raw_response,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rig_core::{
        completion::{ToolDefinition, Usage},
        test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
        tool::{ToolOutput, ToolResult},
        wasm_compat::WasmBoxedFuture,
    };

    struct Echo;

    #[derive(Debug, serde::Deserialize, schemars::JsonSchema, PartialEq)]
    struct StructuredAnswer {
        value: u64,
    }

    impl crate::ToolImplementation for Echo {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "echo".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(&self, arguments: String) -> WasmBoxedFuture<'_, ToolResult> {
            Box::pin(async move { ToolResult::success(ToolOutput::text(arguments)) })
        }
    }

    #[tokio::test]
    pub(crate) async fn local_blocking_exposes_concrete_raw_final() {
        let model = MockCompletionModel::text("hello");
        let request = model.completion_request("hi").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime.run(request, 1).await.expect("run");

        let _: rig_core::test_utils::MockResponse = result.raw_response;
        assert_eq!(result.snapshot.completed_calls, 1);
        assert_eq!(result.snapshot.output.len(), 1);
    }

    #[tokio::test]
    pub(crate) async fn zero_budget_rejects_before_model_dispatch() {
        let model = MockCompletionModel::text("must not run");
        let probe = model.clone();
        let request = model.completion_request("hi").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        assert!(matches!(
            runtime.run(request, 0).await,
            Err(LocalRunError::Runtime(RuntimeError::Terminal))
        ));
        assert_eq!(probe.request_count(), 0);
    }

    #[tokio::test]
    pub(crate) async fn local_streaming_exposes_deltas_then_concrete_raw_final() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("hello"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let request = model.completion_request("hi").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime.stream(request, 1).await.expect("stream");

        let _: rig_core::test_utils::MockResponse = result.raw_response;
        assert!(!result.provisional.is_empty());
        assert_eq!(result.snapshot.completed_calls, 1);
        assert_eq!(result.snapshot.output.len(), 1);
        assert!(
            runtime
                .runtime()
                .events(result.handle)
                .expect("events")
                .iter()
                .any(|event| matches!(event, crate::RuntimeEvent::ProviderFinal { .. }))
        );
    }

    #[tokio::test]
    async fn local_blocking_runs_complete_tool_continuation() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call", "echo", serde_json::json!({"value": "hi"})),
            MockTurn::text("done"),
        ]);
        let request = model.completion_request("hi").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime
            .run_with_tools(request, 2, vec![Arc::new(Echo)])
            .await
            .expect("agent loop");
        assert_eq!(result.snapshot.completed_calls, 2);
        assert_eq!(result.snapshot.history.len(), 4);
    }

    #[tokio::test]
    pub(crate) async fn local_tool_output_is_collision_safe_and_suppresses_execution() {
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "output",
            "__rig_structured_output_2",
            serde_json::json!({"value": 42}),
        )]);
        let mut request = model.completion_request("answer as JSON").build();
        request.tools.push(ToolDefinition {
            name: "__rig_structured_output".to_string(),
            description: "existing application tool".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        });
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime
            .run_tool_output::<StructuredAnswer>(request)
            .await
            .expect("tool output");
        assert_eq!(result.output, StructuredAnswer { value: 42 });
        assert!(matches!(
            result.snapshot.status,
            crate::RunStatus::Terminal(crate::TerminalReason::Completed)
        ));
    }

    #[tokio::test]
    async fn tool_loop_provider_failure_is_terminal() {
        let model = MockCompletionModel::new([MockTurn::error("boom")]);
        let request = model.completion_request("hi").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        assert!(
            runtime
                .run_with_tools(request, 2, vec![Arc::new(Echo)])
                .await
                .is_err()
        );
        let handle = runtime.runtime().handles()[0];
        assert!(matches!(
            runtime.runtime().snapshot(handle).expect("snapshot").status,
            crate::RunStatus::Terminal(crate::TerminalReason::Failed { .. })
        ));
    }

    #[tokio::test]
    async fn stream_without_typed_final_is_terminal() {
        let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::text("partial")]]);
        let request = model.completion_request("hi").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        assert!(runtime.stream(request, 1).await.is_err());
        let handle = runtime.runtime().handles()[0];
        assert!(matches!(
            runtime.runtime().snapshot(handle).expect("snapshot").status,
            crate::RunStatus::Terminal(crate::TerminalReason::Failed { .. })
        ));
    }

    #[tokio::test]
    pub(crate) async fn structured_output_recovers_within_total_budget() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call(
                "bad",
                "__rig_structured_output",
                serde_json::json!({"wrong": 1}),
            ),
            MockTurn::tool_call(
                "good",
                "__rig_structured_output",
                serde_json::json!({"value": 7}),
            ),
        ]);
        let request = model.completion_request("answer").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime
            .run_tool_output::<StructuredAnswer>(request)
            .await
            .expect("recovered output");
        assert_eq!(result.output.value, 7);
        assert_eq!(result.snapshot.completed_calls, 2);
        assert_eq!(result.snapshot.rejected_effects, 1);
    }

    #[tokio::test]
    pub(crate) async fn structured_output_exhaustion_preserves_all_billed_usage() {
        let billed = Usage {
            input_tokens: 2,
            output_tokens: 3,
            total_tokens: 5,
            ..Usage::new()
        };
        let model = MockCompletionModel::new([
            MockTurn::text("not json").with_usage(billed),
            MockTurn::text("still not json").with_usage(billed),
        ]);
        let request = model.completion_request("answer").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        assert!(
            runtime
                .run_structured::<StructuredAnswer>(
                    request,
                    crate::OutputMode::Prompted,
                    false,
                    false,
                )
                .await
                .is_err()
        );
        let handle = runtime.runtime().handles()[0];
        let snapshot = runtime.runtime().snapshot(handle).expect("snapshot");
        assert_eq!(snapshot.completed_calls, 2);
        assert_eq!(snapshot.usage.total_tokens, 10);
        assert_eq!(snapshot.rejected_effects, 2);
    }

    pub(crate) fn all_structured_output_modes_execute() -> usize {
        futures::executor::block_on(async {
            let cases = [
                (crate::OutputMode::Native, true),
                (crate::OutputMode::Prompted, false),
                (crate::OutputMode::Auto, true),
            ];
            let mut completed = 0;
            for (mode, native_supported) in cases {
                let model = MockCompletionModel::new([MockTurn::text(r#"{"value":7}"#)]);
                let request = model.completion_request("answer").build();
                let mut runtime = LocalRuntime::new(model, TenantId::new());
                let result = runtime
                    .run_structured::<StructuredAnswer>(request, mode, native_supported, false)
                    .await
                    .expect("structured mode");
                assert_eq!(result.output.value, 7);
                completed += 1;
            }
            let model = MockCompletionModel::new([MockTurn::tool_call(
                "output",
                "__rig_structured_output",
                serde_json::json!({"value": 7}),
            )]);
            let request = model.completion_request("answer").build();
            let mut runtime = LocalRuntime::new(model, TenantId::new());
            let result = runtime
                .run_structured::<StructuredAnswer>(request, crate::OutputMode::Tool, false, false)
                .await
                .expect("tool mode");
            assert_eq!(result.output.value, 7);
            completed + 1
        })
    }

    #[tokio::test]
    async fn invalid_tool_retry_reprompts_then_executes_advertised_tool() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("bad", "missing", serde_json::json!({})),
            MockTurn::tool_call("good", "echo", serde_json::json!({"value": 1})),
            MockTurn::text("done"),
        ]);
        let request = model.completion_request("answer").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime
            .run_with_tools_policy(
                request,
                3,
                vec![Arc::new(Echo)],
                crate::InvalidToolPolicy::Retry,
            )
            .await
            .expect("retry recovery");
        assert_eq!(result.snapshot.completed_calls, 3);
        assert_eq!(result.snapshot.rejected_effects, 1);
    }

    #[tokio::test]
    async fn invalid_tool_repair_executes_replacement_arguments() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("bad", "missing", serde_json::json!({})),
            MockTurn::text("done"),
        ]);
        let request = model.completion_request("answer").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime
            .run_with_tools_policy(
                request,
                2,
                vec![Arc::new(Echo)],
                crate::InvalidToolPolicy::Repair {
                    replacement_name: "echo".to_string(),
                    replacement_arguments: "{\"repaired\":true}".to_string(),
                },
            )
            .await
            .expect("repair recovery");
        assert!(
            result
                .snapshot
                .history
                .iter()
                .any(|message| format!("{message:?}").contains("repaired"))
        );
    }

    #[tokio::test]
    async fn invalid_tool_skip_commits_skipped_result_without_execution() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("bad", "missing", serde_json::json!({})),
            MockTurn::text("done"),
        ]);
        let request = model.completion_request("answer").build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime
            .run_with_tools_policy(
                request,
                2,
                vec![Arc::new(Echo)],
                crate::InvalidToolPolicy::Skip {
                    reason: "policy skip".to_string(),
                },
            )
            .await
            .expect("skip recovery");
        assert!(
            result
                .snapshot
                .history
                .iter()
                .any(|message| format!("{message:?}").contains("policy skip"))
        );
    }
}
