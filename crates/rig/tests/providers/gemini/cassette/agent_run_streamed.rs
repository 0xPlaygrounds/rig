//! Streamed-turn coverage for [`AgentRun`]: hand-driving
//! [`StreamedTurnAssembler`] over real Gemini SSE streams, mid-stream invalid
//! tool-call recovery, per-call usage recording, and the built-in streaming
//! driver divergences pinned by #1899.

use std::collections::{BTreeSet, VecDeque};

use futures::StreamExt;
use rig::agent::run::{
    AgentRun, AgentRunStep, StreamedInvalidToolCall, StreamedResolution, StreamedTurnAssembler,
    StreamedTurnEvent,
};
use rig::agent::{
    HookAction, InvalidToolCallHookAction, MultiTurnStreamItem, PromptHook, StreamingError,
    ToolCallHookAction,
};
use rig::client::CompletionClient;
use rig::completion::{GetTokenUsage, PromptError, Usage};
use rig::message::{Message, ToolChoice, ToolResult};
use rig::providers::gemini;
use rig::streaming::{StreamedAssistantContent, StreamingCompletion, StreamingPrompt};

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, GeminiAgent, Subtract, Sum, assert_canonical_assistant_order,
    assistant_tool_call_names, execute_pending_calls, history_has_assistant_tool_call,
    is_tool_result_user_message, sum_completion_call_usage, tool_names,
};
use super::super::support::with_gemini_cassette;
use crate::support::{assert_mentions_expected_number, assert_nonempty_response};

/// How one hand-driven streamed turn ended.
#[derive(Debug)]
enum TurnEnd {
    /// The turn was assembled and fed to the machine.
    Finished,
    /// Mid-stream recovery abandoned the turn (retry or skip).
    Abandoned {
        skipped_tool_result: Option<ToolResult>,
    },
}

/// Hand-drive one streamed model turn through [`StreamedTurnAssembler`],
/// mirroring the built-in streaming driver's protocol. Invalid tool calls are
/// resolved with `on_invalid`'s action; streamed text accumulates into
/// `collected_text`.
#[allow(clippy::too_many_arguments)]
async fn run_streamed_turn(
    agent: &GeminiAgent,
    run: &mut AgentRun,
    prompt: Message,
    history: Vec<Message>,
    executable: &BTreeSet<String>,
    allowed: &BTreeSet<String>,
    on_invalid: impl Fn(&StreamedInvalidToolCall) -> InvalidToolCallHookAction,
    collected_text: &mut String,
) -> Result<TurnEnd, PromptError> {
    let mut stream = agent
        .stream_completion(prompt, &history)
        .await
        .expect("stream request should build")
        .stream()
        .await
        .expect("gemini stream should open");
    let mut assembler = StreamedTurnAssembler::new(executable.clone(), allowed.clone());
    let mut recorded = false;

    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        let mut events: VecDeque<StreamedTurnEvent> = assembler
            .ingest(&item)
            .expect("ingest should succeed")
            .into();
        while let Some(event) = events.pop_front() {
            match event {
                StreamedTurnEvent::EmitIngested => {
                    if let StreamedAssistantContent::Text(text) = &item {
                        collected_text.push_str(&text.text);
                    }
                }
                StreamedTurnEvent::EmitToolCallDelta { .. } => {}
                StreamedTurnEvent::Completed { usage, .. } => {
                    if !recorded {
                        run.record_streamed_completion_call(usage)
                            .expect("completion call should record while the turn is pending");
                        recorded = true;
                    }
                }
                StreamedTurnEvent::InvalidToolCall(invalid) => {
                    let partial = assembler.partial_turn(stream.message_id.clone());
                    let context = run.streamed_invalid_tool_call_context(&partial, &invalid);
                    assert!(context.is_streaming);
                    assert_eq!(context.tool_name, invalid.tool_call.function.name);
                    let action = on_invalid(&invalid);
                    match run.resolve_streamed_invalid_tool_call(&partial, &invalid, action) {
                        Err(error) => {
                            // Drain so record mode captures a complete body.
                            while stream.next().await.is_some() {}
                            return Err(error);
                        }
                        Ok(resolution) => {
                            let replayed = assembler.resolve_pending_invalid(&resolution);
                            match resolution {
                                StreamedResolution::Repaired { .. } => {
                                    events.extend(replayed);
                                }
                                StreamedResolution::TurnAbandoned {
                                    skipped_tool_result,
                                } => {
                                    let drained_usage = drain_stream_usage(&mut stream).await;
                                    if !recorded {
                                        run.record_streamed_completion_call(drained_usage).expect(
                                            "abandoned turns may still record their completion call",
                                        );
                                    }
                                    return Ok(TurnEnd::Abandoned {
                                        skipped_tool_result,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        assembler.pending_delta_error().is_none(),
        "the provider stream should end consistently"
    );
    if !recorded {
        run.record_streamed_completion_call(Usage::new())
            .expect("turns without provider usage still record a completion call");
    }
    let streamed_turn = assembler.finish(stream.message_id.clone(), &stream.choice);
    run.streamed_turn(streamed_turn)?;
    Ok(TurnEnd::Finished)
}

async fn drain_stream_usage<R>(stream: &mut rig::streaming::StreamingCompletionResponse<R>) -> Usage
where
    R: Clone + Unpin + GetTokenUsage,
{
    while let Some(item) = stream.next().await {
        if let Ok(StreamedAssistantContent::Final(final_response)) = item {
            return final_response.token_usage();
        }
    }
    Usage::new()
}

#[tokio::test]
async fn streamed_hand_driven_multi_turn_run_completes() {
    with_gemini_cassette(
        "agent_run_streamed/streamed_hand_driven_multi_turn_run_completes",
        |client| async move {
            // Machine-only guard, no IO: a fresh run has no model call to
            // record against.
            let mut fresh = AgentRun::new("unused");
            assert!(
                fresh.record_streamed_completion_call(Usage::new()).is_err(),
                "a phantom completion call must be rejected on a fresh run"
            );

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let names = tool_names(&["add", "subtract"]);

            let mut run = AgentRun::new(
                "Use the tools to compute (7 + 4) - 2: first compute 7 + 4 with the add tool, then subtract 2 from that result with the subtract tool, then state the final result.",
            )
            .max_turns(5);
            let mut streamed_text = String::new();

            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let end = run_streamed_turn(
                            &agent,
                            &mut run,
                            prompt,
                            history,
                            &names,
                            &names,
                            |invalid| {
                                panic!("no invalid tool calls expected: {invalid:?}")
                            },
                            &mut streamed_text,
                        )
                        .await
                        .expect("streamed turn should be accepted");
                        assert!(matches!(end, TurnEnd::Finished));
                    }
                    AgentRunStep::CallTools { calls } => {
                        for call in &calls {
                            assert!(
                                call.internal_call_id.is_some(),
                                "streamed turns persist internal call ids: {call:?}"
                            );
                        }
                        run.tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert_mentions_expected_number(&streamed_text, 9);
            assert_mentions_expected_number(&response.output, 9);
            assert!(run.turn() >= 2, "tool use forces at least two model calls");
            assert_eq!(
                response.completion_calls.len(),
                run.turn(),
                "every streamed model call records exactly one completion call"
            );
            assert_eq!(
                sum_completion_call_usage(&response.completion_calls),
                response.usage
            );
            assert!(
                response.usage.total_tokens > 0,
                "cassette-recorded usage should be non-zero"
            );

            let messages = response.messages.clone().expect("run reports its messages");
            assert!(history_has_assistant_tool_call(&messages, "add"));
            assert!(history_has_assistant_tool_call(&messages, "subtract"));
            // The assembler records streamed turns in canonical replay order.
            assert_canonical_assistant_order(&messages);
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_invalid_tool_call_fails_fast_mid_stream() {
    with_gemini_cassette(
        "agent_run_streamed/streamed_invalid_tool_call_fails_fast_mid_stream",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            let executable = tool_names(&["add"]);
            let nothing_allowed = tool_names(&[]);

            let mut run = AgentRun::new("What is 21 + 21? Use the add tool.").max_turns(2);
            let AgentRunStep::CallModel {
                prompt, history, ..
            } = run.next_step().expect("run should advance")
            else {
                panic!("a fresh run starts with a model call");
            };
            let mut streamed_text = String::new();
            let error = run_streamed_turn(
                &agent,
                &mut run,
                prompt,
                history,
                &executable,
                &nothing_allowed,
                |invalid| {
                    assert_eq!(invalid.tool_call.function.name, "add");
                    InvalidToolCallHookAction::fail()
                },
                &mut streamed_text,
            )
            .await
            .expect_err("the disallowed call must fail the run mid-stream");

            let PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } = error
            else {
                panic!("expected UnknownToolCall, got {error:?}");
            };
            assert_eq!(tool_name, "add");
            assert_eq!(available_tools, vec!["add".to_string()]);
            assert!(allowed_tools.is_empty());
            assert!(
                history_has_assistant_tool_call(&chat_history, "add"),
                "the streamed diagnostic history must include the partial assistant turn: {chat_history:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_repair_continues_the_same_stream() {
    with_gemini_cassette(
        "agent_run_streamed/streamed_repair_continues_the_same_stream",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Sum)
                .build();
            let machine_names = tool_names(&["sum"]);

            let mut run =
                AgentRun::new("Use the add tool to compute 2 + 3, then state the result.")
                    .max_turns(3);
            let mut streamed_text = String::new();
            let mut repaired = false;

            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let end = run_streamed_turn(
                            &agent,
                            &mut run,
                            prompt,
                            history,
                            &machine_names,
                            &machine_names,
                            |invalid| {
                                assert_eq!(invalid.tool_call.function.name, "add");
                                InvalidToolCallHookAction::repair("sum")
                            },
                            &mut streamed_text,
                        )
                        .await
                        .expect("the repaired turn should be accepted");
                        assert!(matches!(end, TurnEnd::Finished));
                    }
                    AgentRunStep::CallTools { calls } => {
                        for call in &calls {
                            assert_eq!(
                                call.tool_call.function.name, "sum",
                                "the repaired name must reach the driver"
                            );
                            repaired = true;
                        }
                        run.tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert!(repaired, "the model should call a tool that gets repaired");
            assert_mentions_expected_number(&response.output, 5);
            let messages = response.messages.clone().expect("run reports its messages");
            let recorded: Vec<String> = messages
                .iter()
                .flat_map(assistant_tool_call_names)
                .collect();
            assert!(
                !recorded.iter().any(|name| name == "add"),
                "the unrepaired name must not be recorded: {recorded:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_skip_abandons_the_turn_and_recovers() {
    with_gemini_cassette(
        "agent_run_streamed/streamed_skip_abandons_the_turn_and_recovers",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .build();
            let executable = tool_names(&["add"]);
            let nothing_allowed = tool_names(&[]);
            const SKIP_REASON: &str = "The add tool is disabled for this request.";

            let mut run = AgentRun::new("What is 21 + 21? Use the add tool.").max_turns(3);
            let mut streamed_text = String::new();
            let mut abandoned = false;

            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt,
                        history,
                        turn,
                    } => {
                        let (allowed, expect_abandon) = if abandoned {
                            (&executable, false)
                        } else {
                            (&nothing_allowed, true)
                        };
                        if abandoned {
                            // The rollback messages from the skipped turn are
                            // already threaded into the retry request.
                            assert!(
                                history_has_assistant_tool_call(&history, "add"),
                                "turn {turn} history should include the abandoned assistant turn: {history:?}"
                            );
                            assert!(
                                is_tool_result_user_message(&prompt),
                                "the retry prompt is the synthetic tool-results message: {prompt:?}"
                            );
                        }
                        let end = run_streamed_turn(
                            &agent,
                            &mut run,
                            prompt,
                            history,
                            &executable,
                            allowed,
                            |invalid| {
                                assert!(expect_abandon, "only the first turn restricts tools");
                                assert_eq!(invalid.tool_call.function.name, "add");
                                InvalidToolCallHookAction::skip(SKIP_REASON)
                            },
                            &mut streamed_text,
                        )
                        .await
                        .expect("the streamed turn should be accepted");
                        match end {
                            TurnEnd::Abandoned {
                                skipped_tool_result,
                            } => {
                                assert!(expect_abandon, "only the first turn should abandon");
                                let tool_result = skipped_tool_result
                                    .expect("a skipped call surfaces its synthetic tool result");
                                assert!(!tool_result.id.is_empty());
                                abandoned = true;
                            }
                            TurnEnd::Finished => {
                                assert!(!expect_abandon, "the first turn must abandon");
                            }
                        }
                    }
                    AgentRunStep::CallTools { calls } => {
                        run.tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert!(abandoned, "the restricted first turn should be abandoned");
            assert_nonempty_response(&response.output);
            assert!(
                run.completion_calls().len() >= 2,
                "the abandoned turn still records its completion call"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn builtin_streaming_max_turns_error_carries_pending_message() {
    with_gemini_cassette(
        "agent_run_streamed/builtin_streaming_max_turns_error_carries_pending_message",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();

            let mut stream = agent
                .stream_prompt("What is 21 + 21? Use the add tool.")
                .multi_turn(0)
                .await;

            let mut prompt_error = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(_) => {}
                    Err(StreamingError::Prompt(error)) => {
                        prompt_error = Some(*error);
                        break;
                    }
                    Err(other) => panic!("expected a prompt error, got {other:?}"),
                }
            }

            let PromptError::MaxTurnsError {
                max_turns,
                chat_history,
                prompt,
            } = prompt_error.expect("the stream should surface MaxTurnsError")
            else {
                panic!("expected MaxTurnsError");
            };
            assert_eq!(max_turns, 0);
            // Pins the divergence resolved by #1899: the streaming error
            // carries the actual pending tool-results message, not a rag-text
            // reconstruction of it.
            assert!(
                is_tool_result_user_message(&prompt),
                "MaxTurnsError must carry the pending tool-results message: {prompt:?}"
            );
            assert!(
                history_has_assistant_tool_call(&chat_history, "add"),
                "the error history must include the assistant tool-call turn: {chat_history:?}"
            );
        },
    )
    .await;
}

#[derive(Clone)]
struct CancelOnToolCall;

impl PromptHook<gemini::completion::CompletionModel> for CancelOnToolCall {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        ToolCallHookAction::Terminate {
            reason: "cancelled by test hook".to_string(),
        }
    }

    async fn on_completion_call(&self, _prompt: &Message, _history: &[Message]) -> HookAction {
        HookAction::cont()
    }
}

#[tokio::test]
async fn builtin_streaming_cancellation_history_includes_assistant_turn() {
    with_gemini_cassette(
        "agent_run_streamed/builtin_streaming_cancellation_history_includes_assistant_turn",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();

            let mut stream = agent
                .stream_prompt("What is 21 + 21? Use the add tool.")
                .with_hook(CancelOnToolCall)
                .multi_turn(2)
                .await;

            let mut prompt_error = None;
            let mut saw_final = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final = true,
                    Ok(_) => {}
                    Err(StreamingError::Prompt(error)) => {
                        prompt_error = Some(*error);
                        break;
                    }
                    Err(other) => panic!("expected a prompt error, got {other:?}"),
                }
            }
            assert!(
                !saw_final,
                "a cancelled run must not produce a final response"
            );

            let PromptError::PromptCancelled {
                chat_history,
                reason,
            } = prompt_error.expect("the hook should cancel the run")
            else {
                panic!("expected PromptCancelled");
            };
            assert!(
                reason.contains("cancelled by test hook"),
                "the hook reason must surface: {reason}"
            );
            // Pins the divergence resolved by #1899: mid-run cancellation
            // history includes the already-recorded assistant turn.
            assert!(
                history_has_assistant_tool_call(&chat_history, "add"),
                "cancellation history must include the recorded assistant turn: {chat_history:?}"
            );
        },
    )
    .await;
}
