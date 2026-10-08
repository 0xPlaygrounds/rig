//! Streamed-turn coverage for [`AgentRun`]: hand-driving
//! [`StreamedTurnAssembler`] over real Gemini SSE streams, mid-stream invalid
//! tool-call recovery, per-call usage recording, and the built-in streaming
//! driver divergences pinned by #1899.

use rig::streaming::Item;
use std::collections::VecDeque;

use futures::StreamExt;
use rig::agent::run::{
    AgentRun, AgentRunStep, CommittedItem, PendingToolCall, StreamedInvalidToolCall,
    StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent, TurnPolicy, project,
};
use rig::agent::{
    AgentHook, DispatchAction, DispatchEvent, InvalidToolCallAction, MultiTurnStreamItem,
};
use rig::completion::PromptError;
use rig::message::{Message, ToolChoice};
use rig::providers::gemini;
use rig::streaming::StreamEvent;
use rig_agent::test_utils::validate_cancelled_failure;

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, GeminiAgent, assistant_tool_call_names, execute_pending_calls,
    history_has_assistant_tool_call, is_tool_result_user_message, policy,
};
use super::super::support::with_gemini_cassette;
use crate::support::{assert_mentions_expected_number, assert_nonempty_response};

/// How one hand-driven streamed turn ended.
#[derive(Debug)]
enum TurnEnd {
    /// The turn was assembled and fed to the machine.
    Finished,
    /// Mid-stream recovery abandoned the turn (retry or skip).
    Abandoned,
}

/// Hand-drive one streamed model turn through [`StreamedTurnAssembler`],
/// mirroring the built-in streaming driver's protocol. Invalid tool calls are
/// resolved with `on_invalid`'s action; streamed text accumulates into
/// `collected_text`.
async fn run_streamed_turn(
    agent: &GeminiAgent,
    run: &mut AgentRun,
    prompt: Message,
    history: Vec<Message>,
    policy: &TurnPolicy,
    on_invalid: impl Fn(&StreamedInvalidToolCall) -> InvalidToolCallAction,
    collected_text: &mut String,
) -> Result<TurnEnd, PromptError> {
    let mut stream = agent
        .model
        .stream(agent.request(prompt, history))
        .expect("gemini stream should open");
    let mut assembler = StreamedTurnAssembler::new(policy.clone());

    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        let mut events: VecDeque<StreamedTurnEvent> = assembler
            .ingest(&item)
            .expect("ingest should succeed")
            .into();
        while let Some(event) = events.pop_front() {
            match event {
                StreamedTurnEvent::EmitIngested => {
                    if let Item::Event(StreamEvent::Text { text, .. }) = &item {
                        collected_text.push_str(text);
                    }
                }
                StreamedTurnEvent::EmitToolCallDelta
                | StreamedTurnEvent::HoldToolCall
                | StreamedTurnEvent::EmitToolCall { .. } => {}
                StreamedTurnEvent::InvalidToolCall(invalid) => {
                    let partial = assembler.partial_turn(&stream.partial());
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
                                StreamedResolution::Repaired { .. }
                                | StreamedResolution::Ignored => {
                                    events.extend(replayed);
                                }
                                StreamedResolution::TurnAbandoned => {
                                    let response = stream
                                        .finish()
                                        .await
                                        .expect("an abandoned turn's stream still ends");
                                    run.record_streamed_completion_call(
                                        response.usage,
                                        response.identity(),
                                        response.finish_reason(),
                                        response.raw,
                                    )
                                    .expect("abandoned turns still record their completion call");
                                    return Ok(TurnEnd::Abandoned);
                                }
                            }
                        }
                    }
                }
                other => panic!("unhandled stream event {other:?}"),
            }
        }
    }

    let response = stream.finish().await.expect("the stream ends");
    run.record_streamed_completion_call(
        response.usage,
        response.identity(),
        response.finish_reason(),
        response.raw.clone(),
    )
    .expect("completion call should record while the turn is pending");
    let streamed_turn = assembler.finish(&response);
    run.streamed_turn(streamed_turn)?;
    Ok(TurnEnd::Finished)
}

#[tokio::test]
async fn streamed_repair_continues_the_same_stream() {
    with_gemini_cassette(
        "agent_run_streamed/streamed_repair_continues_the_same_stream",
        |client| async move {
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add", "sum"],
                None,
            );
            let machine_names = policy(&["sum"]);

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
                            |invalid| {
                                assert_eq!(invalid.tool_call.function.name, "add");
                                InvalidToolCallAction::repair("sum")
                            },
                            &mut streamed_text,
                        )
                        .await
                        .expect("the repaired turn should be accepted");
                        assert!(matches!(end, TurnEnd::Finished));
                    }
                    AgentRunStep::CallTools { calls } => {
                        for call in &calls {
                            assert!(
                                matches!(call, PendingToolCall::Execute(call) if call.name() == "sum"),
                                "the repaired name must reach the driver"
                            );
                            repaired = true;
                        }
                        run.answer_all(execute_pending_calls(calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert!(repaired, "the model should call a tool that gets repaired");
            assert_mentions_expected_number(&response.output(), 5);
            let messages = response.messages;
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
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add"],
                None,
            );
            let executable = policy(&["add"]);
            // The restricted first turn advertises no tool to the machine.
            let nothing_allowed = policy(&[]);
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
                        let cursor = run.messages().len();
                        let end = run_streamed_turn(
                            &agent,
                            &mut run,
                            prompt,
                            history,
                            allowed,
                            |invalid| {
                                assert!(expect_abandon, "only the first turn restricts tools");
                                assert_eq!(invalid.tool_call.function.name, "add");
                                InvalidToolCallAction::skip(SKIP_REASON)
                            },
                            &mut streamed_text,
                        )
                        .await
                        .expect("the streamed turn should be accepted");
                        match end {
                            TurnEnd::Abandoned => {
                                assert!(expect_abandon, "only the first turn should abandon");
                                // A host streams the skip's answer as the
                                // projection of what the run committed.
                                let tool_result = project(&run.messages()[cursor..])
                                    .find_map(|item| match item {
                                        CommittedItem::ToolResult(result) => Some(result),
                                        _ => None,
                                    })
                                    .expect("a skipped call commits its synthetic tool result");
                                // Gemini's wire supplies no tool-call id, and
                                // rig no longer fabricates one from the tool
                                // name — the synthetic result answers the
                                // call's minted correlation handle and
                                // records no provider-issued id.
                                assert!(tool_result.call.is_local());
                                assert!(tool_result.call.provider().is_none());
                                abandoned = true;
                            }
                            TurnEnd::Finished => {
                                assert!(!expect_abandon, "the first turn must abandon");
                            }
                        }
                    }
                    AgentRunStep::CallTools { calls } => {
                        run.answer_all(execute_pending_calls(calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert!(abandoned, "the restricted first turn should be abandoned");
            assert_nonempty_response(&response.output());
            assert!(
                run.completion_calls().len() >= 2,
                "the abandoned turn still records its completion call"
            );
        },
    )
    .await;
}

#[derive(Clone)]
struct CancelOnToolCall;

impl AgentHook for CancelOnToolCall {
    async fn on_dispatch(
        &self,
        _ctx: &rig::agent::HookContext,
        event: DispatchEvent<'_>,
    ) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        DispatchAction::stop("cancelled by test hook")
    }
}

#[tokio::test]
async fn builtin_streaming_cancellation_history_includes_assistant_turn() {
    with_gemini_cassette(
        "agent_run_streamed/builtin_streaming_cancellation_history_includes_assistant_turn",
        |client| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble(FORCE_TOOLS_PREAMBLE)
                    .tool(Add)
                    .tool_choice(ToolChoice::Required)
                    .build();

            let mut stream = agent
                .prompt("What is 21 + 21? Use the add tool.")
                .add_hook(CancelOnToolCall)
                .max_turns(2)
                .stream();

            let mut prompt_error = None;
            let mut saw_final = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final = true,
                    Ok(_) => {}
                    Err(error) => {
                        prompt_error = Some(error);
                        break;
                    }
                }
            }
            assert!(
                !saw_final,
                "a cancelled run must not produce a final response"
            );

            let error = prompt_error.expect("the hook should cancel the run");
            validate_cancelled_failure(&error, "cancelled by test hook", "add")
                .expect("portable cancellation diagnostics should hold");
            let PromptError::Cancelled {
                chat_history,
                reason,
            } = error
            else {
                panic!("expected Cancelled");
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
