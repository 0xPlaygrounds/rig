//! Cassette coverage for `AgentDriver` against real Anthropic traffic.
//!
//! The driver is provider-agnostic; the request it builds is not. Anthropic
//! spells `tool_choice`, tool declarations and the system prompt differently
//! from OpenAI, so the same driver invariants need their own recordings — a
//! per-turn patch that reaches an OpenAI request could silently fail to reach
//! an Anthropic one, and only a recorded body would show it.
//!
//! See `tests/providers/openai/cassette/agent_driver.rs` for the full
//! rationale; this module carries the provider-portable core of that suite.

use futures::StreamExt;
use rig::agent::run::{OutputMode, StreamedTurnAssembler};
use rig::agent::{AgentRun, RequestPatch};
use rig::completion::PromptError;
use rig::message::{Message, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;

use super::super::support::with_anthropic_cassette;
use crate::driver_support::{
    ADD_PROMPT, FORCE_TOOLS_PREAMBLE, dispatch_and_feed, drive_to_completion, expect_done,
    expect_execute_tools, expect_send, expect_send_patched, expect_turn_accepted,
};
use crate::support::{Adder, Subtract};

const MODEL: &str = anthropic::completion::CLAUDE_SONNET_4_6;

/// The whole hand-driven loop over real Anthropic traffic, locking both
/// request bodies.
#[tokio::test]
async fn drive_loop_round_trips_a_tool_call() {
    with_anthropic_cassette("agent_driver/tool_call_round_trip", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, tools, turn) = expect_send(&mut driver).await;
        assert_eq!(turn, 1);
        assert!(tools.executable_tool_names().contains("add"));

        let response = request.send().await.expect("first turn should send");
        expect_turn_accepted(&mut driver, &response);

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        assert!(!pending.is_empty(), "the model should have called the tool");
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let response = drive_to_completion(&mut driver)
            .await
            .expect("run should finish");
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// A custom run's `tool_choice` must reach Anthropic, whose wire spelling
/// differs from OpenAI's.
#[tokio::test]
async fn a_custom_runs_tool_choice_reaches_the_provider() {
    with_anthropic_cassette("agent_driver/run_tool_choice", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .tool(Adder)
            .build();

        let run = AgentRun::new(ADD_PROMPT)
            .max_turns(2)
            .with_tool_choice(ToolChoice::Required);
        let mut driver = agent.drive_run(run);

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);

        let (pending, _) = expect_execute_tools(&mut driver).await;
        assert!(
            !pending.is_empty(),
            "tool_choice=required must force a call"
        );
    })
    .await;
}

/// `ToolChoice::None` forbids tool use for the turn: the run finalizes
/// without ever reaching a tool step.
///
/// Deliberately *not* asserting a non-empty answer. Recording this against
/// Anthropic showed the model returning a genuinely empty turn — forbidden
/// from the only tool that would answer an arithmetic prompt, it emitted no
/// content at all. That is honest provider behavior and `is_empty_assistant_turn`
/// handles it; the invariant under test is that no tool call happened, not
/// that the model had something to say.
#[tokio::test]
async fn tool_choice_none_forbids_tools_on_the_wire() {
    with_anthropic_cassette("agent_driver/tool_choice_none", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble("Answer in plain text.")
            .tool(Adder)
            .build();

        let run = AgentRun::new(ADD_PROMPT)
            .max_turns(2)
            .with_tool_choice(ToolChoice::None);
        let mut driver = agent.drive_run(run);

        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(
            tools.allowed_tool_names().is_empty(),
            "ToolChoice::None allows nothing to be called"
        );
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);

        // Straight to Done: a tool step here would mean the constraint did not
        // reach the provider.
        let response = expect_done(&mut driver).await;
        assert!(
            response
                .content
                .iter()
                .all(|item| !matches!(item, rig::message::AssistantContent::ToolCall(_))),
            "no tool call may survive ToolChoice::None"
        );
    })
    .await;
}

/// `ToolChoice::Specific` names one tool on the Anthropic wire.
#[tokio::test]
async fn tool_choice_specific_names_the_tool_on_the_wire() {
    with_anthropic_cassette("agent_driver/tool_choice_specific", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let run = AgentRun::new(ADD_PROMPT)
            .max_turns(2)
            .with_tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            });
        let mut driver = agent.drive_run(run);

        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(tools.allowed_tool_names().contains("add"));
        assert!(!tools.allowed_tool_names().contains("subtract"));

        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let (pending, _) = expect_execute_tools(&mut driver).await;
        assert_eq!(pending[0].tool_call.function.name, "add");
    })
    .await;
}

/// A patched preamble replaces the agent's system prompt for the turn.
#[tokio::test]
async fn a_patched_preamble_replaces_the_agents_on_the_wire() {
    with_anthropic_cassette("agent_driver/patch_preamble", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble("BASELINE PREAMBLE — must not appear in the request")
            .build();

        let mut driver = agent.drive("Say the word banana.");

        let (request, _, _) = expect_send_patched(
            &mut driver,
            RequestPatch::new().preamble("PATCHED PREAMBLE — reply with one word."),
        )
        .await;
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// `active_tools` narrows the tools Anthropic is shown.
#[tokio::test]
async fn a_patched_active_tools_narrows_the_advertised_set() {
    with_anthropic_cassette("agent_driver/patch_active_tools", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);

        let (request, tools, _) =
            expect_send_patched(&mut driver, RequestPatch::new().active_tools(["add"])).await;
        assert!(tools.executable_tool_names().contains("add"));
        assert!(!tools.executable_tool_names().contains("subtract"));

        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let (pending, tools) = expect_execute_tools(&mut driver).await;
        dispatch_and_feed(&mut driver, &pending, &tools).await;
    })
    .await;
}

/// An explicit patch outranks the run's own choice.
#[tokio::test]
async fn a_patched_tool_choice_outranks_the_runs() {
    with_anthropic_cassette("agent_driver/patch_tool_choice", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .tool(Adder)
            .build();

        let run = AgentRun::new(ADD_PROMPT)
            .max_turns(2)
            .with_tool_choice(ToolChoice::None);
        let mut driver = agent.drive_run(run);

        let (request, tools, _) = expect_send_patched(
            &mut driver,
            RequestPatch::new().tool_choice(ToolChoice::Required),
        )
        .await;
        assert!(tools.allowed_tool_names().contains("add"));
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let (pending, _) = expect_execute_tools(&mut driver).await;
        assert!(!pending.is_empty());
    })
    .await;
}

/// A driver-level history leads the Anthropic request.
#[tokio::test]
async fn driver_history_leads_the_request() {
    with_anthropic_cassette("agent_driver/driver_history", |client| async move {
        let agent = client.agent(MODEL).preamble("Answer briefly.").build();

        let mut driver = agent.drive("What is my name?").history(vec![
            Message::user("My name is Ada."),
            Message::assistant("Nice to meet you, Ada."),
        ]);

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// A run suspended mid-model-call resumes in a fresh driver.
#[tokio::test]
async fn a_run_suspended_awaiting_the_model_resumes_and_accepts_the_reply() {
    with_anthropic_cassette("agent_driver/resume_awaiting_model", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(2)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, _, _) = expect_send(&mut driver).await;

        let serialized = serde_json::to_string(driver.run()).expect("run serializes");
        let response = request.send().await.expect("should send");
        drop(driver);

        let restored: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        assert!(restored.advertised_tools().is_some());
        let mut resumed = agent.drive_run(restored);
        expect_turn_accepted(&mut resumed, &response);

        let (pending, tools) = expect_execute_tools(&mut resumed).await;
        assert!(!pending.is_empty());
        assert!(tools.executable_tool_names().contains("add"));
    })
    .await;
}

/// A run suspended with tool calls pending resumes and completes; the second
/// request is built by the resumed driver.
#[tokio::test]
async fn a_run_suspended_executing_tools_resumes_and_completes() {
    with_anthropic_cassette("agent_driver/resume_executing_tools", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let _ = expect_execute_tools(&mut driver).await;

        let serialized = serde_json::to_string(driver.run()).expect("run serializes");
        drop(driver);

        let restored: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut resumed = agent.drive_run(restored);
        let (pending, tools) = expect_execute_tools(&mut resumed).await;
        dispatch_and_feed(&mut resumed, &pending, &tools).await;

        let response = drive_to_completion(&mut resumed)
            .await
            .expect("resumed run should finish");
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// Tool output mode against Anthropic: the synthetic output tool is advertised
/// and allowed, never executable, and the run finalizes on its call.
#[tokio::test]
async fn tool_output_mode_finalizes_via_the_output_tool() {
    with_anthropic_cassette("agent_driver/output_mode_tool", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble("Reply with the structured answer.")
            .output_schema_raw(
                serde_json::from_value(serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"]
                }))
                .expect("valid schema"),
            )
            .output_mode(OutputMode::Tool)
            .build();

        let mut driver = agent.drive("What is the capital of France?");
        let (request, tools, _) = expect_send(&mut driver).await;
        let output_tool = tools
            .output_tool_name()
            .expect("Tool mode advertises an output tool")
            .to_owned();
        assert!(tools.allowed_tool_names().contains(&output_tool));
        assert!(!tools.executable_tool_names().contains(&output_tool));

        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let response = expect_done(&mut driver).await;
        assert!(response.output.contains("answer"));
    })
    .await;
}

/// A streamed turn driven through the driver against real Anthropic SSE.
#[tokio::test]
async fn a_streamed_turn_drives_through_the_driver() {
    with_anthropic_cassette("agent_driver/streamed_turn", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, tools, _) = expect_send(&mut driver).await;

        let mut assembler = StreamedTurnAssembler::new(
            tools.executable_tool_names().clone(),
            tools.allowed_tool_names().clone(),
        );
        let mut stream = request.stream().await.expect("stream should open");
        while let Some(item) = stream.next().await {
            assembler
                .ingest(&item.expect("stream item"))
                .expect("ingest should succeed");
        }
        let final_content = stream.choice.clone();
        let streamed = assembler.finish(stream.message_id.clone(), &final_content);

        driver
            .run_mut()
            .record_streamed_completion_call(stream.usage())
            .expect("usage recorded");
        driver
            .run_mut()
            .streamed_turn(streamed)
            .expect("streamed turn accepted");

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        assert!(!pending.is_empty());
        dispatch_and_feed(&mut driver, &pending, &tools).await;
    })
    .await;
}

/// A rejected send can be handed back and the turn prepared again.
#[tokio::test]
async fn a_provider_rejection_is_not_retryable() {
    with_anthropic_cassette("agent_driver/provider_rejection", |client| async move {
        // Anthropic requires `max_tokens`; without it the request fails
        // locally and never reaches the provider, which is a different
        // assertion than the one this test is making.
        let agent = client
            .agent("claude-this-model-does-not-exist")
            .preamble(FORCE_TOOLS_PREAMBLE)
            .max_tokens(64)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, _, _) = expect_send(&mut driver).await;
        let error = request
            .send()
            .await
            .expect_err("an unknown model must be rejected");

        // Pin the provider's own envelope: a cassette mock miss is a 404 too.
        let body = error
            .provider_response_body()
            .expect("the provider's rejection body is preserved");
        assert!(
            body.contains("not_found") || body.contains("model"),
            "expected the recorded provider rejection, got: {body}"
        );
        assert!(
            !error.is_retryable(),
            "a rejection is not retryable: {error}"
        );
    })
    .await;
}

/// Exhausting the model-call budget stops the run before a second request.
#[tokio::test]
async fn max_turns_exhaustion_stops_before_a_second_send() {
    with_anthropic_cassette("agent_driver/max_turns", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT).max_turns(1);
        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let error = driver
            .next_step()
            .await
            .expect_err("the budget of one is spent");
        assert!(matches!(error, PromptError::MaxTurnsError { .. }));
    })
    .await;
}
