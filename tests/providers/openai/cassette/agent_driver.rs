//! Cassette coverage for `AgentDriver` against real OpenAI traffic.
//!
//! The driver's job is to build a request and pair it with the tool state that
//! validates and dispatches the model's reply. Unit tests can check the state
//! machine; only recorded traffic can check the *request*. Every request-shape
//! defect found in this API's review (a run's `tool_choice` never reaching the
//! wire, a resumed turn advertising the wrong tool set) was invisible to unit
//! tests precisely because those tests asserted on rig's own view of the
//! request rather than on the bytes.
//!
//! The harness matches each request body against the recorded one, so a
//! request-shape regression fails as a mock miss with a body diff. **That is
//! the assertion these tests exist for**; the response-side asserts are
//! secondary and deliberately structural, since model wording varies between
//! recordings.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::run::{OutputMode, StreamedTurnAssembler};
use rig::agent::{AgentRun, RequestPatch};
use rig::completion::PromptError;
use rig::message::{Message, ToolChoice};
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::{Tool, ToolContext};

use super::super::support::with_openai_completions_cassette;
use crate::driver_support::{
    ADD_PROMPT, FORCE_TOOLS_PREAMBLE, dispatch_and_feed, drive_to_completion, expect_done,
    expect_execute_tools, expect_send,
};
use crate::support::{Adder, Subtract};

/// A counting `add` so tests can assert the tool ran exactly once.
#[derive(Clone)]
struct CountingAdder {
    calls: Arc<AtomicUsize>,
}

#[derive(serde::Deserialize)]
struct AddArgs {
    x: i32,
    y: i32,
}

impl Tool for CountingAdder {
    const NAME: &'static str = "add";
    type Error = std::io::Error;
    type Args = AddArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "x": { "type": "number" }, "y": { "type": "number" } },
            "required": ["x", "y"]
        })
    }

    fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> impl std::future::Future<Output = Result<Self::Output, Self::Error>> + Send {
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(args.x + args.y))
    }
}

// ── Tranche 1: the loop ──────────────────────────────────────────────────

/// The whole hand-driven loop over real traffic: the driver builds both
/// requests, the caller sends them, and the turn that advertised the tool is
/// the turn that dispatches it.
///
/// Locks both request bodies. The second in particular carries the assistant
/// tool call and the tool result the driver threaded back through the run — a
/// shape no unit test observes.
#[tokio::test]
async fn drive_loop_round_trips_a_tool_call() {
    with_openai_completions_cassette("agent_driver/tool_call_round_trip", |client| async move {
        let calls = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(CountingAdder {
                calls: calls.clone(),
            })
            .build();

        let mut driver = agent.drive(ADD_PROMPT);

        let (request, tools, turn) = expect_send(&mut driver).await;
        assert_eq!(turn, 1);
        assert!(tools.executable_tool_names().contains("add"));

        let response = request.send().await.expect("first turn should send");
        driver.model_response(&response).expect("turn accepted");

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        assert!(!pending.is_empty(), "the model should have called the tool");
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let (request, _, turn) = expect_send(&mut driver).await;
        assert_eq!(turn, 2);
        let response = request.send().await.expect("second turn should send");
        driver.model_response(&response).expect("turn accepted");

        let response = expect_done(&mut driver).await;
        assert!(
            !response.output.trim().is_empty(),
            "expected a final answer"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the tool ran exactly once");
    })
    .await;
}

/// Two tools advertised, one called: the request carries both, so a narrowing
/// regression shows up as a body diff.
#[tokio::test]
async fn both_registered_tools_are_advertised() {
    with_openai_completions_cassette("agent_driver/two_tools_advertised", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(tools.executable_tool_names().contains("add"));
        assert!(tools.executable_tool_names().contains("subtract"));

        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let (pending, tools) = expect_execute_tools(&mut driver).await;
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let response = drive_to_completion(&mut driver)
            .await
            .expect("run should finish");
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// A prompt needing two independent tool calls: every pending call dispatches
/// through the same advertising turn, and the request that follows carries all
/// their results.
#[tokio::test]
async fn parallel_tool_calls_all_dispatch_through_one_turn() {
    with_openai_completions_cassette("agent_driver/parallel_tool_calls", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(4)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let mut driver = agent.drive("Compute 2 + 5 and 9 - 3. Use the tools for both.");
        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        assert!(!pending.is_empty());
        // Whatever the model asked for, every call dispatches through the turn
        // that advertised it.
        for call in &pending {
            assert!(
                tools
                    .executable_tool_names()
                    .contains(&call.tool_call.function.name),
                "the model called a tool this turn never advertised"
            );
        }
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let response = drive_to_completion(&mut driver)
            .await
            .expect("run should finish");
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

// ── Tranche 2: per-run and per-turn configuration on the wire ────────────

/// A custom run is taken as-is, so its own `tool_choice` must reach the
/// provider — not merely the run's internal decisions.
///
/// The finding a unit test could only assert against rig's own
/// `CompletionRequest`. Reverting the fix makes this fail as a mock miss whose
/// diff names the missing `tool_choice`.
#[tokio::test]
async fn a_custom_runs_tool_choice_reaches_the_provider() {
    with_openai_completions_cassette("agent_driver/run_tool_choice", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .tool(Adder)
            .build();

        let run = AgentRun::new(ADD_PROMPT)
            .max_turns(2)
            .with_tool_choice(ToolChoice::Required);
        let mut driver = agent.drive_run(run);

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");

        let (pending, _) = expect_execute_tools(&mut driver).await;
        assert!(
            !pending.is_empty(),
            "tool_choice=required must force a call"
        );
    })
    .await;
}

/// `ToolChoice::None` forbids tool use for the turn: the request carries the
/// constraint and the advertised-but-not-allowed set is empty.
#[tokio::test]
async fn tool_choice_none_forbids_tools_on_the_wire() {
    with_openai_completions_cassette("agent_driver/tool_choice_none", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
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
        driver.model_response(&response).expect("turn accepted");
        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// `ToolChoice::Specific` names one tool; the request must carry that name.
#[tokio::test]
async fn tool_choice_specific_names_the_tool_on_the_wire() {
    with_openai_completions_cassette("agent_driver/tool_choice_specific", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
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
        assert!(
            !tools.allowed_tool_names().contains("subtract"),
            "Specific narrows what the model may call"
        );

        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let (pending, _) = expect_execute_tools(&mut driver).await;
        assert_eq!(pending[0].tool_call.function.name, "add");
    })
    .await;
}

/// The driver runs no hooks, so `RequestPatch` is the seam through which a
/// hand-driven turn gets per-turn configuration. Nothing else exercises it, so
/// each field that reaches the request gets a recorded body.
#[tokio::test]
async fn a_patched_preamble_replaces_the_agents_on_the_wire() {
    with_openai_completions_cassette("agent_driver/patch_preamble", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("BASELINE PREAMBLE — must not appear in the request")
            .build();

        let mut driver = agent
            .drive("Say the word banana.")
            .request_patch(RequestPatch::new().preamble("PATCHED PREAMBLE — reply with one word."));

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// An explicit patch outranks the run's own choice.
#[tokio::test]
async fn a_patched_tool_choice_outranks_the_runs() {
    with_openai_completions_cassette("agent_driver/patch_tool_choice", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .tool(Adder)
            .build();

        // The run says None; the patch says Required. The patch wins, so the
        // recorded request carries `required`.
        let run = AgentRun::new(ADD_PROMPT)
            .max_turns(2)
            .with_tool_choice(ToolChoice::None);
        let mut driver = agent
            .drive_run(run)
            .request_patch(RequestPatch::new().tool_choice(ToolChoice::Required));

        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(
            tools.allowed_tool_names().contains("add"),
            "the patch's Required must govern, not the run's None"
        );
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let (pending, _) = expect_execute_tools(&mut driver).await;
        assert!(!pending.is_empty());
    })
    .await;
}

/// `active_tools` narrows the advertised set for the turn, so the request's
/// `tools` array shrinks.
#[tokio::test]
async fn a_patched_active_tools_narrows_the_advertised_set() {
    with_openai_completions_cassette("agent_driver/patch_active_tools", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let mut driver = agent
            .drive(ADD_PROMPT)
            .request_patch(RequestPatch::new().active_tools(["add"]));

        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(tools.executable_tool_names().contains("add"));
        assert!(
            !tools.executable_tool_names().contains("subtract"),
            "active_tools must narrow the advertised set: {:?}",
            tools.executable_tool_names()
        );

        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let (pending, tools) = expect_execute_tools(&mut driver).await;
        dispatch_and_feed(&mut driver, &pending, &tools).await;
        let response = drive_to_completion(&mut driver)
            .await
            .expect("run should finish");
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// Sampling parameters are per-turn request fields; the recorded body pins
/// them.
#[tokio::test]
async fn patched_sampling_parameters_reach_the_request() {
    with_openai_completions_cassette("agent_driver/patch_sampling", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Reply with one word.")
            .temperature(0.9)
            .build();

        let mut driver = agent
            .drive("Say the word banana.")
            .request_patch(RequestPatch::new().temperature(0.0).max_tokens(16));

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let _ = expect_done(&mut driver).await;
    })
    .await;
}

/// Extra context documents are appended to the turn's request.
#[tokio::test]
async fn patched_extra_context_reaches_the_request() {
    with_openai_completions_cassette("agent_driver/patch_extra_context", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Answer using the provided context only.")
            .build();

        let document = rig::completion::Document {
            id: "note-1".to_string(),
            text: "The launch code is banana.".to_string(),
            additional_props: Default::default(),
        };
        let mut driver = agent
            .drive("What is the launch code?")
            .request_patch(RequestPatch::new().extra_context(vec![document]));

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// A patched history replaces the run's for the turn.
#[tokio::test]
async fn a_patched_history_replaces_the_runs_for_the_turn() {
    with_openai_completions_cassette("agent_driver/patch_history", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Answer briefly.")
            .build();

        let mut driver =
            agent
                .drive("What did I just say?")
                .request_patch(RequestPatch::new().history(vec![
                    Message::user("Remember this: the code word is banana."),
                    Message::assistant("Noted."),
                ]));

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// A driver-level history seeds the run and leads the request.
#[tokio::test]
async fn driver_history_leads_the_request() {
    with_openai_completions_cassette("agent_driver/driver_history", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Answer briefly.")
            .build();

        let mut driver = agent.drive("What is my name?").history(vec![
            Message::user("My name is Ada."),
            Message::assistant("Nice to meet you, Ada."),
        ]);

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

// ── Tranche 3: suspend, resume, drift ────────────────────────────────────

/// A run suspended with a model call in flight resumes in a different driver
/// and accepts the reply.
///
/// The request is recorded once; the resumed driver never builds one. Under
/// test is that the turn's advertised names travelled with the run, so the
/// reply is validated against the set the recorded request actually carried.
#[tokio::test]
async fn a_run_suspended_awaiting_the_model_resumes_and_accepts_the_reply() {
    with_openai_completions_cassette("agent_driver/resume_awaiting_model", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
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
        assert!(
            restored.advertised_tools().is_some(),
            "a suspended run carries the turn's advertised names"
        );
        let mut resumed = agent.drive_run(restored);
        resumed
            .model_response(&response)
            .expect("a resumed run accepts the reply to its in-flight call");

        let (pending, tools) = expect_execute_tools(&mut resumed).await;
        assert!(!pending.is_empty());
        assert!(tools.executable_tool_names().contains("add"));
    })
    .await;
}

/// A run suspended with tool calls pending resumes and completes, and the
/// second request is built by the *resumed* driver — so its recorded body is
/// the assertion that resume rebuilds the request faithfully.
#[tokio::test]
async fn a_run_suspended_executing_tools_resumes_and_completes() {
    with_openai_completions_cassette("agent_driver/resume_executing_tools", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
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

/// The resumed turn advertises what *that turn* advertised, not what the
/// resuming process happens to have registered.
///
/// The resuming agent registers an extra tool. If the resumed dispatch target
/// were the current registry rather than the turn's recorded set, the extra
/// tool would leak into a turn that never advertised it.
#[tokio::test]
async fn a_resumed_turn_advertises_its_own_tools_not_the_processs() {
    with_openai_completions_cassette("agent_driver/resume_no_tool_leak", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let _ = expect_execute_tools(&mut driver).await;
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");
        let advertised = driver
            .run()
            .advertised_tools()
            .expect("the turn recorded its names")
            .clone();
        drop(driver);

        // The resuming process has since registered `subtract`.
        let resumed_agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .tool(Subtract)
            .build();
        let restored: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut resumed = resumed_agent.drive_run(restored);

        let (pending, tools) = expect_execute_tools(&mut resumed).await;
        assert_eq!(
            tools.executable_tool_names(),
            &advertised.executable,
            "the resumed turn's tool set is the turn's, not the process's"
        );
        assert!(
            !tools.executable_tool_names().contains("subtract"),
            "a tool registered after suspension must not join this turn"
        );
        dispatch_and_feed(&mut resumed, &pending, &tools).await;
    })
    .await;
}

// ── Tranche 4: failure and recovery ──────────────────────────────────────

/// A real provider rejection is classified non-retryable, so the caller does
/// not hand the turn back and loop.
///
/// A synthetic error cannot falsify the classification; only a real provider
/// rejection can.
#[tokio::test]
async fn a_provider_rejection_is_not_retryable() {
    with_openai_completions_cassette("agent_driver/provider_rejection", |client| async move {
        let agent = client
            .agent("gpt-4o-this-model-does-not-exist")
            .preamble(FORCE_TOOLS_PREAMBLE)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, _, _) = expect_send(&mut driver).await;

        let error = request
            .send()
            .await
            .expect_err("an unknown model must be rejected");

        // A cassette mock miss is *also* a 404, so asserting on the status
        // alone would pass against a cassette that never matched. Pin the
        // provider's own envelope, which the harness cannot fabricate.
        let body = error
            .provider_response_body()
            .expect("the provider's rejection body is preserved");
        assert!(
            body.contains("model_not_found"),
            "expected the recorded provider rejection, got: {body}"
        );
        assert!(
            !error.is_retryable(),
            "a provider rejection must not be classified retryable: {error}"
        );

        assert_eq!(driver.run().turn(), 1, "the turn is still in flight");
        assert_eq!(driver.run().model_call_rollbacks(), 0);
    })
    .await;
}

/// A rejected send can be handed back and the turn prepared again.
///
/// Two recorded interactions: the rejection, then a fresh request. The second
/// body is the assertion — the retry must be a *new build*, not a replay of
/// the request that failed.
#[tokio::test]
async fn a_rejected_send_rolls_back_and_re_prepares() {
    with_openai_completions_cassette("agent_driver/rollback_re_prepares", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(Adder)
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, _, turn) = expect_send(&mut driver).await;
        assert_eq!(turn, 1);

        // Drop the request unsent: the provider never saw anything.
        drop(request);
        driver
            .rollback_model_call()
            .expect("a call that produced nothing can be handed back");
        assert_eq!(driver.run().turn(), 0, "the turn is refunded");
        assert_eq!(driver.run().model_call_rollbacks(), 1);

        // The retry takes the turn the failure did not, and its request is
        // freshly built.
        let (request, _, turn) = expect_send(&mut driver).await;
        assert_eq!(turn, 1);
        let response = request.send().await.expect("the retry should send");
        driver.model_response(&response).expect("turn accepted");

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        dispatch_and_feed(&mut driver, &pending, &tools).await;
        let response = drive_to_completion(&mut driver)
            .await
            .expect("run should finish");
        assert!(!response.output.trim().is_empty());
    })
    .await;
}

/// A preparation failure costs no turn and consumes no interaction.
///
/// The cassette's contribution is the *negative*: exactly one interaction is
/// recorded, and it is consumed only by the retry. A driver that advanced
/// before preparing would burn the turn and the cassette would still have an
/// unconsumed interaction at teardown.
#[tokio::test]
async fn a_preparation_failure_costs_no_turn_and_no_interaction() {
    with_openai_completions_cassette("agent_driver/prepare_failure", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(2)
            .tool(Adder)
            .build();

        // An `active_tools` allow-list naming a tool this turn does not have:
        // preparation fails locally, with no provider round trip.
        let mut driver = agent
            .drive(ADD_PROMPT)
            .request_patch(RequestPatch::new().active_tools(["nonexistent_tool"]));

        let error = driver
            .next_step()
            .await
            .expect_err("active_tools naming an unavailable tool must fail at prepare time");
        assert!(matches!(error, PromptError::CompletionError(_)));
        assert_eq!(
            driver.run().turn(),
            0,
            "a request that never left the process must not consume a turn"
        );

        // Fix the cause and drive the very same step again.
        driver.set_request_patch(RequestPatch::new());
        let (request, tools, turn) = expect_send(&mut driver).await;
        assert_eq!(turn, 1, "the retry takes the turn the failure did not");
        assert!(tools.executable_tool_names().contains("add"));
        let response = request.send().await.expect("the retry should send");
        driver.model_response(&response).expect("turn accepted");
    })
    .await;
}

/// Exhausting the model-call budget stops the run before it builds a second
/// request — so the cassette holds exactly one interaction.
#[tokio::test]
async fn max_turns_exhaustion_stops_before_a_second_send() {
    with_openai_completions_cassette("agent_driver/max_turns", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .tool(Adder)
            .build();

        // One model call only; the tool call cannot be answered.
        let mut driver = agent.drive(ADD_PROMPT).max_turns(1);
        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let error = driver
            .next_step()
            .await
            .expect_err("the budget of one is spent");
        assert!(
            matches!(error, PromptError::MaxTurnsError { .. }),
            "expected MaxTurnsError, got {error:?}"
        );
    })
    .await;
}

// ── Tranche 5: output modes ──────────────────────────────────────────────

/// Tool output mode advertises a synthetic output tool alongside the real
/// ones, and the run finalizes on its call rather than dispatching it.
#[tokio::test]
async fn tool_output_mode_finalizes_via_the_output_tool() {
    with_openai_completions_cassette("agent_driver/output_mode_tool", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
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
        assert!(
            tools.allowed_tool_names().contains(&output_tool),
            "the output tool is allowed"
        );
        assert!(
            !tools.executable_tool_names().contains(&output_tool),
            "the output tool is never executable"
        );

        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");

        // The output-tool call is intercepted by the run, never surfaced.
        let response = expect_done(&mut driver).await;
        assert!(
            response.output.contains("answer"),
            "expected structured output, got {}",
            response.output
        );
    })
    .await;
}

/// Native output mode sets the provider's own structured-output constraint, so
/// the request carries a response format rather than a synthetic tool.
#[tokio::test]
async fn native_output_mode_uses_the_provider_constraint() {
    with_openai_completions_cassette("agent_driver/output_mode_native", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Reply with the structured answer.")
            .output_schema_raw(
                serde_json::from_value(serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"],
                    "additionalProperties": false
                }))
                .expect("valid schema"),
            )
            .output_mode(OutputMode::Native)
            .build();

        let mut driver = agent.drive("What is the capital of France?");
        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(
            tools.output_tool_name().is_none(),
            "Native mode advertises no synthetic tool"
        );

        let response = request.send().await.expect("should send");
        driver.model_response(&response).expect("turn accepted");
        let response = expect_done(&mut driver).await;
        assert!(response.output.contains("answer"));
    })
    .await;
}

// ── Tranche 6: streaming ─────────────────────────────────────────────────

/// A hand-driven **streamed** turn goes through the driver, against real SSE.
///
/// The streamed entry points live on `AgentRun` and take `&mut self`, so this
/// is only expressible because the driver hands out `run_mut()`. Without it a
/// streaming caller has to `into_run()` and rebuild the driver, which discards
/// the per-turn snapshot cache and makes the driver treat a turn prepared in
/// this very process as a resume — drift check included.
#[tokio::test]
async fn a_streamed_turn_drives_through_the_driver() {
    with_openai_completions_cassette("agent_driver/streamed_turn", |client| async move {
        let calls = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .default_max_turns(3)
            .tool(CountingAdder {
                calls: calls.clone(),
            })
            .build();

        let mut driver = agent.drive(ADD_PROMPT);
        let (request, tools, _) = expect_send(&mut driver).await;

        let mut assembler = StreamedTurnAssembler::new(
            tools.executable_tool_names().clone(),
            tools.allowed_tool_names().clone(),
        );
        let mut stream = request.stream().await.expect("stream should open");
        while let Some(item) = stream.next().await {
            let item = item.expect("stream item");
            assembler.ingest(&item).expect("ingest should succeed");
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
        assert!(!pending.is_empty(), "the model should have called the tool");
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let response = drive_to_completion(&mut driver)
            .await
            .expect("run should finish");
        assert!(!response.output.trim().is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    })
    .await;
}

/// A streamed text-only turn: no tool calls, so the run finalizes straight
/// from the assembled stream.
#[tokio::test]
async fn a_streamed_text_turn_finalizes_the_run() {
    with_openai_completions_cassette("agent_driver/streamed_text", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Reply with one short sentence.")
            .build();

        let mut driver = agent.drive("Say hello.");
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

        let response = expect_done(&mut driver).await;
        assert!(!response.output.trim().is_empty());
        assert_eq!(
            driver.run().completion_calls().len(),
            1,
            "a streamed turn records exactly one completion call"
        );
    })
    .await;
}

/// A streamed turn under a `RequestPatch`: the patch reaches the streaming
/// request exactly as it reaches a blocking one.
#[tokio::test]
async fn a_streamed_turn_honors_the_request_patch() {
    with_openai_completions_cassette("agent_driver/streamed_patched", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("BASELINE — must not appear")
            .default_max_turns(2)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let mut driver = agent.drive(ADD_PROMPT).request_patch(
            RequestPatch::new()
                .preamble(FORCE_TOOLS_PREAMBLE)
                .active_tools(["add"]),
        );

        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(!tools.executable_tool_names().contains("subtract"));

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
