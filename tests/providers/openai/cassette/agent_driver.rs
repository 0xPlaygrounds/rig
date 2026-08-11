//! Cassette coverage for `AgentDriver` against real OpenAI traffic.
//!
//! The driver's job is to build a request and pair it with the tool state that
//! validates and dispatches the model's reply. Unit tests can check the state
//! machine, but only recorded traffic can check the *request* — and every
//! request-shape defect found in this API's review so far (a run's
//! `tool_choice` never reaching the wire, a resumed turn advertising the wrong
//! tool set) was invisible to unit tests precisely because they asserted on
//! rig's own view of the request rather than on the bytes.
//!
//! The cassette harness matches each request body against the recorded one, so
//! a request-shape regression fails as a mock miss. That is the assertion these
//! tests exist for; the response-side asserts are secondary.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::run::StreamedTurnAssembler;
use rig::agent::{AgentRun, DriveStep};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::{Tool, ToolContext};

use super::super::support::with_openai_completions_cassette;

#[derive(Debug, serde::Deserialize)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone)]
struct WeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl Tool for WeatherTool {
    const NAME: &'static str = "weather";
    type Error = std::io::Error;
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a city.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }

    fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> impl std::future::Future<Output = Result<Self::Output, Self::Error>> + Send {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(format!("The weather in {} is 22C and sunny", args.city)))
    }
}

/// The whole hand-driven loop over real traffic: the driver builds both
/// requests, the caller sends them, and the turn that advertised the tool is
/// the turn that dispatches it.
///
/// Locks the request bodies for both turns. The second request in particular
/// carries the assistant tool call and the tool result the driver threaded back
/// through the run — a shape no unit test observes.
#[tokio::test]
async fn drive_loop_round_trips_a_tool_call() {
    with_openai_completions_cassette("agent_driver/tool_call_round_trip", |client| async move {
        let calls = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("You are a weather assistant. Always use the weather tool.")
            .default_max_turns(3)
            .tool(WeatherTool {
                call_count: calls.clone(),
            })
            .build();

        let mut driver = agent.drive("What is the weather in Tokyo?");

        let (request, tools, turn) = match driver.next_step().await.expect("first step") {
            DriveStep::SendRequest {
                request,
                tools,
                turn,
            } => (request, tools, turn),
            other => panic!("expected SendRequest, got {other:?}"),
        };
        assert_eq!(turn, 1);
        assert!(tools.executable_tool_names().contains("weather"));

        let response = request.send().await.expect("first turn should send");
        driver.model_response(&response).expect("turn accepted");

        let (pending, tools) = match driver.next_step().await.expect("second step") {
            DriveStep::ExecuteTools { calls, tools } => (calls, tools),
            other => panic!("expected ExecuteTools, got {other:?}"),
        };
        assert!(!pending.is_empty(), "the model should have called the tool");

        let mut context = ToolContext::new();
        let mut results = Vec::new();
        for call in &pending {
            results.push(tools.execute_call(call, &mut context).await);
        }
        driver.tool_results(results).expect("results accepted");

        let request = match driver.next_step().await.expect("third step") {
            DriveStep::SendRequest { request, turn, .. } => {
                assert_eq!(turn, 2);
                request
            }
            other => panic!("expected SendRequest, got {other:?}"),
        };
        let response = request.send().await.expect("second turn should send");
        driver.model_response(&response).expect("turn accepted");

        match driver.next_step().await.expect("final step") {
            DriveStep::Done(response) => {
                assert!(
                    !response.output.trim().is_empty(),
                    "expected a final answer"
                );
            }
            other => panic!("expected Done, got {other:?}"),
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the tool ran exactly once");
    })
    .await;
}

/// A custom run is taken as-is, so its own `tool_choice` must reach the
/// provider — not merely the run's internal decisions.
///
/// This is the finding a unit test could assert only against rig's own
/// `CompletionRequest`. Here the recorded body carries `"tool_choice"`, so a
/// driver that drops it fails as a mock miss: the request rig builds no longer
/// matches the request the provider was actually asked.
#[tokio::test]
async fn a_custom_runs_tool_choice_reaches_the_provider() {
    with_openai_completions_cassette("agent_driver/run_tool_choice", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("You are a weather assistant.")
            .tool(WeatherTool {
                call_count: Arc::new(AtomicUsize::new(0)),
            })
            .build();

        // The choice lives on the run, not on the agent.
        let run = AgentRun::new("What is the weather in Tokyo?")
            .max_turns(2)
            .with_tool_choice(ToolChoice::Required);
        let mut driver = agent.drive_run(run);

        let request = match driver.next_step().await.expect("first step") {
            DriveStep::SendRequest { request, .. } => request,
            other => panic!("expected SendRequest, got {other:?}"),
        };

        let response = request.send().await.expect("request should send");
        driver.model_response(&response).expect("turn accepted");

        // `Required` obliges a tool call, so the run must reach tools.
        match driver.next_step().await.expect("second step") {
            DriveStep::ExecuteTools { calls, .. } => {
                assert!(!calls.is_empty(), "tool_choice=required must force a call");
            }
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
    })
    .await;
}

/// A run suspended with a model call in flight resumes in a different driver
/// and accepts the reply.
///
/// The request is recorded once; the resumed driver never builds one. What is
/// under test is that the turn's advertised tool names travelled with the run,
/// so the reply is validated against the set the recorded request actually
/// carried.
#[tokio::test]
async fn a_run_suspended_awaiting_the_model_resumes_and_accepts_the_reply() {
    with_openai_completions_cassette("agent_driver/resume_awaiting_model", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("You are a weather assistant. Always use the weather tool.")
            .default_max_turns(2)
            .tool(WeatherTool {
                call_count: Arc::new(AtomicUsize::new(0)),
            })
            .build();

        let mut driver = agent.drive("What is the weather in Tokyo?");
        let request = match driver.next_step().await.expect("first step") {
            DriveStep::SendRequest { request, .. } => request,
            other => panic!("expected SendRequest, got {other:?}"),
        };

        // Suspend with the call in flight, then send.
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");
        let response = request.send().await.expect("request should send");
        drop(driver);

        let restored: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        assert!(
            restored.advertised_tools().is_some(),
            "the suspended run carries the turn's advertised names"
        );
        let mut resumed = agent.drive_run(restored);
        resumed
            .model_response(&response)
            .expect("a resumed run accepts the reply to its in-flight call");

        match resumed.next_step().await.expect("second step") {
            DriveStep::ExecuteTools { calls, tools } => {
                assert!(!calls.is_empty());
                assert!(tools.executable_tool_names().contains("weather"));
            }
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
    })
    .await;
}

/// A hand-driven **streamed** turn goes through the driver, against real SSE
/// traffic.
///
/// The streamed entry points live on `AgentRun` and take `&mut self`, so this
/// is only expressible because the driver hands out `run_mut()`. Without it a
/// streaming caller has to `into_run()` and rebuild the driver, which discards
/// the per-turn snapshot cache and makes the driver treat a turn prepared in
/// this very process as a resume — drift check included. The assertion that
/// matters here is the second request body: the streamed turn must thread back
/// into the run exactly as a blocking one does.
#[tokio::test]
async fn a_streamed_turn_drives_through_the_driver() {
    with_openai_completions_cassette("agent_driver/streamed_turn", |client| async move {
        let calls = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("You are a weather assistant. Always use the weather tool.")
            .default_max_turns(3)
            .tool(WeatherTool {
                call_count: calls.clone(),
            })
            .build();

        let mut driver = agent.drive("What is the weather in Tokyo?");

        let (request, tools) = match driver.next_step().await.expect("first step") {
            DriveStep::SendRequest { request, tools, .. } => (request, tools),
            other => panic!("expected SendRequest, got {other:?}"),
        };

        // Stream the turn and assemble it with the names this turn advertised.
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

        // Feed usage and the assembled turn through the driver's run.
        driver
            .run_mut()
            .record_streamed_completion_call(stream.usage())
            .expect("usage recorded");
        driver
            .run_mut()
            .streamed_turn(streamed)
            .expect("streamed turn accepted");

        // The driver still owns the pairing: this turn's snapshot dispatches.
        let (pending, tools) = match driver.next_step().await.expect("second step") {
            DriveStep::ExecuteTools { calls, tools } => (calls, tools),
            other => panic!("expected ExecuteTools, got {other:?}"),
        };
        assert!(!pending.is_empty(), "the model should have called the tool");

        let mut context = ToolContext::new();
        let mut results = Vec::new();
        for call in &pending {
            results.push(tools.execute_call(call, &mut context).await);
        }
        driver.tool_results(results).expect("results accepted");

        let request = match driver.next_step().await.expect("third step") {
            DriveStep::SendRequest { request, .. } => request,
            other => panic!("expected SendRequest, got {other:?}"),
        };
        let response = request.send().await.expect("final turn should send");
        driver.model_response(&response).expect("turn accepted");

        match driver.next_step().await.expect("final step") {
            DriveStep::Done(response) => {
                assert!(!response.output.trim().is_empty());
            }
            other => panic!("expected Done, got {other:?}"),
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    })
    .await;
}

/// A real provider rejection is classified non-retryable, so the caller does
/// not hand the turn back and loop.
///
/// `is_retryable` is the trigger the driver's own docs recommend for
/// `rollback_model_call`, and misclassifying a deterministic failure as
/// retryable is an unbounded loop. A synthetic error cannot falsify the
/// classification — only a real provider rejection can, which is what this
/// records.
#[tokio::test]
async fn a_provider_rejection_is_not_retryable() {
    with_openai_completions_cassette("agent_driver/provider_rejection", |client| async move {
        let agent = client
            // A model name the provider will reject outright.
            .agent("gpt-4o-this-model-does-not-exist")
            .preamble("You are a weather assistant.")
            .build();

        let mut driver = agent.drive("What is the weather in Tokyo?");
        let request = match driver.next_step().await.expect("first step") {
            DriveStep::SendRequest { request, .. } => request,
            other => panic!("expected SendRequest, got {other:?}"),
        };

        let error = request
            .send()
            .await
            .expect_err("an unknown model must be rejected");

        // A cassette mock miss is *also* a 404, so asserting on the status
        // alone would pass on a cassette that never matched. Pin the
        // provider's own error envelope, which the harness cannot fabricate.
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

        // The turn stays in flight: the caller decides, and here the decision
        // is to fail rather than hand the turn back.
        assert_eq!(driver.run().turn(), 1);
        assert_eq!(driver.run().model_call_rollbacks(), 0);
    })
    .await;
}
