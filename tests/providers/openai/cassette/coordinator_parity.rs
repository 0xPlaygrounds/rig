//! Runner/driver parity, as a test rather than a review finding.
//!
//! `AgentRun` has two configured coordinators: `AgentRunner`'s internal loop and
//! the public `AgentDriver`. Both prepare requests, pair turns with tool-registry
//! snapshots and manage structured-output metadata. Nothing structural stops
//! them drifting, and review has repeatedly found places where they had.
//!
//! **One cassette, two coordinators.** Each scenario is recorded twice — once
//! per coordinator — and the harness matches request bodies. If the runner and
//! the driver build different requests for the same configuration, the second
//! pass fails as a mock miss with a body diff naming the field. That is the
//! assertion; the state comparisons below are secondary and catch divergence
//! the wire cannot show, like a differently committed turn budget.
//!
//! When this suite fails after a change to either coordinator, the question is
//! not "which assertion do I relax" but "which of the two did I mean to
//! change".
//!
//! # What the first recording established
//!
//! Diffing the runner half of each cassette against the driver half: the
//! request bodies are **identical**, byte for byte, except for the
//! provider-assigned tool-call id — which differs between any two live runs and
//! is renumbered by the scrubber. So as of this suite's first recording the two
//! coordinators agree completely on *what* they send.
//!
//! # What this suite guards, and what it cannot
//!
//! The commit boundary used to be the interesting divergence: the runner spent
//! its turn before its completion-call hooks, its model selection and its
//! request preparation, each of which can terminate the run, while the driver
//! prepared first and committed last. That is fixed — both now commit only
//! once a request exists — and the commit-boundary unit tests pin it, because
//! it is invisible on the wire, which is exactly why it survived several
//! reviews.
//!
//! What remains is **structural** duplication: two coordinators implementing
//! one protocol, agreeing today because two code paths currently happen to
//! agree. This suite detects the next disagreement; nothing here prevents one.
//! Only consolidating the coordinators would, and until that lands these
//! cassettes are the guard that makes the delay safe. They are therefore worth
//! extending whenever either coordinator grows a behavior the four scenarios
//! below do not exercise — hook termination, per-turn patches, structured
//! output, recovery resolutions, resumed runs.

use rig::agent::AgentRun;
use rig::completion::PromptError;
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_completions_cassette;
use crate::driver_support::{
    ADD_PROMPT, FORCE_TOOLS_PREAMBLE, dispatch_and_feed, drive_to_completion, expect_execute_tools,
    expect_send, expect_turn_accepted,
};
use crate::support::{Adder, Subtract};

/// A plain single-turn run: no tools, no patch. The narrowest possible parity
/// claim, and the one that fails first if either coordinator changes how it
/// builds a baseline request.
#[tokio::test]
async fn a_plain_turn_is_identical_through_both_coordinators() {
    with_openai_completions_cassette("coordinator_parity/plain_turn", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Reply with one short sentence.")
            .temperature(0.0)
            .build();

        // Runner.
        let runner_response = agent
            .runner("Say hello.")
            .run()
            .await
            .expect("the runner should finish");

        // Driver. Same agent, same prompt, no per-turn configuration — so the
        // request must be byte-identical, which the cassette enforces.
        let mut driver = agent.drive("Say hello.");
        let driver_response = drive_to_completion(&mut driver)
            .await
            .expect("the driver should finish");

        assert_eq!(
            driver.run().turn(),
            1,
            "one model call, same as the runner's single completion call"
        );
        assert_eq!(
            runner_response.completion_calls.len(),
            driver.run().completion_calls().len(),
            "both coordinators account one completion call per model call"
        );
        assert!(!runner_response.output.trim().is_empty());
        assert!(!driver_response.output.trim().is_empty());
    })
    .await;
}

/// A two-turn tool round trip. The second request is the interesting one: it
/// carries the assistant tool call and the tool result each coordinator
/// threaded back through the run, so a divergence in how either writes history
/// shows up as a body diff.
#[tokio::test]
async fn a_tool_round_trip_is_identical_through_both_coordinators() {
    with_openai_completions_cassette("coordinator_parity/tool_round_trip", |client| async move {
        let build = |client: &openai::CompletionsClient| {
            client
                .agent(openai::GPT_4O)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .default_max_turns(3)
                .tool(Adder)
                .build()
        };

        let runner_response = build(&client)
            .runner(ADD_PROMPT)
            .run()
            .await
            .expect("the runner should finish");

        let agent = build(&client);
        let mut driver = agent.drive(ADD_PROMPT);
        let driver_response = drive_to_completion(&mut driver)
            .await
            .expect("the driver should finish");

        assert_eq!(
            runner_response.completion_calls.len(),
            driver.run().completion_calls().len(),
            "both coordinators spend the same number of model calls on the same work"
        );
        assert_eq!(driver.run().turn(), 2);
        assert!(!runner_response.output.trim().is_empty());
        assert!(!driver_response.output.trim().is_empty());
    })
    .await;
}

/// `tool_choice` on the agent must reach the provider identically from both.
///
/// `Required` forbids the model from ever answering in text, so a run under it
/// always ends by exhausting its budget — on **both** coordinators, which is
/// the parity claim. Same configuration, same first request body (the cassette
/// enforces it), same terminal condition, same accounting.
#[tokio::test]
async fn a_required_tool_choice_is_identical_through_both_coordinators() {
    with_openai_completions_cassette("coordinator_parity/tool_choice", |client| async move {
        let build = |client: &openai::CompletionsClient| {
            client
                .agent(openai::GPT_4O)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_choice(ToolChoice::Required)
                .tool(Adder)
                .tool(Subtract)
                .build()
        };

        let runner_error = build(&client)
            .runner(ADD_PROMPT)
            .max_turns(1)
            .run()
            .await
            .expect_err("Required with a budget of one cannot finish");
        assert!(
            matches!(
                runner_error,
                PromptError::MaxTurnsError { max_turns: 1, .. }
            ),
            "expected the runner to exhaust its budget, got {runner_error:?}"
        );

        let agent = build(&client);
        let mut driver = agent.drive(ADD_PROMPT).max_turns(1);
        let (request, tools, _) = expect_send(&mut driver).await;
        assert!(tools.allowed_tool_names().contains("add"));
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);

        // The turn recorded what it resolved to, and it is the agent's choice —
        // the same one the runner sent.
        let prepared = driver
            .run()
            .prepared_turn()
            .expect("the committed turn records its metadata")
            .clone();
        assert_eq!(prepared.tool_choice, Some(ToolChoice::Required));

        let (pending, tools) = expect_execute_tools(&mut driver).await;
        assert!(!pending.is_empty(), "Required must force a tool call");
        dispatch_and_feed(&mut driver, &pending, &tools).await;

        let driver_error = driver
            .next_step()
            .await
            .expect_err("the driver exhausts the same budget");
        assert!(
            matches!(
                driver_error,
                PromptError::MaxTurnsError { max_turns: 1, .. }
            ),
            "expected the driver to exhaust its budget, got {driver_error:?}"
        );
        assert_eq!(
            driver.run().completion_calls().len(),
            1,
            "one model call spent, same as the runner"
        );
    })
    .await;
}

/// A custom `AgentRun` driven by hand reaches the same request the runner
/// builds from the equivalent agent configuration.
///
/// The run carries the choice rather than the agent, which is the path that
/// used to drop it silently. Recorded for each, so a regression there is a mock
/// miss rather than a behavioral difference nobody notices.
#[tokio::test]
async fn a_custom_run_matches_the_runners_equivalent_configuration() {
    with_openai_completions_cassette("coordinator_parity/custom_run", |client| async move {
        // Runner: the choice comes from the agent.
        let runner_error = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool_choice(ToolChoice::Required)
            .tool(Adder)
            .build()
            .runner(ADD_PROMPT)
            .max_turns(1)
            .run()
            .await
            .expect_err("Required with a budget of one cannot finish");
        assert!(matches!(
            runner_error,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));

        // Driver: an equivalent run carrying the choice itself, on an agent
        // that has none.
        let bare = client
            .agent(openai::GPT_4O)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .build();
        let run = AgentRun::new(ADD_PROMPT)
            .max_turns(1)
            .with_tool_choice(ToolChoice::Required);
        let mut driver = bare.drive_run(run);

        let (request, _, _) = expect_send(&mut driver).await;
        let response = request.send().await.expect("should send");
        expect_turn_accepted(&mut driver, &response);
        let (pending, _) = expect_execute_tools(&mut driver).await;
        assert!(
            !pending.is_empty(),
            "the run's own Required must have reached the provider"
        );
    })
    .await;
}
