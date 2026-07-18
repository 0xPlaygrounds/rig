//! Hook-system stress suite: `RequestPatch` steering on `CompletionCall` —
//! preamble override, `tool_choice`, per-turn `history` replacement, and
//! multi-field patches. Recorded against real Gemini; each patch effect is
//! proven by a downstream-observable change (the model can't echo settings).

use rig::agent::RequestPatch;
use rig::completion::Prompt;
use rig::message::{Message, ToolChoice};
use rig::prelude::AgentClientExt;
use rig::providers::gemini;

use super::super::hook_stress_support::{ApplyPatch, FirstTurnPatch, fact_doc};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::CountingAdd;
use crate::support::assert_nonempty_response;

const CODEWORD: &str = "ZULU-99";

#[tokio::test]
async fn preamble_override_forces_codeword_blocking() {
    with_gemini_cassette(
        "hook_stress_patch/preamble_override_forces_codeword_blocking",
        |client| async move {
            // The agent's own preamble says nothing about a codeword.
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a terse assistant.")
                .build();

            // A hook overrides the preamble for this turn to require a codeword
            // suffix — a behavior change only the injected preamble can cause.
            let response = agent
                .prompt("Greet me in one short sentence.")
                .max_turns(2)
                .add_hook(ApplyPatch(
                    RequestPatch::new()
                        .preamble(format!(
                            "You are a terse assistant. End every reply with the exact token \
                             {CODEWORD} on its own, verbatim."
                        ))
                        .temperature(0.0),
                ))
                .await
                .expect("preamble-override run should succeed");

            assert!(
                response.contains(CODEWORD),
                "the overridden preamble must change behavior; answer: {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn tool_choice_required_forces_a_tool_call_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();

    with_gemini_cassette(
        "hook_stress_patch/tool_choice_required_forces_a_tool_call_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant.")
                .tool(add)
                .build();

            // Force tool_choice = Required on the FIRST turn only, so the model
            // must call the tool up front. (Forcing it every turn would force a
            // tool call on every turn and loop until max_turns — a real footgun
            // this stress test surfaced.)
            let response = agent
                .prompt("Use the add tool to compute 12 plus 30, then report the number.")
                .max_turns(4)
                .add_hook(FirstTurnPatch(
                    RequestPatch::new()
                        .tool_choice(ToolChoice::Required)
                        .temperature(0.0),
                ))
                .await
                .expect("tool_choice=Required run should succeed");

            assert_nonempty_response(&response);
            assert!(
                add_calls.count() >= 1,
                "tool_choice=Required (via RequestPatch) must force a tool call"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn history_replacement_injects_prior_fact_blocking() {
    with_gemini_cassette(
        "hook_stress_patch/history_replacement_injects_prior_fact_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a helpful assistant. Use the conversation so far to answer.")
                .build();

            // A hook replaces the messages sent this turn with a synthetic prior
            // exchange that establishes a fact — the model answers from it.
            let response = agent
                .prompt("What is the passphrase?")
                .max_turns(2)
                .add_hook(ApplyPatch(
                    RequestPatch::new()
                        .history([Message::user(
                            "For this session, the passphrase is OMEGA-7. Acknowledge and remember it.",
                        )])
                        .temperature(0.0),
                ))
                .await
                .expect("history-replacement run should succeed");

            assert!(
                response.contains("OMEGA-7"),
                "the per-turn history view must reach the model; answer: {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn multi_field_patch_applies_preamble_and_context_blocking() {
    with_gemini_cassette(
        "hook_stress_patch/multi_field_patch_applies_preamble_and_context_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a terse assistant.")
                .build();

            // One patch sets BOTH the preamble and an extra_context document; both
            // fields must take effect.
            let response = agent
                .prompt("What is the depot code? Keep it short.")
                .max_turns(2)
                .add_hook(ApplyPatch(
                    RequestPatch::new()
                        .preamble(format!(
                            "You are a terse assistant. End every reply with the exact token \
                             {CODEWORD}."
                        ))
                        .context(fact_doc("depot", "The depot code is GAMMA-33."))
                        .temperature(0.0),
                ))
                .await
                .expect("multi-field patch run should succeed");

            assert!(
                response.contains("GAMMA-33"),
                "the patch's extra_context must reach the model; answer: {response:?}"
            );
            assert!(
                response.contains(CODEWORD),
                "the patch's preamble override must also take effect; answer: {response:?}"
            );
        },
    )
    .await;
}
