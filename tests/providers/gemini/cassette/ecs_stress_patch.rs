//! Native Gemini request patch stress cases, preserving original assertions.
use super::super::hook_stress_support::fact_doc;
use super::super::tools_support::CountingAdd;
use super::ecs_stress_runtime as runtime;
use rig::message::{Message, ToolChoice};
const CODEWORD: &str = "ZULU-99";
use super::super::support::with_gemini_cassette;
use crate::support::assert_nonempty_response;
use rig::{prelude::*, providers::gemini};
#[tokio::test]
async fn preamble_override_forces_codeword_blocking() {
    with_gemini_cassette(
        "hook_stress_patch/preamble_override_forces_codeword_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a terse assistant.",
                Some("stress-agent"),
                None,
            );
            runtime::install_patch(
                &mut ecs,
                rig_ecs::agent::RequestPatch {
                    preamble: Some(format!(
                        "You are a terse assistant. End every reply with the exact token \
                             {CODEWORD} on its own, verbatim."
                    )),
                    temperature: Some(0.0),
                    ..Default::default()
                },
                false,
            );
            let response = runtime::prompt(
                &mut ecs,
                "Greet me in one short sentence.",
                2,
                vec![],
                vec![],
            )
            .await;
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
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant.",
                Some("stress-agent"),
                None,
            );
            ecs.tool(add);
            runtime::install_patch(
                &mut ecs,
                rig_ecs::agent::RequestPatch {
                    tool_choice: Some(ToolChoice::Required),
                    temperature: Some(0.0),
                    ..Default::default()
                },
                true,
            );
            let response = runtime::prompt(
                &mut ecs,
                "Use the add tool to compute 12 plus 30, then report the number.",
                4,
                vec![],
                vec![],
            )
            .await;
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
                let mut ecs = runtime::agent(
                    client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                    "You are a helpful assistant. Use the conversation so far to answer.",
                    Some("stress-agent"),
                    None,
                );
                runtime::install_patch(
                    &mut ecs,
                    rig_ecs::agent::RequestPatch {
                        history: Some(
                            ([
                                Message::user(
                                    "For this session, the passphrase is OMEGA-7. Acknowledge and remember it.",
                                ),
                            ])
                                .iter()
                                .map(|m| {
                                    rig_ecs::agent::MessageParts::from_message(m)
                                        .expect("history message parts")
                                })
                                .collect(),
                        ),
                        temperature: Some(0.0),
                        ..Default::default()
                    },
                    false,
                );
                let response = runtime::prompt(
                        &mut ecs,
                        "What is the passphrase?",
                        2,
                        vec![],
                        vec![],
                    )
                    .await;
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
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a terse assistant.",
                Some("stress-agent"),
                None,
            );
            runtime::install_patch(
                &mut ecs,
                rig_ecs::agent::RequestPatch {
                    preamble: Some(format!(
                        "You are a terse assistant. End every reply with the exact token \
                             {CODEWORD}."
                    )),
                    extra_context: vec![fact_doc("depot", "The depot code is GAMMA-33.")],
                    temperature: Some(0.0),
                    ..Default::default()
                },
                false,
            );
            let response = runtime::prompt(
                &mut ecs,
                "What is the depot code? Keep it short.",
                2,
                vec![],
                vec![],
            )
            .await;
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
