//! Matrix C of the effect corpus, the OpenAI cell: two tool-call turns on
//! the dual-id wire with a runner concurrency of two. One call per turn,
//! so the concurrency has nothing to overlap and the trace is the
//! `openai_tool_call_turns` golden's; the cassette is the same.

use rig::effect::EffectFamily;
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_cassette;
use crate::goldens::families;
use crate::support::{Adder, Subtract};

const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";
const CHAIN_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
     subtract tool. Report the final number.";

#[tokio::test]
async fn two_turns_concurrency_two_effect_log_is_the_golden_fixture() {
    with_openai_cassette("effect_corpus/tool_call_turns", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .name("golden")
            .preamble(CHAIN_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .tool(Subtract)
            .record_effects()
            .build();
        let response = agent
            .prompt(CHAIN_PROMPT)
            .max_turns(6)
            .tool_concurrency(2)
            .await
            .expect("the agent answers");
        assert!(response.output.contains("21"), "{}", response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        crate::goldens::golden_effects("openai_serving_two_turns_concurrency_two", &log);
    })
    .await;
}
