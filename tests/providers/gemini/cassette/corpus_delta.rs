//! Matrix K of the effect corpus, the live interactions wire: a tool call
//! streamed as a name delta and argument deltas (`interactions_api()`), dispatched,
//! answered. Producer of the golden `crates/rig-verify/tests/corpus_delta.rs`
//! replays by both interpreters. A new recording under
//! `tests/cassettes/gemini/corpus_delta/`.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::{Delta, StreamEvent};

use super::super::support::with_gemini_corpus_delta_cassette;
use crate::goldens::families;
use crate::support::{Adder, TOOLS_PREAMBLE};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

#[tokio::test]
async fn interactions_baseline_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_delta_cassette("corpus_delta/interactions_baseline", |client| async move {
        let agent = client
            .interactions_api()
            .agent(gemini::completion::GEMINI_2_5_FLASH)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .record_effects_with_events()
            .build();
        let mut stream = agent.stream_prompt(ADD_PROMPT).max_turns(3).stream().await;
        let mut output = None;
        while let Some(item) = stream.next().await {
            if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
                output = Some(response.output);
            }
        }
        drop(stream);
        let output = output.expect("a final response");
        assert!(output.contains("42"), "{output}");
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        let events = log.records[0].events.as_ref().expect("events are kept");
        assert!(
            events.iter().any(|event| matches!(
                event,
                StreamEvent::BlockDelta {
                    delta: Delta::ToolName { .. },
                    ..
                }
            )),
            "the wire streams the tool's name as a delta"
        );
        crate::goldens::golden_effects("gemini_delta_interactions_baseline", &log);
    })
    .await;
}
