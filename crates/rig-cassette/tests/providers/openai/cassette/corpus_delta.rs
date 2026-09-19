//! Matrix K of the effect corpus, the live chat-completions wire: a tool call
//! streamed as a name delta and argument deltas (the chat wire), dispatched,
//! answered. Producer of the golden `crates/rig-cassette/tests/corpus_delta.rs`
//! replays by both interpreters. A new recording under
//! `crates/rig-cassette/fixtures/cassettes/openai/corpus_delta/`.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{Delta, StreamEvent};
use rig_cassette::agent::AgentReplayExt;

use super::super::support::with_openai_corpus_delta_cassette;
use crate::goldens::families;
use crate::support::{Adder, TOOLS_PREAMBLE};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

#[tokio::test]
async fn chat_baseline_effect_log_is_the_golden_fixture() {
    with_openai_corpus_delta_cassette("corpus_delta/chat_baseline", |client| async move {
        let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
        let agent = client
            .chat
            .agent(openai::GPT_4O)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .record_to(recorder.clone())
            .build();
        let mut stream = agent.prompt(ADD_PROMPT).max_turns(3).stream();
        let mut output = None;
        while let Some(item) = stream.next().await {
            if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
                output = Some(response.output);
            }
        }
        drop(stream);
        let output = output.expect("a final response");
        assert!(output.contains("42"), "{output}");
        let log = agent.stamp(recorder.take());
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
        crate::goldens::golden_effects("openai_delta_chat_baseline", &log);
    })
    .await;
}
