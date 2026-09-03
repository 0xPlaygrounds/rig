//! The effect corpus's OpenAI recordings. The Responses wire gives a tool
//! call two ids (`call_id` and the item's `id`), so these goldens are the
//! proof that a dual provider id survives record, fold and replay verbatim.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_cassette;
use crate::support::{Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract};

pub(crate) const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";
pub(crate) const CHAIN_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
     subtract tool. Report the final number.";

fn families(log: &rig::effect_log::EffectLog) -> Vec<EffectFamily> {
    log.records
        .iter()
        .map(|record| record.kind.family())
        .collect()
}

/// Every tool call in the log's completion outcomes, with its provider id.
fn tool_calls(log: &rig::effect_log::EffectLog) -> Vec<rig::message::ToolCall> {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(rig::effect::Outcome::Completion(response)) => Some(response),
            _ => None,
        })
        .flat_map(|response| response.choice.iter())
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

fn assert_dual_ids(calls: &[rig::message::ToolCall]) {
    assert!(!calls.is_empty(), "the program calls tools");
    for call in calls {
        let provider = call
            .provider
            .as_ref()
            .unwrap_or_else(|| panic!("a Responses call carries a provider id: {call:?}"));
        assert!(
            provider.item_id.is_some(),
            "a Responses call carries the item id too: {call:?}"
        );
    }
}

/// A streamed turn with one tool call, events kept.
#[tokio::test]
async fn streaming_with_events_effect_log_is_the_golden_fixture() {
    with_openai_cassette("effect_corpus/streaming_with_events", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .name("golden")
            .preamble(STREAMING_TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .tool(Subtract)
            .record_effects_with_events()
            .build();
        let mut stream = agent
            .stream_prompt(STREAMING_TOOLS_PROMPT)
            .max_turns(3)
            .stream()
            .await;
        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                saw_final = true;
            }
        }
        assert!(saw_final, "the stream yields a final response");
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        assert!(
            log.records
                .iter()
                .filter(|record| record.kind.family() == EffectFamily::Completion)
                .all(|record| record
                    .events
                    .as_ref()
                    .is_some_and(|events| !events.is_empty())),
            "every streamed completion keeps its events"
        );
        assert_dual_ids(&tool_calls(&log));
        crate::goldens::golden_effects("openai_streaming_with_events", &log);
    })
    .await;
}

/// Two tool-call turns, blocking: the second turn's request carries the
/// first turn's dual-id call and its result back to the wire.
#[tokio::test]
async fn tool_call_turns_effect_log_is_the_golden_fixture() {
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
            ],
            "add, then subtract, then the answer"
        );
        let calls = tool_calls(&log);
        assert_eq!(calls.len(), 2);
        assert_dual_ids(&calls);
        crate::goldens::golden_effects("openai_tool_call_turns", &log);
    })
    .await;
}
