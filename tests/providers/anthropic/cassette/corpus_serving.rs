//! Matrix C of the effect corpus: serving policy, routing and bus
//! ownership. Producers of the goldens
//! `crates/rig-verify/tests/corpus_serving.rs` replays by both
//! interpreters; the enumeration lives there.
//!
//! The serving cells re-record nothing: the bus policy changes how a
//! program is served, not what it asks, so the same cassette serves every
//! policy and the record is the proof that the trace is the same. The
//! routing and host-bus cells are new recordings under
//! `tests/cassettes/anthropic/corpus_serving/`.

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::effect::{EffectFamily, HandlerKey};
use rig::prelude::*;
use rig::providers::anthropic::completion::{CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6};

use super::super::support::{with_anthropic_cassette, with_anthropic_corpus_serving_cassette};
use crate::goldens::{RouteAfterFirstTurn, families};
use crate::support::{
    Adder, AlphaSignal, BetaSignal, TOOLS_PREAMBLE, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT,
};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const TWO_TOOLS: [EffectFamily; 4] = [
    EffectFamily::Completion,
    EffectFamily::Tool,
    EffectFamily::Tool,
    EffectFamily::Completion,
];

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

/// The two-tool stream program under a bus policy and a runner
/// concurrency: the record is in dispatch order whatever the policy.
async fn two_tools(
    client: rig::providers::anthropic::Client,
    bus: rig::bus::BusConfig,
    concurrency: usize,
    events: bool,
) -> rig::effect_log::EffectLog {
    let mut builder = client
        .agent(CLAUDE_SONNET_4_6)
        .name("golden")
        .configure_bus(bus)
        .preamble(TWO_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .tool(BetaSignal);
    builder = if events {
        builder.record_effects_with_events()
    } else {
        builder.record_effects()
    };
    let agent = builder.build();
    let mut stream = agent
        .stream_prompt(TWO_TOOL_STREAM_PROMPT)
        .max_turns(8)
        .tool_concurrency(concurrency)
        .stream()
        .await;
    let output = tokio::time::timeout(std::time::Duration::from_secs(5), final_output(&mut stream))
        .await
        .expect("two tools never wait on each other");
    drop(stream);
    assert!(!output.is_empty());
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), TWO_TOOLS);
    assert_eq!(
        log.header.bus,
        Some(bus),
        "the header says how it was served"
    );
    let names: Vec<_> = log
        .records
        .iter()
        .filter_map(|record| match &record.kind {
            rig::effect::EffectKind::ToolCall { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        ["lookup_harbor_label", "lookup_orchard_label"],
        "dispatch order, whatever the policy"
    );
    log
}

#[tokio::test]
async fn serial_concurrency_one_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let serial = rig::bus::BusConfig {
                serial_per_handler: true,
                ..rig::bus::BusConfig::default()
            };
            let log = two_tools(client, serial, 1, false).await;
            crate::goldens::golden_effects("anthropic_serving_serial_concurrency_one", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn concurrent_concurrency_one_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let log = two_tools(client, rig::bus::BusConfig::default(), 1, false).await;
            crate::goldens::golden_effects("anthropic_serving_concurrent_concurrency_one", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn concurrent_concurrency_two_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let log = two_tools(client, rig::bus::BusConfig::default(), 2, false).await;
            crate::goldens::golden_effects("anthropic_serving_concurrent_concurrency_two", &log);
        },
    )
    .await;
}

/// Events kept under concurrent dispatch: the stream's delivery is the
/// record, and buffering does not reorder it.
#[tokio::test]
async fn concurrent_concurrency_two_events_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let log = two_tools(client, rig::bus::BusConfig::default(), 2, true).await;
            crate::goldens::golden_effects(
                "anthropic_serving_concurrent_concurrency_two_events",
                &log,
            );
        },
    )
    .await;
}

/// Every buffer at one: the park points are exercised, the trace is the
/// same.
#[tokio::test]
async fn capacity_one_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let bus = rig::bus::BusConfig {
                command_capacity: 1,
                stream_capacity: 1,
                serial_per_handler: false,
            };
            let log = two_tools(client, bus, 2, false).await;
            crate::goldens::golden_effects("anthropic_serving_capacity_one", &log);
        },
    )
    .await;
}

/// Serial serving over memory and a tool: three keys, each served one
/// command at a time, in dispatch order.
#[tokio::test]
async fn serial_memory_tools_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("corpus_hooks/observe_everything", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .configure_bus(rig::bus::BusConfig {
                serial_per_handler: true,
                ..rig::bus::BusConfig::default()
            })
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .memory(rig::memory::InMemoryConversationMemory::new())
            .conversation("golden-conversation")
            .record_effects()
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(response.output.contains("42"), "{}", response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion,
                EffectFamily::Memory,
            ]
        );
        assert_eq!(log.header.bus.map(|bus| bus.serial_per_handler), Some(true));
        crate::goldens::golden_effects("anthropic_serving_serial_memory_tools", &log);
    })
    .await;
}

/// A second model registered as the route `fast` and selected by the hook
/// on every turn after the first: the tool-call turn goes to the default
/// model, the answer to the route, and the header's required row names
/// both.
#[tokio::test]
async fn model_route_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_serving_cassette("corpus_serving/model_route", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .model_route("fast", client.completion_model(CLAUDE_HAIKU_4_5))
            .tool(Adder)
            .add_hook(RouteAfterFirstTurn)
            .record_effects()
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(response.output.contains("42"), "{}", response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        assert_eq!(log.records[0].key.as_str(), "golden/model:default");
        assert_eq!(log.records[2].key.as_str(), "golden/model:fast");
        assert_eq!(
            log.header
                .required
                .get(&HandlerKey::from("golden/model:fast")),
            Some(&EffectFamily::Completion),
            "the route is in the required row"
        );
        crate::goldens::golden_effects("anthropic_serving_model_route", &log);
    })
    .await;
}

/// The route registered and never selected: the required row still names
/// it, the record never dispatches to it, and the replay must advertise it
/// from the row alone.
#[tokio::test]
async fn model_route_unselected_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("effect_corpus/tool_call_turn", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .model_route("fast", client.completion_model(CLAUDE_HAIKU_4_5))
            .tool(Adder)
            .record_effects()
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(response.output.contains("42"), "{}", response.output);
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
                .all(|record| record.key.as_str() != "golden/model:fast"),
            "the route was never selected"
        );
        assert_eq!(
            log.header
                .required
                .get(&HandlerKey::from("golden/model:fast")),
            Some(&EffectFamily::Completion),
            "the route is in the required row"
        );
        crate::goldens::golden_effects("anthropic_serving_model_route_unselected", &log);
    })
    .await;
}

/// The same tool-call program over a host's bus: the host registers the
/// model under the agent's key, drives the bus and records; the agent
/// stamps the log, whose header names no bus policy (the host's).
async fn over_host_bus(
    client: rig::providers::anthropic::Client,
    streamed: bool,
) -> rig::effect_log::EffectLog {
    let (dispatcher, registrar, mut driver) = rig::bus::Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            rig::serve::ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                client.completion_model(CLAUDE_SONNET_4_6),
            )),
        )
        .expect("a fresh key");
    let recorder = if streamed {
        rig::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig::effect_log::EffectLogRecorder::new()
    };
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool(Adder)
        .build();
    assert_eq!(agent.bus_config(), None, "the policy is the host's");
    let output = if streamed {
        let mut stream = agent.stream_prompt(ADD_PROMPT).max_turns(3).stream().await;
        let output = final_output(&mut stream).await;
        drop(stream);
        output
    } else {
        agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers")
            .output
    };
    assert!(output.contains("42"), "{output}");
    // The run is settled, so the recorder holds every record; the agent
    // stamps the log, then every dispatcher clone (the agent's among them)
    // is dropped so the host's driver can finish.
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert_eq!(log.header.bus, None);
    assert!(log.header.run_spec.is_some(), "the agent stamped its spec");
    assert_eq!(log.header.required.len(), 2, "{:?}", log.header.required);
    log
}

#[tokio::test]
async fn host_bus_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_serving_cassette("corpus_serving/host_bus", |client| async move {
        let log = over_host_bus(client, false).await;
        crate::goldens::golden_effects("anthropic_serving_host_bus", &log);
    })
    .await;
}

#[tokio::test]
async fn host_bus_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_serving_cassette(
        "corpus_serving/host_bus_streamed",
        |client| async move {
            let log = over_host_bus(client, true).await;
            crate::goldens::golden_effects("anthropic_serving_host_bus_streamed", &log);
        },
    )
    .await;
}
