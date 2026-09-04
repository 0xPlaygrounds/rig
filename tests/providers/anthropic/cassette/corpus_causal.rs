//! Matrix Q's live cells: a tool that dispatches a nested completion
//! through its sink's dispatcher (`CLAUDE_SONNET_4_6`, temperature 0),
//! over a host bus, under both serving policies and streamed. The host
//! registers the model under the agent's key and records; the `lookup`
//! tool is the agent's, registered through its tool server; the agent
//! stamps the log. The enumeration and the replays live in
//! `crates/rig-verify/tests/corpus_causal.rs`. Every cell is a new
//! recording under `tests/cassettes/anthropic/corpus_causal/`.

use futures::StreamExt;
use rig::agent::tool::server::ToolServer;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::bus::Bus;
use rig::effect::{EffectFamily, HandlerKey};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::serve::ServingPolicy;
use rig::tool::RegisteredTool;

use super::super::support::with_anthropic_corpus_causal_cassette;
use crate::goldens::{Lookup, NestedChild, Nesting, families, parent_positions};

const TOOLS_PREAMBLE: &str = "You are a research assistant. Use the lookup tool to answer.";
const PROMPT: &str = "Use the lookup tool with q set to exactly \"What is the capital of France?\" and reply with just the lookup result.";

struct Host {
    serial: bool,
    streamed: bool,
}

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

async fn over_host(
    client: rig::providers::anthropic::Client,
    host: Host,
) -> rig::effect_log::EffectLog {
    let config = ServingPolicy {
        serial_per_handler: host.serial,
        ..ServingPolicy::default()
    };
    let (dispatcher, registrar, mut driver) = Bus::channel_with(config);
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
    let recorder = if host.streamed {
        rig::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig::effect_log::EffectLogRecorder::new()
    };
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let server = ToolServer::new()
        .owner("golden")
        .registered_tool(
            RegisteredTool::from_handler(Lookup {
                nesting: Nesting {
                    child: NestedChild::Completion,
                    from_thread: false,
                    detached: false,
                },
                model_key: model_key.clone(),
            })
            .expect("a tool-family handler"),
        )
        .run();
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool_server_handle(server)
        .build();
    let output = if host.streamed {
        let mut stream = agent.stream_prompt(PROMPT).max_turns(3).stream().await;
        let output = final_output(&mut stream).await;
        drop(stream);
        output
    } else {
        agent
            .prompt(PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers")
            .output
    };
    assert!(output.contains("Paris"), "{output}");
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(log.header.bus, None, "the policy is the host's");
    // The tool's completion is the model key's second record, made from
    // the tool's record.
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion,
            EffectFamily::Completion
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, Some(1), None]);
    log
}

#[tokio::test]
async fn completion_serial_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_causal_cassette("corpus_causal/completion_serial", |client| async move {
        let log = over_host(
            client,
            Host {
                serial: true,
                streamed: false,
            },
        )
        .await;
        crate::goldens::golden_effects("anthropic_causal_completion_serial", &log);
    })
    .await;
}

#[tokio::test]
async fn completion_concurrent_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_causal_cassette(
        "corpus_causal/completion_concurrent",
        |client| async move {
            let log = over_host(
                client,
                Host {
                    serial: false,
                    streamed: false,
                },
            )
            .await;
            crate::goldens::golden_effects("anthropic_causal_completion_concurrent", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn completion_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_causal_cassette(
        "corpus_causal/completion_streamed",
        |client| async move {
            let log = over_host(
                client,
                Host {
                    serial: false,
                    streamed: true,
                },
            )
            .await;
            // The run's completions are streamed with their events; the
            // tool's nested completion is unary.
            assert!(log.records[0].events.is_some());
            assert!(log.records[2].events.is_none());
            assert!(log.records[3].events.is_some());
            crate::goldens::golden_effects("anthropic_causal_completion_streamed", &log);
        },
    )
    .await;
}
