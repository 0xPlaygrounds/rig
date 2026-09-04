//! Matrix T's cells: a layer's denial under every consumer mapping — on a
//! tool (the model sees the skipped result), on a completion (the run
//! fails), on a memory `Load` (the run fails at the record), on a custom
//! effect from a hook (the hook sees `Denied`) — and the leftovers of the
//! #2443 review: a host effect that does not serialize dispatched from a
//! hook (L3), required host keys of the embed, rerank and custom families
//! described from the handler table on replay (L4), and the recorder
//! under five thousand kept events on one key beside two hundred other
//! records (L2). Mock-scripted: the decision, the wire form and the
//! recorder are the cells, not the provider. The enumeration and the
//! replays live in `crates/rig-verify/tests/corpus_leftovers.rs`.

use rig::agent::AgentBuilder;
use rig::bus::Bus;
use rig::effect::{EffectFamily, HandlerKey};
use rig::serve::ErasedHandler;
use rig::test_utils::{MockCompletionModel, MockTurn};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use serde_json::json;

use crate::goldens::{
    CONVERSATION, DenyAllLayer, HOST_DENY_REASON, MockRerank, NOTE_KEY, NeverAsked,
    NoteDeniedAtStart, NoteTaker, NoteUnserializableAtStart, NotesAtStart, UNSERIALIZABLE_KEY,
    add_tool_under, families,
};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const PROMPT: &str = "Reply with the single word: ready.";

fn tool_result_texts(log: &EffectLog) -> Vec<String> {
    let request = match &log.records.last().expect("a record").kind {
        rig::effect::EffectKind::Completion { request, .. } => request,
        other => panic!("the last record is a completion, not {other:?}"),
    };
    request
        .chat_history
        .iter()
        .filter_map(|message| match message {
            rig::message::Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|content| match content {
            rig::message::UserContent::ToolResult(result) => Some(
                result
                    .content
                    .iter()
                    .map(|part| match part {
                        rig::message::ToolResultContent::Text(text) => text.text.clone(),
                        other => format!("{other:?}"),
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn denied_tool_effect_log_is_the_golden_fixture() {
    let server = add_tool_under(|adder| adder.layered(DenyAllLayer));
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "add", json!({"x": 17, "y": 25})),
        MockTurn::text("I could not add them."),
    ]))
    .name("golden")
    .preamble(TOOLS_PREAMBLE)
    .temperature(0.0)
    .tool_server_handle(server)
    .record_effects()
    .build();
    let response = agent
        .prompt(ADD_PROMPT)
        .max_turns(3)
        .await
        .expect("the model sees the skipped result and answers");
    assert_eq!(response.output, "I could not add them.");
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    assert_eq!(tool_result_texts(&log), [HOST_DENY_REASON]);
    assert_eq!(log.header.hooks, ["DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_tool", &log);
}

#[tokio::test]
async fn denied_completion_effect_log_is_the_golden_fixture() {
    // A host's layer on the model key denies: the run fails with the
    // report, and the log holds no record.
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                MockCompletionModel::text("never asked"),
            ))
            .layered(DenyAllLayer),
        )
        .expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .build();
    let error = agent
        .prompt(PROMPT)
        .await
        .expect_err("the denial fails the run");
    assert!(
        matches!(&error, rig::completion::PromptError::Report(report) if report.kind == rig::error::ErrorKind::Denied),
        "{error:?}"
    );
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert!(log.records.is_empty(), "no record for a denied completion");
    assert_eq!(log.header.hooks, ["DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_completion", &log);
}

#[tokio::test]
async fn denied_memory_load_effect_log_is_the_golden_fixture() {
    let memory = ErasedHandler::new(rig::serve::adapters::MemoryAdapter::new(
        rig::memory::InMemoryConversationMemory::new(),
    ))
    .layered(DenyAllLayer);
    let agent = AgentBuilder::new(MockCompletionModel::text("never asked"))
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .memory_handler(memory)
        .conversation(CONVERSATION)
        .record_effects()
        .build();
    let error = agent
        .prompt(PROMPT)
        .await
        .expect_err("the denied load fails the run");
    assert!(
        matches!(
            &error,
            rig::completion::PromptError::MemoryError(rig::memory::MemoryError::Policy(reason))
                if reason == HOST_DENY_REASON
        ),
        "{error:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert!(log.records.is_empty(), "no record for a denied load");
    assert_eq!(log.header.hooks, ["DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_memory_load", &log);
}

#[tokio::test]
async fn denied_custom_from_hook_effect_log_is_the_golden_fixture() {
    // The host's layer on its own note key denies the hook's note: the
    // hook sees `Denied` and the run goes on; the log holds the completion
    // only.
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                MockCompletionModel::text("ready"),
            )),
        )
        .expect("a fresh key");
    driver
        .register_erased(
            HandlerKey::from(NOTE_KEY),
            ErasedHandler::new(NoteTaker).layered(DenyAllLayer),
        )
        .expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .add_hook(NoteDeniedAtStart)
        .build();
    let response = agent.prompt(PROMPT).await.expect("the run goes on");
    assert_eq!(response.output, "ready");
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    assert_eq!(log.header.hooks, ["NoteDeniedAtStart", "DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_custom_from_hook", &log);
}

/// A host bus with the mock model under the agent's key and `host`
/// registered; the run's log stamped by the agent.
async fn over_host(
    model: MockCompletionModel,
    streamed: bool,
    host: impl FnOnce(&mut rig::bus::BusDriver),
    agent: impl FnOnce(AgentBuilder<rig::agent::NoToolConfig>) -> rig::agent::Agent,
    prompt: &str,
) -> (EffectLog, String) {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default", model,
            )),
        )
        .expect("a fresh key");
    host(&mut driver);
    let recorder = if streamed {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let builder =
        AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
            .name("golden")
            .temperature(0.0);
    let agent = agent(builder);
    let output = if streamed {
        use futures::StreamExt;
        let mut stream = agent.stream_prompt(prompt).max_turns(200).stream().await;
        let mut output = None;
        while let Some(item) = stream.next().await {
            if let rig::agent::MultiTurnStreamItem::FinalResponse(response) =
                item.expect("the stream yields")
            {
                output = Some(response.output);
            }
        }
        drop(stream);
        output.expect("a final response")
    } else {
        agent
            .prompt(prompt)
            .max_turns(200)
            .await
            .expect("the agent answers")
            .output
    };
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    (log, output)
}

#[tokio::test]
async fn unserializable_from_hook_effect_log_is_the_golden_fixture() {
    // L3: the hook's effect has no wire form; the run sees `Request` with
    // the serde message, the log has no record, the handler was never
    // entered.
    let reached = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&reached);
    let (log, output) = over_host(
        MockCompletionModel::text("ready"),
        false,
        move |driver| {
            driver
                .register_erased(
                    HandlerKey::from(UNSERIALIZABLE_KEY),
                    ErasedHandler::new(NeverAsked { reached: counter }),
                )
                .expect("a fresh key");
        },
        |builder| {
            builder
                .preamble(BASIC_PREAMBLE)
                .add_hook(NoteUnserializableAtStart)
                .build()
        },
        PROMPT,
    )
    .await;
    assert_eq!(output, "ready");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    assert_eq!(
        reached.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "never entered"
    );
    assert!(
        log.header
            .handlers
            .iter()
            .any(|h| h.key.as_str() == UNSERIALIZABLE_KEY),
        "the host's handler is in the table, never in a record"
    );
    crate::goldens::golden_effects("mock_leftovers_unserializable_from_hook", &log);
}

/// L4: a host key the program registers and never dispatches to: in the
/// handler table, in no record; on replay it is described from the table.
async fn required_host_key(register: impl FnOnce(&mut rig::bus::BusDriver)) -> EffectLog {
    let (log, output) = over_host(
        MockCompletionModel::text("ready"),
        false,
        register,
        |builder| builder.preamble(BASIC_PREAMBLE).build(),
        PROMPT,
    )
    .await;
    assert_eq!(output, "ready");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    log
}

#[tokio::test]
async fn required_embed_effect_log_is_the_golden_fixture() {
    let log = required_host_key(|driver| {
        driver
            .register_erased(
                HandlerKey::from("host/embed"),
                ErasedHandler::new(rig::serve::adapters::EmbedAdapter::new(
                    "host",
                    rig::test_utils::MockEmbeddingModel,
                )),
            )
            .expect("a fresh key");
    })
    .await;
    crate::goldens::golden_effects("mock_leftovers_required_embed", &log);
}

#[tokio::test]
async fn required_rerank_effect_log_is_the_golden_fixture() {
    let log = required_host_key(|driver| {
        driver
            .register_erased(
                HandlerKey::from("host/rerank"),
                ErasedHandler::new(rig::serve::adapters::RerankAdapter::new("host", MockRerank)),
            )
            .expect("a fresh key");
    })
    .await;
    crate::goldens::golden_effects("mock_leftovers_required_rerank", &log);
}

#[tokio::test]
async fn required_custom_effect_log_is_the_golden_fixture() {
    let log = required_host_key(|driver| {
        driver
            .register_erased(HandlerKey::from(NOTE_KEY), ErasedHandler::new(NoteTaker))
            .expect("a fresh key");
    })
    .await;
    crate::goldens::golden_effects("mock_leftovers_required_custom", &log);
}

#[tokio::test]
async fn five_thousand_events_effect_log_is_the_golden_fixture() {
    // L2: the recorder finds its slot from the back. Two hundred host notes
    // from the run-start hook (two hundred records) and then the answer,
    // streamed as five thousand deltas on the one key: every event lands
    // in the last slot, in order.
    use rig::test_utils::MockStreamEvent;
    let mut answer: Vec<MockStreamEvent> = (0..5000)
        .map(|n| MockStreamEvent::text(format!("{n} ")))
        .collect();
    answer.push(MockStreamEvent::final_response_with_total_tokens(5000));
    let model = MockCompletionModel::from_stream_turns([answer]);
    let (log, output) = over_host(
        model,
        true,
        |driver| {
            driver
                .register_erased(HandlerKey::from(NOTE_KEY), ErasedHandler::new(NoteTaker))
                .expect("a fresh key");
        },
        |builder| {
            builder
                .preamble(BASIC_PREAMBLE)
                .add_hook(NotesAtStart(200))
                .build()
        },
        PROMPT,
    )
    .await;
    assert_eq!(log.len(), 201, "two hundred notes and the answer");
    let expected: String = (0..5000).map(|n| format!("{n} ")).collect();
    assert_eq!(output, expected);
    let events = log.records[200].events.as_ref().expect("kept");
    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            rig::streaming::StreamEvent::BlockDelta {
                delta: rig::streaming::Delta::Text { text },
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas.len(), 5000);
    assert!(
        deltas
            .iter()
            .enumerate()
            .all(|(n, delta)| *delta == format!("{n} ")),
        "every event in the last slot, in order"
    );
    assert_eq!(log.header.hooks, ["NotesAtStart(200)"]);
    crate::goldens::golden_effects("mock_leftovers_five_thousand_events", &log);
}

#[tokio::test]
async fn denied_tool_streamed_effect_log_is_the_golden_fixture() {
    let server = add_tool_under(|adder| adder.layered(DenyAllLayer));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            rig::test_utils::MockStreamEvent::tool_call("call-1", "add", json!({"x": 17, "y": 25})),
            rig::test_utils::MockStreamEvent::final_response_with_total_tokens(1),
        ],
        vec![
            rig::test_utils::MockStreamEvent::text("I could not add them."),
            rig::test_utils::MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let (log, output) = over_host(
        model,
        true,
        |_| {},
        |builder| {
            builder
                .preamble(TOOLS_PREAMBLE)
                .tool_server_handle(server)
                .build()
        },
        ADD_PROMPT,
    )
    .await;
    assert_eq!(output, "I could not add them.");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    assert_eq!(tool_result_texts(&log), [HOST_DENY_REASON]);
    assert!(log.records[0].events.is_some());
    crate::goldens::golden_effects("mock_leftovers_denied_tool_streamed", &log);
}

#[tokio::test]
async fn denied_custom_from_hook_streamed_effect_log_is_the_golden_fixture() {
    let model = MockCompletionModel::from_stream_turns([vec![
        rig::test_utils::MockStreamEvent::text("ready"),
        rig::test_utils::MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let (log, output) = over_host(
        model,
        true,
        |driver| {
            driver
                .register_erased(
                    HandlerKey::from(NOTE_KEY),
                    ErasedHandler::new(NoteTaker).layered(DenyAllLayer),
                )
                .expect("a fresh key");
        },
        |builder| {
            builder
                .preamble(BASIC_PREAMBLE)
                .add_hook(NoteDeniedAtStart)
                .build()
        },
        PROMPT,
    )
    .await;
    assert_eq!(output, "ready");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    assert!(log.records[0].events.is_some());
    crate::goldens::golden_effects("mock_leftovers_denied_custom_from_hook_streamed", &log);
}
