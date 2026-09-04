//! Matrix Q's mock-scripted cells: causal dispatch. The `lookup` tool
//! dispatches from inside its own service through its sink's dispatcher —
//! a host note, the host's relay (which nests once more), its own key, or
//! the host's never-answering handler — over a host bus whose model is a
//! scripted mock (the wire does not change what the chain records; the
//! completion child is recorded live in `tests/providers/anthropic`). The
//! enumeration and the replays live in `crates/rig-verify/tests/corpus_causal.rs`.
//!
//! Scripted, not live: the cancelled cells need the run dropped at the
//! moment the child is reached, the same-key cells need a model that calls
//! a tool with `leaf` semantics no prompt can promise, and none of them
//! asks anything of a provider.

use std::time::Duration;

use rig::agent::AgentBuilder;
use rig::agent::tool::server::ToolServer;
use rig::bus::Bus;
use rig::effect::{EffectFamily, EffectKind, HandlerKey};
use rig::serve::ServingPolicy;
use rig::test_utils::{MockCompletionModel, MockTurn};
use rig::tool::RegisteredTool;
use rig_effect_log::{EffectLog, EffectLogRecorder};
use serde_json::json;

use crate::goldens::{
    Lookup, NEVER_KEY, NestedChild, Nesting, Never, NoteTaker, RELAY_KEY, Relay, families,
    parent_positions,
};

const TOOLS_PREAMBLE: &str = "You are a research assistant. Use the lookup tool to answer.";
const PROMPT: &str = "Look up the capital of France and reply with just the lookup result.";
const QUESTION: &str = "What is the capital of France?";

/// The model's script: a call to `lookup`, then the answer (a mock answers
/// turns in order, so a nested completion — none here — would take the
/// second turn).
fn script() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "lookup", json!({"q": QUESTION})),
        MockTurn::text("Paris"),
    ])
}

/// What a cell asks of the host.
struct Host {
    nesting: Nesting,
    serial: bool,
    /// Drop the run once the never-answering child is reached.
    cancel_at_child: bool,
}

const fn plain(child: NestedChild) -> Host {
    Host {
        nesting: Nesting {
            child,
            from_thread: false,
            detached: false,
        },
        serial: false,
        cancel_at_child: false,
    }
}

/// The program over the host's bus: the mock model under the agent's key,
/// the note taker, the relay and the never-answering handler under the
/// host's, the `lookup` tool registered through the agent's tool server,
/// the host's recorder tapping; the agent stamps the log.
async fn over_host(host: Host) -> EffectLog {
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
                script(),
            )),
        )
        .expect("a fresh key");
    driver
        .register_erased(
            HandlerKey::from(crate::goldens::NOTE_KEY),
            rig::serve::ErasedHandler::new(NoteTaker),
        )
        .expect("a fresh key");
    driver
        .register_erased(
            HandlerKey::from(RELAY_KEY),
            rig::serve::ErasedHandler::new(Relay),
        )
        .expect("a fresh key");
    let reached = std::sync::Arc::new(tokio::sync::Notify::new());
    driver
        .register_erased(
            HandlerKey::from(NEVER_KEY),
            rig::serve::ErasedHandler::new(Never {
                reached: std::sync::Arc::clone(&reached),
            }),
        )
        .expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let server = ToolServer::new()
        .owner("golden")
        .registered_tool(
            RegisteredTool::from_handler(Lookup {
                nesting: host.nesting,
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
    if host.cancel_at_child {
        // The run is dropped once the child is reached: the tool call and
        // its chain are cancelled together, and the run never finishes.
        let run = agent.prompt(PROMPT).max_turns(3);
        tokio::select! {
            finished = tokio::time::timeout(Duration::from_secs(5), run) => {
                panic!("the run finished before the child was reached: {finished:?}")
            }
            reached = tokio::time::timeout(Duration::from_secs(5), reached.notified()) => {
                reached.expect("the child is reached within the guard");
            }
        }
        // The driver resolves the cancelled dispatches on its own task.
        for _ in 0..200 {
            tokio::task::yield_now().await;
            if recorder.in_flight() == 0 {
                break;
            }
        }
        assert_eq!(recorder.in_flight(), 0, "every begun dispatch resolved");
    } else {
        let started = std::time::Instant::now();
        let output =
            tokio::time::timeout(Duration::from_secs(5), agent.prompt(PROMPT).max_turns(3))
                .await
                .expect("a nested dispatch never hangs the run")
                .expect("the agent answers")
                .output;
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "refused or served, never queued behind itself"
        );
        assert_eq!(output, "Paris");
    }
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(log.header.bus, None, "the policy is the host's");
    log
}

/// The tool's recorded text output.
fn tool_output(log: &EffectLog, position: usize) -> String {
    match &log.records[position].outcome {
        Ok(rig::effect::Outcome::ToolResult { result, .. }) => result.output().render(),
        other => panic!("a tool result at {position}, not {other:?}"),
    }
}

fn custom_kind(log: &EffectLog, position: usize) -> String {
    match &log.records[position].kind {
        EffectKind::Custom { kind, .. } => kind.to_string(),
        other => panic!("a custom effect at {position}, not {other:?}"),
    }
}

fn is_cancelled(log: &EffectLog, position: usize) -> bool {
    matches!(&log.records[position].outcome, Err(report) if report.kind == rig::error::ErrorKind::Cancelled)
}

#[tokio::test]
async fn note_serial_effect_log_is_the_golden_fixture() {
    let log = over_host(Host {
        serial: true,
        ..plain(NestedChild::Note)
    })
    .await;
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Custom,
            EffectFamily::Completion
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, Some(1), None]);
    assert_eq!(tool_output(&log, 1), "noted:lookup");
    crate::goldens::golden_effects("mock_causal_note_serial", &log);
}

#[tokio::test]
async fn note_concurrent_effect_log_is_the_golden_fixture() {
    let log = over_host(plain(NestedChild::Note)).await;
    assert_eq!(parent_positions(&log), [None, None, Some(1), None]);
    assert_eq!(tool_output(&log, 1), "noted:lookup");
    crate::goldens::golden_effects("mock_causal_note_concurrent", &log);
}

#[tokio::test]
async fn depth_two_effect_log_is_the_golden_fixture() {
    let log = over_host(plain(NestedChild::Relay)).await;
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Custom,
            EffectFamily::Custom,
            EffectFamily::Completion
        ]
    );
    // The chain of three: tool → relay → note.
    assert_eq!(parent_positions(&log), [None, None, Some(1), Some(2), None]);
    assert_eq!(custom_kind(&log, 2), "corpus:relay");
    assert_eq!(custom_kind(&log, 3), "corpus:note");
    assert_eq!(tool_output(&log, 1), "relayed:relay<lookup");
    crate::goldens::golden_effects("mock_causal_depth_two", &log);
}

#[tokio::test]
async fn same_key_serial_refused_effect_log_is_the_golden_fixture() {
    let log = over_host(Host {
        serial: true,
        ..plain(NestedChild::Same)
    })
    .await;
    // The refused child leaves no record: the parent's outcome carries the
    // refusal.
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, None]);
    assert_eq!(tool_output(&log, 1), "refused:Request");
    crate::goldens::golden_effects("mock_causal_same_key_serial_refused", &log);
}

#[tokio::test]
async fn same_key_concurrent_served_effect_log_is_the_golden_fixture() {
    let log = over_host(plain(NestedChild::Same)).await;
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, Some(1), None]);
    assert_eq!(tool_output(&log, 1), "served:leaf");
    assert_eq!(tool_output(&log, 2), "leaf");
    crate::goldens::golden_effects("mock_causal_same_key_concurrent_served", &log);
}

#[tokio::test]
async fn same_key_from_thread_refused_effect_log_is_the_golden_fixture() {
    // The case a thread-keyed re-entrancy check could not see: the nested
    // dispatch is made from a spawned OS thread. The chain refuses it; the
    // wall-clock guard in `over_host` is the "not hung" half.
    let log = over_host(Host {
        nesting: Nesting {
            child: NestedChild::Same,
            from_thread: true,
            detached: false,
        },
        serial: true,
        cancel_at_child: false,
    })
    .await;
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert_eq!(tool_output(&log, 1), "refused:Request");
    crate::goldens::golden_effects("mock_causal_same_key_from_thread_refused", &log);
}

#[tokio::test]
async fn parent_cancelled_child_in_flight_effect_log_is_the_golden_fixture() {
    let log = over_host(Host {
        cancel_at_child: true,
        ..plain(NestedChild::Never)
    })
    .await;
    // Both cancelled, the child's record after the parent's in dispatch
    // order; the run never answered, so no second completion.
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Custom
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, Some(1)]);
    assert!(is_cancelled(&log, 1), "{:?}", log.records[1].outcome);
    assert!(is_cancelled(&log, 2), "{:?}", log.records[2].outcome);
    crate::goldens::golden_effects("mock_causal_parent_cancelled_child_in_flight", &log);
}

#[tokio::test]
async fn parent_cancelled_child_queued_effect_log_is_the_golden_fixture() {
    let log = over_host(Host {
        serial: true,
        cancel_at_child: true,
        ..plain(NestedChild::NeverTwice)
    })
    .await;
    // Two children were dispatched; under the serial host the second was
    // queued behind the first when the run was dropped, and it never
    // began: one child record, not two.
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Custom
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, Some(1)]);
    assert!(is_cancelled(&log, 1), "{:?}", log.records[1].outcome);
    assert!(is_cancelled(&log, 2), "{:?}", log.records[2].outcome);
    crate::goldens::golden_effects("mock_causal_parent_cancelled_child_queued", &log);
}

#[tokio::test]
async fn detached_resolver_effect_log_is_the_golden_fixture() {
    // The tool detaches its sink; a spawned task dispatches the note
    // through the detached sink's dispatcher and answers. The parent is
    // set from the detached sink.
    let log = over_host(Host {
        nesting: Nesting {
            child: NestedChild::Note,
            from_thread: false,
            detached: true,
        },
        serial: false,
        cancel_at_child: false,
    })
    .await;
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Custom,
            EffectFamily::Completion
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, Some(1), None]);
    assert_eq!(tool_output(&log, 1), "noted:lookup");
    crate::goldens::golden_effects("mock_causal_detached_resolver", &log);
}
