//! Record and replay: a run's every dispatch is recorded, in dispatch
//! order, and a fresh bus with no model and no tool behind the keys replays
//! the run to the same answer from the log alone.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use rig_agent::{
    AgentBuilder,
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_core::{
    bus::{Bus, EffectLogReplayer},
    effect::EffectFamily,
    test_utils::{MockCompletionModel, MockTurn},
};
use serde::Deserialize;
use serde_json::json;

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a run over the bus never hangs")
}

#[derive(Deserialize)]
struct SlowArgs {
    #[serde(default)]
    delay_ms: u64,
    tag: String,
}

/// Records the order tool calls *complete* in, after a per-call delay.
#[derive(Clone, Default)]
struct Slow {
    completed: Arc<Mutex<Vec<String>>>,
}

impl Tool for Slow {
    const NAME: &'static str = "slow";
    type Args = SlowArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "sleeps then answers".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"delay_ms": {"type": "integer"}, "tag": {"type": "string"}}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: SlowArgs,
    ) -> Result<String, Self::Error> {
        tokio::time::sleep(Duration::from_millis(args.delay_ms)).await;
        self.completed.lock().expect("lock").push(args.tag.clone());
        Ok(args.tag)
    }
}

fn two_tool_calls_then_done() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::from_contents([
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-1",
                rig_core::message::ToolFunction::new(
                    "slow".to_owned(),
                    json!({"delay_ms": 40, "tag": "first"}),
                ),
            )),
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-2",
                rig_core::message::ToolFunction::new(
                    "slow".to_owned(),
                    json!({"delay_ms": 0, "tag": "second"}),
                ),
            )),
        ]),
        MockTurn::text("done"),
    ])
}

#[tokio::test]
async fn a_run_records_every_dispatch_and_replays_from_the_log() {
    let recorded = AgentBuilder::new(two_tool_calls_then_done())
        .tool(Slow::default())
        .record_effects()
        .build();
    let response = within(recorded.prompt("go").max_turns(3).run())
        .await
        .expect("recorded run");
    assert_eq!(response.output, "done");
    let log = recorded.take_effect_log().expect("recording was enabled");
    assert_eq!(log.len(), 4, "two completions and two tool calls");
    assert_eq!(log[0].kind.family(), EffectFamily::Completion);
    assert_eq!(log[1].kind.family(), EffectFamily::Tool);
    assert_eq!(log[2].kind.family(), EffectFamily::Tool);
    assert_eq!(log[3].kind.family(), EffectFamily::Completion);
    assert!(log.iter().all(|record| record.outcome.is_ok()));

    // The examples-doc flow: save the session, restore it, replay it with
    // no model and no tool behind the keys.
    let saved = serde_json::to_string(&log).expect("log serializes");
    let restored: rig_core::effect::EffectLog = serde_json::from_str(&saved).expect("restores");
    let (dispatcher, registrar, mut driver) = Bus::channel();
    EffectLogReplayer::register_all(&restored, &mut driver).expect("fresh keys");
    let replay_task = tokio::spawn(driver);

    // The replayed agent advertises the same tool (its handler is the
    // replayer under the recorded key) and dispatches to the recorded keys.
    let model_key = log[0].key.clone();
    let tool_key = log[1].key.clone();
    let tool_replayer =
        EffectLogReplayer::for_key(&restored, &tool_key).expect("the log has the tool's records");
    let replayed =
        AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "replay", model_key)
            .tool_server_handle({
                let server = rig_agent::tool::server::ToolServer::new().run();
                // The registration's handler *is* the replayer, under the recorded
                // key: the catalog advertises the tool, the bus answers from the
                // log, and no `Slow` exists on this side at all.
                server.add_registered_tool(
                    rig_agent::tool::RegisteredTool::from_handler(tool_replayer)
                        .expect("a tool-family replayer"),
                );
                server
            })
            .build();
    let response = within(replayed.prompt("go").max_turns(3).run())
        .await
        .expect("replayed run");
    assert_eq!(
        response.output, "done",
        "the same PromptResponse from the record"
    );
    drop(replayed);
    drop(dispatcher);
    within(replay_task).await.expect("replay driver ends");
}
