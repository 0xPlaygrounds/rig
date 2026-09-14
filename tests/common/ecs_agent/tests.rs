//! Synthetic controls for the success runner's observation contract. These
//! are harness checks, not genuine provider captures or migrated scenarios.

use super::EcsAgent;

#[tokio::test]
async fn expected_budget_failure_retains_completed_effects_and_run_identity() {
    use rig_core::{effect::EffectFamily, test_utils::MockTurn};
    use rig_ecs::{agent::Failure, systems::spawn_run};
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("first", "add", serde_json::json!({"x":1,"y":2})),
        MockTurn::tool_call("second", "add", serde_json::json!({"x":3,"y":4})),
    ]);
    let mut ecs = EcsAgent::for_golden(model, "", false);
    ecs.tool(crate::support::Adder);
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        "add twice",
        false,
        Some(2),
    );
    assert_eq!(
        ecs.wait_for_outcome(run).await,
        Err(Failure::MaxTurns { limit: 2 })
    );
    let log = ecs.effect_log();
    assert_eq!(
        log.records
            .iter()
            .map(|r| r.kind.family())
            .collect::<Vec<_>>(),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion,
            EffectFamily::Tool
        ]
    );
    assert!(log.records.iter().all(|r| r.outcome.is_ok()));
    assert!(!log.header.programs.is_empty());
}
use rig_core::{
    completion::Usage,
    test_utils::{MockCompletionModel, MockError, MockStreamEvent, mock_final},
};

fn stream(late_error: bool) -> MockCompletionModel {
    let mut events = vec![
        MockStreamEvent::text("answer"),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ];
    if late_error {
        events.push(MockStreamEvent::Error(MockError::provider("after final")));
    }
    MockCompletionModel::from_stream_turns([events])
}

#[tokio::test]
async fn clean_stream_returns_final_answer() {
    let mut ecs = EcsAgent::new(stream(false), "", 1);
    assert_eq!(ecs.prompt("prompt", true).await, "answer");
}

#[tokio::test]
#[should_panic(expected = "successful prompt must not contain stream error items")]
async fn error_after_final_is_not_success() {
    let mut ecs = EcsAgent::new(stream(true), "", 1);
    ecs.prompt("prompt", true).await;
}

#[tokio::test]
#[should_panic(expected = "successful prompt must not contain stream error items")]
async fn error_after_final_without_recorded_events_is_not_success() {
    let mut ecs = EcsAgent::for_golden(stream(true), "", false);
    ecs.prompt("prompt", true).await;
}

#[tokio::test]
async fn run_turn_override_does_not_change_agent_default() {
    use rig_core::test_utils::MockTurn;
    use rig_ecs::agent::{DefaultMaxTurns, MaxTurns};

    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("call", "add", serde_json::json!({"x": 1, "y": 2})),
        MockTurn::text("3"),
    ]);
    let mut ecs = EcsAgent::new(model, "", 1);
    ecs.tool(crate::support::Adder);
    assert_eq!(ecs.prompt_with_max_turns("add", false, Some(2)).await, "3");
    assert_eq!(ecs.app.world().get::<MaxTurns>(ecs.agent).unwrap().0, 1);
    assert_eq!(
        ecs.app.world().get::<DefaultMaxTurns>(ecs.agent).unwrap().0,
        Some(1)
    );
}

struct GatedMemory {
    started: std::sync::Arc<tokio::sync::Notify>,
    release: std::sync::Arc<tokio::sync::Notify>,
}
impl rig_core::serve::Serve for GatedMemory {
    type Family = rig_core::effect::family::Memory;
    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: "memory".into(),
            family: rig_core::effect::FamilyDescriptor::Memory {},
            layers: vec![],
        }
    }
    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        use rig_core::effect::{EffectKind, MemoryOp, MemoryOutcome, Outcome};
        let outcome = match kind {
            EffectKind::Memory {
                op: MemoryOp::Load { .. },
            } => MemoryOutcome::Loaded { messages: vec![] },
            EffectKind::Memory {
                op: MemoryOp::Append { .. },
            } => {
                self.started.notify_one();
                self.release.notified().await;
                MemoryOutcome::Appended
            }
            other => panic!("unexpected memory request {other:?}"),
        };
        rig_core::serve::Reply::Outcome(Ok(Outcome::Memory(outcome)))
    }
}

#[tokio::test]
async fn successful_response_waits_for_actual_memory_append() {
    use rig_core::test_utils::MockTurn;
    use rig_ecs::{
        agent::{Conversation, Remembers},
        bus::Handlers,
    };
    use std::sync::Arc;
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mut ecs = EcsAgent::new(
        MockCompletionModel::from_turns([MockTurn::text("answer")]),
        "",
        1,
    );
    let memory = Handlers::with(ecs.app.world_mut(), |handlers| {
        handlers.register(
            "memory",
            super::RuntimeHandler {
                inner: Arc::new(GatedMemory {
                    started: started.clone(),
                    release: release.clone(),
                }),
                runtime: tokio::runtime::Handle::current(),
            },
        )
    })
    .expect("bus")
    .expect("fresh memory");
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert((Remembers(memory), Conversation("gated".into())));
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let mut answer = Box::pin(ecs.prompt("prompt", false));
        tokio::select! {
            output=&mut answer => panic!("response returned before append release: {output}"),
            ()=started.notified() => {}
        }
        assert!(
            futures::poll!(&mut answer).is_pending(),
            "settled answer is not append acknowledgement"
        );
        release.notify_one();
        assert_eq!(answer.await, "answer");
    })
    .await
    .expect("gated memory completes");
    assert!(matches!(
        ecs.effect_log()
            .records
            .last()
            .expect("append record")
            .outcome,
        Ok(rig_core::effect::Outcome::Memory(
            rig_core::effect::MemoryOutcome::Appended
        ))
    ));
}
