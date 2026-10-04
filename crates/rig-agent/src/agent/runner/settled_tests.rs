//! A run ending the effect corpus's endings matrix found unsettled.

use std::sync::{Arc, Mutex};

use rig_core::test_utils::MockCompletionModel;

use crate::agent::{
    AgentBuilder, AgentHook, HookContext, RunSettled, RunStart, RunStartAction, SettledOutcome,
};
use crate::completion::PromptError;

struct StopAtStart;
impl AgentHook for StopAtStart {
    async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        RunStartAction::stop("stopped at start")
    }
}

#[derive(Clone, Default)]
struct Settled(Arc<Mutex<Option<String>>>);
impl AgentHook for Settled {
    async fn on_run_settled(&self, _ctx: &HookContext, event: RunSettled<'_>) {
        let seen = match event.outcome {
            SettledOutcome::Response(response) => format!("response:{}", response.output()),
            SettledOutcome::Error(reason) => format!("error:{reason}"),
        };
        *self.0.lock().expect("settled") = Some(seen);
    }
}

/// A blocking run a hook stops settles: the fold drains the engine after
/// the error it yields, so `on_run_settled` sees the error. (It used to
/// return at the yield and drop the engine before the hook fired.)
#[tokio::test]
async fn a_blocking_run_a_hook_stops_settles_with_the_error() {
    let settled = Settled::default();
    let agent = AgentBuilder::new(MockCompletionModel::text("never asked"))
        .add_hook(StopAtStart)
        .add_hook(settled.clone())
        .build();
    let error = agent.prompt("go").await.expect_err("stopped");
    assert!(
        matches!(error, PromptError::Cancelled { ref reason, .. } if reason == "stopped at start")
    );
    let seen = settled.0.lock().expect("settled").clone();
    assert!(
        seen.as_deref()
            .is_some_and(|seen| seen.starts_with("error:") && seen.ends_with("stopped at start")),
        "{seen:?}"
    );
}

mod slow_stream {}

#[derive(Clone, Default)]
struct CommittedHistory(Arc<Mutex<Option<Vec<rig_core::message::Message>>>>);

impl AgentHook for CommittedHistory {
    async fn on_run_settled(&self, _ctx: &HookContext, event: RunSettled<'_>) {
        let SettledOutcome::Error(error) = event.outcome else {
            panic!("the reasoning-only capped turn fails");
        };
        assert!(error.contains(&rig_core::completion::FinishReason::Length.no_answer_message()));
        let previous = self
            .0
            .lock()
            .expect("history")
            .replace(event.messages.expect("the run was constructed").to_vec());
        assert!(previous.is_none(), "exactly one settlement");
    }
}

/// The hook must expose actual committed state on both error surfaces.
/// A controlled model pins the no-answer decision independently of whether
/// a live provider happens to spend its small cap entirely on reasoning;
/// the six-wire capped cassette cells use the same settlement field.
#[tokio::test]
async fn capped_reasoning_settlement_exposes_only_the_committed_prompt() {
    use futures::StreamExt;
    use rig_core::completion::{FinishReason, Usage};
    use rig_core::message::{AssistantContent, Message, Reasoning};
    use rig_core::operation::Finish;
    use rig_core::test_utils::{MockStreamEvent, MockTurn, mock_final};

    for streamed in [false, true] {
        let model = if streamed {
            MockCompletionModel::from_stream_turns([[
                MockStreamEvent::reasoning("unfinished reasoning"),
                MockStreamEvent::FinalResponse(Finish {
                    reason: Some(FinishReason::Length),
                    ..mock_final(Usage::default())
                }),
            ]])
        } else {
            MockCompletionModel::from_turns([MockTurn::from_content(AssistantContent::Reasoning(
                Reasoning::new("unfinished reasoning"),
            ))
            .with_finish_reason(FinishReason::Length)])
        };
        let captured = CommittedHistory::default();
        let agent = AgentBuilder::new(model).add_hook(captured.clone()).build();
        if streamed {
            let mut stream = agent.prompt("solve this").stream();
            let mut failed = false;
            while let Some(item) = stream.next().await {
                if item.is_err() {
                    failed = true;
                    break;
                }
            }
            assert!(failed);
        } else {
            agent
                .prompt("solve this")
                .await
                .expect_err("capped reasoning has no answer");
        }
        assert_eq!(
            captured.0.lock().expect("history").as_deref(),
            Some([Message::user("solve this")].as_slice())
        );
    }
}
