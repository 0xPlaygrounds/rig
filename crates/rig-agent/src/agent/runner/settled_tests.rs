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
            SettledOutcome::Response(response) => format!("response:{}", response.output),
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
        matches!(error, PromptError::PromptCancelled { ref reason, .. } if reason == "stopped at start")
    );
    let seen = settled.0.lock().expect("settled").clone();
    assert!(
        seen.as_deref()
            .is_some_and(|seen| seen.starts_with("error:") && seen.ends_with("stopped at start")),
        "{seen:?}"
    );
}

mod slow_stream {
    use std::time::Duration;

    use rig_core::{
        effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey},
        serve::{Dispatch, Reply, Serve},
    };

    use crate::agent::{AgentBuilder, AgentHook, HookContext, ObservationAction, TextDelta};
    use futures::StreamExt;

    /// A model that streams three deltas a while apart, then finishes.
    struct Slow;

    impl Serve for Slow {
        type Family = rig_core::effect::family::Dynamic;

        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("golden/model:default"),
                family: FamilyDescriptor::Completion {
                    model: rig_core::completion::ModelRef::new("slow"),
                    capabilities: rig_core::completion::ProviderCapabilities::default(),
                },
                layers: Vec::new(),
            }
        }

        async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
            Reply::written(|mut out| async move {
                for word in ["one", "two", "three"] {
                    if out.text(word).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                let _ = out
                    .finish(rig_core::test_utils::mock_final(
                        rig_core::completion::Usage::default(),
                    ))
                    .await;
            })
        }
    }

    struct StopOnTextDelta;
    impl AgentHook for StopOnTextDelta {
        async fn on_text_delta(
            &self,
            _ctx: &HookContext,
            _event: TextDelta<'_>,
        ) -> ObservationAction {
            ObservationAction::stop("stopped on a delta")
        }
    }

    /// A stop on a delta cancels the dispatch in flight: the engine drops
    /// the model's stream before surfacing the stop, so a model still
    /// streaming is cut off and the record is the cancel.
    #[tokio::test]
    async fn a_delta_stop_cancels_a_dispatch_still_streaming() {
        let (dispatcher, registrar, mut driver) = crate::bus::Bus::channel();
        let key = HandlerKey::from("golden/model:default");
        driver
            .register_erased(key.clone(), rig_core::serve::ErasedHandler::new(Slow))
            .expect("register");
        let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
        driver.record_to(recorder.clone());
        let task = tokio::spawn(driver);
        let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", key)
            .add_hook(StopOnTextDelta)
            .build();
        let mut stream = agent.prompt("go").stream();
        let mut stopped = false;
        while let Some(item) = stream.next().await {
            if let Err(crate::agent::StreamingError::Prompt(error)) = item {
                stopped = matches!(
                    error,
                    crate::completion::PromptError::PromptCancelled { .. }
                );
            }
        }
        drop(stream);
        assert!(stopped);
        drop((agent, dispatcher, registrar));
        task.await.expect("driver");
        let log = recorder.take();
        assert_eq!(log.len(), 1);
        let report = log[0].outcome.as_ref().expect_err("cancelled in flight");
        assert_eq!(
            report.kind,
            rig_core::error::ErrorKind::Cancelled,
            "{report:?}"
        );
    }

    struct SelectSlow;
    impl AgentHook for SelectSlow {
        fn on_model_select(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::ModelSelection<'_>,
        ) -> crate::agent::ModelSelectionAction {
            crate::agent::ModelSelectionAction::select("slow")
        }
    }

    /// The same stop over an agent's own bus: nothing polls an owned driver
    /// between runs, so the cancelled dispatch's record used to be lost —
    /// the run's drive now settles in-flight cancels before it finishes,
    /// and the log holds the cancel.
    #[tokio::test]
    async fn a_delta_stop_on_an_owned_bus_is_in_the_log() {
        let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
        let agent = AgentBuilder::new(rig_core::test_utils::MockCompletionModel::text("never"))
            .name("golden")
            .model_route_handler("slow", Slow)
            .add_hook(SelectSlow)
            .add_hook(StopOnTextDelta)
            .record_to(recorder.clone())
            .build();
        let mut stream = agent.prompt("go").stream();
        while let Some(item) = stream.next().await {
            if let Err(crate::agent::StreamingError::Prompt(error)) = item {
                assert!(matches!(
                    error,
                    crate::completion::PromptError::PromptCancelled { .. }
                ));
            }
        }
        drop(stream);
        let log = recorder.take();
        assert_eq!(log.len(), 1, "the cancelled dispatch is a record");
        let report = log[0].outcome.as_ref().expect_err("cancelled in flight");
        assert_eq!(
            report.kind,
            rig_core::error::ErrorKind::Cancelled,
            "{report:?}"
        );
    }
}

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
    use rig_core::test_utils::{MockStreamEvent, MockTurn, mock_final};

    for streamed in [false, true] {
        let model = if streamed {
            MockCompletionModel::from_stream_turns([[
                MockStreamEvent::reasoning("unfinished reasoning"),
                MockStreamEvent::FinalResponse(
                    mock_final(Usage::default()).with_finish_reason(FinishReason::Length),
                ),
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
