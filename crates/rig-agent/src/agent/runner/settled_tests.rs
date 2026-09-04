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
        serve::{OutcomeSink, Serve},
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
            }
        }

        async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
            let mut out = sink.writer();
            for word in ["one", "two", "three"] {
                if out.text(word).await.is_err() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            let _ = out
                .finish(rig_core::test_utils::mock_final(
                    rig_core::completion::Usage::new(),
                ))
                .await;
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
        let (dispatcher, registrar, mut driver) = rig_bus::Bus::channel();
        let key = HandlerKey::from("golden/model:default");
        driver
            .register_erased(key.clone(), rig_core::serve::ErasedHandler::new(Slow))
            .expect("register");
        let recorder = rig_effect_log::EffectLogRecorder::new();
        driver.record_to(recorder.clone());
        let task = tokio::spawn(driver);
        let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", key)
            .add_hook(StopOnTextDelta)
            .build();
        let mut stream = agent.stream_prompt("go").stream().await;
        let mut stopped = false;
        while let Some(item) = stream.next().await {
            if let Err(crate::agent::StreamingError::Prompt(error)) = item {
                stopped = matches!(
                    *error,
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
        let agent = AgentBuilder::new(rig_core::test_utils::MockCompletionModel::text("never"))
            .name("golden")
            .model_route_handler("slow", Slow)
            .add_hook(SelectSlow)
            .add_hook(StopOnTextDelta)
            .record_effects()
            .build();
        let mut stream = agent.stream_prompt("go").stream().await;
        while let Some(item) = stream.next().await {
            if let Err(crate::agent::StreamingError::Prompt(error)) = item {
                assert!(matches!(
                    *error,
                    crate::completion::PromptError::PromptCancelled { .. }
                ));
            }
        }
        drop(stream);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(log.len(), 1, "the cancelled dispatch is a record");
        let report = log[0].outcome.as_ref().expect_err("cancelled in flight");
        assert_eq!(
            report.kind,
            rig_core::error::ErrorKind::Cancelled,
            "{report:?}"
        );
    }
}
