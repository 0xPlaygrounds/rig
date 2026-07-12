//! Bounded correlated child-agent orchestration.

use crate::{
    runtime::{RunContext, RunControlHandle},
    wasm_compat::WasmCompatSend,
};
use std::{future::Future, sync::Arc};
use tracing::Instrument;

/// Whether a child inherits the parent run or starts a fresh correlated run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildContextMode {
    Inherit,
    Fresh,
}
/// Child lifecycle status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChildStatus {
    Queued,
    Running,
    Completed,
    Cancelled,
    Failed,
}
/// Typed child handoff with correlation.
#[derive(Clone, Debug)]
pub struct ChildHandoff<T, E = ()> {
    pub child_run_id: String,
    pub parent_call_id: Option<String>,
    pub status: ChildStatus,
    pub output: Option<T>,
    /// Typed child failure, when status is [`ChildStatus::Failed`].
    pub error: Option<E>,
}

/// Agent-native child failure.
#[derive(Debug, thiserror::Error)]
pub enum ChildAgentError {
    #[error(transparent)]
    Prompt(#[from] crate::completion::PromptError),
    #[error(transparent)]
    Session(#[from] crate::session::SessionError),
}
/// Progress observer for child work.
pub trait ChildProgress:
    crate::wasm_compat::WasmCompatSend + crate::wasm_compat::WasmCompatSync
{
    fn status(&self, status: ChildStatus);
}

/// Concurrency/depth bounded child executor.
#[derive(Clone)]
pub struct SubagentOrchestrator {
    semaphore: Arc<tokio::sync::Semaphore>,
    max_depth: usize,
}
/// History supplied to an Agent-native child run.
#[derive(Clone, Debug, Default)]
pub enum ChildHistory {
    /// Start without parent transcript history.
    #[default]
    Fresh,
    /// Seed the child with a selected parent transcript.
    Inherited(Vec<crate::completion::Message>),
}

impl SubagentOrchestrator {
    pub fn new(max_concurrency: usize, max_depth: usize) -> Self {
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrency.max(1))),
            max_depth,
        }
    }

    /// Run typed child work with inherited cancellation/deadline and correlation.
    pub async fn run_child<T, E, F, Fut>(
        &self,
        parent: &RunContext,
        name: &str,
        mode: ChildContextMode,
        progress: &dyn ChildProgress,
        work: F,
    ) -> ChildHandoff<T, E>
    where
        T: WasmCompatSend,
        E: WasmCompatSend,
        F: FnOnce(RunContext) -> Fut + WasmCompatSend,
        Fut: Future<Output = Result<T, E>> + WasmCompatSend,
    {
        progress.status(ChildStatus::Queued);
        let parent_call_id = parent.current_call_id().map(str::to_owned);
        if parent.ancestry().len() + 1 > self.max_depth || parent.should_stop() {
            progress.status(ChildStatus::Cancelled);
            return ChildHandoff {
                child_run_id: parent.run_id().to_string(),
                parent_call_id,
                status: ChildStatus::Cancelled,
                output: None,
                error: None,
            };
        }
        let permit = self.semaphore.acquire();
        let stopped = parent.stopped();
        futures::pin_mut!(permit, stopped);
        let _permit = match futures::future::select(permit, stopped).await {
            futures::future::Either::Left((Ok(permit), _)) => permit,
            futures::future::Either::Left((Err(_), _)) => {
                progress.status(ChildStatus::Failed);
                return ChildHandoff {
                    child_run_id: parent.run_id().to_string(),
                    parent_call_id,
                    status: ChildStatus::Failed,
                    output: None,
                    error: None,
                };
            }
            futures::future::Either::Right(_) => {
                progress.status(ChildStatus::Cancelled);
                return ChildHandoff {
                    child_run_id: parent.run_id().to_string(),
                    parent_call_id,
                    status: ChildStatus::Cancelled,
                    output: None,
                    error: None,
                };
            }
        };
        if parent.should_stop() {
            progress.status(ChildStatus::Cancelled);
            return ChildHandoff {
                child_run_id: parent.run_id().to_string(),
                parent_call_id,
                status: ChildStatus::Cancelled,
                output: None,
                error: None,
            };
        }
        let context = match mode {
            ChildContextMode::Inherit => parent.child(name.to_string(), crate::id::generate()),
            ChildContextMode::Fresh => parent.fresh_child(name.to_string(), crate::id::generate()),
        };
        let child_run_id = context.run_id().to_string();
        progress.status(ChildStatus::Running);
        match work(context).await {
            Ok(output) => {
                progress.status(ChildStatus::Completed);
                ChildHandoff {
                    child_run_id,
                    parent_call_id,
                    status: ChildStatus::Completed,
                    output: Some(output),
                    error: None,
                }
            }
            Err(error) => {
                progress.status(ChildStatus::Failed);
                ChildHandoff {
                    child_run_id,
                    parent_call_id,
                    status: ChildStatus::Failed,
                    output: None,
                    error: Some(error),
                }
            }
        }
    }

    /// Run an [`Agent`](crate::agent::Agent) as a correlated child using the
    /// same bounded scheduler and inherited cancellation/deadline.
    pub async fn run_agent<M>(
        &self,
        agent: &crate::agent::Agent<M>,
        parent: &RunContext,
        prompt: impl Into<crate::completion::Message>,
        history: ChildHistory,
        mode: ChildContextMode,
        progress: &dyn ChildProgress,
    ) -> ChildHandoff<crate::agent::prompt_request::PromptResponse, ChildAgentError>
    where
        M: crate::completion::CompletionModel + 'static,
    {
        let prompt = prompt.into();
        let parent_run_id = parent.run_id().to_string();
        self.run_child(
            parent,
            agent.name.as_deref().unwrap_or("child"),
            mode,
            progress,
            move |context| async move {
                let child_run_id = context.run_id().to_string();
                let span = tracing::info_span!("run_child_agent", %parent_run_id, %child_run_id);
                async move {
                let mut parent_sequence = None;
                if let (Some(store), Some(session_id)) = (&agent.session_store, &agent.session_id) {
                    parent_sequence = Some(store.append(
                        session_id, &agent.session_branch, None, Some(parent_run_id.clone()),
                        crate::session::SessionEventKind::Lifecycle {
                            status: "child_started".into(),
                            detail: Some(serde_json::json!({"parent_run_id": parent_run_id, "child_run_id": child_run_id})),
                        },
                    ).await?.sequence);
                }
                let (control, _) = RunControlHandle::new(
                    context.conversation_id().map(str::to_owned),
                    context.deadline(),
                );
                let mut runner = agent.runner(prompt).inherit_runtime(control, context);
                if let ChildHistory::Inherited(messages) = history {
                    runner = runner.history(messages);
                }
                let result = runner.run().await.map_err(ChildAgentError::from);
                if let (Some(store), Some(session_id)) = (&agent.session_store, &agent.session_id) {
                    store.append(
                        session_id, &agent.session_branch, parent_sequence, Some(child_run_id.clone()),
                        crate::session::SessionEventKind::Lifecycle {
                            status: if result.is_ok() { "child_completed" } else { "child_failed" }.into(),
                            detail: Some(serde_json::json!({"parent_run_id": parent_run_id, "child_run_id": child_run_id})),
                        },
                    ).await?;
                }
                result
                }.instrument(span).await
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    struct Progress(Mutex<Vec<ChildStatus>>);
    impl ChildProgress for Progress {
        fn status(&self, status: ChildStatus) {
            self.0.lock().unwrap().push(status);
        }
    }
    #[tokio::test]
    async fn typed_child_inherits_context_and_reports_progress() {
        let (_, context) = RunControlHandle::new(Some("conversation".into()), None);
        let progress = Progress(Mutex::new(Vec::new()));
        let result = SubagentOrchestrator::new(1, 2)
            .run_child(
                &context,
                "child",
                ChildContextMode::Inherit,
                &progress,
                |child| async move { Ok::<_, ()>((child.run_id().to_string(), 42u32)) },
            )
            .await;
        assert_eq!(result.output.as_ref().map(|(_, value)| *value), Some(42));
        assert_eq!(result.status, ChildStatus::Completed);
        assert_eq!(
            *progress.0.lock().unwrap(),
            vec![
                ChildStatus::Queued,
                ChildStatus::Running,
                ChildStatus::Completed
            ]
        );
    }

    #[tokio::test]
    async fn queued_child_cancels_before_permit_acquisition() {
        let orchestrator = SubagentOrchestrator::new(1, 2);
        let permit = orchestrator.semaphore.acquire().await.unwrap();
        let (control, context) = RunControlHandle::new(None, None);
        let progress = Progress(Mutex::new(Vec::new()));
        let child = orchestrator.run_child(
            &context,
            "queued",
            ChildContextMode::Inherit,
            &progress,
            |_| async { Ok::<_, ()>(42u32) },
        );
        futures::pin_mut!(child);
        assert!(futures::poll!(&mut child).is_pending());
        control.cancel();
        let handoff = child.await;
        assert_eq!(handoff.status, ChildStatus::Cancelled);
        assert!(handoff.output.is_none());
        drop(permit);
    }

    #[tokio::test]
    async fn agent_native_child_returns_typed_handoff() {
        let (_, context) = RunControlHandle::new(Some("conversation".into()), None);
        let agent = crate::agent::AgentBuilder::new(crate::test_utils::MockCompletionModel::text(
            "child answer",
        ))
        .build();
        let progress = Progress(Mutex::new(Vec::new()));
        let handoff = SubagentOrchestrator::new(1, 2)
            .run_agent(
                &agent,
                &context,
                "child prompt",
                ChildHistory::Fresh,
                ChildContextMode::Fresh,
                &progress,
            )
            .await;
        assert_eq!(
            handoff.output.as_ref().map(|output| output.output.as_str()),
            Some("child answer")
        );
        assert_ne!(handoff.child_run_id, context.run_id().to_string());
        assert_eq!(handoff.status, ChildStatus::Completed);
    }
}
