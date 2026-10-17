//! A Rig agent as an A2A agent executor.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use a2a::{
    A2AError, Artifact, Message, Part, Role, StreamResponse, Task, TaskArtifactUpdateEvent, TaskId,
    TaskState, TaskStatus, TaskStatusUpdateEvent, new_message_id,
};
use a2a_server::{AgentExecutor, ExecutorContext};
use futures::StreamExt;
use futures::stream::{AbortHandle, Abortable, BoxStream};
use rig_agent::Agent;
use rig_agent::agent::MultiTurnStreamItem;
use rig_core::streaming::{Item, StreamEvent};

use crate::parts::user_message;

/// The id of the text artifact each task's reply streams into.
const RESPONSE_ARTIFACT: &str = "response";

type Contexts = Arc<Mutex<HashMap<String, Vec<rig_core::message::Message>>>>;
type Running = Arc<Mutex<HashMap<TaskId, AbortHandle>>>;

/// A Rig [`Agent`] as an A2A [`AgentExecutor`].
///
/// Each execution prompts the agent with the incoming message and the history
/// of its `contextId`, streams the reply's text into one artifact, and ends the
/// task completed, failed, or rejected when a part has no Rig content kind.
/// A completed run's messages join the context's history, which is kept in
/// memory for the life of the executor and shared by its clones. Cancelling a
/// task drops its run.
#[derive(Clone)]
pub struct A2aExecutor {
    agent: Agent,
    contexts: Contexts,
    running: Running,
}

impl A2aExecutor {
    /// An executor that answers every task with `agent`.
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            contexts: Arc::default(),
            running: Arc::default(),
        }
    }
}

impl AgentExecutor for A2aExecutor {
    fn execute(
        &self,
        ctx: ExecutorContext,
    ) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        let (handle, registration) = AbortHandle::new_pair();
        lock(&self.running).insert(ctx.task_id.clone(), handle);
        let registered = Registered {
            running: Arc::clone(&self.running),
            task_id: ctx.task_id.clone(),
        };
        let run = run(
            self.agent.clone(),
            Arc::clone(&self.contexts),
            ctx,
            registered,
        );
        Abortable::new(run, registration).boxed()
    }

    fn cancel(&self, ctx: ExecutorContext) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        if let Some(handle) = lock(&self.running).remove(&ctx.task_id) {
            handle.abort();
        }
        let ids = Ids {
            task: ctx.task_id,
            context: ctx.context_id,
        };
        futures::stream::iter([Ok(ids.status(TaskState::Canceled, None))]).boxed()
    }
}

/// Unregisters a task's abort handle when its run is dropped.
struct Registered {
    running: Running,
    task_id: TaskId,
}

impl Drop for Registered {
    fn drop(&mut self) {
        lock(&self.running).remove(&self.task_id);
    }
}

fn run(
    agent: Agent,
    contexts: Contexts,
    ctx: ExecutorContext,
    registered: Registered,
) -> impl futures::Stream<Item = Result<StreamResponse, A2AError>> + Send + 'static {
    async_stream::stream! {
        let _registered = registered;
        let ids = Ids {
            task: ctx.task_id,
            context: ctx.context_id,
        };
        let Some(message) = ctx.message else {
            yield Ok(ids.status(TaskState::Failed, Some(ids.agent_message("No message to answer."))));
            return;
        };
        let mut history = ctx
            .stored_task
            .and_then(|task| task.history)
            .unwrap_or_default();
        if history.last().map(|last| &last.message_id) != Some(&message.message_id) {
            history.push(message.clone());
        }
        let prompt = match user_message(&message) {
            Ok(prompt) => prompt,
            Err(unsupported) => {
                let reason = ids.agent_message(unsupported.to_string());
                yield Ok(StreamResponse::Task(ids.task(TaskState::Rejected, Some(reason), history)));
                return;
            }
        };
        yield Ok(StreamResponse::Task(ids.task(TaskState::Working, None, history)));

        let prior = lock(&contexts).get(&ids.context).cloned().unwrap_or_default();
        let mut stream = agent.prompt(prompt).history(prior).stream();
        let mut streamed = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                    text,
                    ..
                }))) => {
                    yield Ok(ids.chunk(text, streamed, false));
                    streamed = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    lock(&contexts)
                        .entry(ids.context.clone())
                        .or_default()
                        .extend(response.messages.iter().cloned());
                    // The final answer replaces the streamed text, which can
                    // include retried turns and text written before tool calls.
                    yield Ok(ids.chunk(response.output(), false, true));
                    yield Ok(ids.status(TaskState::Completed, None));
                    return;
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(task_id = %ids.task, %error, "A2A task failed");
                    yield Ok(ids.status(TaskState::Failed, Some(ids.agent_message("The agent run failed."))));
                    return;
                }
            }
        }
        yield Ok(ids.status(TaskState::Failed, Some(ids.agent_message("The agent run ended without an answer."))));
    }
}

/// The task and context one execution reports on.
struct Ids {
    task: TaskId,
    context: String,
}

impl Ids {
    fn agent_message(&self, text: impl Into<String>) -> Message {
        Message {
            message_id: new_message_id(),
            context_id: Some(self.context.clone()),
            task_id: Some(self.task.clone()),
            role: Role::Agent,
            parts: vec![Part::text(text)],
            metadata: None,
            extensions: None,
            reference_task_ids: None,
        }
    }

    fn task(&self, state: TaskState, message: Option<Message>, history: Vec<Message>) -> Task {
        Task {
            id: self.task.clone(),
            context_id: self.context.clone(),
            status: status(state, message),
            artifacts: None,
            history: Some(history),
            metadata: None,
        }
    }

    fn status(&self, state: TaskState, message: Option<Message>) -> StreamResponse {
        StreamResponse::StatusUpdate(TaskStatusUpdateEvent {
            task_id: self.task.clone(),
            context_id: self.context.clone(),
            status: status(state, message),
            metadata: None,
        })
    }

    /// A piece of the response artifact: appended to the pieces before it, or
    /// replacing them when `append` is false.
    fn chunk(&self, text: String, append: bool, last_chunk: bool) -> StreamResponse {
        StreamResponse::ArtifactUpdate(TaskArtifactUpdateEvent {
            task_id: self.task.clone(),
            context_id: self.context.clone(),
            artifact: Artifact {
                artifact_id: RESPONSE_ARTIFACT.to_owned(),
                name: Some(RESPONSE_ARTIFACT.to_owned()),
                description: None,
                parts: vec![Part::text(text)],
                metadata: None,
                extensions: None,
            },
            append: Some(append),
            last_chunk: Some(last_chunk),
            metadata: None,
        })
    }
}

fn status(state: TaskState, message: Option<Message>) -> TaskStatus {
    TaskStatus {
        state,
        message,
        timestamp: Some(chrono::Utc::now()),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
