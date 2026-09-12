//! The entry-point contract: `Agent::prompt` is the one way to a runner and
//! the terminal call alone chooses the medium; `Agent::resume` continues a
//! run without a prompt; `stream()` does nothing until it is polled.
//!
//! The resume tests here continue a pristine run (turn 0, nothing pending):
//! they pin the entry point, not mid-flight resumption, which
//! `rig-verify`'s `durable_execution` suite covers.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use futures::StreamExt;

use crate::{
    agent::{AgentBuilder, AgentHook, HookContext, MultiTurnStreamItem, RunStart, RunStartAction},
    completion::Message,
    run::AgentRun,
    test_utils::{CountingMemory, MockCompletionModel, MockStreamEvent},
};

/// Records what the run-scoped context said about the medium.
#[derive(Clone, Default)]
struct MediumProbe {
    seen: Arc<AtomicBool>,
    streaming: Arc<AtomicBool>,
}

impl AgentHook for MediumProbe {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        self.seen.store(true, Ordering::SeqCst);
        self.streaming.store(ctx.is_streaming(), Ordering::SeqCst);
        RunStartAction::Continue
    }
}

fn unary_model() -> MockCompletionModel {
    MockCompletionModel::text("collected")
}

fn streaming_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("streamed"),
        MockStreamEvent::final_response(crate::completion::Usage::new()),
    ]])
}

/// An awaited runner asks the provider for a complete response and tells its
/// hooks so: the mock scripted only unary turns, and it answers.
#[tokio::test]
async fn an_awaited_prompt_runs_unary() {
    let probe = MediumProbe::default();
    let agent = AgentBuilder::new(unary_model())
        .add_hook(probe.clone())
        .build();

    let response = agent.prompt("go").await.expect("a unary run");

    assert_eq!(response.output, "collected");
    assert!(probe.seen.load(Ordering::SeqCst));
    assert!(!probe.streaming.load(Ordering::SeqCst));
}

/// The same runner, streamed, asks the provider for a stream instead: the
/// mock scripted only stream turns, and it answers.
#[tokio::test]
async fn a_streamed_prompt_runs_streaming() {
    let probe = MediumProbe::default();
    let agent = AgentBuilder::new(streaming_model())
        .add_hook(probe.clone())
        .build();

    let mut stream = agent.prompt("go").stream();
    let mut final_output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("a stream item") {
            final_output = Some(response.output);
        }
    }

    assert_eq!(final_output.as_deref(), Some("streamed"));
    assert!(probe.seen.load(Ordering::SeqCst));
    assert!(probe.streaming.load(Ordering::SeqCst));
}

/// The medium is not a property of the runner: a runner built for one
/// medium and driven in the other reaches the provider through the other
/// operation. The mock has no turn scripted for it, so the run fails at the
/// provider — nothing in between silently switched medium.
#[tokio::test]
async fn the_terminal_alone_chooses_the_medium() {
    let agent = AgentBuilder::new(unary_model()).build();
    let mut stream = agent.prompt("go").stream();
    let first = stream.next().await.expect("the stream yields its error");
    let error = first.expect_err("a unary-only mock cannot serve a stream");
    assert!(
        error.to_string().contains("no scripted streaming turn"),
        "the provider was asked for a stream: {error}"
    );

    let agent = AgentBuilder::new(streaming_model()).build();
    let error = agent
        .prompt("go")
        .await
        .expect_err("a stream-only mock cannot serve a unary call");
    assert!(
        error.to_string().contains("no scripted completion turn"),
        "the provider was asked for a completion: {error}"
    );
}

/// `stream()` is lazy: building the stream loads no memory and touches no
/// hook; dropping it unpolled leaves the run unstarted.
#[tokio::test]
async fn a_stream_does_nothing_until_polled() {
    let memory = CountingMemory::default();
    let probe = MediumProbe::default();
    let agent = AgentBuilder::new(streaming_model())
        .memory(memory.clone())
        .add_hook(probe.clone())
        .build();

    let stream = agent.prompt("go").conversation("thread").stream();
    assert_eq!(memory.load_count(), 0, "no load before the first poll");
    assert!(!probe.seen.load(Ordering::SeqCst));
    drop(stream);
    assert_eq!(memory.load_count(), 0, "an unpolled stream does nothing");
    assert_eq!(memory.append_count(), 0);

    let mut stream = agent.prompt("go").conversation("thread").stream();
    let _ = stream.next().await.expect("the first poll starts the run");
    assert_eq!(memory.load_count(), 1, "the first poll loads memory");
    while stream.next().await.is_some() {}
    assert_eq!(memory.append_count(), 1, "a completed run saves once");
}

/// `Agent::resume` continues a run from its own state: no prompt is
/// supplied, the run's prompt reaches the provider, and what the runner
/// still supplies — its hooks — applies.
#[tokio::test]
async fn resume_continues_a_run_without_a_prompt() {
    let model = unary_model();
    let recorded = model.clone();
    let probe = MediumProbe::default();
    let agent = AgentBuilder::new(model).build();
    let run = AgentRun::from_spec(&agent.run_spec(), Message::user("from the run"), None);

    let response = agent
        .resume(run)
        .add_hook(probe.clone())
        .await
        .expect("the resumed run completes");

    assert_eq!(response.output, "collected");
    assert!(
        probe.seen.load(Ordering::SeqCst),
        "a hook added on the resuming runner fires"
    );
    let requests = recorded.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]
            .chat_history
            .last()
            .and_then(Message::rag_text)
            .as_deref(),
        Some("from the run"),
        "the run's own prompt is what the provider saw"
    );
}

/// The streamed twin of the above: a resumed run streams from its own
/// state, its prompt reaches the provider, and its hooks see the medium.
#[tokio::test]
async fn resume_streams_a_run_without_a_prompt() {
    let model = streaming_model();
    let recorded = model.clone();
    let probe = MediumProbe::default();
    let agent = AgentBuilder::new(model).build();
    let run = AgentRun::from_spec(&agent.run_spec(), Message::user("from the run"), None);

    let mut stream = agent.resume(run).add_hook(probe.clone()).stream();
    let mut final_output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("a stream item") {
            final_output = Some(response.output);
        }
    }

    assert_eq!(final_output.as_deref(), Some("streamed"));
    assert!(probe.streaming.load(Ordering::SeqCst));
    let requests = recorded.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]
            .chat_history
            .last()
            .and_then(Message::rag_text)
            .as_deref(),
        Some("from the run")
    );
}

/// A resumed run carries its history: even with memory and a conversation
/// configured, nothing is loaded and nothing is appended — on either medium.
#[tokio::test]
async fn resume_neither_loads_nor_saves_memory() {
    let memory = CountingMemory::default();
    let agent = AgentBuilder::new(unary_model())
        .memory(memory.clone())
        .build();
    let run = AgentRun::from_spec(&agent.run_spec(), Message::user("from the run"), None);

    let response = agent
        .resume(run)
        .conversation("thread")
        .await
        .expect("the resumed run completes");

    assert_eq!(response.output, "collected");
    assert_eq!(memory.load_count(), 0);
    assert_eq!(memory.append_count(), 0);

    let agent = AgentBuilder::new(streaming_model())
        .memory(memory.clone())
        .build();
    let run = AgentRun::from_spec(&agent.run_spec(), Message::user("from the run"), None);
    let mut stream = agent.resume(run).conversation("thread").stream();
    let mut final_output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("a stream item") {
            final_output = Some(response.output);
        }
    }
    assert_eq!(final_output.as_deref(), Some("streamed"));
    assert_eq!(memory.load_count(), 0);
    assert_eq!(memory.append_count(), 0);
}

/// The number of hook events is the same whichever medium drives the run:
/// one run start each.
#[tokio::test]
async fn both_media_fire_run_start_once() {
    let counter = Arc::new(AtomicUsize::new(0));
    #[derive(Clone)]
    struct Count(Arc<AtomicUsize>);
    impl AgentHook for Count {
        async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
            self.0.fetch_add(1, Ordering::SeqCst);
            RunStartAction::Continue
        }
    }

    let agent = AgentBuilder::new(unary_model())
        .add_hook(Count(counter.clone()))
        .build();
    agent.prompt("go").await.expect("unary");
    assert_eq!(counter.load(Ordering::SeqCst), 1);

    let agent = AgentBuilder::new(streaming_model())
        .add_hook(Count(counter.clone()))
        .build();
    let mut stream = agent.prompt("go").stream();
    while stream.next().await.is_some() {}
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}
