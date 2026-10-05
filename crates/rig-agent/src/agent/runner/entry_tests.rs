//! The entry-point contract: `Agent::prompt` is the one way to a runner and
//! the terminal call alone chooses the medium; `Agent::resume` continues a
//! run without a prompt; `stream()` does nothing until it is polled.
//!
//! The resume tests here continue a pristine run (turn 0, nothing pending):
//! they pin the entry point, not mid-flight resumption, which
//! `rig-cassette`'s `durable_execution` suite covers.

use futures::StreamExt;

use crate::{
    agent::{AgentBuilder, MultiTurnStreamItem},
    completion::Message,
    run::AgentRun,
    test_utils::{CountingMemory, MockCompletionModel, MockStreamEvent},
};

fn unary_model() -> MockCompletionModel {
    MockCompletionModel::text("collected")
}

fn streaming_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("streamed"),
        MockStreamEvent::final_response(crate::completion::Usage::default()),
    ]])
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

/// A resumed run carries its history, so nothing is loaded, but with memory
/// and a conversation configured the messages it adds are appended once, on
/// either medium (#2244).
#[tokio::test]
async fn resume_appends_its_messages_without_loading() {
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

    assert_eq!(response.output(), "collected");
    assert_eq!(memory.load_count(), 0, "the run carries its history");
    assert_eq!(
        memory.append_count(),
        1,
        "the run's messages are appended once"
    );

    let agent = AgentBuilder::new(streaming_model())
        .memory(memory.clone())
        .build();
    let run = AgentRun::from_spec(&agent.run_spec(), Message::user("from the run"), None);
    let mut stream = agent.resume(run).conversation("thread").stream();
    let mut final_output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("a stream item") {
            final_output = Some(response.output());
        }
    }
    assert_eq!(final_output.as_deref(), Some("streamed"));
    assert_eq!(memory.load_count(), 0, "the run carries its history");
    assert_eq!(
        memory.append_count(),
        2,
        "the streamed run appends once too"
    );
}
