//! The entry-point contract: `Agent::prompt` is the one way to a runner and
//! the terminal call alone chooses the medium; `Agent::resume` continues a
//! run without a prompt; `stream()` does nothing until it is polled.
//!
//! The resume tests here continue a pristine run (turn 0, nothing pending):
//! they pin the entry point, not mid-flight resumption, which
//! `rig-cassette`'s `durable_execution` suite covers.

use futures::StreamExt;

use crate::{
    agent::AgentBuilder,
    test_utils::{MockCompletionModel, MockStreamEvent},
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
