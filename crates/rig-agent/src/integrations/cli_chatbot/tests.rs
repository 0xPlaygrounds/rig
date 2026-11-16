use super::ChatBot;
use crate::AgentBuilder;
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use std::io::BufRead;

/// End of input stops the loop: one prompt then EOF makes one model request,
/// even with more answers scripted.
#[tokio::test]
async fn end_of_input_ends_the_chat() {
    let model = MockCompletionModel::from_stream_turns((0..3).map(|i| {
        [
            MockStreamEvent::text(format!("answer {i}")),
            MockStreamEvent::final_response(Usage::default()),
        ]
    }));
    let agent = AgentBuilder::new(model.clone()).build();

    let mut input = &b"hello\n"[..];

    let result = ChatBot::new(agent)
        .run_on(|line| input.read_line(line))
        .await;

    assert!(result.is_ok(), "{result:?}");
    assert_eq!(model.request_count(), 1);
}

/// End of input before any prompt makes no request and is not an error.
#[tokio::test]
async fn end_of_input_before_a_prompt_ends_the_chat() {
    let model = MockCompletionModel::from_stream_turns(Vec::<[MockStreamEvent; 0]>::new());
    let agent = AgentBuilder::new(model.clone()).build();

    let mut input = &b""[..];

    let result = ChatBot::new(agent)
        .run_on(|line| input.read_line(line))
        .await;

    assert!(result.is_ok(), "{result:?}");
    assert_eq!(model.request_count(), 0);
}

/// `ChatBot::run`'s future can be spawned on a multi-threaded runtime.
#[cfg(not(target_family = "wasm"))]
#[test]
fn run_is_send() {
    fn assert_send<T: Send>(_: T) {}
    let agent = AgentBuilder::new(MockCompletionModel::text("unused")).build();
    assert_send(ChatBot::new(agent).run());
}
