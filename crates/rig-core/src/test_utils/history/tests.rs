use super::*;
use crate::message::AssistantContent;
use crate::operation::Finish;
use crate::test_utils::{MockFrame, MockScript, MockStreamEvent};

fn whole() -> CompletionResponse {
    CompletionResponse::new(
        vec![
            AssistantContent::reasoning("think"),
            AssistantContent::text("answer"),
        ],
        Default::default(),
        crate::message::Origin::new("test.api", "mock", ""),
        serde_json::Value::Null,
    )
}

#[test]
fn a_whole_reply_and_its_stream_fold_into_one_turn() {
    assert_restated_agrees(
        &MockScript::default(),
        [MockFrame::Response(Box::new(whole()))],
        [
            MockFrame::Event(MockStreamEvent::reasoning_delta("think")),
            MockFrame::Event(MockStreamEvent::text("answer")),
            MockFrame::Event(MockStreamEvent::FinalResponse(Finish::default())),
        ],
    );
}

#[test]
#[should_panic(expected = "the same turn")]
fn a_stream_that_drops_a_block_disagrees() {
    assert_restated_agrees(
        &MockScript::default(),
        [MockFrame::Response(Box::new(whole()))],
        [
            MockFrame::Event(MockStreamEvent::text("answer")),
            MockFrame::Event(MockStreamEvent::FinalResponse(Finish::default())),
        ],
    );
}
