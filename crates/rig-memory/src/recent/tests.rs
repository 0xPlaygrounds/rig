use rig_core::completion::Message;
use rig_core::conversation::pairing::validate;
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, ToolCall, ToolFunction, ToolName, UserContent,
};

use super::{LastMessages, TokenBudget, view};
use crate::{HeuristicTokenCounter, TokenCounter};

fn call(id: &str) -> Message {
    let function = ToolFunction::new(ToolName::new("t").expect("a name"), serde_json::json!({}));
    let call = ToolCall::new(CallId::from_wire(id), function);
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        call,
    )]))
}

/// The results of `ids` in one user message, between texts.
fn results(ids: &[&str]) -> Message {
    let mut content = vec![UserContent::text("Results follow:")];
    for id in ids {
        let name = ToolName::new("t").expect("a name");
        let output = vec![rig_core::message::ToolResultContent::text("ok")];
        content.push(UserContent::tool_result(
            CallId::from_wire(*id),
            name,
            output,
        ));
    }
    Message::User { content }
}

#[test]
fn a_view_never_keeps_a_result_without_its_call() {
    let history = [
        Message::user("start"),
        call("1"),
        Message::system("a note"),
        results(&["1"]),
        Message::assistant("done"),
        Message::user("next"),
    ]
    .map(std::sync::Arc::new);
    // The newest four start at the system message, before a result whose
    // call they cut off: the view moves past it.
    let kept = view(&history, &LastMessages(4));
    assert_eq!(kept, &history[4..]);
    assert!(validate(kept).is_ok());
    // A rule that keeps the call keeps the result.
    assert_eq!(view(&history, &LastMessages(5)), &history[1..]);
    // A budget cuts the same way.
    let counter = HeuristicTokenCounter::default();
    let newest: usize = history[2..]
        .iter()
        .map(|message| counter.count(message))
        .sum();
    let budget = TokenBudget::new(newest);
    assert_eq!(view(&history, &budget), &history[4..]);
    assert_eq!(view(&history, &TokenBudget::new(usize::MAX)), &history[..]);
}
