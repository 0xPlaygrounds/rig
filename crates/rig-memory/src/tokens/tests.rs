use std::sync::Arc;

use rig_core::completion::{Message, ToolDefinition, Usage};
use rig_core::conversation::Conversation;

use super::*;
use crate::{ClearToolOutputs, SummaryState, apply_summary, summary_author};

fn tools() -> Vec<Arc<ToolDefinition>> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"path": {"type": "string", "description": "x".repeat(4_000)}},
    });
    let name = rig_core::message::ToolName::new("read").expect("a name");
    vec![Arc::new(ToolDefinition::new(name, "Reads a file.", schema))]
}

fn prepared<'a>(
    model: &'a str,
    tools: &'a [Arc<ToolDefinition>],
    history: &'a Conversation,
) -> Prepared<'a> {
    Prepared {
        model,
        prompt: Some("You are a coding agent."),
        tools,
        history,
    }
}

/// While the reported usage describes the request, the estimate is that
/// usage plus what came after the reply, the tool schemas not counted
/// again; after a compaction, a cleared copy or a model change it is the
/// whole request, schemas included.
#[test]
fn the_estimate_counts_tool_schemas_once_and_is_whole_after_a_compaction() {
    let tools = tools();
    let schemas = tools
        .iter()
        .map(|tool| HeuristicTokenCounter::default().count_tool(tool) as u64)
        .sum::<u64>();
    assert!(schemas > 900, "{schemas}");
    let mut history = Conversation::new();
    history.push(Message::user("read the file"));
    let sent = prepared("m", &tools, &history).start();
    let reply = history.push(Message::assistant("x".repeat(400)));
    let anchor = Anchor {
        usage: Usage::new().input_tokens(5_000).output_tokens(100),
        through: reply,
    };
    history.push(Message::user("y".repeat(4_000)));
    let after = HeuristicTokenCounter::default().count(&Message::user("y".repeat(4_000))) as u64;

    let request = prepared("m", &tools, &history);
    let anchored = next_request_tokens(Some(&anchor), Some(sent), &request);
    assert_eq!(anchored, 5_100 + after, "schemas are in the reported input");
    let whole = request.whole_tokens();
    assert!(whole > schemas, "the whole estimate counts the schemas");
    assert_eq!(next_request_tokens(Some(&anchor), None, &request), whole);
    assert_eq!(next_request_tokens(None, Some(sent), &request), whole);

    // Another model.
    let other = prepared("n", &tools, &history);
    assert_eq!(
        next_request_tokens(Some(&anchor), Some(sent), &other),
        other.whole_tokens()
    );

    // A cleared copy: its ids stay, its entries changed.
    let mut cleared = history.view();
    cleared.edit_each(summary_author(), |entry| {
        (entry.index == 0).then(|| Message::user("read it"))
    });
    let request = prepared("m", &tools, &cleared);
    assert_eq!(
        next_request_tokens(Some(&anchor), Some(sent), &request),
        request.whole_tokens()
    );
    let mut copy = history.view();
    ClearToolOutputs::new(0).clear_conversation(&mut copy, summary_author());
    assert_eq!(
        next_request_tokens(Some(&anchor), Some(sent), &prepared("m", &tools, &copy)),
        anchored,
        "nothing to clear changes nothing"
    );

    // A compaction that keeps the reply.
    let state = SummaryState {
        summary: "s".to_owned(),
        ..SummaryState::default()
    };
    apply_summary(&mut history, Some(reply), &state);
    let request = prepared("m", &tools, &history);
    assert_eq!(
        next_request_tokens(Some(&anchor), Some(sent), &request),
        request.whole_tokens()
    );
}

#[test]
fn an_image_costs_about_twelve_hundred_tokens() {
    let image = rig_core::message::UserContent::image_base64(
        "aGk=",
        Some(rig_core::message::ImageMediaType::PNG),
        None,
    );
    let counter = HeuristicTokenCounter::default();
    assert_eq!(counter.count_user(&image), ATTACHMENT_TOKENS);
}
