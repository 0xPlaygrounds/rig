use rig_core::message::{AssistantMessage, CallId, ToolCall, ToolFunction, ToolName};

use super::*;

fn call(id: &str, tool: &str, path: &str) -> Message {
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(
                ToolName::new(tool).expect("tool name"),
                serde_json::json!({ "path": path }),
            ),
        ),
    )]))
}

fn result(id: &str, output: &str) -> Message {
    Message::tool_result(
        CallId::from_wire(id),
        ToolName::new("read").expect("tool name"),
        output,
    )
}

/// A history of `turns` turns drawn from `seed`: each a user message, then replies and tool
/// exchanges of random sizes.
fn history(seed: u64, turns: usize) -> Vec<Arc<Message>> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = |below: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % below
    };
    let mut messages = Vec::new();
    for turn in 0..turns {
        messages.push(Message::user(format!("question {turn}")));
        for step in 0..next(5) {
            let id = format!("{turn}-{step}");
            messages.push(call(&id, "read", &format!("{id}.rs")));
            let size = usize::try_from(next(20_000)).unwrap_or(0);
            messages.push(result(&id, &"x".repeat(size)));
        }
        let size = usize::try_from(next(4_000)).unwrap_or(0);
        messages.push(Message::assistant("y".repeat(size)));
    }
    messages.into_iter().map(Arc::new).collect()
}

/// Over generated histories and budgets, the cut keeps from a message that is no tool result,
/// and a split turn names a turn start before it.
#[test]
fn the_cut_never_lands_on_a_tool_result() {
    for seed in 0..200 {
        let messages = history(seed, 1 + usize::try_from(seed % 7).unwrap_or(0));
        for keep in [1, 500, 5_000, 20_000, 1_000_000] {
            let cut = find_cut_point(&messages, 0, messages.len(), keep);
            let kept = messages.get(cut.first_kept);
            assert!(
                kept.is_some_and(|message| !is_tool_result(message)),
                "seed {seed}, keep {keep}: {cut:?}"
            );
            if let Some(start) = cut.turn_start {
                assert!(start < cut.first_kept, "seed {seed}, keep {keep}: {cut:?}");
                let starts = messages
                    .get(start)
                    .is_some_and(|message| is_turn_start(message));
                assert!(starts, "seed {seed}, keep {keep}: {cut:?}");
            }
        }
    }
}

/// The files of earlier compactions and of the summarized calls are listed, a changed file only
/// as changed.
#[test]
fn file_lists_carry_across_compactions() {
    let mut conversation = Conversation::from_messages([
        Message::user("start"),
        call("1", "read", "a.rs"),
        result("1", "a"),
        call("2", "edit", "b.rs"),
        result("2", "ok"),
        Message::assistant("done"),
        Message::user("again"),
        Message::assistant("x".repeat(400)),
    ]);
    let previous = SummaryState {
        summary: "earlier".to_owned(),
        read_files: vec!["b.rs".to_owned(), "old.rs".to_owned()],
        modified_files: Vec::new(),
    };
    let first = conversation.id(0);
    crate::apply_summary(&mut conversation, first, &previous).expect("applied");
    let prepared = prepare_compaction(&conversation, 10).expect("something to summarize");
    assert_eq!(prepared.previous, previous);
    assert_eq!(prepared.files.read_files, ["a.rs", "old.rs"]);
    assert_eq!(prepared.files.modified_files, ["b.rs"]);
}

/// A summary state saved by an earlier version, with its tracked sets, reads as its summary.
#[test]
fn an_old_summary_state_reads_as_its_summary() {
    let old = serde_json::json!({
        "summary": "the old checkpoint",
        "tracked": [{"name": "read-files", "values": ["a.rs"]}],
    });
    let state: SummaryState = serde_json::from_value(old).expect("read");
    assert_eq!(state.summary, "the old checkpoint");
    assert!(state.read_files.is_empty());
    assert!(
        state
            .message()
            .is_some_and(|text| text.contains("the old checkpoint"))
    );
}

/// A reply that failed, was cut at the output limit or called a tool is no summary.
#[test]
fn a_failed_or_cut_summary_is_refused() {
    use rig_core::completion::{FinishReason, Usage};
    let reply = |finish| {
        let text = vec![AssistantContent::text("## Goal")];
        let origin = rig_core::message::Origin::new("test.api", "mock", "");
        let reply = CompletionResponse::new(text, Usage::default(), origin, serde_json::json!({}));
        reply.with_finish_reason(finish)
    };
    let label = "Summarization";
    let whole = summary_text(&reply(FinishReason::Stop), label);
    assert_eq!(whole, Ok("## Goal".to_owned()));
    let cut = summary_text(&reply(FinishReason::Length), label);
    assert_eq!(cut, Err(SummaryFailure::Truncated(label)));
    let filtered = summary_text(&reply(FinishReason::ContentFilter), label);
    assert_eq!(
        filtered.map_err(|why| why.to_string()),
        Err("Summarization failed: Provider finish_reason: content_filter".to_owned())
    );
}
