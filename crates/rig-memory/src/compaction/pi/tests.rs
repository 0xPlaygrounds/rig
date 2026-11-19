use rig_core::message::{AssistantMessage, CallId, ToolCall, ToolFunction, ToolName};

use super::*;
use crate::SummaryState;

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

/// Whether keeping from `cut` keeps a result whose call it summarizes.
fn orphans(messages: &[Arc<Message>], cut: usize) -> bool {
    let kept = messages.get(cut..).unwrap_or_default();
    let summarized = messages.get(..cut).unwrap_or_default();
    kept.iter().any(|message| {
        message.results().any(|result| {
            summarized.iter().any(|older| match &**older {
                Message::Assistant(reply) => reply.tool_calls().any(|call| call.id == result.call),
                _ => false,
            })
        })
    })
}

/// A history of `turns` turns drawn from `seed`: each a user message, then replies and tool
/// exchanges of random sizes, some with a system message between a call and its result, and
/// some ending with a large result.
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
            if next(3) == 0 {
                messages.push(Message::system("keep going"));
            }
            let size = usize::try_from(next(20_000)).unwrap_or(0);
            messages.push(result(&id, &"x".repeat(size)));
        }
        if turn + 1 < turns || next(2) == 0 {
            let size = usize::try_from(next(4_000)).unwrap_or(0);
            messages.push(Message::assistant("y".repeat(size)));
        }
    }
    messages.into_iter().map(Arc::new).collect()
}

/// Over generated histories and budgets, the cut never keeps a tool result without its call, and
/// a split turn names a turn start before it.
#[test]
fn the_cut_never_separates_a_call_from_its_result() {
    for seed in 0..200 {
        let messages = history(seed, 1 + usize::try_from(seed % 7).unwrap_or(0));
        for keep in [1, 500, 5_000, 20_000, 1_000_000] {
            for overflowed in [false, true] {
                let cut = find_cut_point(&messages, (0, messages.len()), keep, overflowed);
                let at = format!("seed {seed}, keep {keep}, overflowed {overflowed}: {cut:?}");
                assert!(!orphans(&messages, cut.first_kept), "{at}");
                if let Some(start) = cut.turn_start {
                    assert!(start < cut.first_kept, "{at}");
                    let starts = messages
                        .get(start)
                        .is_some_and(|message| is_turn_start(message));
                    assert!(starts, "{at}");
                }
            }
        }
    }
}

/// A system message between a reply's calls and their results is no place to cut, nor is a
/// trailing result over the budget alone: the cut keeps the reply that made the calls.
#[test]
fn a_cut_never_lands_inside_a_tool_exchange() {
    let mut batch = AssistantMessage::new(Vec::new());
    for id in ["1", "2"] {
        if let Message::Assistant(reply) = call(id, "read", "a.rs") {
            batch.content.extend(reply.content);
        }
    }
    let messages = [
        Message::user("start"),
        Message::Assistant(batch),
        Message::system("keep going"),
        result("1", "x"),
        Message::system("keep going"),
        result("2", &"x".repeat(2_000)),
    ]
    .map(Arc::new);
    for keep in [1, 100, 400] {
        let cut = find_cut_point(&messages, (0, messages.len()), keep, false);
        assert_eq!(cut.first_kept, 1, "keep {keep}");
    }
}

/// After an overflow, a last message over the budget alone is summarized as the turn's earlier
/// part rather than sent again.
#[test]
fn an_oversized_last_message_is_summarized_after_an_overflow() {
    let messages = [
        Message::user("first"),
        Message::assistant("done"),
        Message::user("x".repeat(4_000)),
    ]
    .map(Arc::new);
    let range = (0, messages.len());
    assert_eq!(
        find_cut_point(&messages, range, 100, false),
        CutPoint {
            first_kept: 2,
            turn_start: None
        }
    );
    assert_eq!(
        find_cut_point(&messages, range, 100, true),
        CutPoint {
            first_kept: 3,
            turn_start: Some(2)
        }
    );
}

/// Tokens are counted in UTF-16 code units, each tool result rounded apart, and a long result is
/// cut at 2000 units without splitting a character.
#[test]
fn counting_and_cutting_go_by_utf16_units() {
    let emoji = "😀".repeat(1_500);
    assert_eq!(estimate_message_tokens(&Message::user(emoji.as_str())), 750);
    let mut two = result("1", "a");
    if let (Message::User { content }, Message::User { content: more }) =
        (&mut two, result("2", "b"))
    {
        content.extend(more);
    }
    assert_eq!(estimate_message_tokens(&two), 2);
    let cut = truncate_for_summary(&emoji, TOOL_RESULT_MAX_CHARS);
    assert!(cut.starts_with(&"😀".repeat(1_000)));
    assert!(
        cut.ends_with("[... 1000 more characters truncated]"),
        "{cut}"
    );
    let odd = format!("a{emoji}");
    let cut = truncate_for_summary(&odd, TOOL_RESULT_MAX_CHARS);
    assert!(
        cut.starts_with(&format!("a{}\n", "😀".repeat(999))),
        "{cut}"
    );
}

/// Text parts of a user message or a tool result are joined with nothing between them, as pi
/// serializes them; results come before the user's text.
#[test]
fn serialized_text_parts_are_joined_as_pi_joins_them() {
    let mut message = result("1", "abc");
    if let Message::User { content } = &mut message {
        content.push(UserContent::text("de"));
        content.push(UserContent::text("f"));
    }
    let text = serialize_conversation(&[message]);
    assert_eq!(text, "[Tool result]: abc\n\n[User]: def");
}

/// The files of earlier compactions and of the summarized calls are listed, a changed file only
/// as changed, an empty path not at all, and the previous summary goes as it was.
#[test]
fn file_lists_carry_across_compactions() {
    let mut conversation = Conversation::from_messages([
        Message::user("start"),
        call("1", "read", "a.rs"),
        result("1", "a"),
        call("2", "edit", "b.rs"),
        result("2", "ok"),
        call("3", "read", ""),
        result("3", "nothing"),
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
    let prepared = prepare_compaction(&conversation, 10, false).expect("something to summarize");
    assert_eq!(
        prepared.previous.as_deref(),
        Some("earlier\n\n<read-files>\nb.rs\nold.rs\n</read-files>")
    );
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

/// A reply that failed, was cut at the output limit or called a tool is no summary; a whole one
/// joins its text parts with newlines.
#[test]
fn a_failed_or_cut_summary_is_refused() {
    use rig_core::completion::{FinishReason, Usage};
    let reply = |finish| {
        let text = vec![
            AssistantContent::text("## Goal"),
            AssistantContent::text("- ship it"),
        ];
        let origin = rig_core::message::Origin::new("test.api", "mock", "");
        let reply = CompletionResponse::new(text, Usage::default(), origin, serde_json::json!({}));
        reply.with_finish_reason(finish)
    };
    let label = "Summarization";
    let whole = summary_text(&reply(FinishReason::Stop), label);
    assert_eq!(whole, Ok("## Goal\n- ship it".to_owned()));
    let cut = summary_text(&reply(FinishReason::Length), label);
    assert_eq!(cut, Err(SummaryFailure::Truncated(label)));
    let filtered = summary_text(&reply(FinishReason::ContentFilter), label);
    assert_eq!(
        filtered.map_err(|why| why.to_string()),
        Err("Summarization failed: Provider finish_reason: content_filter".to_owned())
    );
}

/// A summary request writes no prompt cache and asks for no more output than the window leaves.
#[test]
fn a_summary_request_fits_its_output_and_writes_no_cache() {
    let spec = rig_core::providers::registry::ProviderId::catalog("anthropic")
        .map(|vendor| {
            rig_core::catalog::ModelSpec::new(vendor, "test")
                .with_context_window(40_000)
                .with_max_output_tokens(32_000)
        })
        .expect("a vendor");
    let messages = [Message::user("x".repeat(80_000))];
    let request = summary_request(&messages, (None, ""), &spec, None).expect("a request");
    assert_eq!(request.options.cache, None);
    let most = request.max_tokens.expect("an output limit");
    assert!((1..32_000).contains(&most), "{most}");
    assert_eq!(clamp_max_tokens(&spec, 1_000, 8_000), 8_000);
}

/// A summary request reads no more than fits the summarizer's input limit, its oldest text left
/// out as it says, so a smaller model the agent switched to can still write it.
#[test]
fn a_summary_request_fits_a_smaller_model() {
    let spec = rig_core::providers::registry::ProviderId::catalog("ollama")
        .map(|vendor| {
            rig_core::catalog::ModelSpec::new(vendor, "small").with_context_window(40_000)
        })
        .expect("a vendor");
    let messages: Vec<Message> = (0..10)
        .map(|n| Message::user(format!("TURN-{n} {}", "x".repeat(40_000))))
        .collect();
    let request = summary_request(&messages, (None, ""), &spec, None).expect("a request");
    let text = serde_json::to_string(&request.chat_history).unwrap_or_default();
    assert!(
        text.contains("earlier characters left out"),
        "the cut says so"
    );
    assert!(
        text.contains("TURN-9") && !text.contains("TURN-0"),
        "the newest stay"
    );
    let input = estimate_tokens(&request.chat_history);
    assert!(input < 40_000, "{input}");
}
