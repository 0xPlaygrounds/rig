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

/// What a compaction of `conversation` keeping `keep` tokens summarizes right after an earlier
/// one, which [`prepare_compaction`] plans nothing for.
fn cut_again(conversation: &Conversation, keep: u64) -> Option<CompactionPreparation> {
    let messages = conversation.messages();
    let range = (summarized_prefix(conversation), messages.len());
    prepare_compaction_at(conversation, find_cut_point(messages, range, keep, false))
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
                assert!(!orphans(&messages, cut), "{at}");
                if let Some(start) = split_turn_start(&messages, 0, cut) {
                    assert!(start < cut, "{at}");
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
        assert_eq!(cut, 1, "keep {keep}");
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
    assert_eq!(find_cut_point(&messages, range, 100, false), 2);
    assert_eq!(split_turn_start(&messages, 0, 2), None);
    assert_eq!(find_cut_point(&messages, range, 100, true), 3);
    assert_eq!(split_turn_start(&messages, 0, 3), Some(2));
}

/// Every index a plugin may cut at, over generated histories with and without an earlier
/// summary, is refused or summarizes exactly the messages before it, each once, and keeps no
/// tool result without its call; the summary then goes in only at such an index.
#[test]
fn every_custom_cut_is_refused_or_keeps_calls_with_their_results() {
    for seed in 0..60 {
        let messages = history(seed, 1 + usize::try_from(seed % 5).unwrap_or(0));
        let mut conversation = Conversation::from_messages(messages.iter().cloned());
        if seed % 2 == 0 {
            let read = conversation.messages()[..1].to_vec();
            crate::apply_summary(&mut conversation, &read, &SummaryState::default())
                .expect("applied");
        }
        let start = summarized_prefix(&conversation);
        let messages = conversation.messages().to_vec();
        assert_eq!(
            crate::apply_summary(&mut conversation.clone(), &[], &SummaryState::default()),
            None
        );
        for cut in 0..messages.len() + 3 {
            let at = format!("seed {seed}, cut {cut}");
            let read = messages.get(..cut.min(messages.len())).unwrap_or_default();
            let applied =
                crate::apply_summary(&mut conversation.clone(), read, &SummaryState::default());
            let Some(prepared) = prepare_compaction_at(&conversation, cut) else {
                let fine = cut > start && cut <= messages.len() && !orphans(&messages, cut);
                assert!(!fine, "{at}: refused");
                assert!(applied.is_none() || !orphans(&messages, cut), "{at}");
                continue;
            };
            assert!(!orphans(&messages, cut), "{at}");
            assert!(applied.is_some(), "{at}");
            assert_eq!(Some(&*prepared.replaces), messages.get(..cut), "{at}");
            let summarized = prepared.to_summarize.iter().chain(&prepared.turn_prefix);
            let summarized: Vec<_> = summarized.cloned().collect();
            assert_eq!(Some(&*summarized), messages.get(start..cut), "{at}");
        }
    }
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
        Message::user("before"),
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
    let read = conversation.messages()[..1].to_vec();
    crate::apply_summary(&mut conversation, &read, &previous).expect("applied");
    let prepared = cut_again(&conversation, 10).expect("something to summarize");
    let sent = prepared.previous.and_then(|it| it.previous_summary());
    let lists = "earlier\n\n<read-files>\nb.rs\nold.rs\n</read-files>";
    assert_eq!(sent.as_deref(), Some(lists));
    assert_eq!(prepared.files.read_files, ["a.rs", "old.rs"]);
    assert_eq!(prepared.files.modified_files, ["b.rs"]);
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

/// A summary request writes no prompt cache, carries the reasoning effort when the model takes
/// one, and asks for no more output than pi's clamp leaves.
#[test]
fn a_summary_request_fits_its_output_and_writes_no_cache() {
    let vendor = rig_core::providers::registry::ProviderId::catalog("anthropic").expect("a vendor");
    let spec = rig_core::catalog::ModelSpec::new(vendor, "test")
        .with_context_window(40_000)
        .with_max_output_tokens(32_000)
        .with_reasoning(rig_core::catalog::ReasoningSupport::Unknown { default: None });
    let effort = Reasoning::Effort(rig_core::completion::Effort::Low);
    let messages = [Message::user("x".repeat(80_000))];
    let request = summary_request(&messages, (None, ""), &spec, Some(effort)).expect("a request");
    // The test spec lists no prompt cache, so no cache option is sent at all.
    assert_eq!(request.options.cache, None);
    assert_eq!(request.options.reasoning, Some(effort));
    // About 23k tokens read, at pi-ai's 3.5 characters a token, leaves less than the output limit
    // in the 40k window.
    let asked = request.max_tokens.unwrap_or(0);
    assert!((12_000..13_500).contains(&asked), "{asked}");
    // 40000 - 30000 - 4500 leaves under 32768, so only the fixed 4096 margin is kept.
    assert_eq!(clamp_max_tokens(&spec, 30_000, 32_000), 5_904);
    assert_eq!(clamp_max_tokens(&spec, 1_000, 8_000), 8_000);
}

/// A saved checkpoint with no text is still the previous one, as pi keeps the empty string: with
/// file lists it is sent in `<previous-summary>`; a split turn's history is that empty summary,
/// without the lists, not "No prior history.".
#[test]
fn a_blank_checkpoint_is_still_the_previous_one() {
    let with_files = SummaryState {
        summary: String::new(),
        read_files: vec!["a.rs".to_owned()],
        modified_files: Vec::new(),
    };
    for state in [with_files.clone(), SummaryState::default()] {
        let mut conversation = Conversation::from_messages([
            Message::user("start"),
            Message::assistant("done"),
            Message::user("next"),
            Message::assistant("x".repeat(400)),
        ]);
        let read = conversation
            .messages()
            .get(..2)
            .unwrap_or_default()
            .to_vec();
        crate::apply_summary(&mut conversation, &read, &state).expect("applied");
        let prepared = cut_again(&conversation, 10).expect("something to summarize");
        assert_eq!(prepared.previous.as_ref(), Some(&state));
        assert_eq!(no_history(prepared.previous.as_ref()), "");
    }
    assert_eq!(no_history(None), "No prior history.");
    let spec = rig_core::providers::registry::ProviderId::catalog("ollama")
        .map(|vendor| rig_core::catalog::ModelSpec::new(vendor, "test"))
        .expect("a vendor");
    let request = summary_request(
        &[Message::user("more")],
        (Some(&with_files), ""),
        &spec,
        None,
    )
    .expect("a request");
    let text = serde_json::to_string(&request.chat_history).unwrap_or_default();
    assert!(
        text.contains("<previous-summary>") && text.contains("a.rs"),
        "{text}"
    );
}

/// However often a turn is split, and though the model copies the previous summary's file lists
/// into each history summary, the summary's entry holds each list once, at its end, and reads
/// back as the state that went in; so does a path ending in a carriage return, and a tag block
/// the model wrote is no list.
#[test]
fn repeated_split_summaries_hold_one_block_of_file_lists() {
    let mut conversation = Conversation::from_messages([Message::user("start")]);
    for round in 0..4 {
        let id = format!("{round}");
        conversation.extend([
            Message::user("a question"),
            Message::assistant("an answer"),
            Message::user("a task"),
            call(&id, "read", &format!("{round}.rs")),
            result(&id, "text"),
            call(&format!("e{id}"), "edit", "changed.rs"),
            result(&format!("e{id}"), "ok"),
            Message::assistant("x".repeat(400)),
        ]);
        let cut = conversation.messages().len() - 1;
        let prepared = prepare_compaction_at(&conversation, cut).expect("a split turn");
        assert!(!prepared.turn_prefix.is_empty(), "round {round}");
        let previous = prepared.previous.as_ref();
        let echoed = previous.and_then(SummaryState::previous_summary);
        let history = match prepared.to_summarize.is_empty() {
            true => no_history(previous),
            false => format!(
                "the goal, see <read-files> below\n\n{}",
                echoed.unwrap_or_default()
            ),
        };
        let state = SummaryState {
            summary: merge_split_turn(&history, "the turn so far"),
            read_files: prepared.files.read_files,
            modified_files: prepared.files.modified_files,
        };
        crate::apply_summary(&mut conversation, &prepared.replaces, &state).expect("applied");
        let text = conversation.messages().first().and_then(|it| it.rag_text());
        let text = text.unwrap_or_default();
        assert_eq!(text.matches("\n<read-files>\n").count(), 1, "{text}");
        assert_eq!(text.matches("\n<modified-files>\n").count(), 1, "{text}");
        assert_eq!(crate::summary_state(&conversation), state, "round {round}");
        assert_eq!(state.read_files.len(), round + 1);
    }
    let echo = SummaryState {
        summary: "done\n\n<read-files>\nmodel.rs\n</read-files>\n".to_owned(),
        read_files: vec!["a\nb".to_owned(), "c\r".to_owned()],
        modified_files: Vec::new(),
    };
    let read = SummaryState::read(&echo.message().unwrap_or_default());
    assert_eq!(read.summary, "done");
    assert_eq!(read.read_files, ["c\r"]);
}

/// The output limit is fitted by pi-ai's estimate of the request at 3.5 characters a token, each
/// tool result apart, or the reported tokens plus an estimate of what follows them.
#[test]
fn the_output_clamp_counts_the_request_as_pi_ai_does() {
    let messages = [
        Message::user("x".repeat(3_500)),
        result("1", &"y".repeat(7)),
        Message::assistant("z".repeat(35)),
    ];
    assert_eq!(
        request_tokens((None, None), &messages, None),
        1_000 + 2 + 10
    );
    let system = "s".repeat(350);
    assert_eq!(
        request_tokens((Some(system.as_str()), Some("[{}]")), &messages, None),
        1_012 + 100 + 2
    );
    assert_eq!(
        request_tokens((None, None), &messages, Some((5_000, 1))),
        5_010
    );
}
