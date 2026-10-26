use super::*;
use rig_core::message::{AssistantMessage, CallId, ToolCall, ToolFunction, ToolName, ToolResult};

fn call(id: &str, tool: &str, path: &str) -> Message {
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(
                ToolName::new(tool).expect("tool name"),
                rig_core::serde_json::json!({ "path": path }),
            ),
        ),
    )]))
}

fn result(id: &str, output: &str) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            is_error: false,
            call: CallId::from_wire(id),
            name: ToolName::new("read").expect("tool name"),
            content: vec![ToolResultContent::text(output)],
        })],
    }
}

#[test]
fn a_later_set_overrides_an_earlier_one_in_the_summary() {
    let mut state = SummaryState {
        summary: "did things".to_owned(),
        tracked: Vec::new(),
    };
    let messages = [
        call("1", "read", "a.rs"),
        result("1", "ok"),
        call("2", "read", "b.rs"),
        result("2", "ok"),
        call("3", "edit", "b.rs"),
        result("3", "ok"),
    ];
    state.track(
        &messages,
        &[
            TrackArgument::new("read", "path", "read-files"),
            TrackArgument::new("edit", "path", "modified-files"),
        ],
    );
    let text = state.message().expect("a summary");
    assert!(text.contains("<read-files>\na.rs\n</read-files>"), "{text}");
    assert!(
        text.contains("<modified-files>\nb.rs\n</modified-files>"),
        "{text}"
    );
}

#[test]
fn a_cut_never_separates_a_call_from_its_result() {
    let messages = [
        Message::user("start"),
        call("1", "read", "a.rs"),
        result("1", "x"),
        Message::assistant("done"),
    ]
    .map(Arc::new);
    let counter = HeuristicTokenCounter::default();
    assert_eq!(cut_at(&messages, 0, 0, false, &counter), Some(3));
    assert_eq!(cut_at(&messages, 0, usize::MAX, false, &counter), None);
    assert_eq!(cut_at(&messages, 0, usize::MAX, true, &counter), Some(3));
}

fn spec(window: u32) -> Option<ModelSpec> {
    rig_core::providers::registry::ProviderId::catalog("ollama")
        .map(|vendor| ModelSpec::new(vendor, "test").with_context_window(window))
}

/// Where `reason` cuts `messages` for a model with `window` tokens.
fn cut(messages: &[Arc<Message>], window: u32, reason: &CompactReason) -> Option<usize> {
    let spec = spec(window);
    assert!(spec.is_some());
    CompactionPolicy::default().cut(messages, 0, reason, spec.as_ref())
}

/// A conversation of turns, each a question and a long answer.
fn turns(count: usize) -> Vec<Arc<Message>> {
    let answer = "word ".repeat(2_000);
    (0..count)
        .flat_map(|turn| {
            [
                Message::user(format!("question {turn}")),
                Message::assistant(answer.clone()),
            ]
        })
        .map(Arc::new)
        .collect()
}

#[test]
fn a_compaction_keeps_the_recent_work_unless_asked() {
    let (short, long) = (turns(2), turns(20));
    let asked = CompactReason::Asked {
        focus: String::new(),
    };
    // Asked, or refused as too long, a short conversation is summarized
    // all but its newest reply; near the window, it is left alone.
    assert_eq!(cut(&short, 200_000, &asked), Some(short.len() - 1));
    let overflow = CompactReason::Overflow;
    assert_eq!(cut(&short, 200_000, &overflow), Some(short.len() - 1));
    assert_eq!(cut(&short, 200_000, &CompactReason::Threshold), None);
    // Otherwise the newest 20k tokens stay, a quarter of the window at most.
    let at = cut(&long, 200_000, &overflow);
    assert!(at.is_some_and(|at| at < long.len() - 2), "{at:?}");
    let policy = CompactionPolicy::default();
    let keep = |window| policy.keep(&CompactReason::Threshold, spec(window).as_ref());
    assert_eq!((keep(200_000), keep(40_000)), (20_000, 10_000));
}

#[test]
fn clearing_keeps_the_newest_outputs_and_the_last_message() {
    let big = "x".repeat(4_000);
    let mut messages = [
        call("1", "read", "a.rs"),
        result("1", &big),
        call("2", "read", "b.rs"),
        result("2", &big),
        result("3", &big),
    ]
    .map(Arc::new);
    let policy = ClearToolOutputs::new(1_500);
    let cleared = policy.clear(&mut messages);
    assert_eq!(cleared.results, 1);
    assert_eq!(cleared.tokens, 1_000);
    assert!(is_cleared(match &*messages[1] {
        Message::User { content } => match &content[0] {
            UserContent::ToolResult(result) => &result.content,
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }));
    assert_eq!(policy.clear(&mut messages), Cleared::default());
}

/// Clearing skips error results, which are short and still needed, and
/// in a message with several parts replaces only the results it clears;
/// an empty history clears nothing.
#[test]
fn clearing_skips_errors_and_leaves_the_other_parts_alone() {
    let big = "x".repeat(4_000);
    let output = |id: &str, is_error: bool| {
        UserContent::ToolResult(ToolResult {
            is_error,
            call: CallId::from_wire(id),
            name: ToolName::new("read").expect("tool name"),
            content: vec![ToolResultContent::text(big.as_str())],
        })
    };
    let mut messages = [
        call("1", "read", "a.rs"),
        Message::User {
            content: vec![
                UserContent::text("see"),
                output("1", false),
                output("2", false),
            ],
        },
        Message::User {
            content: vec![output("3", true)],
        },
        Message::assistant("done"),
    ]
    .map(Arc::new);
    let policy = ClearToolOutputs::new(1_500);
    assert_eq!(policy.clear(&mut messages).results, 1);
    let cleared = |message: &Message| -> Vec<bool> {
        let Message::User { content } = message else {
            return Vec::new();
        };
        content
            .iter()
            .map(|item| matches!(item, UserContent::ToolResult(result) if is_cleared(&result.content)))
            .collect()
    };
    let parts: Vec<Vec<bool>> = messages.iter().map(|message| cleared(message)).collect();
    assert_eq!(
        parts,
        [vec![], vec![false, true, false], vec![false], vec![]]
    );
    assert_eq!(policy.clear(&mut []), Cleared::default());
}

/// A policy is plain serde data, one flat table: each field it does not
/// name is the default's, and an unknown one is refused.
#[test]
fn a_policy_reads_from_a_table_keeping_what_it_does_not_name() {
    let base = CompactionPolicy {
        tracked: vec![TrackArgument::new("read", "path", "read-files")],
        ..CompactionPolicy::default()
    };
    let json = serde_json::to_value(&base).expect("serde");
    assert!(json.get("limits").is_some(), "{json}");
    let back: CompactionPolicy = serde_json::from_value(json).expect("serde");
    assert_eq!(back, base);

    let policy: CompactionPolicy =
        serde_json::from_value(serde_json::json!({"keep_recent": 1000})).expect("table");
    assert_eq!(policy.keep_recent, 1000);
    assert_eq!(policy.summarizer, Summarizer::DEFAULT);
    let reserve: CompactionPolicy =
        serde_json::from_value(serde_json::json!({"limits": {"reserve": 1}})).expect("table");
    let limits = reserve.summarizer.limits;
    assert_eq!(
        (limits.reserve, limits.summary_tokens),
        (1, SummaryLimits::DEFAULT.summary_tokens)
    );
    assert!(serde_json::from_value::<CompactionPolicy>(serde_json::json!({"keep": 1})).is_err());
}

#[test]
fn a_summary_takes_the_place_of_what_it_summarized_by_id() {
    let big = "x".repeat(4_000);
    let mut conversation = Conversation::from_messages([
        Message::user("start"),
        call("1", "read", "a.rs"),
        result("1", &big),
        call("2", "read", "b.rs"),
        result("2", &big),
        Message::assistant("done"),
    ]);
    let ids: Vec<MessageId> = conversation.entries().map(|entry| entry.id).collect();
    let cleared =
        ClearToolOutputs::new(1_500).clear_conversation(&mut conversation, summary_author());
    assert_eq!(cleared.results, 1);
    let kept: Vec<MessageId> = conversation.entries().map(|entry| entry.id).collect();
    assert_eq!(kept, ids, "clearing keeps every id");
    let last = |conversation: &mut Conversation| {
        let mut changes = conversation.take_unsaved().into_iter().rev();
        changes.find_map(|recorded| summary_in(&recorded.change))
    };
    assert_eq!(last(&mut conversation), None, "clearing is no compaction");

    // A summary planned up to the newest reply survives an edit before it.
    let state = SummaryState {
        summary: "read two files".to_owned(),
        tracked: Vec::new(),
    };
    let upto = conversation.id(5);
    conversation.splice(0..1, [], Author::plugin("other"));
    let spliced = apply_summary(&mut conversation, upto, &state).expect("applied");
    assert_eq!(spliced.removed.len(), 4);
    assert_eq!(conversation.messages().len(), 2);
    assert_eq!(summarized_prefix(&conversation), 1);
    assert_eq!(conversation.author(0), Some(&summary_author()));
    assert_eq!(last(&mut conversation), conversation.id(0));
    // Once the kept message is gone, a summary replaces nothing.
    conversation.splice(1.., [], Author::plugin("other"));
    assert_eq!(apply_summary(&mut conversation, upto, &state), None);
}

#[test]
fn compactions_are_undone_newest_first_with_their_ids_and_states() {
    use crate::{MemoryStore, load, save};

    let store = MemoryStore::default();
    let mut conversation = load(&store, "agent").expect("loaded");
    for text in ["one", "two", "three", "four"] {
        conversation.push(Message::user(text));
    }
    let original = conversation.clone();
    let state = |summary: &str| SummaryState {
        summary: summary.to_owned(),
        tracked: Vec::new(),
    };
    // The first summarizes "one" and "two", the second it and "three".
    let upto = conversation.id(2);
    apply_summary(&mut conversation, upto, &state("first")).expect("applied");
    save(&store, "agent", &mut conversation).expect("saved");
    let upto = conversation.id(2);
    apply_summary(&mut conversation, upto, &state("second")).expect("applied");
    save(&store, "agent", &mut conversation).expect("saved");
    assert_eq!(conversation.messages().len(), 2);
    assert_eq!(Compacted::of(&conversation).count(), 2);

    let by = Author::plugin("undo");
    let undone = undo_summary(&mut conversation, &store, "agent", by.clone()).expect("undone");
    assert_eq!(undone.state, state("first"));
    assert_eq!(summary_state(&conversation), state("first"));
    assert_eq!(undone.restored.len(), 2, "the first summary and \"three\"");
    assert_eq!(conversation.messages().len(), 3);
    save(&store, "agent", &mut conversation).expect("saved");
    let undone = undo_summary(&mut conversation, &store, "agent", by.clone()).expect("undone");
    assert_eq!(undone.state, SummaryState::default());
    assert_eq!(summary_state(&conversation), SummaryState::default());
    assert_eq!(conversation.messages(), original.messages());
    let ids = |conversation: &Conversation| -> Vec<MessageId> {
        conversation.entries().map(|entry| entry.id).collect()
    };
    assert_eq!(ids(&conversation), ids(&original));
    assert_eq!(Compacted::of(&conversation).count(), 0);
    assert!(matches!(
        undo_summary(&mut conversation, &store, "agent", by),
        Err(UndoError::Nothing)
    ));
    save(&store, "agent", &mut conversation).expect("saved");
    assert_eq!(load(&store, "agent").expect("loaded"), conversation);
}

/// A plan clears by message id, leaves outputs under `keep_under` whole,
/// and puts its notes after the outputs of the newest message it clears,
/// where a later clearing, which touches only tool outputs, leaves them.
#[test]
fn a_clear_plan_keeps_small_outputs_and_its_notes_survive_the_next_clearing() {
    let (big, small) = ("x".repeat(4_000), "y".repeat(400));
    let mut conversation = Conversation::from_messages([
        Message::user("start"),
        call("1", "read", "a.rs"),
        result("1", &big),
        call("2", "read", "b.rs"),
        result("2", &small),
        call("3", "read", "c.rs"),
        result("3", &big),
        call("4", "read", "d.rs"),
        result("4", &big),
        Message::assistant("done"),
    ]);
    let ids: Vec<MessageId> = conversation.entries().map(|entry| entry.id).collect();
    let clearing = ClearToolOutputs {
        keep_tokens: 1_500,
        keep_under: 500,
    };
    let plan = clearing.plan(&conversation);
    let planned: Vec<CallId> = plan
        .results(&conversation)
        .map(|result| result.call.clone())
        .collect();
    assert_eq!(planned, [CallId::from_wire("1"), CallId::from_wire("3")]);
    assert_eq!(plan.through, conversation.id(6));
    let cleared = plan.apply(
        &mut conversation,
        Some("a.rs and c.rs hold x"),
        summary_author(),
    );
    assert_eq!(cleared, plan.cleared);
    let kept: Vec<MessageId> = conversation.entries().map(|entry| entry.id).collect();
    assert_eq!(kept, ids);
    // What each message holds: a cleared output, a whole one, notes.
    let parts = |conversation: &Conversation| -> Vec<String> {
        let messages = conversation.messages().iter();
        let content = messages.filter_map(|message| match &**message {
            Message::User { content } => Some(content),
            _ => None,
        });
        content
            .flatten()
            .map(|part| match part {
                UserContent::ToolResult(result) if is_cleared(&result.content) => "cleared".into(),
                UserContent::ToolResult(_) => "output".into(),
                UserContent::Text(text) if text.text.contains("a.rs and c.rs hold x") => {
                    "notes".into()
                }
                _ => "text".into(),
            })
            .collect()
    };
    let expected = ["text", "cleared", "output", "cleared", "notes", "output"];
    assert_eq!(parts(&conversation), expected);
    let everything = ClearToolOutputs {
        keep_tokens: 0,
        keep_under: 0,
    };
    everything.clear_conversation(&mut conversation, summary_author());
    let expected = ["text", "cleared", "cleared", "cleared", "notes", "cleared"];
    assert_eq!(parts(&conversation), expected);
}

/// A summary of its own reads the conversation and fits the model,
/// whatever share of the window its output limit takes.
#[test]
fn a_summary_of_its_own_on_a_small_model_reads_the_conversation() {
    for (window, output, reason, asked) in [
        (32_768, 4_096, CompactReason::Threshold, 4_096),
        (32_768, 16_384, CompactReason::Threshold, 8_192),
        (40_960, 32_768, CompactReason::Threshold, 12_288),
        (65_536, 32_768, CompactReason::Overflow, 8_192),
        // Just above the reserve, and after an overflow, half the limit.
        (16_385, 16_384, CompactReason::Threshold, 4_096),
        (40_960, 32_768, CompactReason::Overflow, 5_120),
    ] {
        let spec = spec(window).map(|spec| spec.with_max_output_tokens(output));
        let request = spec.map(|spec| Summarizer::default().request(&turns(3), "", &reason, &spec));
        assert!(
            matches!(&request, Some(Ok(request)) if request.max_tokens == Some(asked)
                && format!("{:?}", request.chat_history).contains("question 2")
                && estimate_tokens(&request.chat_history) + asked <= u64::from(window)),
            "{window}, {output}: {request:?}"
        );
    }
}

/// A model with neither its window nor its output limit known is asked for
/// a summary with no output limit, as a turn is, even when it knows its
/// input limit.
#[test]
fn a_summary_of_its_own_for_a_model_of_unknown_limits_sends_no_output_limit() {
    let spec = rig_core::providers::registry::ProviderId::catalog("ollama")
        .map(|vendor| ModelSpec::new(vendor, "test"));
    let reason = CompactReason::Threshold;
    for spec in [
        spec.clone(),
        spec.map(|spec| spec.with_max_input_tokens(8_000)),
    ] {
        let request = spec.map(|spec| Summarizer::default().request(&turns(3), "", &reason, &spec));
        assert!(
            matches!(&request, Some(Ok(request)) if request.max_tokens.is_none()),
            "{request:?}"
        );
    }
}

/// A reply cut at the output limit, stopped for tool calls or for a reason
/// the provider does not name is no summary, whatever text it has.
#[test]
fn a_summary_cut_at_the_output_limit_is_not_used() {
    use rig_core::completion::{AssistantContent, CompletionResponse, FinishReason, Usage};
    let reply = |finish| {
        let text = vec![AssistantContent::text("## Goal\nAnswer.")];
        let origin = rig_core::message::Origin::new("test.api", "mock", "");
        let reply = CompletionResponse::new(text, Usage::default(), origin, serde_json::json!({}));
        reply.with_finish_reason(finish)
    };
    let cut = Summarizer::summary_text(&reply(FinishReason::Length));
    assert_eq!(cut, Err(SummaryError::Truncated));
    let called = Summarizer::summary_text(&reply(FinishReason::ToolCalls));
    assert_eq!(called, Err(SummaryError::CalledTool));
    let unknown = Summarizer::summary_text(&reply(FinishReason::Other("pause".to_owned())));
    assert!(
        matches!(unknown, Err(SummaryError::Failed(_))),
        "{unknown:?}"
    );
    let whole = Summarizer::summary_text(&reply(FinishReason::Stop));
    assert_eq!(whole.as_deref(), Ok("## Goal\nAnswer."));
}

#[test]
fn only_successful_paired_calls_track_files() {
    let failure = |id: &str| {
        let mut message = result(id, "failed");
        if let Message::User { content } = &mut message {
            for part in content {
                if let UserContent::ToolResult(result) = part {
                    result.is_error = true;
                }
            }
        }
        message
    };
    let mut aborted = call("aborted", "write", "aborted.rs");
    if let Message::Assistant(reply) = &mut aborted {
        reply.stop = Some(StopReason::Aborted("stopped".to_owned()));
    }
    let messages = [
        call("same", "read", "missing.txt"),
        failure("same"),
        call("same", "read", "a.rs"),
        result("same", "ok"),
        call("same", "edit", "a.rs"),
        failure("same"),
        call("missing", "write", "no-result.rs"),
        aborted,
        result("aborted", "orphan success"),
        Message::assistant("boundary"),
        result("missing", "late success"),
    ];
    let mut state = SummaryState {
        summary: "Done".to_owned(),
        ..Default::default()
    };
    state.track(
        &messages,
        &[
            TrackArgument::new("read", "path", "read-files"),
            TrackArgument::new("edit", "path", "modified-files"),
            TrackArgument::new("write", "path", "modified-files"),
        ],
    );
    assert_eq!(
        state.tracked.first().map(|set| &set.values),
        Some(&["a.rs".to_owned()].into())
    );
    assert!(
        state
            .tracked
            .get(1)
            .is_some_and(|set| set.values.is_empty())
    );
}
