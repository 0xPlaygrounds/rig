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
        call("2", "read", "b.rs"),
        call("3", "edit", "b.rs"),
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
    ];
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
fn cut(messages: &[Message], window: u32, reason: &CompactReason) -> Option<usize> {
    let spec = spec(window);
    assert!(spec.is_some());
    CompactionPolicy::default().cut(messages, 0, reason, spec.as_ref(), &CutRules::default())
}

/// A conversation of turns, each a question and a long answer.
fn turns(count: usize) -> Vec<Message> {
    let answer = "word ".repeat(2_000);
    (0..count)
        .flat_map(|turn| {
            [
                Message::user(format!("question {turn}")),
                Message::assistant(answer.clone()),
            ]
        })
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
    let mut messages = vec![
        call("1", "read", "a.rs"),
        result("1", &big),
        call("2", "read", "b.rs"),
        result("2", &big),
        result("3", &big),
    ];
    let policy = ClearToolOutputs::new(1_500);
    let cleared = policy.clear(&mut messages);
    assert_eq!(cleared.results, 1);
    assert_eq!(cleared.tokens, 1_000);
    assert!(is_cleared(match &messages[1] {
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
    let mut messages = vec![
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
    ];
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
    let parts: Vec<Vec<bool>> = messages.iter().map(cleared).collect();
    assert_eq!(
        parts,
        [vec![], vec![false, true, false], vec![false], vec![]]
    );
    assert_eq!(policy.clear(&mut []), Cleared::default());
}

#[test]
fn a_named_cut_rule_chooses_the_first_kept_message() {
    // Keep from the newest question on, whatever the budget says.
    let mut rules = CutRules::default();
    rules.add("newest-question", |messages: &[Message], _, _, _| {
        messages
            .iter()
            .rposition(|message| matches!(message, Message::User { .. }))
    });
    let policy = CompactionPolicy {
        cut: Cut::Named("newest-question".to_owned()),
        ..CompactionPolicy::default()
    };
    let messages = turns(3);
    let threshold = CompactReason::Threshold;
    let window = spec(200_000);
    let budget = CompactionPolicy::default().cut(&messages, 0, &threshold, window.as_ref(), &rules);
    assert_eq!(budget, None, "the budget keeps a short conversation");
    assert_eq!(policy.cut(&messages, 0, &threshold, None, &rules), Some(4));
    // An answer that summarizes nothing new counts as none.
    assert_eq!(policy.cut(&messages, 4, &threshold, None, &rules), None);
    // A name no rule has cuts by the budget.
    let none = CutRules::default();
    assert_eq!(
        policy.cut(&messages, 0, &threshold, window.as_ref(), &none),
        None
    );
}

/// A policy is plain serde data, and overrides set only what they name:
/// `{"keep_recent": 1000}` keeps the base's tracked arguments and prompts.
#[test]
fn a_policy_saves_and_overrides_keep_what_they_do_not_name() {
    let base = CompactionPolicy {
        tracked: vec![TrackArgument::new("read", "path", "read-files")],
        cut: Cut::Named("mine".to_owned()),
        ..CompactionPolicy::default()
    };
    let json = serde_json::to_value(&base).expect("serde");
    assert_eq!(json.get("cut"), Some(&serde_json::json!("mine")));
    let back: CompactionPolicy = serde_json::from_value(json).expect("serde");
    assert_eq!(back, base);

    let own: CompactionOverrides =
        serde_json::from_value(serde_json::json!({"keep_recent": 1000})).expect("overrides");
    let policy = own.apply(&base);
    assert_eq!(policy.keep_recent, 1000);
    assert_eq!(policy.tracked, base.tracked);
    assert_eq!(policy.summarizer, base.summarizer);
    let reserve: CompactionOverrides =
        serde_json::from_value(serde_json::json!({"limits": {"reserve": 1}})).expect("overrides");
    let limits = reserve.apply(&base).summarizer.limits;
    assert_eq!(
        (limits.reserve, limits.summary_tokens),
        (1, SummaryLimits::DEFAULT.summary_tokens)
    );
    assert!(serde_json::from_value::<CompactionOverrides>(serde_json::json!({"keep": 1})).is_err());
    assert!(CompactionOverrides::default().is_empty());
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

/// Each knob `/compaction` names is set over the policy, shows the value
/// it was set to, and once unset shows the base's again; a value a knob
/// does not take, and a knob that is none of them, are refused with why.
#[test]
fn every_knob_sets_shows_and_unsets_its_value() {
    let base = CompactionPolicy::default();
    let mut own = CompactionOverrides::default();
    let values = ["100", "200", "300", "400", "5", "mine"];
    for (knob, value) in CompactionOverrides::KNOBS.iter().zip(values) {
        assert!(!own.sets(knob), "{knob}");
        assert_eq!(own.set(knob, value, &base), Ok(()));
        assert!(own.sets(knob), "{knob}");
        assert_eq!(own.apply(&base).knob(knob), value);
    }
    let limits = own.apply(&base).summarizer.limits;
    assert_eq!(limits.summary_tokens, base.summarizer.limits.summary_tokens);
    for knob in CompactionOverrides::KNOBS {
        assert_eq!(own.unset(knob), Ok(()));
        assert_eq!(own.apply(&base).knob(knob), base.knob(knob));
    }
    assert!(own.is_empty());

    let refused = |result: Result<(), String>, why: &str| {
        assert!(
            result.as_ref().is_err_and(|it| it.contains(why)),
            "{result:?}"
        );
    };
    refused(own.set("keep", "lots", &base), "takes a number of tokens");
    refused(own.set("per-turn", "-1", &base), "takes a count");
    refused(own.set("bogus", "1", &base), "there is no knob `bogus`");
    refused(own.unset("bogus"), "the knobs are reserve, keep");
    assert!(own.is_empty());
}
