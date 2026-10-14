use serde_json::json;

use super::super::{Aside, AsideId, Author, Conversation, Halt, MessageKey, Recorded};
use super::{ACTIVE, BRANCH, Branch, BranchError, LINE, MAIN, MergeAt, shared_prefix};
use crate::completion::Message;
use crate::message::{AssistantContent, CallId, ToolName};

fn keys(conversation: &Conversation) -> Vec<MessageKey> {
    conversation.entries().map(|entry| entry.key()).collect()
}

fn titles(conversation: &Conversation) -> Vec<String> {
    let branches = conversation.branches().into_iter();
    branches.map(|kept| kept.branch.title).collect()
}

fn fold(changes: &[Recorded]) -> Conversation {
    let mut folded = Conversation::new();
    for change in changes {
        let applied = folded.apply(change);
        assert_eq!(applied, Ok(()));
    }
    folded
}

/// Switching to a branch and back puts every entry back with its id and
/// made-time, and every aside with its id; the branch left is kept, and a
/// fold of the records gives the same value.
#[test]
fn a_branch_switched_away_and_back_keeps_ids_made_times_and_asides() {
    let by = Author::plugin("tree");
    let mut conversation = Conversation::new();
    conversation.dated(1_000, |conversation| {
        conversation.extend([Message::user("Name a colour."), Message::assistant("Red.")]);
    });
    let red = conversation.id(1);
    let rating = Aside::new("rating", Author::plugin("ratings"), json!(5)).after(red);
    let rated = conversation.add_aside(rating.clone());
    conversation.dated(2_000, |conversation| {
        conversation.extend([Message::user("Another?"), Message::assistant("Green.")]);
    });
    let before = keys(&conversation);
    let mut log = conversation.take_unsaved();

    let first = conversation.branch_off(1, "blue", by.clone());
    assert!(first.is_ok(), "{first:?}");
    let first = first.unwrap_or_default();
    assert_eq!(conversation.messages(), [Message::user("Name a colour.")]);
    assert_eq!(conversation.aside(rated), None, "the rating went with it");
    conversation.push(Message::assistant("Blue."));
    let blue = keys(&conversation);

    let switched = conversation.switch_branch(first, by.clone());
    assert_eq!(switched.map(|spliced| spliced.restored.len()), Ok(3));
    assert_eq!(keys(&conversation), before);
    assert_eq!(conversation.aside(rated), Some(&rating));
    assert_eq!(conversation.aside(first), None);
    assert_eq!(titles(&conversation), ["blue"]);
    assert_eq!(conversation.branch_title(), MAIN);

    let second = conversation
        .branches()
        .first()
        .map(|kept| (kept.id, kept.at));
    assert_eq!(
        second.map(|(_, at)| at),
        Some(1),
        "it continues from the first entry"
    );
    let second = second.map_or(AsideId::default(), |(id, _)| id);
    assert!(conversation.switch_branch(second, by).is_ok());
    assert_eq!(keys(&conversation), blue);
    assert_eq!(conversation.aside(rated), None);
    assert_eq!(titles(&conversation), [MAIN]);
    assert_eq!(conversation.branch_title(), "blue");

    log.extend(conversation.take_unsaved());
    let folded = fold(&log);
    assert_eq!(folded, conversation);
    assert_eq!(keys(&folded), blue);
}

/// A branch is refused where there is no entry, where it would part a call
/// from its result, and on an aside that is not a branch; a branch aside
/// reads back as a branch.
#[test]
fn a_branch_is_refused_where_it_would_not_fit() {
    let by = Author::plugin("tree");
    let shell = ToolName::new("shell").expect("a tool name");
    let call = AssistantContent::tool_call("c1", shell.clone(), json!({}));
    let mut conversation = Conversation::from_messages([
        Message::user("List files."),
        Message::Assistant(crate::message::AssistantMessage::new(vec![call])),
        Message::tool_result(CallId::from_wire("c1"), shell, "a.rs"),
    ]);
    assert_eq!(
        conversation.branch_off(4, "none", by.clone()),
        Err(BranchError::NoEntry(4))
    );
    assert_eq!(
        conversation.branch_off(2, "parts", by.clone()),
        Err(BranchError::PartsCall)
    );
    let note = conversation.add_aside(Aside::new("note", by.clone(), json!("hi")));
    assert!(conversation.switch_branch(note, by.clone()).is_err());
    assert_eq!(conversation.messages().len(), 3, "nothing changed");

    let id = conversation.branch_off(1, "call", by);
    let aside = id.ok().and_then(|id| conversation.aside(id));
    assert_eq!(aside.map(|aside| aside.kind.as_str()), Some(BRANCH));
    let branch = aside.and_then(Branch::of);
    assert_eq!(branch.map(|branch| branch.entries.len()), Some(2));
}

/// Switching from a stopped line with nothing after the point to an empty
/// branch that was not halted clears the halt and keeps the stopped line;
/// switching back brings the halt back.
#[test]
fn switching_between_empty_continuations_carries_the_halt_both_ways() {
    let by = Author::plugin("tree");
    let mut conversation = Conversation::from_messages([
        Message::user("Name a colour."),
        Message::assistant("Red."),
        Message::user("Another?"),
    ]);
    assert!(conversation.halt(Halt::Stopped));
    let last = conversation.id(2);
    let empty = Branch {
        title: "empty".into(),
        ..Branch::default()
    };
    let empty = conversation.add_aside(Aside::new(BRANCH, by.clone(), json!(empty)).after(last));
    let mut log = Vec::new();

    assert!(conversation.switch_branch(empty, by.clone()).is_ok());
    assert_eq!(conversation.halted(), None);
    assert_eq!(titles(&conversation), [MAIN]);

    let stopped = conversation.branches().first().map(|kept| kept.id);
    let stopped = stopped.unwrap_or_default();
    assert!(conversation.switch_branch(stopped, by).is_ok());
    assert_eq!(conversation.halted(), Some(Halt::Stopped));
    assert_eq!(titles(&conversation), ["empty"]);

    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);
}

/// A branch started at the end keeps the line left as an empty branch that
/// comes back when switched to, with ids, made-times and titles; the title
/// is one aside that follows no entry.
#[test]
fn a_branch_at_the_end_keeps_the_empty_line_it_leaves() {
    let by = Author::plugin("tree");
    let mut conversation = Conversation::new();
    conversation.dated(1_000, |conversation| {
        conversation.extend([Message::user("Fix the bug."), Message::assistant("Done.")]);
    });
    let mut log = conversation.take_unsaved();
    let main = keys(&conversation);

    let len = conversation.messages().len();
    let left = conversation.branch_off(len, "try", by.clone());
    assert!(left.is_ok(), "{left:?}");
    assert_eq!(keys(&conversation), main, "nothing was taken out");
    assert_eq!(conversation.branch_title(), "try");
    conversation.dated(2_000, |conversation| {
        conversation.push(Message::user("Try another way."));
    });
    let tried = keys(&conversation);

    let main_id = conversation.branches().first().map(|kept| kept.id);
    assert!(
        conversation
            .switch_branch(main_id.unwrap_or_default(), by.clone())
            .is_ok()
    );
    assert_eq!(keys(&conversation), main);
    assert_eq!(conversation.branch_title(), MAIN);
    assert_eq!(titles(&conversation), ["try"]);
    let titled = conversation
        .asides()
        .filter(|(_, aside)| aside.kind == ACTIVE);
    assert_eq!(titled.count(), 0, "main needs no title aside");

    let tried_id = conversation.branches().first().map(|kept| kept.id);
    assert!(
        conversation
            .switch_branch(tried_id.unwrap_or_default(), by.clone())
            .is_ok()
    );
    assert_eq!(keys(&conversation), tried);
    assert_eq!(
        titles(&conversation),
        [MAIN],
        "the empty line is kept again"
    );

    // The title follows no entry, so a branch off before it leaves it here.
    assert!(conversation.branch_off(1, "short", by).is_ok());
    let titled: Vec<_> = conversation
        .asides()
        .filter(|(_, aside)| aside.kind == ACTIVE)
        .collect();
    assert_eq!(titled.len(), 1);
    assert_eq!(titled.first().and_then(|(_, aside)| aside.after), None);
    assert_eq!(conversation.branch_title(), "short");

    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);
}

/// A branch kept two deep is listed inside the one keeping it and reached
/// by one switch; its line is the same before and after.
#[test]
fn a_nested_branch_is_listed_and_reached_in_one_switch() {
    let by = Author::plugin("tree");
    let mut conversation = Conversation::from_messages([
        Message::user("one"),
        Message::assistant("two"),
        Message::user("three"),
    ]);
    let mut log = conversation.take_unsaved();
    // a: "three" kept as "main"; now on "b" from "two".
    assert!(conversation.branch_off(2, "b", by.clone()).is_ok());
    conversation.push(Message::user("b-three"));
    // "b" kept from "one"; now on "c".
    assert!(conversation.branch_off(1, "c", by.clone()).is_ok());
    conversation.push(Message::assistant("c-two"));

    let kept = conversation.branches();
    let found: Vec<(&str, bool, usize)> = kept
        .iter()
        .map(|kept| (kept.branch.title.as_str(), kept.inside.is_some(), kept.at))
        .collect();
    assert_eq!(found, [("b", false, 1), (MAIN, true, 2)]);
    let main = kept.get(1).map(|kept| kept.id).unwrap_or_default();
    let line = conversation.line(Some(main));
    let texts = |entries: &[crate::conversation::Entry]| -> Vec<Message> {
        entries.iter().map(|entry| entry.message.clone()).collect()
    };
    assert_eq!(
        line.as_deref().map(texts),
        Some(vec![
            Message::user("one"),
            Message::assistant("two"),
            Message::user("three")
        ])
    );

    assert!(conversation.switch_branch(main, by).is_ok());
    assert_eq!(conversation.branch_title(), MAIN);
    assert_eq!(conversation.line(None), line);
    assert_eq!(titles(&conversation), ["c", "b"]);
    assert_eq!(conversation.line(Some(AsideId::default())), None);

    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);
}

/// A line value set on a line with no entry of its own goes with that line:
/// a branch at the end starts with a copy, each line keeps its own through
/// switches both ways, and the old line's value never becomes the new one's.
#[test]
fn a_line_value_goes_with_its_line_through_switches() {
    let by = Author::plugin("tree");
    let mut conversation =
        Conversation::from_messages([Message::user("Name a colour."), Message::assistant("Red.")]);
    let value = |conversation: &Conversation| {
        let values = conversation.line_values();
        values.and_then(|values| values.get("k")).cloned()
    };
    conversation.set_line_value("k", Some(json!(1)), by.clone());
    let mut log = conversation.take_unsaved();
    assert!(!conversation.parts_at_end());

    let len = conversation.messages().len();
    let main = conversation.branch_off(len, "split", by.clone());
    assert!(main.is_ok(), "{main:?}");
    assert!(conversation.parts_at_end());
    assert_eq!(
        value(&conversation),
        Some(json!(1)),
        "the split starts as of its point"
    );
    conversation.set_line_value("k", Some(json!(2)), by.clone());

    let main = conversation.branches().first().map(|kept| kept.id);
    assert!(
        conversation
            .switch_branch(main.unwrap_or_default(), by.clone())
            .is_ok()
    );
    assert_eq!(
        value(&conversation),
        Some(json!(1)),
        "the old line kept its own"
    );
    let split = conversation.branches().first().map(|kept| kept.id);
    assert!(
        conversation
            .switch_branch(split.unwrap_or_default(), by.clone())
            .is_ok()
    );
    assert_eq!(value(&conversation), Some(json!(2)));

    conversation.set_line_value("k", None, by);
    assert_eq!(conversation.line_values(), None);
    let lines = conversation
        .asides()
        .filter(|(_, aside)| aside.kind == LINE);
    assert_eq!(lines.count(), 0, "an empty line keeps no aside");

    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);
}

/// Six messages, one pair a call and its result, with a rating after the
/// fourth.
fn six(by: &Author) -> (Conversation, Option<AsideId>) {
    let shell = ToolName::new("shell").expect("a tool name");
    let call = AssistantContent::tool_call("c1", shell.clone(), json!({}));
    let mut conversation = Conversation::from_messages([
        Message::user("one"),
        Message::assistant("two"),
        Message::user("three"),
        Message::Assistant(crate::message::AssistantMessage::new(vec![call])),
        Message::tool_result(CallId::from_wire("c1"), shell, "four"),
        Message::assistant("five"),
    ]);
    let after = conversation.id(4);
    let rating = Aside::new("rating", by.clone(), json!(5)).after(after);
    let rated = conversation.add_aside(rating);
    (conversation, Some(rated))
}

/// A range moved out leaves its pointer where it was, and merged back it
/// is where it was, ids, made-times and asides kept; a range that would
/// part a call from its result, at either end, is refused.
#[test]
fn a_range_moved_out_and_merged_back_is_where_it_was() {
    let by = Author::plugin("tree");
    let (mut conversation, rated) = six(&by);
    let before = keys(&conversation);
    let rating = rated.and_then(|rated| conversation.aside(rated).cloned());
    let mut log = conversation.take_unsaved();
    assert_eq!(
        conversation.move_out(2..4, "parts", by.clone()),
        Err(BranchError::PartsCall)
    );
    assert_eq!(
        conversation.move_out(4..6, "parts", by.clone()),
        Err(BranchError::PartsCall)
    );
    assert_eq!(
        conversation.move_out(5..7, "past", by.clone()),
        Err(BranchError::NoEntry(6))
    );
    assert_eq!(
        conversation.move_out(3..3, "empty", by.clone()),
        Err(BranchError::NoEntry(3))
    );

    let moved = conversation.move_out(2..5, "aside", by.clone());
    assert!(moved.is_ok(), "{moved:?}");
    let moved = moved.unwrap_or_default();
    let messages = [
        Message::user("one"),
        Message::assistant("two"),
        Message::assistant("five"),
    ];
    assert_eq!(conversation.messages(), messages);
    let kept = conversation.branches();
    let listed: Vec<_> = kept
        .iter()
        .map(|kept| {
            (
                kept.id,
                kept.at,
                kept.branch.moved,
                kept.branch.title.as_str(),
            )
        })
        .collect();
    assert_eq!(listed, [(moved, 2, true, "aside")]);
    assert_eq!(rated.and_then(|rated| conversation.aside(rated)), None);
    assert_eq!(
        conversation.branch_title(),
        MAIN,
        "the line keeps its title"
    );

    let merged = conversation.merge_branch(moved, by);
    assert_eq!(merged.map(|spliced| spliced.restored.len()), Ok(3));
    assert_eq!(keys(&conversation), before);
    assert_eq!(
        rated.and_then(|rated| conversation.aside(rated).cloned()),
        rating
    );
    assert!(conversation.branches().is_empty());
    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);
}

/// Merging a branch that holds another appends its own entries with their
/// ids, makes the branch it held one of the active line's, and brings its
/// line values back.
#[test]
fn a_merged_branch_brings_its_entries_nested_branches_and_line_values() {
    let by = Author::plugin("tree");
    let mut conversation = Conversation::from_messages([
        Message::user("u1"),
        Message::assistant("a1"),
        Message::user("u2"),
        Message::assistant("a2"),
    ]);
    let ids: Vec<_> = conversation.entries().map(|entry| entry.id).collect();
    let main = conversation.branch_off(2, "x", by.clone());
    conversation.push(Message::user("u3"));
    assert!(
        conversation
            .switch_branch(main.unwrap_or_default(), by.clone())
            .is_ok()
    );
    // Main goes on as `y` from u2, keeping a2 as a branch inside it.
    assert!(conversation.branch_off(3, "y", by.clone()).is_ok());
    conversation.set_line_value("k", Some(json!(1)), by.clone());
    let x = conversation
        .branches()
        .iter()
        .find(|kept| kept.branch.title == "x")
        .map(|kept| kept.id);
    assert!(
        conversation
            .switch_branch(x.unwrap_or_default(), by.clone())
            .is_ok()
    );
    assert_eq!(conversation.line_values(), None);
    let y = conversation
        .branches()
        .into_iter()
        .find(|kept| kept.branch.title == "y");
    assert_eq!(y.as_ref().map(|y| y.at), Some(2));

    let merged = conversation.merge_branch(y.map(|y| y.id).unwrap_or_default(), by.clone());
    assert!(merged.is_ok(), "{merged:?}");
    let titles: Vec<_> = conversation
        .branches()
        .iter()
        .map(|kept| (kept.inside, kept.branch.title.clone()))
        .collect();
    assert_eq!(
        titles,
        [(None, MAIN.to_owned())],
        "a2's branch is the active line's now"
    );
    let values = conversation
        .line_values()
        .and_then(|values| values.get("k"))
        .cloned();
    assert_eq!(values, Some(json!(1)));
    let main = conversation.branches().first().map(|kept| kept.id);
    assert!(
        conversation
            .merge_branch(main.unwrap_or_default(), by)
            .is_ok()
    );
    let now: Vec<_> = conversation.entries().map(|entry| entry.id).collect();
    let u3 = now.get(2).copied();
    let expected: Vec<_> = [ids.first(), ids.get(1), u3.as_ref(), ids.get(2), ids.get(3)]
        .into_iter()
        .flatten()
        .copied()
        .collect();
    assert_eq!(now, expected, "every entry keeps its id");
}

/// A line as a conversation of its own keeps its entries' keys and the
/// asides that follow them, not the branches on the way to it, and its
/// line values only through its end; merged into a conversation it parted
/// from, its own entries come at the end with new ids, their asides
/// following them, or, for a range moved out, back where it was with its
/// ids.
#[test]
fn a_line_copied_out_merges_back_by_appending_or_where_it_was_moved_from() {
    let by = Author::plugin("tree");
    let (mut conversation, _) = six(&by);
    // A line that parts at the call's result: the copy is its own.
    let side = conversation.branch_off(5, "side", by.clone());
    conversation.push(Message::assistant("six"));
    conversation.set_line_value("k", Some(json!(2)), by.clone());
    assert!(
        conversation
            .switch_branch(side.unwrap_or_default(), by.clone())
            .is_ok()
    );
    let side = conversation.branches().first().map(|kept| kept.id);
    let line = conversation.line(side).unwrap_or_default();
    let copy = conversation.line_conversation(side, line.len());
    assert_eq!(copy.as_ref().map(Conversation::branch_title), Some("side"));
    let copied: Vec<_> = copy.as_ref().map(keys).unwrap_or_default();
    let line_keys: Vec<_> = line
        .iter()
        .filter_map(|entry| {
            Some(MessageKey {
                id: entry.id?,
                made: entry.made?,
            })
        })
        .collect();
    assert_eq!(copied, line_keys);
    let kinds: Vec<_> = copy
        .iter()
        .flat_map(|copy| copy.asides().map(|(_, aside)| aside.kind.clone()))
        .collect();
    assert_eq!(kinds, [ACTIVE, LINE, "rating"], "no branch on the way");
    let short = conversation.line_conversation(side, 3);
    assert_eq!(short.as_ref().and_then(Conversation::line_values), None);
    assert_eq!(
        shared_prefix(&line, &conversation.line(None).unwrap_or_default()),
        5
    );
    let unsaid = [crate::conversation::Entry::new(Message::user("one"))];
    assert_eq!(
        shared_prefix(&unsaid, &unsaid),
        0,
        "no id, shared with none"
    );

    // The copy goes on, then comes back at the end with new ids.
    let Some(mut copy) = copy else { return };
    let seven = copy.push(Message::user("seven"));
    copy.add_aside(Aside::new("note", by.clone(), json!("n")).after(Some(seven)));
    let mut log = conversation.take_unsaved();
    let merged = conversation.merge_line(&copy, MergeAt::End, by.clone());
    assert!(merged.is_ok(), "{merged:?}");
    let tail: Vec<Message> = conversation.messages().iter().skip(6).cloned().collect();
    assert_eq!(tail, [Message::assistant("six"), Message::user("seven")]);
    let last = conversation.id(7);
    assert!(last.is_some());
    assert_ne!(last, Some(seven), "a new id: it is not this conversation's");
    let note = conversation
        .asides()
        .find(|(_, aside)| aside.kind == "note");
    assert_eq!(note.map(|(_, aside)| aside.after), Some(last));
    assert_eq!(
        conversation
            .line_values()
            .and_then(|values| values.get("k"))
            .cloned(),
        Some(json!(2))
    );
    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);

    // A range moved into a copy comes back where it was, with its ids.
    let (mut conversation, _) = six(&by);
    let all = keys(&conversation);
    let copy = conversation.line_conversation(None, 5);
    let pointer = conversation.move_out(2..5, "away", by.clone());
    assert!(pointer.is_ok(), "{pointer:?}");
    let pointer = pointer.unwrap_or_default();
    let back = MergeAt::Back { pointer, moved: 3 };
    let merged = copy.map(|copy| conversation.merge_line(&copy, back, by.clone()));
    assert!(matches!(merged, Some(Ok(_))), "{merged:?}");
    assert_eq!(keys(&conversation), all);
}

/// A branch or a line merged in after a call still waiting for its result
/// is refused, and nothing changes.
#[test]
fn a_merge_after_an_unanswered_call_is_refused() {
    let by = Author::plugin("tree");
    let shell = ToolName::new("shell").expect("a tool name");
    let call = AssistantContent::tool_call("c1", shell, json!({}));
    let mut conversation =
        Conversation::from_messages([Message::user("one"), Message::assistant("two")]);
    let main = conversation.branch_off(2, "calls", by.clone());
    let main = main.unwrap_or_default();
    let copy = conversation.line_conversation(Some(main), 2);
    conversation.push(Message::Assistant(crate::message::AssistantMessage::new(
        vec![call],
    )));
    let before = conversation.clone();
    assert_eq!(
        conversation.merge_branch(main, by.clone()),
        Err(BranchError::PartsCall)
    );
    let merged = copy.map(|copy| conversation.merge_line(&copy, MergeAt::End, by));
    assert_eq!(merged, Some(Err(BranchError::PartsCall)));
    assert_eq!(conversation, before, "nothing changed");
}

/// An aside put in a kept branch, at any depth, is there when the branch
/// comes back, and its id stays its own.
#[test]
fn an_aside_put_in_a_kept_branch_comes_back_with_it() {
    let by = Author::plugin("tree");
    let (mut conversation, _) = six(&by);
    let first = conversation.branch_off(2, "b", by.clone());
    let first = first.unwrap_or_default();
    let point = conversation.id(1);
    let mut log = conversation.take_unsaved();
    let put = conversation.add_aside_in(
        Some(first),
        Aside::new("pointer", by.clone(), json!(1)).after(point),
    );
    assert!(put.is_ok(), "{put:?}");
    let put = put.unwrap_or_default();
    let later = conversation.add_aside(Aside::new("note", by.clone(), json!(2)));
    assert_ne!(later, put);
    assert_eq!(conversation.aside(put), None);
    assert!(conversation.switch_branch(first, by.clone()).is_ok());
    assert_eq!(
        conversation.aside(put).map(|aside| aside.kind.as_str()),
        Some("pointer")
    );
    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);

    // Two deep: "main" kept inside "b", reached in one switch.
    let mut conversation = Conversation::from_messages([
        Message::user("one"),
        Message::assistant("two"),
        Message::user("three"),
    ]);
    let three = conversation.id(2);
    assert!(conversation.branch_off(2, "b", by.clone()).is_ok());
    assert!(conversation.branch_off(1, "c", by.clone()).is_ok());
    let main = conversation
        .branches()
        .into_iter()
        .find(|kept| kept.inside.is_some());
    let main = main.map(|kept| kept.id).unwrap_or_default();
    let note = Aside::new("note", by.clone(), json!(3)).after(three);
    let put = conversation.add_aside_in(Some(main), note);
    assert!(put.is_ok(), "{put:?}");
    let put = put.unwrap_or_default();
    assert_eq!(conversation.aside(put), None);
    assert!(conversation.switch_branch(main, by).is_ok());
    assert_eq!(
        conversation
            .aside(put)
            .map(|aside| (aside.kind.as_str(), aside.after)),
        Some(("note", three))
    );
}
