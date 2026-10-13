use serde_json::json;

use super::super::{Aside, AsideId, Author, Conversation, Halt, MessageKey, Recorded};
use super::{ACTIVE, BRANCH, Branch, BranchError, MAIN};
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
