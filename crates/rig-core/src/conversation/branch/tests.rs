use serde_json::json;

use super::super::{Aside, AsideId, Author, Conversation, Halt, MessageKey, Recorded};
use super::{BRANCH, Branch, BranchError};
use crate::completion::Message;
use crate::message::{AssistantContent, CallId, ToolName};

fn keys(conversation: &Conversation) -> Vec<MessageKey> {
    conversation.entries().map(|entry| entry.key()).collect()
}

fn titles(conversation: &Conversation) -> Vec<String> {
    let branches = conversation.branches();
    branches.map(|(_, _, branch)| branch.title).collect()
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

    let first = conversation.branch_off(1, "red", by.clone());
    assert!(first.is_ok(), "{first:?}");
    let first = first.unwrap_or_default();
    assert_eq!(conversation.messages(), [Message::user("Name a colour.")]);
    assert_eq!(conversation.aside(rated), None, "the rating went with it");
    conversation.push(Message::assistant("Blue."));
    let blue = keys(&conversation);

    let switched = conversation.switch_branch(first, "blue", by.clone());
    assert_eq!(switched.map(|spliced| spliced.restored.len()), Ok(3));
    assert_eq!(keys(&conversation), before);
    assert_eq!(conversation.aside(rated), Some(&rating));
    assert_eq!(conversation.aside(first), None);
    assert_eq!(titles(&conversation), ["blue"]);

    let second = conversation.branches().next().map(|(id, at, _)| (id, at));
    assert_eq!(
        second.map(|(_, at)| at),
        Some(1),
        "it continues from the first entry"
    );
    let second = second.map_or(AsideId::default(), |(id, _)| id);
    assert!(conversation.switch_branch(second, "red", by).is_ok());
    assert_eq!(keys(&conversation), blue);
    assert_eq!(conversation.aside(rated), None);
    assert_eq!(titles(&conversation), ["red"]);

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
        conversation.branch_off(3, "none", by.clone()),
        Err(BranchError::NoEntry(3))
    );
    assert_eq!(
        conversation.branch_off(2, "parts", by.clone()),
        Err(BranchError::PartsCall)
    );
    let note = conversation.add_aside(Aside::new("note", by.clone(), json!("hi")));
    assert!(conversation.switch_branch(note, "x", by.clone()).is_err());
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

    assert!(
        conversation
            .switch_branch(empty, "stopped", by.clone())
            .is_ok()
    );
    assert_eq!(conversation.halted(), None);
    assert_eq!(titles(&conversation), ["stopped"]);

    let stopped = conversation.branches().next().map(|(id, _, _)| id);
    let stopped = stopped.unwrap_or_default();
    assert!(conversation.switch_branch(stopped, "empty", by).is_ok());
    assert_eq!(conversation.halted(), Some(Halt::Stopped));
    assert_eq!(titles(&conversation), ["empty"]);

    log.extend(conversation.take_unsaved());
    assert_eq!(fold(&log), conversation);
}
