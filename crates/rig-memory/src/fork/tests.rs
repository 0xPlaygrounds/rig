use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::{Author, Conversation, MessageId};

use crate::{
    MemoryStore, Session, SummaryState, Writer, apply_summary, as_of, history, load, undo_summary,
};

/// A running writer on a fresh store, started.
fn writer() -> (Arc<MemoryStore>, Writer) {
    let store = Arc::new(MemoryStore::default());
    let (writer, _) = Writer::open(store.clone()).expect("opened");
    writer.start();
    (store, writer)
}

/// Pushes each of `messages` to `conversation` and saves it to the log
/// `log` after each, so each is a record of its own.
fn say(writer: &Writer, log: &str, conversation: &mut Conversation, messages: &[Message]) {
    for message in messages {
        conversation.push(message.clone());
        writer.save(log, conversation);
    }
}

/// Messages of `texts`: a user's question for each ending in `?`, an
/// answer for the others.
fn said(texts: &[&str]) -> Vec<Arc<Message>> {
    let message = |text: &&str| match text.ends_with('?') {
        true => Message::user(*text),
        false => Message::assistant(*text),
    };
    texts.iter().map(message).map(Arc::new).collect()
}

#[test]
fn a_fork_excludes_a_partially_answered_batch_with_repeated_ids() {
    use rig_core::conversation::Halt;
    use rig_core::message::{
        AssistantContent, AssistantMessage, CallId, ToolCall, ToolFunction, ToolName, ToolResult,
        ToolResultContent, UserContent,
    };

    let (_, writer) = writer();
    let mut main = Conversation::new();
    let name = ToolName::new("lookup").expect("tool name");
    let calls = (0..2)
        .map(|index| {
            AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire("same"),
                ToolFunction::new(name.clone(), serde_json::json!({"index": index})),
            ))
        })
        .collect();
    say(
        &writer,
        "main",
        &mut main,
        &[
            Message::user("one?"),
            Message::assistant("one."),
            Message::user("Look up both"),
            Message::Assistant(AssistantMessage::new(calls)),
            Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    is_error: false,
                    call: CallId::from_wire("same"),
                    name,
                    content: vec![ToolResultContent::text("first result")],
                })],
            },
        ],
    );
    assert_eq!(main.pending_calls().len(), 1);
    assert!(main.halt(Halt::Stopped));
    writer.save("main", &mut main);
    let fork = writer.fork_log("main", "fork", None).expect("forked");
    assert_eq!(fork.conversation.messages(), said(&["one?", "one."]));
    assert!(fork.conversation.validate().is_ok());
}

/// A fork copies the log up to a turn boundary at or before the message it
/// is given: the whole log, through an answer, just before a question (a
/// question that asks for a turn is at no boundary), or nothing; and it
/// goes on as a log of its own, read back as it was written.
#[test]
fn a_fork_copies_the_log_up_to_a_turn_boundary_and_goes_on_by_itself() {
    let (store, writer) = writer();
    let mut main = Conversation::new();
    let turns = [
        Message::user("one?"),
        Message::assistant("one."),
        Message::user("two?"),
        Message::assistant("two."),
    ];
    say(&writer, "main", &mut main, &turns);
    let id = |at: usize| main.id(at).expect("an id");
    let fork = |to: &str, through: Option<MessageId>| writer.fork_log("main", to, through);

    let whole = fork("whole", None).expect("forked");
    assert_eq!(whole.conversation.messages(), main.messages());
    let answered = fork("answered", Some(id(1))).expect("forked");
    assert_eq!(answered.conversation.messages(), said(&["one?", "one."]));
    let asked = fork("asked", Some(id(2))).expect("forked");
    assert_eq!(asked.conversation.messages(), said(&["one?", "one."]));
    let first = fork("first", Some(id(0))).expect("forked");
    assert!(first.conversation.messages().is_empty());
    let nowhere: MessageId = "m99".parse().expect("an id");
    assert!(fork("nowhere", Some(nowhere)).is_err());
    assert!(fork("whole", None).is_err(), "a log that exists is refused");

    // The copy keeps the ids and goes on in its own log, with no reset.
    assert_eq!(answered.conversation.id(1), Some(id(1)));
    let mut copy = answered.conversation;
    let saved = {
        copy.push(Message::user("three?"));
        writer.save("answered", &mut copy)
    };
    assert_eq!(saved.len(), 1, "one push, not a reset");
    writer.flush().expect("flushed");
    assert_eq!(load(&*store, "answered").expect("read"), copy);
    assert_eq!(load(&*store, "main").expect("read"), main);
    let read = Session::read(&*store).expect("read");
    assert_eq!(read.agents.len(), 5);
}

/// A fork made after a compaction undoes it: the copy holds the records
/// that made the summary, with their stamps.
#[test]
fn a_fork_made_after_a_compaction_can_undo_it() {
    let (store, writer) = writer();
    let mut main = Conversation::new();
    let turns = [
        Message::user("one?"),
        Message::assistant("one."),
        Message::user("two?"),
        Message::assistant("two."),
    ];
    say(&writer, "main", &mut main, &turns);
    let kept = main.id(2);
    let state = SummaryState {
        summary: "Asked one.".to_owned(),
        ..SummaryState::default()
    };
    assert!(apply_summary(&mut main, kept, &state).is_some());
    writer.save("main", &mut main);
    say(&writer, "main", &mut main, &[Message::user("three?")]);
    say(&writer, "main", &mut main, &[Message::assistant("three.")]);

    let mut copy = writer.fork_log("main", "fork", None).expect("forked");
    assert_eq!(copy.conversation.messages(), main.messages());
    let by = Author::plugin("test");
    let undone = undo_summary(&mut copy.conversation, &*store, "fork", by);
    assert!(undone.is_ok(), "{undone:?}");
    let all = ["one?", "one.", "two?", "two.", "three?", "three."];
    assert_eq!(copy.conversation.messages(), said(&all));
    // The copy's history reads as the source's did, record for record.
    let copied = history(&*store, "fork").expect("history");
    let source = history(&*store, "main").expect("history");
    assert_eq!(copied, source);
    let (stamp, _) = copied.get(2).expect("a third record");
    let then = as_of(&*store, "fork", stamp.n).expect("as of");
    assert_eq!(then.messages(), said(&["one?", "one."]));
}
