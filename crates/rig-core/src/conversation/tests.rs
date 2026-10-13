use super::*;

fn ids(conversation: &Conversation) -> Vec<MessageId> {
    conversation.entries().map(|entry| entry.id).collect()
}

fn fold(changes: &[Recorded]) -> Conversation {
    let mut folded = Conversation::new();
    for change in changes {
        folded
            .apply(change)
            .expect("a recorded change fits its fold");
    }
    folded
}

/// Ids survive splices: an entry taken out and spliced back changed keeps
/// its id, a second copy of it and an id from outside the range get new
/// ones, and nothing merges, not even two user messages side by side.
#[test]
fn ids_survive_splices_and_nothing_merges() {
    let mut conversation = Conversation::new();
    let first = conversation.push(Message::user("one"));
    let second = conversation.push(Message::user("two"));
    let third = conversation.push(Entry::new(Message::assistant("three")).by(Author::rig()));
    assert_eq!(
        conversation.messages().len(),
        3,
        "user messages do not merge"
    );
    assert!(first < second && second < third);

    let mut edited = conversation.entry(1).expect("entry 1");
    edited.message = Message::user("two, edited");
    let copy = edited.clone();
    let outsider = conversation.entry(2).expect("entry 2");
    let fresh = Entry::new(Message::user("new"));
    let spliced = conversation.splice(
        1..2,
        [edited, copy, outsider, fresh],
        Author::plugin("surgeon"),
    );

    let now = ids(&conversation);
    assert_eq!(now.len(), 6);
    assert_eq!(now.first(), Some(&first));
    assert_eq!(now.get(1), Some(&second), "the edited entry keeps its id");
    assert_eq!(now.last(), Some(&third), "the original stays where it was");
    let new: Vec<_> = now.iter().skip(2).take(3).copied().collect();
    assert!(
        new.iter().all(|id| *id > third),
        "copies get new ids: {new:?}"
    );
    assert_eq!(spliced.removed, Vec::<MessageId>::new());
    assert_eq!(spliced.changed, now.get(1..5).unwrap_or_default());
    assert_eq!(conversation.author(1), None);
    assert_eq!(conversation.author(3), Some(&Author::rig()));

    // Cutting it out, and an id taken out is gone for good.
    let cut = conversation.splice(..2, [], Author::plugin("surgeon"));
    assert_eq!(cut.removed, [first, second]);
    assert_eq!(conversation.index_of(second), None);
    assert_eq!(conversation.start_of(second), 0);
    let Some(Recorded { by, .. }) = conversation.unsaved().last() else {
        panic!("the cut is recorded");
    };
    assert_eq!(by.as_ref(), Some(&Author::plugin("surgeon")));
    let again = conversation.push(Message::user("again"));
    assert!(again > *new.iter().max().expect("new ids"));
}

/// Folding a conversation's outbox, also through its saved JSON form,
/// gives the live value and its revision; a change that does not fit its
/// fold fails without changing it.
#[test]
fn a_fold_of_the_outbox_equals_the_live_value() {
    let mut live = Conversation::from_messages([Message::user("hi"), Message::assistant("hey")]);
    live.push(Message::user("do it"));
    live.halt(Halt::Kept);
    live.push(Entry::new(Message::user("note")).by(Author::agent("a1b2c3d4e5", None)));
    let note = Aside::new(
        "note",
        Author::plugin("btw"),
        serde_json::json!({"text": "x"}),
    )
    .after(live.id(1));
    let note = live.add_aside(note);
    let scratch =
        live.add_aside(Aside::new("scratch", Author::rig(), serde_json::Value::Null).unsaved());
    let moved = live
        .aside(note)
        .cloned()
        .expect("the note")
        .after(live.id(2));
    assert!(live.replace_aside(note, moved));
    live.splice(
        0..1,
        [Entry::new(Message::user("summary"))],
        Author::plugin("compact"),
    );

    let lines: Vec<String> = live
        .unsaved()
        .iter()
        .map(|change| serde_json::to_string(change).expect("a change serializes"))
        .collect();
    let read: Vec<Recorded> = lines
        .iter()
        .map(|line| serde_json::from_str(line).expect("a change reads back"))
        .collect();
    let folded = fold(&read);
    assert_eq!(folded, live);
    assert_eq!(folded.revision(), live.revision());
    assert_eq!(
        folded.revision(),
        u64::try_from(read.len()).unwrap_or(u64::MAX)
    );
    assert!(live.aside(scratch).is_some() && folded.aside(scratch).is_none());
    assert_eq!(
        folded.aside_position(folded.aside(note).expect("the note")),
        3
    );

    let mut wrong = folded.clone();
    let stale = Recorded {
        change: Change::Splice {
            from: 0,
            to: 1,
            removed: vec![MessageId(99)],
            with: Vec::new(),
        },
        by: None,
        at: None,
    };
    assert!(wrong.apply(&stale).is_err());
    assert_eq!(wrong, folded);
    assert_eq!(wrong.revision(), folded.revision());
    assert_eq!(live.take_unsaved().len(), read.len());
    assert!(live.unsaved().is_empty());
}

/// A reset replaces the whole value, next ids too, so a fold goes on as
/// the live value does, and survives its saved form; a saved state whose
/// ids do not fit is refused.
#[test]
fn a_reset_replaces_the_value_and_the_fold_numbers_as_the_live_one() {
    let mut log = Conversation::from_messages([Message::user("a"), Message::user("b")]);
    log.push(Message::user("c"));
    let replacement = Conversation::from_messages([Message::user("other")]);
    let reset = Recorded {
        change: Change::Reset {
            state: Box::new(replacement.view()),
        },
        by: Some(Author::plugin("snapshot")),
        at: None,
    };
    let reset: Recorded =
        serde_json::from_str(&serde_json::to_string(&reset).expect("serializes")).expect("reads");
    let before = log.revision();
    log.apply(&reset).expect("a reset fits anything");
    assert_eq!(log, replacement);
    assert_eq!(log.revision(), before + 1);
    let mut live = replacement;
    let next = log.push(Message::user("d"));
    assert_eq!(next, live.push(Message::user("d")));

    let mut saved = serde_json::to_value(&log).expect("serializes");
    let back: Conversation = serde_json::from_value(saved.clone()).expect("reads");
    assert_eq!(back, log);
    if let Some(meta) = saved.get_mut("meta") {
        *meta = serde_json::json!([]);
    }
    assert!(serde_json::from_value::<Conversation>(saved).is_err());
}

/// A view is the same value with nothing to save, and editing it leaves
/// the original alone.
#[test]
fn a_view_has_an_empty_outbox() {
    let conversation = Conversation::from_messages([Message::user("hi")]);
    let mut view = conversation.view();
    assert_eq!(view, conversation);
    assert!(view.unsaved().is_empty() && !conversation.unsaved().is_empty());
    view.push(Message::assistant("hello"));
    assert_eq!(view.unsaved().len(), 1);
    assert_eq!(conversation.messages().len(), 1);
}

/// After a stopped request the next push puts the stopped note, by rig,
/// first, in the same change; a kept halt adds none; a splice reaching the
/// end clears the halt and one short of it does not.
#[test]
fn a_stopped_request_gets_its_note_and_halts_clear() {
    let mut conversation = Conversation::from_messages([Message::user("delete it all")]);
    assert!(conversation.halt(Halt::Stopped));
    assert!(!conversation.awaits_model());
    conversation.take_unsaved();
    let next = conversation.push(Message::user("never mind"));
    assert_eq!(conversation.halted(), None);
    assert_eq!(conversation.index_of(next), Some(2));
    let stopped = conversation.entries().nth(1).expect("the stopped note");
    assert_eq!(stopped.message, &Message::user(STOPPED));
    assert_eq!(stopped.author, Some(&Author::rig()));
    let changes: Vec<_> = conversation.take_unsaved();
    let pushed = changes
        .iter()
        .filter(|recorded| matches!(recorded.change, Change::Push { .. }));
    assert_eq!(pushed.count(), 1, "the note and the message are one push");
    assert!(
        changes
            .iter()
            .any(|recorded| recorded.change == Change::Halt { halt: None })
    );
    assert_eq!(
        conversation.turn_start(),
        conversation.id(0).expect("first")
    );

    assert!(conversation.halt(Halt::Kept));
    conversation.splice(0..1, [], Author::plugin("trim"));
    assert_eq!(conversation.halted(), Some(Halt::Kept));
    conversation.push(Message::user("more"));
    assert_eq!(conversation.messages().len(), 3, "a kept halt adds no note");
    assert!(conversation.halt(Halt::Kept));
    conversation.splice(2.., [Entry::new(Message::user("redo"))], Author::user());
    assert_eq!(conversation.halted(), None);
    assert!(conversation.awaits_model());
}

/// The turn after a stopped one answers the stopped reply's results and
/// rig's note before the user's message: the request is the user's.
#[test]
fn a_turn_after_a_stop_is_asked_by_the_users_message() {
    use crate::message::{AssistantContent, AssistantMessage, CallId, ToolName};
    let shell = || ToolName::new("shell").expect("tool name");
    let mut conversation = Conversation::from_messages([
        Message::user("fix it"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::tool_call(
            "call-1",
            shell(),
            serde_json::json!({}),
        )])),
        Message::tool_result(CallId::from_wire("call-1"), shell(), "interrupted"),
    ]);
    assert!(conversation.halt(Halt::Stopped));
    let asked = conversation.push(Message::user("never mind"));
    let start = conversation.turn_start();

    let notes: Vec<bool> = conversation
        .asked(start)
        .map(|entry| entry.is_stopped_note())
        .collect();
    assert_eq!(notes, [false, true, false]);
    assert_eq!(
        conversation.request(start).map(|entry| entry.id),
        Some(asked)
    );
}
