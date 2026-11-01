use super::*;

fn ids(conversation: &Conversation) -> Vec<MessageId> {
    conversation.entries().map(|entry| entry.id).collect()
}

fn fold(changes: &[ChangeRecord]) -> Conversation {
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
    edited.message = Message::user("two, edited").into();
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
    let Some(ChangeRecord { by, .. }) = conversation.unsaved().last() else {
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
    let read: Vec<ChangeRecord> = lines
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
    let stale = ChangeRecord {
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
    let reset = ChangeRecord {
        change: Change::Reset {
            state: Box::new(replacement.view()),
        },
        by: Some(Author::plugin("snapshot")),
        at: None,
    };
    let reset: ChangeRecord =
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

/// A reply stopped halfway is kept aborted and never read as an answer, so
/// the user's message is still owed one: a stop halts it, and the next message
/// comes after the stopped note.
#[test]
fn a_reply_stopped_halfway_leaves_the_request_owed() {
    use crate::message::{AssistantContent, AssistantMessage};
    let partial = vec![AssistantContent::text("The answer is")];
    let aborted = AssistantMessage::aborted(None, partial, "stopped by the user");
    let mut conversation =
        Conversation::from_messages([Message::user("what is it?"), Message::Assistant(aborted)]);
    assert!(conversation.awaits_model());
    assert!(conversation.halt(Halt::Stopped));
    conversation.push(Message::user("never mind"));
    let notes: Vec<bool> = conversation
        .entries()
        .map(|entry| entry.is_stop_marker())
        .collect();
    assert_eq!(notes, [false, false, true, false]);
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
        .map(|entry| entry.is_stop_marker())
        .collect();
    assert_eq!(notes, [false, true, false]);
    assert_eq!(
        conversation.request(start).map(|entry| entry.id),
        Some(asked)
    );
}

/// An edit of many entries in place is one recorded change by its author:
/// the rewritten entries keep their ids and authors, the fold matches, and
/// an edit that rewrites nothing records nothing.
#[test]
fn edit_each_rewrites_in_place_as_one_change() {
    let mut conversation = Conversation::from_messages(["a", "b", "c", "d"].map(Message::user));
    conversation.push(Entry::new(Message::user("e")).by(Author::plugin("notes")));
    let before = ids(&conversation);
    let shout = |entry: EntryRef<'_>| {
        [1, 4]
            .contains(&entry.index)
            .then(|| Message::user(format!("{}!", entry.index)))
    };
    let changed = conversation.edit_each(Author::plugin("shout"), shout);
    assert_eq!(
        changed,
        [before.get(1), before.get(4)]
            .into_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
    );
    assert_eq!(ids(&conversation), before);
    assert_eq!(
        conversation.messages().get(4),
        Some(&Message::user("4!").into())
    );
    assert_eq!(
        conversation.messages().get(2),
        Some(&Message::user("c").into())
    );
    assert_eq!(conversation.author(4), Some(&Author::plugin("notes")));
    let unsaved = conversation.take_unsaved();
    assert_eq!(unsaved.len(), 3, "the push, the extend and one edit");
    assert_eq!(fold(&unsaved).messages(), conversation.messages());
    assert!(
        conversation
            .edit_each(Author::plugin("none"), |_| None)
            .is_empty()
    );
    assert!(conversation.unsaved().is_empty());
}

/// Each entry keeps when it was made: an entry spliced back changed keeps
/// its time, a fold of the outbox and a reset (through its serde form)
/// give the same times, and equality leaves them out.
#[test]
fn made_times_fold_as_live_survive_edits_and_resets_and_stay_out_of_equality() {
    let keys = |conversation: &Conversation| -> Vec<MessageKey> {
        conversation.entries().map(|entry| entry.key()).collect()
    };
    let mut live = Conversation::new();
    live.dated(1_000, |conversation| {
        conversation.extend([Message::user("a"), Message::assistant("b")]);
    });
    let first = live.key(0).expect("a key");
    assert_eq!(first.made, 1_000);
    let mut edited = live.entry(0).expect("entry 0");
    edited.message = Message::user("a, edited").into();
    let fresh = Entry::new(Message::user("new"));
    live.dated(2_000, |conversation| {
        conversation.splice(0..1, [edited, fresh], Author::plugin("edit"))
    });
    assert_eq!(live.key(0), Some(first), "an edited entry keeps its time");
    assert_eq!(live.key(1).map(|key| key.made), Some(2_000));
    assert_eq!(live.index_of_key(first), Some(0));
    let other_time = MessageKey {
        made: 3,
        id: first.id,
    };
    assert_eq!(live.index_of_key(other_time), None);
    assert_eq!(keys(&fold(live.unsaved())), keys(&live));

    let reset = ChangeRecord {
        change: Change::Reset {
            state: Box::new(live.view()),
        },
        by: None,
        at: Some(9_000),
    };
    let reset: ChangeRecord =
        serde_json::from_str(&serde_json::to_string(&reset).expect("serializes")).expect("reads");
    let mut replaced = Conversation::from_messages([Message::user("x")]);
    replaced.apply(&reset).expect("a reset fits anything");
    assert_eq!(keys(&replaced), keys(&live));

    let mut later = Conversation::new();
    later.dated(5_000, |conversation| {
        conversation.extend([Message::user("a"), Message::assistant("b")]);
    });
    let now = Conversation::from_messages([Message::user("a"), Message::assistant("b")]);
    assert_eq!(later, now);
    assert_ne!(keys(&later), keys(&now));
}

/// Entries put back with `restore` keep their ids and made-times, an aside
/// that followed one follows it again, and a fold of the outbox (read back
/// from text) accepts them. `splice` still gives such entries new ids, and
/// `restore` gives one to an id held outside the range. `set_entries`
/// replaces only the range that changed.
#[test]
fn restore_puts_entries_back_with_their_ids_and_the_fold_accepts_them() {
    let keys = |conversation: &Conversation| -> Vec<MessageKey> {
        conversation.entries().map(|entry| entry.key()).collect()
    };
    let mut live = Conversation::new();
    live.dated(1_000, |conversation| {
        conversation.extend([
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
            Message::assistant("d"),
        ]);
    });
    let before = keys(&live);
    let note = Aside::new("note", Author::plugin("rate"), serde_json::Value::Null);
    let note = live.add_aside(note.after(live.id(2)));
    let originals: Vec<Entry> = (1..3).filter_map(|at| live.entry(at)).collect();
    let summary = Entry::new(Message::user("summary"));
    live.splice(1..3, [summary], Author::plugin("compact"));
    assert_eq!(live.aside_position(live.aside(note).expect("the note")), 1);

    let copies = live
        .clone()
        .splice(1..2, originals.clone(), Author::plugin("x"));
    assert!(copies.restored.is_empty() && copies.changed.iter().all(|id| id.0 >= 5));
    let outsider = live.entry(0).expect("entry 0");
    let mut with = originals.clone();
    with.push(outsider);
    let undone = live.dated(2_000, |conversation| {
        conversation.restore(1..2, with, Author::plugin("undo"))
    });
    let restored: Vec<MessageId> = originals.iter().filter_map(|entry| entry.id).collect();
    assert_eq!(undone.restored, restored);
    assert_eq!(undone.removed, [MessageId(4)]);
    let mut now = keys(&live);
    let copy = now.remove(3);
    assert_eq!(now, before, "a, b, c and d as they were");
    assert_eq!(
        (copy.id, copy.made),
        (MessageId(5), 2_000),
        "the outsider is new"
    );
    assert_eq!(live.aside_position(live.aside(note).expect("the note")), 3);

    let lines: Vec<String> = live
        .unsaved()
        .iter()
        .map(|change| serde_json::to_string(change).expect("a change serializes"))
        .collect();
    let read: Vec<ChangeRecord> = lines
        .iter()
        .map(|line| serde_json::from_str(line).expect("a change reads back"))
        .collect();
    let folded = fold(&read);
    assert_eq!(folded, live);
    assert_eq!(keys(&folded), keys(&live));

    let mut wanted: Vec<Entry> = (0..3).filter_map(|at| live.entry(at)).collect();
    if let Some(last) = wanted.last_mut() {
        last.message = Message::user("c, edited").into();
    }
    let revision = live.revision();
    let set = live.set_entries(wanted.clone(), Author::plugin("editor"));
    assert_eq!(set.changed, [MessageId(2)]);
    assert_eq!(set.removed, [MessageId(5), MessageId(3)]);
    assert_eq!(live.revision(), revision + 1, "one change");
    assert_eq!(
        live.set_entries(wanted, Author::plugin("editor")),
        Spliced::default()
    );
}

/// `normalize` saves the repair with ids kept: an aside on a merged entry
/// follows the entry it went into, one on a dropped entry follows the
/// entry before it, each move by the normalizer, and a fold of the outbox
/// equals the live value.
#[test]
fn normalize_keeps_ids_and_reanchors_asides() {
    use crate::message::{CallId, ToolName};
    let ghost = Message::tool_result(
        CallId::from_wire("ghost"),
        ToolName::new("add").expect("a tool name"),
        "3",
    );
    let mut live = Conversation::from_messages([
        Message::user("q"),
        Message::assistant("a"),
        ghost,
        Message::assistant("b"),
    ]);
    let before = ids(&live);
    let at = |index: usize| before.get(index).copied();
    let aside = |kind: &str, after| {
        Aside::new(kind, Author::plugin("rate"), serde_json::Value::Null).after(after)
    };
    let rating = live.add_aside(aside("rating", at(3)));
    let note = live.add_aside(aside("note", at(2)));
    let scratch = live.add_aside(aside("scratch", at(3)).unsaved());
    let saved = live.take_unsaved();

    let normalized = live.normalize(Author::plugin("tidy"));
    assert_eq!(
        normalized.merged,
        at(1).zip(at(3)).into_iter().collect::<Vec<_>>()
    );
    assert_eq!(normalized.dropped, at(2).into_iter().collect::<Vec<_>>());
    assert_eq!(
        ids(&live),
        [at(0), at(1)].into_iter().flatten().collect::<Vec<_>>()
    );
    for aside in [rating, note, scratch] {
        assert_eq!(live.aside(aside).and_then(|aside| aside.after), at(1));
    }
    let moves = live
        .unsaved()
        .iter()
        .filter(|recorded| matches!(recorded.change, Change::Aside { .. }));
    assert!(
        moves.clone().count() == 2
            && moves
                .clone()
                .all(|recorded| recorded.by == Some(Author::plugin("tidy")))
    );
    let mut all = saved;
    all.extend(live.unsaved().iter().cloned());
    assert_eq!(fold(&all), live);
}

/// `reread` takes what the log holds, lineage and revision too, and keeps
/// the value's unsaved asides with their ids; it is refused while the
/// value has unsaved changes.
#[test]
fn reread_takes_the_log_and_keeps_unsaved_asides() {
    let mut live = Conversation::from_messages([Message::user("my secret")]);
    let note = Aside::new("note", Author::plugin("n"), serde_json::json!("secret"));
    let note = live.add_aside(note);
    let scratch = Aside::new("scratch", Author::rig(), serde_json::Value::Null).unsaved();
    let scratch = live.add_aside(scratch);
    let mut log = fold(live.unsaved());
    log.edit_each(Author::plugin("redact"), |_| {
        Some(Message::user("my [redacted]"))
    });
    let redacted = Aside::new("note", Author::plugin("n"), serde_json::json!("[redacted]"));
    log.replace_aside(note, redacted.clone());
    let changes = u64::try_from(log.take_unsaved().len()).unwrap_or_default();
    log.continue_log(Lineage::fresh(), changes);

    assert!(live.reread(log.clone()).is_err(), "unsaved changes");
    live.take_unsaved();
    live.reread(log.clone()).expect("reread");
    assert_eq!(live, log);
    assert_eq!(
        (live.lineage(), live.revision()),
        (log.lineage(), log.revision())
    );
    assert_eq!(live.aside(note), Some(&redacted));
    assert!(live.aside(scratch).is_some_and(|aside| !aside.saved));
    let next = live.add_aside(Aside::new("next", Author::rig(), serde_json::Value::Null));
    assert!(next > scratch);
}

/// A turn record opens after the last entry, numbered after the last one,
/// and ends once: a record already ended, or none, is left alone. The
/// records fold back from the outbox.
#[test]
fn turn_records_open_numbered_and_end_once() {
    let mut conversation = Conversation::from_messages([Message::user("hi")]);
    let asked = conversation.id(0).expect("the request");
    let first = conversation.open_turn();
    assert_eq!((first.index, first.start, first.end), (0, asked, None));
    let answer = conversation.push(Message::assistant("hello"));
    assert!(conversation.end_turn(TurnEnd::Answered {
        answer: Some(answer)
    }));
    assert!(!conversation.end_turn(TurnEnd::Failed { why: "late".into() }));
    conversation.push(Message::user("and?"));
    assert_eq!(conversation.open_turn().index, 1);
    let stopped = TurnEnd::Stopped {
        by: Some(StopCause::User),
        at: conversation.entries().next_back().map(|entry| entry.id),
        calls: vec![CallRef {
            reply: answer,
            index: 0,
        }],
    };
    assert!(conversation.end_turn(stopped.clone()));

    let records: Vec<_> = conversation.turns().map(|(_, record)| record).collect();
    let ends: Vec<_> = records.iter().map(|record| record.end.clone()).collect();
    let stopped = Some(stopped);
    assert_eq!(
        ends,
        [
            Some(TurnEnd::Answered {
                answer: Some(answer)
            }),
            stopped
        ]
    );
    let second = records
        .get(1)
        .and_then(|record| conversation.request(record.start));
    assert_eq!(
        second.map(|entry| entry.message),
        Some(&Message::user("and?"))
    );
    assert_eq!(fold(&conversation.take_unsaved()), conversation);
}

/// A stop's cause reads back as it was saved, and a stop saved before
/// causes and call places were recorded reads as recording neither,
/// whatever text and call ids it kept.
#[test]
fn a_stop_cause_reads_back_and_an_old_stop_has_none() {
    let causes = [
        StopCause::User,
        StopCause::Exit("the run ended".into()),
        StopCause::Restarted,
        StopCause::Plugin {
            by: "budget".into(),
            why: "over budget".into(),
        },
    ];
    for cause in causes {
        let end = TurnEnd::Stopped {
            by: Some(cause),
            at: None,
            calls: Vec::new(),
        };
        let saved = serde_json::to_value(&end).expect("a turn end serializes");
        let read: TurnEnd = serde_json::from_value(saved).expect("a turn end reads back");
        assert_eq!(read, end);
    }
    let old = serde_json::json!({"how": "stopped", "why": "compacted", "calls": ["c1"]});
    let read: TurnEnd = serde_json::from_value(old).expect("an old stop reads");
    assert_eq!(
        read,
        TurnEnd::Stopped {
            by: None,
            at: None,
            calls: Vec::new()
        }
    );
}

/// Ids have a plain form: their number, and `m4`/`a4` text that reads back,
/// with or without the letter; anything else is refused.
#[test]
fn ids_read_back_from_their_plain_form() {
    let mut conversation = Conversation::new();
    conversation.push(Message::user("one"));
    let id = conversation.push(Message::user("two"));
    assert_eq!(id.get(), 1);
    assert_eq!(id.to_string(), "m1");
    assert_eq!("m1".parse(), Ok(id));
    assert_eq!("1".parse(), Ok(id));
    assert!("a1".parse::<MessageId>().is_err());

    let aside = conversation.add_aside(Aside::new("note", Author::rig(), serde_json::json!(1)));
    assert_eq!(aside.to_string(), format!("a{}", aside.get()));
    assert_eq!(aside.to_string().parse(), Ok(aside));
    assert_eq!("m0".parse::<AsideId>(), Err(ParseIdError("m0".to_owned())));
}
