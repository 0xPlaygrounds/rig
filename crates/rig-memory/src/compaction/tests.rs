use std::sync::Arc;

use super::*;

/// A summary goes in by id, so an edit before its cut does not move it, and replaces nothing
/// once the message it keeps from is gone.
#[test]
fn a_summary_takes_the_place_of_what_it_summarized_by_id() {
    let mut conversation = Conversation::from_messages([
        Message::user("start"),
        Message::assistant("one"),
        Message::user("next"),
        Message::assistant("two"),
        Message::user("more"),
        Message::assistant("done"),
    ]);
    let last = |conversation: &mut Conversation| {
        let mut changes = conversation.take_unsaved().into_iter().rev();
        changes.find_map(|recorded| summary_in(&recorded.change))
    };
    let state = SummaryState {
        summary: "read two files".to_owned(),
        ..SummaryState::default()
    };
    let upto = conversation.id(5);
    conversation.splice(0..1, [], Author::plugin("other"));
    let spliced = apply_summary(&mut conversation, upto, &state).expect("applied");
    assert_eq!(spliced.removed.len(), 4);
    assert_eq!(conversation.messages().len(), 2);
    assert_eq!(summarized_prefix(&conversation), 1);
    assert_eq!(conversation.author(0), Some(&summary_author()));
    assert_eq!(last(&mut conversation), conversation.id(0));
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
        ..SummaryState::default()
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

/// One pass over the log reads what each compaction replaced, and a
/// summary an undo put back keeps what it first replaced.
#[test]
fn the_log_holds_what_each_summary_first_replaced() {
    use crate::{MemoryStore, load, save};

    let store = MemoryStore::default();
    let mut conversation = load(&store, "agent").expect("loaded");
    for text in ["one", "two", "three", "four"] {
        conversation.push(Message::user(text));
    }
    let ids = |conversation: &Conversation| -> Vec<MessageId> {
        conversation.entries().map(|entry| entry.id).collect()
    };
    let before = ids(&conversation);
    let state = SummaryState {
        summary: "summary".to_owned(),
        ..SummaryState::default()
    };
    let compact = |conversation: &mut Conversation| {
        let upto = conversation.id(2);
        let spliced = apply_summary(conversation, upto, &state).expect("applied");
        save(&store, "agent", conversation).expect("saved");
        spliced.changed.first().copied().expect("a summary")
    };
    let first = compact(&mut conversation);
    let after_first = ids(&conversation);
    let second = compact(&mut conversation);
    let by = Author::plugin("undo");
    undo_summary(&mut conversation, &store, "agent", by).expect("undone");
    save(&store, "agent", &mut conversation).expect("saved");
    let third = compact(&mut conversation);

    let Summarized {
        replaced: found,
        records,
    } = summarized_in_log(&store, "agent").expect("read");
    let replaced = |summary| -> Option<Vec<MessageId>> {
        let entries = found.get(&summary)?;
        entries.iter().map(|entry| entry.id).collect()
    };
    assert_eq!(found.len(), 3);
    assert_eq!(replaced(first).as_deref(), before.get(..2));
    assert_eq!(replaced(second).as_deref(), after_first.get(..2));
    assert_eq!(replaced(third).as_deref(), after_first.get(..2));
    let texts = found.get(&first).into_iter().flatten();
    let texts: Vec<_> = texts.map(|entry| entry.message.clone()).collect();
    assert_eq!(
        texts,
        [Message::user("one"), Message::user("two")].map(Arc::new)
    );
    assert_eq!(
        records.get(&first).map(|record| &record.state),
        Some(&state)
    );
}

/// A reset keeps what summaries replaced only for a value whose records
/// prove it. A new conversation whose summary reuses an id, even one equal
/// to the old summary, takes nothing from the old one, saved or not. The
/// same value saved again undoes all the way down, edited or not, and so
/// does a value gone back to after another took its place. A record saved
/// before hashes were kept undoes from its log but proves nothing.
#[test]
fn a_reset_keeps_what_a_summary_replaced_only_for_the_same_conversation() {
    use crate::{MemoryStore, Writer};

    let logged = || {
        let store = Arc::new(MemoryStore::default());
        let (writer, _) = Writer::open(store.clone()).expect("opened");
        writer.start();
        (store, writer)
    };
    let save = |writer: &Writer, conversation: &mut Conversation| {
        writer.save("agent", conversation);
        writer.flush().expect("flushed");
    };
    let state = SummaryState {
        summary: "summary".to_owned(),
        ..SummaryState::default()
    };
    let compact_to = |conversation: &mut Conversation, cut: Option<usize>| {
        let upto = cut.and_then(|at| conversation.id(at));
        conversation.dated(42, |conversation| {
            apply_summary(conversation, upto, &state).expect("applied");
        });
    };
    let compact = |conversation: &mut Conversation| compact_to(conversation, Some(2));
    let messages = |prefix: &str, count: usize| {
        let texts = ["one", "two", "three", "four", "five"]
            .into_iter()
            .take(count);
        Conversation::from_messages(texts.map(|text| Message::user(format!("{prefix} {text}"))))
    };
    let reread = |conversation: &Conversation| -> Conversation {
        let value = serde_json::to_value(conversation).expect("serialized");
        serde_json::from_value(value).expect("deserialized")
    };
    let by = Author::plugin("undo");

    // A new conversation compacted after the reset undoes to its own.
    let (store, writer) = logged();
    let mut old = messages("old", 3);
    save(&writer, &mut old);
    compact(&mut old);
    save(&writer, &mut old);
    let mut current = messages("current", 3);
    let original = current.clone();
    save(&writer, &mut current);
    compact(&mut current);
    save(&writer, &mut current);
    assert_eq!(
        current.id(0),
        old.id(0),
        "the reset reused the summary's id"
    );
    undo_summary(&mut current, &*store, "agent", by.clone()).expect("undone");
    assert_eq!(current.messages(), original.messages());

    // One that came already compacted, with an equal summary, has no
    // originals in this log, saved or not, whichever of the two kept a tail.
    let cuts = [
        (Some(2), Some(2)),
        (Some(2), None),
        (None, Some(2)),
        (None, None),
    ];
    for (old_cut, current_cut) in cuts {
        let (store, writer) = logged();
        let mut old = messages("old", 3);
        save(&writer, &mut old);
        compact_to(&mut old, old_cut);
        save(&writer, &mut old);
        let mut current = messages("current", 3);
        compact_to(&mut current, current_cut);
        assert_eq!(current.entry(0), old.entry(0), "the summaries are equal");
        for _ in ["unsaved", "saved"] {
            let imported = current.clone();
            let undone = undo_summary(&mut current, &*store, "agent", by.clone());
            assert!(matches!(undone, Err(UndoError::NotInLog)), "{undone:?}");
            assert_eq!(current, imported);
            save(&writer, &mut current);
        }
    }

    // Its summary put back by an undo still has none, though the undo names it.
    let (store, writer) = logged();
    save(&writer, &mut messages("old", 3));
    let mut current = messages("current", 4);
    compact(&mut current);
    save(&writer, &mut current);
    compact(&mut current);
    save(&writer, &mut current);
    undo_summary(&mut current, &*store, "agent", by.clone()).expect("undone");
    save(&writer, &mut current);
    let arrived = current.clone();
    let undone = undo_summary(&mut current, &*store, "agent", by.clone());
    assert!(matches!(undone, Err(UndoError::NotInLog)), "{undone:?}");
    assert_eq!(current, arrived);

    // Three nested summaries saved again as a reset undo down to the start.
    let (store, writer) = logged();
    let mut nested = messages("nested", 5);
    let original = nested.clone();
    save(&writer, &mut nested);
    for _ in 0..3 {
        compact(&mut nested);
        save(&writer, &mut nested);
    }
    let mut nested = reread(&nested);
    save(&writer, &mut nested);
    for _ in 0..3 {
        undo_summary(&mut nested, &*store, "agent", by.clone()).expect("undone");
        save(&writer, &mut nested);
    }
    assert_eq!(nested.messages(), original.messages());

    // The same value edited before it was saved again still undoes.
    let (store, writer) = logged();
    let mut edited = messages("edited", 3);
    let original = edited.clone();
    save(&writer, &mut edited);
    compact(&mut edited);
    save(&writer, &mut edited);
    let mut edited = reread(&edited);
    edited.edit_each(Author::plugin("redact"), |_| {
        Some(Message::user("redacted"))
    });
    save(&writer, &mut edited);
    undo_summary(&mut edited, &*store, "agent", by.clone()).expect("undone");
    assert_eq!(edited.messages().get(..2), original.messages().get(..2));

    // Another conversation's outer summary can equal this one's, with what
    // it replaced, but its inner summary takes nothing from this log.
    let dated = |first: &str, second: &str| {
        let mut conversation = Conversation::new();
        conversation.dated(10, |conversation| {
            conversation.extend([first, second, "tail"].map(Message::user));
        });
        conversation
    };
    let (store, writer) = logged();
    let mut old = dated("a", "b");
    save(&writer, &mut old);
    compact(&mut old);
    save(&writer, &mut old);
    compact_to(&mut old, None);
    save(&writer, &mut old);
    let mut current = dated("x", "y");
    compact(&mut current);
    let inner = current.id(0).expect("the inner summary");
    compact_to(&mut current, None);
    assert_eq!(current.entry(0), old.entry(0), "the summaries are equal");
    save(&writer, &mut current);
    let found = summarized_in_log(&*store, "agent").expect("read");
    assert!(!found.replaced.contains_key(&inner));
    undo_summary(&mut current, &*store, "agent", by.clone()).expect("undone");
    let outer_undone = current.clone();
    let undone = undo_summary(&mut current, &*store, "agent", by.clone());
    assert!(matches!(undone, Err(UndoError::NotInLog)), "{undone:?}");
    assert_eq!(current, outer_undone);

    // A record with no hash undoes from its log, but not across a reset,
    // even under a newer compaction that has one.
    let compact_unhashed = |conversation: &mut Conversation| {
        let summary = Entry::new(Message::user("summary")).by(summary_author());
        let spliced = conversation.splice(..2, [summary], summary_author());
        let id = spliced.changed.first().copied();
        let value = serde_json::json!({"summary": id, "replaced": spliced.removed, "state": state});
        conversation.add_aside(Aside::new(COMPACTED, summary_author(), value).after(id));
    };
    let (store, writer) = logged();
    let mut under = messages("under", 3);
    save(&writer, &mut under);
    compact_unhashed(&mut under);
    save(&writer, &mut under);
    compact_to(&mut under, None);
    save(&writer, &mut under);
    let mut under = reread(&under);
    save(&writer, &mut under);
    undo_summary(&mut under, &*store, "agent", by.clone()).expect("undone");
    let outer_undone = under.clone();
    let undone = undo_summary(&mut under, &*store, "agent", by.clone());
    assert!(matches!(undone, Err(UndoError::NotInLog)), "{undone:?}");
    assert_eq!(under, outer_undone);

    let (store, writer) = logged();
    let mut unhashed = messages("unhashed", 3);
    let original = unhashed.clone();
    save(&writer, &mut unhashed);
    compact_unhashed(&mut unhashed);
    save(&writer, &mut unhashed);
    let mut reset = reread(&unhashed);
    undo_summary(&mut unhashed, &*store, "agent", by.clone()).expect("undone");
    assert_eq!(unhashed.messages(), original.messages());
    save(&writer, &mut reset);
    let unproved = reset.clone();
    let undone = undo_summary(&mut reset, &*store, "agent", by.clone());
    assert!(matches!(undone, Err(UndoError::NotInLog)), "{undone:?}");
    assert_eq!(reset, unproved);

    // A value gone back to, from before a later compaction and a
    // replacement, undoes to what it replaced.
    let (store, writer) = logged();
    let mut kept = messages("kept", 3);
    let original = kept.clone();
    save(&writer, &mut kept);
    compact(&mut kept);
    save(&writer, &mut kept);
    let mut back = reread(&kept);
    kept.push(Message::user("kept four"));
    save(&writer, &mut kept);
    compact(&mut kept);
    save(&writer, &mut kept);
    save(&writer, &mut messages("other", 3));
    save(&writer, &mut back);
    undo_summary(&mut back, &*store, "agent", by).expect("undone");
    assert_eq!(back.messages(), original.messages());
}
