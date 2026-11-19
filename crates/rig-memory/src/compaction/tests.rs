use std::sync::Arc;

use super::*;

/// The first `end` messages of `conversation`, as a compaction captures what it replaces.
fn captured(conversation: &Conversation, end: usize) -> Vec<Arc<Message>> {
    conversation
        .messages()
        .get(..end)
        .unwrap_or_default()
        .to_vec()
}

/// A summary replaces exactly the messages its compaction read, and only while they are as it
/// read them: a message added meanwhile stays, and an edit or a removal among them, or an
/// earlier summary replaced, refuses the summary and changes nothing.
#[test]
fn a_summary_replaces_exactly_what_it_read_and_only_if_unchanged() {
    let fresh = || {
        Conversation::from_messages([
            Message::user("start"),
            Message::assistant("one"),
            Message::user("next"),
            Message::assistant("two"),
            Message::user("more"),
        ])
    };
    let state = |summary: &str| SummaryState {
        summary: summary.to_owned(),
        ..SummaryState::default()
    };

    let mut conversation = fresh();
    let prepared = prepare_compaction_at(
        &conversation,
        CutPoint {
            first_kept: 4,
            turn_start: None,
        },
    )
    .expect("something to summarize");
    assert_eq!(prepared.replaces, captured(&conversation, 4));
    conversation.push(Message::assistant("added while the summary was written"));
    let spliced = apply_summary(&mut conversation, &prepared.replaces, &state("first"));
    assert_eq!(spliced.expect("applied").removed.len(), 4);
    let texts: Vec<_> = conversation.messages().iter().skip(1).cloned().collect();
    let kept = [
        Message::user("more"),
        Message::assistant("added while the summary was written"),
    ];
    assert_eq!(texts, kept.map(Arc::new));
    assert_eq!(summarized_prefix(&conversation), 1);
    let mut changes = conversation.take_unsaved().into_iter().rev();
    let summary = changes.find_map(|recorded| summary_in(&recorded.change));
    assert_eq!(summary, conversation.id(0));

    // A summary of every message leaves one added while it was written.
    let mut whole = fresh();
    let read = captured(&whole, 5);
    whole.push(Message::assistant("added"));
    apply_summary(&mut whole, &read, &state("all")).expect("applied");
    let after: Vec<_> = whole.messages().iter().skip(1).cloned().collect();
    assert_eq!(after, [Arc::new(Message::assistant("added"))]);

    // The next one read the first summary: once another summary took its place, it is refused.
    let read = captured(&conversation, 2);
    let other = captured(&conversation, 1);
    apply_summary(&mut conversation, &other, &state("second")).expect("applied");
    let before = conversation.clone();
    assert_eq!(
        apply_summary(&mut conversation, &read, &state("stale")),
        None
    );
    assert_eq!(conversation, before);

    let mut edited = fresh();
    let read = captured(&edited, 4);
    edited.edit_each(Author::plugin("redact"), |_| {
        Some(Message::user("redacted"))
    });
    let before = edited.clone();
    assert_eq!(apply_summary(&mut edited, &read, &state("stale")), None);
    assert_eq!(edited, before);

    let mut shorter = fresh();
    let read = captured(&shorter, 4);
    shorter.splice(0..1, [], Author::plugin("other"));
    assert_eq!(apply_summary(&mut shorter, &read, &state("stale")), None);
    shorter.splice(1.., [], Author::plugin("other"));
    assert_eq!(apply_summary(&mut shorter, &read, &state("stale")), None);
}

/// Right after a compaction nothing is planned again, however much it kept, as pi refuses a
/// compaction when nothing was added since the last; a message added after it is planned for.
#[test]
fn nothing_is_compacted_again_until_something_is_added() {
    let long = |text: &str| Message::user(format!("{text}: {}", "word ".repeat(400)));
    let mut conversation = Conversation::from_messages([
        long("one"),
        Message::assistant("first"),
        long("two"),
        Message::assistant("second"),
        long("three"),
        Message::assistant("third"),
    ]);
    let prepared = prepare_compaction(&conversation, 600, false).expect("planned");
    let state = SummaryState {
        summary: "summary".to_owned(),
        ..SummaryState::default()
    };
    apply_summary(&mut conversation, &prepared.replaces, &state).expect("applied");
    assert!(conversation.messages().len() > 2, "more than one turn kept");
    assert!(already_compacted(&conversation));
    assert_eq!(prepare_compaction(&conversation, 10, false), None);
    conversation.push(long("four"));
    assert!(!already_compacted(&conversation));
    assert!(prepare_compaction(&conversation, 10, false).is_some());
}

/// One pass over the log reads what each compaction replaced.
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
        let read = captured(conversation, 2);
        let spliced = apply_summary(conversation, &read, &state).expect("applied");
        save(&store, "agent", conversation).expect("saved");
        spliced.changed.first().copied().expect("a summary")
    };
    let first = compact(&mut conversation);
    let after_first = ids(&conversation);
    let second = compact(&mut conversation);

    let Summarized {
        replaced: found,
        records,
    } = summarized_in_log(&store, "agent").expect("read");
    let replaced = |summary| -> Option<Vec<MessageId>> {
        let entries = found.get(&summary)?;
        entries.iter().map(|entry| entry.id).collect()
    };
    assert_eq!(found.len(), 2);
    assert_eq!(replaced(first).as_deref(), before.get(..2));
    assert_eq!(replaced(second).as_deref(), after_first.get(..2));
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
/// to the old summary, takes nothing from the old one. The same value
/// saved again keeps every nested summary's, and so does a value gone back
/// to after another took its place. A record saved before hashes were kept
/// proves nothing across a reset.
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
        let read = captured(conversation, cut.unwrap_or(conversation.messages().len()));
        conversation.dated(42, |conversation| {
            apply_summary(conversation, &read, &state).expect("applied");
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
    let replaced = |store: &MemoryStore, summary: Option<MessageId>| -> Option<Vec<Message>> {
        let found = summarized_in_log(store, "agent").expect("read").replaced;
        let entries = found.get(&summary?)?;
        Some(
            entries
                .iter()
                .map(|entry| (*entry.message).clone())
                .collect(),
        )
    };
    let first_two = |prefix: &str| -> Option<Vec<Message>> {
        Some(vec![
            Message::user(format!("{prefix} one")),
            Message::user(format!("{prefix} two")),
        ])
    };

    // A new conversation compacted after the reset has its own.
    let (store, writer) = logged();
    let mut old = messages("old", 3);
    save(&writer, &mut old);
    compact(&mut old);
    save(&writer, &mut old);
    let mut current = messages("current", 3);
    save(&writer, &mut current);
    compact(&mut current);
    save(&writer, &mut current);
    assert_eq!(
        current.id(0),
        old.id(0),
        "the reset reused the summary's id"
    );
    assert_eq!(replaced(&store, current.id(0)), first_two("current"));

    // One that came already compacted, with an equal summary, has nothing
    // in this log, whichever of the two kept a tail.
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
        save(&writer, &mut current);
        assert_eq!(replaced(&store, current.id(0)), None);
    }

    // Three nested summaries saved again as a reset keep the first one's.
    let (store, writer) = logged();
    let mut nested = messages("nested", 5);
    save(&writer, &mut nested);
    compact(&mut nested);
    let first = nested.id(0);
    save(&writer, &mut nested);
    for _ in 0..2 {
        compact(&mut nested);
        save(&writer, &mut nested);
    }
    let newest = nested.id(0);
    let mut nested = reread(&nested);
    save(&writer, &mut nested);
    assert_eq!(replaced(&store, newest).map(|taken| taken.len()), Some(2));
    assert_eq!(replaced(&store, first), first_two("nested"));

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

    // A record with no hash proves nothing across a reset.
    let (store, writer) = logged();
    let mut unhashed = messages("unhashed", 3);
    save(&writer, &mut unhashed);
    let summary = Entry::new(Message::user("summary")).by(summary_author());
    let spliced = unhashed.splice(..2, [summary], summary_author());
    let id = spliced.changed.first().copied();
    let value = serde_json::json!({"summary": id, "replaced": spliced.removed, "state": state});
    unhashed.add_aside(Aside::new(COMPACTED, summary_author(), value).after(id));
    save(&writer, &mut unhashed);
    assert_eq!(replaced(&store, id), first_two("unhashed"));
    save(&writer, &mut reread(&unhashed));
    assert_eq!(replaced(&store, id), None);

    // A value gone back to, from before a later compaction and a
    // replacement, keeps what it replaced.
    let (store, writer) = logged();
    let mut kept = messages("kept", 3);
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
    assert_eq!(replaced(&store, back.id(0)), first_two("kept"));
}

/// Another plugin's request on a small model (a 4096-token window, a 1024-token output limit) keeps
/// conversation text and the answer room it fitted, as before compaction followed pi.
#[test]
fn a_small_model_keeps_text_and_room_for_another_plugins_summary() {
    let spec = rig_core::providers::registry::ProviderId::catalog("ollama")
        .map(|vendor| {
            ModelSpec::new(vendor, "small")
                .with_context_window(4_096)
                .with_max_output_tokens(1_024)
        })
        .expect("a vendor");
    let text = format!("OLDEST {} NEWEST", "word ".repeat(2_000));
    let (fitted, most) = fit_to_model(&text, &spec, 4_000);
    assert_eq!(most, Some(1_024));
    assert!(
        fitted.ends_with("NEWEST") && !fitted.contains("OLDEST"),
        "{fitted}"
    );
    assert!(fitted.chars().count() > 2_000, "{}", fitted.chars().count());
    let request =
        crate::summarization_request(("Write a brief.", fitted.into_owned()), &spec, None, most)
            .expect("a request");
    assert_eq!(request.max_tokens, Some(1_024));
}
