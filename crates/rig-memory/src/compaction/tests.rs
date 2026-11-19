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
    let prepared = prepare_compaction_at(&conversation, 4).expect("something to summarize");
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

/// One pass over the log reads what each compaction replaced, and the conversation's summary
/// entry holds the state the next one builds on.
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
        read_files: vec!["a.rs".to_owned()],
        modified_files: Vec::new(),
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

    let found = summarized_in_log(&store, "agent").expect("read");
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
        summary_state(&load(&store, "agent").expect("loaded")),
        state
    );
}

/// A log cut after any line, as a crash leaves it, or inside one, reopens as the conversation
/// was right after one of its edits: never a summary without what it replaced or with another
/// state than it went in with, and never halted with a newer message in.
#[test]
fn a_log_cut_anywhere_reopens_as_the_conversation_once_was() {
    use crate::{MemoryStore, Store, load, save};

    let store = MemoryStore::default();
    let mut live = load(&store, "agent").expect("loaded");
    let state = |summary: &str| SummaryState {
        summary: summary.to_owned(),
        read_files: vec![format!("{summary}.rs")],
        modified_files: Vec::new(),
    };
    let compact = |it: &mut Conversation, end: usize, summary: &str| {
        apply_summary(it, &captured(it, end), &state(summary)).expect("applied");
    };
    let edits: [&dyn Fn(&mut Conversation); 9] = [
        &|it| it.extend(["one", "two"].map(Message::user)),
        &|it| assert!(it.halt(rig_core::conversation::Halt::Stopped)),
        &|it| it.extend([Message::user("three")]),
        &|it| compact(it, 2, "first"),
        &|it| assert!(it.halt(rig_core::conversation::Halt::Stopped)),
        &|it| compact(it, 3, "second"),
        &|it| it.extend([Message::user("four")]),
        &|it| assert!(it.halt(rig_core::conversation::Halt::Kept)),
        &|it| compact(it, 1, "third"),
    ];
    let mut seen = vec![(live.clone(), Vec::new())];
    for edit in edits {
        edit(&mut live);
        save(&store, "agent", &mut live).expect("saved");
        let replaced: Vec<_> = summarized_in_log(&store, "agent")
            .expect("read")
            .into_keys()
            .collect();
        seen.push((live.clone(), replaced));
    }
    let log = store.read_log("agent").expect("the log");
    let ends = log.iter().enumerate().filter(|(_, byte)| **byte == b'\n');
    let ends: Vec<usize> = ends.map(|(at, _)| at + 1).collect();
    assert_eq!(ends.len(), seen.len(), "a header, then one line an edit");
    for (line, end) in ends.iter().enumerate() {
        let (then, replaced) = seen.get(line).expect("as many values as lines");
        for torn in [0, 7] {
            let cut = MemoryStore::default();
            let bytes = log.get(..(end + torn).min(log.len())).unwrap_or_default();
            cut.append_log("agent", 0, bytes).expect("written");
            let reopened = load(&cut, "agent").expect("loaded");
            assert_eq!(&reopened, then, "after {line} lines and {torn} bytes");
            let found = summarized_in_log(&cut, "agent").expect("read");
            assert_eq!(&found.into_keys().collect::<Vec<_>>(), replaced);
            let summary = reopened.id(0).filter(|_| summarized_prefix(&reopened) > 0);
            assert!(summary.is_none_or(|summary| replaced.contains(&summary)));
        }
    }
    let halts: Vec<bool> = seen
        .iter()
        .map(|(then, _)| then.halted().is_some())
        .collect();
    let halted = [
        false, false, true, false, false, true, false, false, true, true,
    ];
    assert_eq!(halts, halted, "a summary short of the end keeps the halt");
    assert_eq!(summary_state(&live), state("third"));
}

/// Nothing is carried across a reset: a value that takes the log's place, even one whose
/// summary equals the old one's entry for entry, or the same value saved again or gone back
/// to, shows none of the messages the log's summaries replaced before it, so a conversation
/// never shows another's. A compaction made after the reset holds what it replaced.
#[test]
fn nothing_a_summary_replaced_is_carried_across_a_reset() {
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
        let found = summarized_in_log(store, "agent").expect("read");
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
    assert_eq!(replaced(&store, old.id(0)), first_two("old"));
    let mut current = messages("current", 3);
    save(&writer, &mut current);
    assert_eq!(summarized_in_log(&*store, "agent").expect("read").len(), 0);
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

    // Another conversation's nested summaries can equal this one's: neither
    // its outer nor its inner summary takes anything from this log.
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
    compact_to(&mut current, None);
    assert_eq!(current.entry(0), old.entry(0), "the summaries are equal");
    save(&writer, &mut current);
    assert_eq!(summarized_in_log(&*store, "agent").expect("read").len(), 0);

    // The same value saved again as a reset, and a value gone back to, keep
    // nothing either; what is compacted after it is held again.
    let (store, writer) = logged();
    let mut kept = messages("kept", 5);
    save(&writer, &mut kept);
    compact(&mut kept);
    save(&writer, &mut kept);
    let mut back = reread(&kept);
    compact(&mut kept);
    save(&writer, &mut kept);
    assert_eq!(summarized_in_log(&*store, "agent").expect("read").len(), 2);
    save(&writer, &mut reread(&kept));
    assert_eq!(summarized_in_log(&*store, "agent").expect("read").len(), 0);
    save(&writer, &mut back);
    assert_eq!(replaced(&store, back.id(0)), None);
    let inner = back.entry(0).map(|entry| (*entry.message).clone());
    compact(&mut back);
    save(&writer, &mut back);
    let held = inner.into_iter().chain([Message::user("kept three")]);
    assert_eq!(replaced(&store, back.id(0)), Some(held.collect()));
}

/// A secret taken out of a session's messages is in no summary state and nothing a summary
/// replaced afterwards: the summary's entry is the compaction's only copy of it.
#[test]
fn a_redacted_secret_is_in_no_summary_state_or_replaced_message() {
    use crate::{MemoryStore, Messages, load, rewrite, save};
    use rig_core::message::UserContent;

    let store = MemoryStore::default();
    let mut conversation = load(&store, "agent").expect("loaded");
    conversation.extend(["the key is hunter2", "two", "three", "four"].map(Message::user));
    let state = SummaryState {
        summary: "The key is hunter2.".to_owned(),
        read_files: vec!["hunter2.txt".to_owned()],
        modified_files: Vec::new(),
    };
    for _ in 0..2 {
        let read = captured(&conversation, 2);
        apply_summary(&mut conversation, &read, &state).expect("applied");
        save(&store, "agent", &mut conversation).expect("saved");
    }
    let redact = Messages(|message: &mut Message| {
        let Message::User { content } = message else {
            return false;
        };
        let mut changed = false;
        for item in content.iter_mut() {
            if let UserContent::Text(text) = item
                && text.text.contains("hunter2")
            {
                text.text = text.text.replace("hunter2", "[gone]");
                changed = true;
            }
        }
        changed
    });
    rewrite(&store, Author::plugin("redact"), &redact).expect("rewritten");
    let mut conversation = load(&store, "agent").expect("loaded");
    conversation.push(Message::user("five"));
    let state = summary_state(&conversation);
    assert_eq!(state.summary, "The key is [gone].");
    assert_eq!(state.read_files, ["[gone].txt"]);
    let prepared = prepare_compaction(&conversation, 1, false).expect("something to summarize");
    assert_eq!(prepared.previous, Some(state));
    let replaced = summarized_in_log(&store, "agent").expect("read");
    assert_eq!(replaced.len(), 2);
    let all = (
        serde_json::to_string(&conversation),
        serde_json::to_string(&replaced),
    );
    assert!(!format!("{all:?}").contains("hunter2"), "{all:?}");
}

/// A session saved when each compaction also wrote a `rig.compaction` aside still loads: the
/// aside is kept and read by nothing, and the state is the summary entry's.
#[test]
fn a_session_with_an_earlier_versions_compaction_asides_still_loads() {
    use crate::{MemoryStore, load, save};
    use rig_core::conversation::Aside;

    let store = MemoryStore::default();
    let mut conversation = load(&store, "agent").expect("loaded");
    conversation.extend(["one", "two", "three"].map(Message::user));
    let state = SummaryState {
        summary: "summary".to_owned(),
        read_files: vec!["a.rs".to_owned()],
        modified_files: Vec::new(),
    };
    let read = captured(&conversation, 2);
    let spliced = apply_summary(&mut conversation, &read, &state).expect("applied");
    let id = spliced.changed.first().copied();
    let value = serde_json::json!({
        "summary": id,
        "replaced": spliced.removed,
        "hash": "00000000000000000000000000000000",
        "state": {"summary": "another text", "read_files": ["stale.rs"]},
    });
    conversation.add_aside(Aside::new("rig.compaction", summary_author(), value).after(id));
    save(&store, "agent", &mut conversation).expect("saved");

    let mut loaded = load(&store, "agent").expect("loaded");
    assert_eq!(loaded, conversation);
    assert_eq!(loaded.asides().count(), 1, "the aside stays in the session");
    assert_eq!(summary_state(&loaded), state);
    let replaced = summarized_in_log(&store, "agent").expect("read");
    assert_eq!(replaced.get(&id.expect("a summary")).map(Vec::len), Some(2));
    loaded.push(Message::user("four"));
    let prepared = prepare_compaction(&loaded, 1, false).expect("something to summarize");
    assert_eq!(prepared.previous, Some(state));
    assert_eq!(prepared.files.read_files, ["a.rs"]);
    save(&store, "agent", &mut loaded).expect("saved");
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
