use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::{Aside, Author, Change, Conversation, Entry, Halt};
use rig_core::message::{ImageMediaType, UserContent};
use serde_json::json;

use super::{Record, SavedAgent, Session, as_of, fork, history, load, save};
use crate::{Files, LoadError, MemoryStore, Repair, Store, Writer};

/// A conversation with an image, an authored entry, a splice that keeps an
/// id, a stopped request, and a saved and an unsaved aside.
fn edited() -> Conversation {
    let mut conversation = Conversation::new();
    edit(&mut conversation);
    conversation
}

/// Makes those edits to `conversation`.
fn edit(conversation: &mut Conversation) {
    conversation.push(Message::User {
        content: vec![
            UserContent::text("look"),
            UserContent::image_base64("cG5n", Some(ImageMediaType::PNG), None),
        ],
    });
    conversation.push(Message::assistant("a cat"));
    conversation.push(Entry::new(Message::user("report")).by(Author::plugin("ci")));
    let mut first = conversation.entry(0).expect("an entry");
    first.message = Message::user("look again");
    conversation.splice(0..1, [first], Author::plugin("editor"));
    conversation.halt(Halt::Stopped);
    conversation.push(Message::user("go on"));
    let note = Aside::new("note", Author::plugin("btw"), json!({ "text": "hi" }));
    conversation.add_aside(note.clone());
    conversation.add_aside(note.unsaved());
}

#[test]
fn a_saved_conversation_folds_back_to_itself() {
    let store = MemoryStore::default();
    let mut conversation = load(&store, "agent").expect("an empty log");
    conversation.push(Message::user("first"));
    save(&store, "agent", &mut conversation).expect("saved");
    edit(&mut conversation);
    conversation.splice(0..1, [], Author::plugin("editor"));
    conversation.halt(Halt::Kept);
    save(&store, "agent", &mut conversation).expect("saved again");
    assert!(conversation.unsaved().is_empty());
    let loaded = load(&store, "agent").expect("loaded");
    assert_eq!(loaded, conversation);
    assert_eq!(loaded.revision(), conversation.revision());
    assert_eq!(
        store.blobs().expect("blobs").len(),
        1,
        "the image is a blob"
    );
    let log = String::from_utf8(store.read_log("agent").expect("a log")).expect("text");
    assert!(log.starts_with(r#"{"seq":0,"n":0,"#), "{log}");
    assert!(log.contains(r#""format":"rig-session/1""#), "{log}");
    assert!(!log.contains("cG5n"), "the image data is not in the log");

    // A running writer's log folds back the same, with saved values.
    let store = Arc::new(MemoryStore::default());
    let (writer, session) = Writer::open(store.clone()).expect("an empty session");
    assert_eq!(session, Session::default());
    writer.start();
    let mut live = edited();
    let saved = writer.save("main", &mut live);
    assert!(saved.iter().all(|change| change.at.is_some()));
    writer.state("main", "mode", Some(json!("plan")), false);
    writer.state(crate::SESSION, "theme", Some(json!("dark")), false);
    writer.flush();
    let session = Session::read(&*store).expect("read");
    let main = session.agent("main").expect("the agent");
    assert_eq!(main.conversation, live);
    assert_eq!(main.state, [("mode".to_owned(), json!("plan"))].into());
    assert_eq!(session.state, [("theme".to_owned(), json!("dark"))].into());
    assert_eq!(main.stamps.len(), live.messages().len());
}

#[test]
fn saving_refuses_a_log_that_moved_on() {
    let store = MemoryStore::default();
    let mut first = load(&store, "agent").expect("loaded");
    let mut second = load(&store, "agent").expect("loaded");
    first.push(Message::user("one"));
    save(&store, "agent", &mut first).expect("saved");
    let before = store.read_log("agent").expect("a log");
    second.push(Message::user("two"));
    let refused = save(&store, "agent", &mut second);
    assert!(matches!(refused, Err(LoadError::Changed { log }) if log == "agent"));
    assert_eq!(store.read_log("agent").expect("a log"), before);
}

#[test]
fn a_writer_saves_a_replaced_conversation_as_one_reset() {
    let store = Arc::new(MemoryStore::default());
    let (writer, _) = Writer::open(store.clone()).expect("opened");
    writer.start();
    let mut conversation = edited();
    writer.save("main", &mut conversation);
    let mut replaced = Conversation::from_messages([Message::user("elsewhere")]);
    let saved = writer.save("main", &mut replaced);
    assert!(matches!(saved.as_slice(), [only] if matches!(only.change, Change::Reset { .. })));
    // It goes on from there.
    replaced.push(Message::assistant("ok"));
    writer.save("main", &mut replaced);
    writer.flush();
    let session = Session::read(&*store).expect("read");
    assert_eq!(session.agent("main").expect("agent").conversation, replaced);
}

#[test]
fn a_torn_tail_is_reported_then_set_aside() {
    let store = Arc::new(MemoryStore::default());
    let mut conversation = Conversation::from_messages([Message::user("hello")]);
    save(&*store, "main", &mut conversation).expect("saved");
    let whole = store.read_log("main").expect("a log");
    store.append_log("main", br#"{"seq":2,"n":"#).expect("torn");

    // Reading reports it and changes nothing.
    let session = Session::read(&*store).expect("read");
    let torn = Repair::TornTail {
        log: "main".to_owned(),
        kept: whole.len() as u64,
        cut: 13,
        set_aside: None,
    };
    assert_eq!(session.repairs, [torn]);
    assert_eq!(
        session.agent("main").expect("agent").conversation,
        conversation
    );
    assert!(store.files().expect("files").is_empty());

    // Opening it to write sets the tail aside and cuts it off.
    let (writer, session) = Writer::open(store.clone()).expect("opened");
    let part = "recovered/main-1.part";
    assert!(matches!(
        session.repairs.as_slice(),
        [Repair::TornTail { set_aside: Some(file), cut: 13, .. }] if file == part
    ));
    assert_eq!(
        store.read_file(part).expect("set aside"),
        br#"{"seq":2,"n":"#
    );
    assert_eq!(store.read_log("main").expect("a log"), whole);
    writer.start();
    let mut conversation = session
        .agents
        .into_iter()
        .next()
        .expect("agent")
        .conversation;
    conversation.push(Message::assistant("hi"));
    writer.save("main", &mut conversation);
    writer.flush();
    assert_eq!(load(&*store, "main").expect("loaded"), conversation);
}

#[test]
fn a_session_it_cannot_read_whole_is_refused() {
    let store = MemoryStore::default();
    let mut conversation = Conversation::from_messages([Message::user("a"), Message::user("b")]);
    save(&store, "main", &mut conversation).expect("saved");
    store
        .append_log(
            "main",
            b"{\"seq\":2,\"n\":2,\"t\":0,\"type\":\"mystery\"}\n",
        )
        .expect("written");
    store
        .append_log(
            "main",
            b"{\"seq\":3,\"n\":3,\"t\":0,\"type\":\"halt\",\"halt\":\"kept\"}\n",
        )
        .expect("written");
    let refused = Session::read(&store);
    assert!(
        matches!(&refused, Err(LoadError::Record { log, line: 3, .. }) if log == "main"),
        "{refused:?}"
    );
    // A log from before this format names none.
    let old = MemoryStore::default();
    old.append_log(
        "old",
        b"{\"seq\":0,\"n\":0,\"t\":0,\"type\":\"header\",\"agent\":\"old\"}\n",
    )
    .expect("written");
    let refused = load(&old, "old");
    assert!(
        matches!(refused, Err(LoadError::Format { found: None, .. })),
        "{refused:?}"
    );
}

#[test]
fn fork_history_and_as_of_cut_by_the_record_that_made_a_message() {
    let store = MemoryStore::default();
    store
        .write_file("plugins/notes.json", b"{}")
        .expect("written");
    store.write_file("meta.json", b"{}").expect("written");
    let mut session = Session::default();
    session.agents.push(SavedAgent {
        id: "main".to_owned(),
        conversation: Conversation::from_messages([
            Message::user("one"),
            Message::assistant("two"),
        ]),
        ..SavedAgent::default()
    });
    session.save(&store).expect("saved");
    let main = session.agent_mut("main").expect("main");
    let three = main.conversation.push(Message::user("three"));
    session.agents.push(SavedAgent {
        id: "sub".to_owned(),
        parent: Some("main".to_owned()),
        conversation: Conversation::from_messages([Message::user("work")]),
        ..SavedAgent::default()
    });
    session.save(&store).expect("saved again");

    let records = history(&store, "main").expect("history");
    let made = records.iter().find_map(|(stamp, record)| match record {
        Record::Change(recorded) => match &recorded.change {
            Change::Push { entries } if entries.iter().any(|entry| entry.id == Some(three)) => {
                Some(stamp.n)
            }
            _ => None,
        },
        _ => None,
    });
    let made = made.expect("the record that made it");
    let before = as_of(&store, "main", made).expect("as of");
    assert_eq!(
        before.messages(),
        &[Message::user("one"), Message::assistant("two")]
    );

    let forked = MemoryStore::default();
    fork(&store, &forked, Some(("main", three))).expect("forked");
    let read = Session::read(&forked).expect("read");
    let ids: Vec<&str> = read.agents.iter().map(|agent| agent.id.as_str()).collect();
    assert_eq!(ids, ["main"], "the subagent came later");
    assert_eq!(
        read.agents.first().map(|agent| &agent.conversation),
        Some(&before)
    );
    assert_eq!(forked.files().expect("files"), ["plugins/notes.json"]);

    let whole = MemoryStore::default();
    fork(&store, &whole, None).expect("forked");
    assert_eq!(Session::read(&whole).expect("read").agents.len(), 2);
}
