use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::{Aside, Author, Change, Conversation, Entry, Halt};
use rig_core::message::{ImageMediaType, UserContent};
use serde_json::json;

use super::{LogRecord, SavedAgent, Session, as_of, history, load, save};
use crate::{Files, MemoryStore, Repair, SessionError, Store, Writer};

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
    first.message = Message::user("look again").into();
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
        crate::blobs::blob_names(&store).expect("blobs").len(),
        1,
        "the image is a blob"
    );
    let log = String::from_utf8(store.read_log("agent").expect("a log")).expect("text");
    assert!(log.starts_with(r#"{"seq":0,"n":0,"#), "{log}");
    assert!(log.contains(r#""format":"rig-session/4""#), "{log}");
    assert!(!log.contains("cG5n"), "the image data is not in the log");

    // A running writer's log folds back the same, with saved values.
    let store = Arc::new(MemoryStore::default());
    let (writer, session) = Writer::open(store.clone()).expect("an empty session");
    assert_eq!(session, Session::default());
    writer.start();
    let mut live = edited();
    let saved = writer.save("main", &mut live);
    assert!(saved.iter().all(|change| change.at.is_some()));
    writer.save_value("main", "mode", Some(json!("plan")), false);
    writer.save_value(crate::SESSION, "theme", Some(json!("dark")), false);
    writer.flush().expect("flushed");
    let session = Session::read(&*store).expect("read");
    let main = session.agent("main").expect("the agent");
    assert_eq!(main.conversation, live);
    assert_eq!(main.saved, [("mode".to_owned(), json!("plan"))].into());
    assert_eq!(session.saved, [("theme".to_owned(), json!("dark"))].into());
    assert!(main.rewrites.is_empty());
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
    assert!(matches!(refused, Err(SessionError::Changed { log }) if log == "agent"));
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
    writer.flush().expect("flushed");
    let session = Session::read(&*store).expect("read");
    assert_eq!(session.agent("main").expect("agent").conversation, replaced);
}

/// Only the conversation a log holds continues it, even when another has
/// made as many changes: one saved to another log, one read back from its
/// serde form. Saving
/// another refuses, writing nothing, unless the log is empty.
#[test]
fn only_the_conversation_a_log_holds_continues_it() {
    let store = MemoryStore::default();
    let mut a = load(&store, "a").expect("an empty log");
    a.push(Message::user("for a"));
    save(&store, "a", &mut a).expect("saved");
    let mut b = load(&store, "b").expect("an empty log");
    b.push(Message::user("for b"));
    save(&store, "b", &mut b).expect("saved");
    let before = store.read_log("a").expect("a log");
    b.push(Message::assistant("moved"));
    let refused = save(&store, "a", &mut b);
    assert!(
        matches!(&refused, Err(SessionError::Changed { log }) if log == "a"),
        "{refused:?}"
    );
    assert_eq!(store.read_log("a").expect("a log"), before);

    // Read back from its serde form, it is saved whole onto an empty log.
    let json = serde_json::to_string(&a).expect("serialized");
    let mut copy: Conversation = serde_json::from_str(&json).expect("deserialized");
    copy.push(Message::assistant("copied"));
    save(&store, "c", &mut copy).expect("an empty log takes any conversation");
    assert_eq!(load(&store, "c").expect("loaded"), copy);
    assert!(matches!(
        save(&store, "a", &mut copy),
        Err(SessionError::Changed { .. })
    ));
}

/// A running writer writes a conversation that does not continue a log
/// as one reset, whatever its number of changes, so the log folds back to
/// it: one moved from another agent's log, one read back from its serde
/// form.
#[test]
fn a_writer_resets_a_log_to_a_conversation_it_does_not_hold() {
    let store = Arc::new(MemoryStore::default());
    let (writer, _) = Writer::open(store.clone()).expect("opened");
    writer.start();
    let mut a = Conversation::from_messages([Message::user("for a")]);
    writer.save("a", &mut a);
    let mut b = Conversation::from_messages([Message::user("for b")]);
    writer.save("b", &mut b);
    let saved = writer.save("a", &mut b);
    assert!(matches!(saved.as_slice(), [only] if matches!(only.change, Change::Reset { .. })));
    b.push(Message::assistant("moved"));
    writer.save("a", &mut b);

    let json = serde_json::to_string(&a).expect("serialized");
    let mut copy: Conversation = serde_json::from_str(&json).expect("deserialized");
    copy.push(Message::assistant("copied"));
    writer.save("c", &mut copy);
    writer.flush().expect("flushed");
    let session = Session::read(&*store).expect("read");
    assert_eq!(
        session.agent("a").map(|agent| &agent.conversation),
        Some(&b)
    );
    assert_eq!(
        session.agent("c").map(|agent| &agent.conversation),
        Some(&copy)
    );
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
    writer.flush().expect("flushed");
    assert_eq!(load(&*store, "main").expect("loaded"), conversation);
}

/// A log whose first write was cut short holds no record yet: reading
/// reports it torn whole, and opening it to write sets it aside, so the
/// header written next starts a line and the session still reads.
#[test]
fn a_log_holding_only_a_torn_first_line_is_set_aside_before_it_is_written() {
    let store = Arc::new(MemoryStore::default());
    store
        .append_log("main", br#"{"seq":0,"n":0,"t":1,"ty"#)
        .expect("torn");
    let session = Session::read(&*store).expect("read");
    assert!(session.agent("main").is_none());
    let torn = Repair::TornTail {
        log: "main".to_owned(),
        kept: 0,
        cut: 24,
        set_aside: None,
    };
    assert_eq!(session.repairs, [torn]);

    let (writer, session) = Writer::open(store.clone()).expect("opened");
    assert!(matches!(
        session.repairs.as_slice(),
        [Repair::TornTail {
            kept: 0,
            set_aside: Some(_),
            ..
        }]
    ));
    assert_eq!(
        store.read_file("recovered/main-1.part").expect("set aside"),
        br#"{"seq":0,"n":0,"t":1,"ty"#
    );
    writer.start();
    let mut conversation = Conversation::from_messages([Message::user("hello")]);
    writer.save("main", &mut conversation);
    writer.flush().expect("flushed");
    assert_eq!(load(&*store, "main").expect("loaded"), conversation);
}

/// The library saves set a torn last line aside before they append, as
/// [`Writer::open`] does, so the next record starts a line of its own.
#[test]
fn the_library_saves_set_a_torn_tail_aside_before_they_append() {
    let tear = br#"{"seq":2,"n":"#;

    // `save` onto a log whose last record was cut short.
    let store = MemoryStore::default();
    let mut conversation = Conversation::from_messages([Message::user("hello")]);
    save(&store, "main", &mut conversation).expect("saved");
    store.append_log("main", tear).expect("torn");
    conversation.push(Message::assistant("hi"));
    save(&store, "main", &mut conversation).expect("saved after the tear");
    assert_eq!(load(&store, "main").expect("loaded"), conversation);
    assert_eq!(
        store.read_file("recovered/main-1.part").expect("set aside"),
        tear
    );

    // `save` onto a log whose first write was cut short.
    let store = MemoryStore::default();
    store.append_log("main", tear).expect("torn");
    save(&store, "main", &mut conversation).expect("saved after the tear");
    assert_eq!(load(&store, "main").expect("loaded"), conversation);
    assert_eq!(
        store.read_file("recovered/main-1.part").expect("set aside"),
        tear
    );

    // `Session::save` onto a log whose last record was cut short.
    store.append_log("main", tear).expect("torn");
    let mut session = Session::read(&store).expect("read");
    if let Some(agent) = session.agent_mut("main") {
        agent.conversation.push(Message::user("again"));
    }
    session.save(&store).expect("saved after the tear");
    let read = Session::read(&store).expect("read again");
    assert_eq!(read.repairs, []);
    assert_eq!(read.agent("main"), session.agent("main"));
    assert_eq!(
        store.read_file("recovered/main-2.part").expect("set aside"),
        tear
    );
}

#[test]
fn a_session_it_cannot_read_whole_is_refused_unless_read_leniently() {
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
    let mut other = Conversation::from_messages([Message::user("c")]);
    save(&store, "other", &mut other).expect("saved");
    let refused = Session::read(&store);
    assert!(
        matches!(&refused, Err(SessionError::Record { log, line: 3, .. }) if log == "main"),
        "{refused:?}"
    );
    // Read leniently, the log that reads is kept and the other named.
    let (session, unreadable) = Session::read_lenient(&store).expect("listed");
    let read: Vec<&str> = session
        .agents
        .iter()
        .map(|agent| agent.id.as_str())
        .collect();
    assert_eq!(read, ["other"]);
    let names: Vec<&str> = unreadable.iter().map(|log| log.log.as_str()).collect();
    assert_eq!(names, ["main"]);
    // A log from before this format names none.
    let old = MemoryStore::default();
    old.append_log(
        "old",
        b"{\"seq\":0,\"n\":0,\"t\":0,\"type\":\"header\",\"agent\":\"old\"}\n",
    )
    .expect("written");
    let refused = load(&old, "old");
    assert!(
        matches!(refused, Err(SessionError::Format { found: None, .. })),
        "{refused:?}"
    );
}

#[test]
fn history_and_as_of_cut_by_the_record_that_made_a_message() {
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
        LogRecord::Change(recorded) => match &recorded.change {
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
        [Message::user("one"), Message::assistant("two")].map(std::sync::Arc::new)
    );
}

/// A store reached asynchronously, over a [`MemoryStore`]: every call waits
/// one poll, as one behind a network does, and appends wait for good while
/// `held`.
#[derive(Default)]
struct Remote {
    store: MemoryStore,
    held: std::sync::atomic::AtomicBool,
}

impl Remote {
    async fn call<T>(&self, call: impl FnOnce(&MemoryStore) -> T) -> T {
        let mut waited = false;
        std::future::poll_fn(|cx| {
            if std::mem::replace(&mut waited, true) {
                return std::task::Poll::Ready(());
            }
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        })
        .await;
        call(&self.store)
    }
}

impl crate::AsyncStore for Remote {
    async fn logs(&self) -> std::io::Result<Vec<String>> {
        self.call(Store::logs).await
    }

    async fn read_log(&self, name: &str) -> std::io::Result<Vec<u8>> {
        self.call(|store| Store::read_log(store, name)).await
    }

    async fn append_log(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        if self.held.load(std::sync::atomic::Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        self.call(|store| Store::append_log(store, name, bytes))
            .await
    }

    async fn replace_log(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        self.call(|store| Store::replace_log(store, name, bytes))
            .await
    }

    async fn files(&self) -> std::io::Result<Vec<String>> {
        self.call(Files::files).await
    }

    async fn read_file(&self, name: &str) -> std::io::Result<Vec<u8>> {
        self.call(|store| Files::read_file(store, name)).await
    }

    async fn write_file(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        self.call(|store| Files::write_file(store, name, bytes))
            .await
    }
}

/// Polls `work` up to `polls` times; its output if it finished.
fn poll<T>(work: impl Future<Output = T>, polls: usize) -> Option<T> {
    let mut work = std::pin::pin!(work);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    (0..polls).find_map(|_| match work.as_mut().poll(&mut cx) {
        std::task::Poll::Ready(output) => Some(output),
        std::task::Poll::Pending => None,
    })
}

#[test]
fn an_async_store_keeps_a_conversation_as_a_local_one_does() {
    let remote = Remote::default();
    let mut conversation = edited();
    poll(super::save_async(&remote, "agent", &mut conversation), 100)
        .expect("finished")
        .expect("saved");
    let loaded = poll(super::load_async(&remote, "agent"), 100)
        .expect("finished")
        .expect("loaded");
    assert_eq!(loaded, conversation);
    assert_eq!(load(&remote.store, "agent").expect("read locally"), loaded);
    assert_eq!(
        crate::blobs::blob_names(&remote.store)
            .expect("blobs")
            .len(),
        1
    );
}

#[test]
fn a_save_cut_short_appends_nothing() {
    let remote = Remote::default();
    let mut conversation = Conversation::new();
    conversation.push(Message::user("first"));
    poll(super::save_async(&remote, "agent", &mut conversation), 100)
        .expect("finished")
        .expect("saved");
    let before = remote.store.read_log("agent").expect("a log");

    remote.held.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut cut = conversation.clone();
    cut.extend([Message::assistant("a turn"), Message::user("cut short")]);
    assert!(poll(super::save_async(&remote, "agent", &mut cut), 100).is_none());
    assert_eq!(remote.store.read_log("agent").expect("a log"), before);

    remote
        .held
        .store(false, std::sync::atomic::Ordering::SeqCst);
    conversation.push(Message::assistant("the next turn"));
    poll(super::save_async(&remote, "agent", &mut conversation), 100)
        .expect("finished")
        .expect("saved after");
    let loaded = poll(super::load_async(&remote, "agent"), 100).expect("finished");
    assert_eq!(loaded.expect("loaded"), conversation);
}
