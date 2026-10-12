use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::message::UserContent;
use serde_json::json;

use super::{
    SESSION, SavedAgent, SavedSession, Stamp, Writer, fork_session, now_ms, read_session,
    write_session,
};
use crate::journal::{
    Entry, Halt, JournalStore, LogLines, MemoryStore, MessageId, Origin, RequestId, STOPPED,
};

/// A session rig-ecs wrote before the log format moved here: a summary
/// spliced over the first messages, a report from a subagent, a stopped
/// request and a kept one, saved values, a subagent and a retired one.
fn written_before() -> MemoryStore {
    let store = MemoryStore::default();
    let mut log = None;
    for line in include_str!("saved-session.txt").lines() {
        if let Some(name) = line.strip_prefix("=== ") {
            log = Some(name);
        } else if let Some(blob) = log.and_then(|name| name.strip_prefix("blob ")) {
            store.put_blob(blob, line.as_bytes()).unwrap();
        } else if let Some(agent) = log {
            store.append(agent, format!("{line}\n").as_bytes()).unwrap();
        }
    }
    store
}

const MAIN: &str = "cd9a3280-91f0-4fb4-aa5c-c1c3629a0f32";

fn text(content: &[&str]) -> Message {
    Message::User {
        content: content
            .iter()
            .map(|text| UserContent::text(*text))
            .collect(),
    }
}

#[test]
fn a_session_written_before_the_move_reads_and_restores_the_same() {
    let store = written_before();
    let session = read_session(&store).unwrap();
    assert!(session.unreadable.is_empty() && session.lost_images.is_empty());
    assert_eq!(
        session.resources,
        [("rig_ecs::journal::tests::Mode".to_owned(), json!("plan"))].into()
    );
    let ids: Vec<_> = session
        .agents
        .iter()
        .map(|agent| agent.id.as_str())
        .collect();
    assert_eq!(ids, [MAIN, "child-agent", "gone-agent"]);
    let [main, child, gone] = session.agents.as_slice() else {
        panic!("three agents");
    };
    let report = Origin {
        request: Some(RequestId("call-1".to_owned())),
        ..Origin::agent("child-agent", None)
    }
    .titled("Fix the parser");
    let conversation = &main.conversation;
    let expected = [
        text(&["Summary.", "report"]),
        Message::assistant("thanks"),
        text(&["question 3", STOPPED, "go on"]),
        Message::assistant("done"),
        text(&["last"]),
    ];
    assert_eq!(conversation.messages(), expected);
    let ids: Vec<_> = (0..5).filter_map(|at| conversation.id(at)).collect();
    assert_eq!(ids, [2, 3, 4, 6, 7].map(MessageId));
    assert_eq!(
        conversation.origin(0, 0),
        Some(&Origin::plugin("compaction"))
    );
    assert_eq!(conversation.origin(0, 1), Some(&report));
    assert_eq!(conversation.halted(), Some(Halt::Kept));
    let mut going_on = conversation.clone();
    going_on.append(Message::assistant("next"), None);
    assert_eq!(going_on.id(5), Some(MessageId(8)), "new ids go on");
    assert_eq!(
        main.saved.get("rig_ecs::journal::tests::Counter"),
        Some(&json!(3))
    );
    assert_eq!(main.started_ms, 1_791_779_409_181);
    assert_eq!(child.parent.as_deref(), Some(MAIN));
    assert_eq!(child.conversation.messages(), [text(&["work"])]);
    assert!(gone.retired && !child.retired && !main.retired);

    // A restore leaves the retired agent out, and goes on numbering where
    // the logs stopped.
    let writer = Writer::default();
    let restored = writer.restore(Arc::new(store.clone())).unwrap();
    let ids: Vec<_> = restored
        .agents
        .iter()
        .map(|agent| agent.id.as_str())
        .collect();
    assert_eq!(ids, [MAIN, "child-agent"]);
    writer.message(MAIN, &Message::assistant("not logged before start"), None);
    writer.start();
    writer.message(MAIN, &Message::assistant("again"), None);
    writer.flush();
    let log = LogLines::read(&store, MAIN).unwrap();
    let last: serde_json::Value = serde_json::from_slice(log.iter().last().unwrap().1).unwrap();
    assert_eq!(
        (last["seq"].clone(), last["n"].clone()),
        (json!(14), json!(23))
    );
    assert_eq!(log.iter().count(), 15);
}

/// The JSON line of a record, `seq` in its log and `n` in the session, all
/// written in the same millisecond.
fn line(seq: u64, n: u64, record: &serde_json::Value) -> String {
    let mut line = json!({ "seq": seq, "n": n, "t": 1_000 });
    line.as_object_mut()
        .unwrap()
        .extend(record.as_object().unwrap().clone());
    format!("{line}\n")
}

fn said(message: Message) -> serde_json::Value {
    json!({ "type": "message", "message": message })
}

#[test]
fn a_fork_keeps_what_other_agents_logged_before_its_cut_in_the_same_millisecond() {
    let header = |agent: &str| json!({ "type": "header", "agent": agent });
    let from = MemoryStore::default();
    let a = [
        line(0, 0, &header("a")),
        line(1, 1, &said(Message::user("a first"))),
        line(2, 4, &said(Message::assistant("a second"))),
    ];
    let b = [
        line(0, 2, &header("b")),
        line(1, 3, &said(Message::user("b first"))),
        line(2, 5, &said(Message::assistant("b second"))),
    ];
    from.append("a", a.concat().as_bytes()).unwrap();
    from.append("b", b.concat().as_bytes()).unwrap();
    let to = MemoryStore::default();
    fork_session(&from, &to, Some(("a", MessageId(1)))).unwrap();
    let session = read_session(&to).unwrap();
    let kept = |agent: &str| -> Vec<Message> {
        let agent = session.agents.iter().find(|saved| saved.id == agent);
        agent.map_or_else(Vec::new, |agent| agent.conversation.messages().to_vec())
    };
    assert_eq!(kept("a"), [Message::user("a first")]);
    assert_eq!(kept("b"), [Message::user("b first")]);
    let nowhere = fork_session(&from, &MemoryStore::default(), Some(("a", MessageId(9))));
    assert!(nowhere.is_err());
}

/// `session` as read back, without when its logs and messages were made.
fn read_back(store: &MemoryStore) -> SavedSession {
    let mut session = read_session(store).unwrap();
    for agent in &mut session.agents {
        agent.started_ms = 0;
        agent.stamps.clear();
    }
    session
}

#[test]
fn a_session_is_written_from_data_then_appended_to() {
    let mut main = SavedAgent {
        id: "main".to_owned(),
        saved: [("model".to_owned(), json!("gpt"))].into(),
        ..SavedAgent::default()
    };
    let conversation = &mut main.conversation;
    conversation.append(Message::user("Fix the parser"), None);
    conversation.append(Message::assistant("Fixed."), None);
    conversation.append(Message::user("Thanks"), Some(Origin::plugin("import")));
    conversation.halt(Halt::Kept);
    let mut helper = SavedAgent {
        id: "sub".to_owned(),
        parent: Some("main".to_owned()),
        ..SavedAgent::default()
    };
    helper.conversation.append(Message::user("Look"), None);
    let mut session = SavedSession {
        agents: vec![main, helper],
        resources: [("mode".to_owned(), json!("plan"))].into(),
        ..SavedSession::default()
    };
    let store = MemoryStore::default();
    write_session(&store, &session).unwrap();
    assert_eq!(read_back(&store), session);
    assert!(store.read(SESSION).is_ok());

    // Written again unchanged, nothing is added; with more messages and a
    // model changed, only those are.
    let lines = |agent: &str| LogLines::read(&store, agent).unwrap().iter().count();
    let before = lines("main");
    write_session(&store, &session).unwrap();
    assert_eq!(lines("main"), before);
    let main = session.agents.first_mut().unwrap();
    main.conversation
        .append(Message::assistant("You're welcome."), None);
    main.conversation.append(Message::user("One more"), None);
    main.saved.insert("model".to_owned(), json!("claude"));
    let first = main.conversation.id(0);
    write_session(&store, &session).unwrap();
    assert_eq!(
        lines("main"),
        before + 3,
        "a record per message and a saved value"
    );
    let read = read_back(&store);
    assert_eq!(read, session);
    let main = read.agents.first().unwrap();
    assert_eq!(main.conversation.id(0), first, "ids stay");

    // A replaced message replaces what follows it.
    let edited = Entry::from(Message::user("Fix the lexer"), Origin::plugin("import"));
    let main = session.agents.first_mut().unwrap();
    main.conversation.splice(0, 4, vec![edited]);
    main.retired = true;
    let edited = main.conversation.clone();
    write_session(&store, &session).unwrap();
    let read = read_back(&store);
    let main = read.agents.first().unwrap();
    assert!(main.retired);
    assert_eq!(main.conversation.messages().len(), 1);
    assert_eq!(main.conversation.entry(0), edited.entry(0));
}

/// Messages written with stamps are logged at their times, and one added
/// without is logged now; each is read back with the record that made it.
#[test]
fn messages_are_written_at_their_stamps_and_read_back_with_them() {
    let mut agent = SavedAgent {
        id: "imported".to_owned(),
        stamps: vec![Stamp { n: 0, t: 1_000 }, Stamp { n: 0, t: 2_000 }],
        ..SavedAgent::default()
    };
    agent
        .conversation
        .append(Message::user("Fix the parser"), None);
    agent
        .conversation
        .append(Message::assistant("Fixed."), None);
    let session = SavedSession {
        agents: vec![agent],
        ..SavedSession::default()
    };
    let store = MemoryStore::default();
    write_session(&store, &session).unwrap();
    let mut read = read_session(&store).unwrap();
    let stamps = |session: &SavedSession| session.agents.first().unwrap().stamps.clone();
    let times: Vec<u64> = stamps(&read).iter().map(|stamp| stamp.t).collect();
    assert_eq!(times, [1_000, 2_000]);

    let agent = read.agents.first_mut().unwrap();
    agent.conversation.append(Message::user("Thanks"), None);
    let now = now_ms();
    write_session(&store, &read).unwrap();
    let stamps = stamps(&read_session(&store).unwrap());
    assert_eq!(
        stamps.get(..2),
        read.agents.first().unwrap().stamps.get(..2)
    );
    let [first, second, third] = *stamps.as_slice() else {
        panic!("three stamps, got {stamps:?}");
    };
    assert!(first.n < second.n && second.n < third.n);
    assert!(third.t >= now);
}
