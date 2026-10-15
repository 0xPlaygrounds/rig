use std::collections::BTreeSet;
use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::{Aside, Author, Change, Conversation};
use serde_json::{Value, json};

use crate::format::stamp_of;
use crate::{
    Files, LogRecord, MemoryStore, NoRewrite, Session, SessionError, Store, Strings, Writer, as_of,
    fork, history, load, save,
};

fn rating(value: Value) -> Aside {
    Aside::new("rating", Author::plugin("ratings"), value)
}

fn values(conversation: &Conversation) -> Vec<(String, Value)> {
    let asides = conversation.asides();
    asides
        .map(|(_, aside)| (aside.kind.clone(), aside.value.clone()))
        .collect()
}

/// A fork before a message copies the records made before it, each
/// through the transform, then carries what the asides of the copied
/// messages say now (a rating changed, removed or added after the cut),
/// but not rig's own marks, which stay as they were at the cut, nor an
/// aside on a message made after it.
#[test]
fn a_fork_carries_the_asides_of_the_copied_messages_as_they_are_now() {
    let store = MemoryStore::default();
    store
        .write_file("plugins/notes.json", br#"{"note":"sk-1"}"#)
        .expect("written");
    let mut main = load(&store, "main").expect("an empty log");
    let one = main.push(Message::user("one"));
    let two = main.push(Message::assistant("two sk-1"));
    let changed = main.add_aside(rating(json!("good")).after(Some(two)));
    let gone = main.add_aside(rating(json!("meh")).after(Some(one)));
    let call = Aside::new("rig.call", Author::rig(), json!({ "open": true }));
    let call = main.add_aside(call.after(Some(two)));
    let three = main.push(Message::user("three"));
    main.replace_aside(changed, rating(json!("bad")).after(Some(two)));
    main.remove_aside(gone);
    main.add_aside(rating(json!("great")).after(Some(one)));
    main.add_aside(rating(json!("later")).after(Some(three)));
    let closed = Aside::new("rig.call", Author::rig(), json!({ "open": false }));
    main.replace_aside(call, closed.after(Some(two)));
    main.add_aside(Aside::new("note", Author::plugin("btw"), json!("on top")));
    save(&store, "main", &mut main).expect("saved");

    let to = MemoryStore::default();
    let redact = Strings(|text: &mut String| {
        let hit = text.contains("sk-1");
        if hit {
            *text = text.replace("sk-1", "[R]");
        }
        hit
    });
    let forked = fork(&store, &to, Some(("main", three)), &redact).expect("forked");
    assert_eq!(forked.carried, 4);
    let copy = load(&to, "main").expect("loaded");
    assert_eq!(
        copy.messages(),
        [Message::user("one"), Message::assistant("two [R]")].map(std::sync::Arc::new)
    );
    assert_eq!(
        values(&copy),
        [
            ("rating".to_owned(), json!("bad")),
            ("rig.call".to_owned(), json!({ "open": true })),
            ("rating".to_owned(), json!("great")),
            ("note".to_owned(), json!("on top")),
        ]
    );
    assert_eq!(
        to.read_file("plugins/notes.json").expect("copied"),
        br#"{"note":"[R]"}"#
    );

    // As of the cut, the copy is what the session was then.
    let records = history(&store, "main").expect("history");
    let cut = records.iter().find_map(|(stamp, record)| match record {
        LogRecord::Change(recorded) => match &recorded.change {
            Change::Push { entries } if entries.iter().any(|entry| entry.id == Some(three)) => {
                Some(stamp.n)
            }
            _ => None,
        },
        _ => None,
    });
    let cut = cut.expect("the record that made it");
    let then = as_of(&to, "main", cut).expect("as of");
    assert_eq!(
        values(&then),
        [
            ("rating".to_owned(), json!("good")),
            ("rating".to_owned(), json!("meh")),
            ("rig.call".to_owned(), json!({ "open": true })),
        ]
    );
    // Each carried record keeps the time and author of the one it carries.
    let carried = history(&to, "main").expect("history");
    let carried: Vec<_> = carried
        .iter()
        .filter(|(stamp, _)| stamp.n >= cut)
        .map(|(stamp, record)| match record {
            LogRecord::Change(recorded) => (stamp.t, recorded.by.clone()),
            _ => (0, None),
        })
        .collect();
    let parent: Vec<_> = records
        .iter()
        .filter_map(|(stamp, record)| match record {
            LogRecord::Change(recorded)
                if matches!(recorded.change, Change::Aside { .. })
                    && recorded.by != Some(Author::rig())
                    && stamp.n > cut =>
            {
                Some((stamp.t, recorded.by.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(carried.len(), 4);
    assert!(carried.iter().all(|record| parent.contains(record)));

    // The copy goes on as a session of its own, every record numbered once.
    let to = Arc::new(to);
    let (writer, session) = Writer::open(to.clone()).expect("opened");
    writer.start();
    let mut copy = session
        .agents
        .into_iter()
        .next()
        .expect("main")
        .conversation;
    copy.push(Message::user("four"));
    writer.save("main", &mut copy);
    writer.flush();
    assert_eq!(load(&*to, "main").expect("loaded"), copy);
    let lines = to.read_log("main").expect("a log");
    let numbers: Vec<u64> = lines
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| stamp_of(line).expect("a stamp").0.n)
        .collect();
    let unique: BTreeSet<u64> = numbers.iter().copied().collect();
    assert_eq!(unique.len(), numbers.len(), "{numbers:?}");
    assert_ne!(copy.lineage(), main.lineage());
}

/// A session saved in another format is refused, and nothing is copied.
#[test]
fn a_fork_of_a_session_in_format_1_is_refused_and_writes_nothing() {
    let store = MemoryStore::default();
    let mut main = Conversation::from_messages([Message::user("one")]);
    save(&store, "main", &mut main).expect("saved");
    store
        .write_file("plugins/notes.json", b"{}")
        .expect("written");
    store
        .append_log(
            "old",
            b"{\"seq\":0,\"n\":5,\"t\":0,\"type\":\"header\",\"format\":\"rig-session/1\",\"log\":\"old\",\"lineage\":7}\n",
        )
        .expect("written");
    let to = MemoryStore::default();
    let refused = fork(&store, &to, None, &NoRewrite);
    assert!(
        matches!(&refused, Err(SessionError::Format { found: Some(found), .. }) if found == "rig-session/1"),
        "{refused:?}"
    );
    let empty = |listed: std::io::Result<Vec<String>>| listed.expect("listed").is_empty();
    assert!(empty(to.logs()) && empty(to.blobs()) && empty(to.files()));
    assert!(Session::read(&to).expect("read").agents.is_empty());
}
