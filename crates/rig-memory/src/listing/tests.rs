use rig_core::completion::Message;
use rig_core::conversation::{Author, Conversation};

use crate::{ForkedFrom, Listing, MemoryStore, SavedAgent, Session, SessionMeta, Store, read_line};

/// Every log of `store` and its bytes.
fn logs(store: &MemoryStore) -> Vec<(String, Vec<u8>)> {
    let mut names = store.logs().unwrap_or_default();
    names.sort();
    let read = names.into_iter().map(|name| {
        let bytes = store.read_log(&name).unwrap_or_default();
        (name, bytes)
    });
    read.collect()
}

/// A session saved in this format lists from its files alone, and any
/// agent's line of it reads (the main agent's found from the headers, a
/// kept branch by its aside) with ids kept, writing nothing.
#[test]
fn a_saved_session_lists_and_any_line_of_it_reads_without_writing() {
    let mut main =
        Conversation::from_messages([Message::user("Name a colour."), Message::assistant("Red.")]);
    let red = main.id(1);
    let by = Author::plugin("test");
    let kept = main.branch_off(1, "blue", by.clone()).unwrap();
    main.push(Message::assistant("Blue."));
    let sub = Conversation::from_messages([Message::user("Look it up.")]);
    let agent = |id: &str, parent: Option<&str>, conversation: Conversation| SavedAgent {
        id: id.to_owned(),
        parent: parent.map(str::to_owned),
        conversation,
        ..SavedAgent::default()
    };
    let mut session = Session {
        agents: vec![
            agent("0-sub", Some("b-main"), sub),
            agent("b-main", None, main),
        ],
        ..Session::default()
    };
    let store = MemoryStore::default();
    session.save(&store).unwrap();
    let meta = SessionMeta {
        title: "Name a colour.".to_owned(),
        messages: 2,
        ..SessionMeta::default()
    };
    meta.write(&store).unwrap();
    let parent = ForkedFrom {
        session: "1700000000-1".to_owned(),
        ..ForkedFrom::default()
    };
    parent.write(&store).unwrap();
    let before = logs(&store);

    let listing = Listing::read(&store);
    assert_eq!(listing.meta, meta);
    assert_eq!(listing.parent, Some(parent));

    let (id, active) = read_line(&store, None, None).unwrap();
    assert_eq!(id, "b-main");
    assert_eq!(active.branch_title(), "blue");
    assert_eq!(active.messages().last(), Some(&Message::assistant("Blue.")));
    let (_, red_line) = read_line(&store, None, Some(kept)).unwrap();
    assert_eq!(
        red_line.messages().last(),
        Some(&Message::assistant("Red."))
    );
    assert_eq!(red_line.id(1), red, "ids are kept");
    let (_, sub) = read_line(&store, Some("0-sub"), None).unwrap();
    assert_eq!(sub.messages(), [Message::user("Look it up.")]);
    assert!(read_line(&store, Some("nobody"), None).is_err());
    assert!(read_line(&MemoryStore::default(), None, None).is_err());
    assert_eq!(logs(&store), before, "reading wrote nothing");
}
