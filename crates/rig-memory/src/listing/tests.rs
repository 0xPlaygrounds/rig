use rig_core::completion::Message;
use rig_core::conversation::Conversation;

use crate::{MemoryStore, SavedAgent, Session, Store, read_conversation};

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

/// Any agent's conversation of a saved session reads (the main agent's
/// found from the headers) with ids kept, writing nothing.
#[test]
fn any_agent_of_a_saved_session_reads_without_writing() {
    let main =
        Conversation::from_messages([Message::user("Name a colour."), Message::assistant("Red.")]);
    let red = main.id(1);
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
    let before = logs(&store);

    let (id, main) = read_conversation(&store, None).unwrap();
    assert_eq!(id, "b-main");
    assert_eq!(
        main.messages().last(),
        Some(&Message::assistant("Red.").into())
    );
    assert_eq!(main.id(1), red, "ids are kept");
    let (_, sub) = read_conversation(&store, Some("0-sub")).unwrap();
    assert_eq!(
        sub.messages(),
        [Message::user("Look it up.")].map(std::sync::Arc::new)
    );
    assert!(read_conversation(&store, Some("nobody")).is_err());
    assert!(read_conversation(&MemoryStore::default(), None).is_err());
    assert_eq!(logs(&store), before, "reading wrote nothing");
}
