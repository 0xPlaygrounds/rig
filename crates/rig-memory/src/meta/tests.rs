use std::collections::BTreeMap;

use crate::{Files, ForkedFrom, MemoryStore, SessionMeta};

/// A listing written by an older rig, with a field this one does not know
/// (a name), reads with it skipped; a fork's parent reads and writes in
/// the shape forks were always written in.
#[test]
fn a_listing_and_a_parent_read_back_and_unknown_fields_are_skipped() {
    let store = MemoryStore::default();
    assert_eq!(SessionMeta::read(&store), None);
    let old = br#"{"name":"release","title":"fix the parser","labels":{"spent":"$1"},"updated":7}"#;
    assert!(store.write_file("meta.json", old).is_ok());
    let meta = SessionMeta {
        title: "fix the parser".to_owned(),
        labels: BTreeMap::from([("spent".to_owned(), "$1".to_owned())]),
        updated: 7,
    };
    assert_eq!(SessionMeta::read(&store).as_ref(), Some(&meta));
    assert!(SessionMeta::default().write(&store).is_ok());
    assert_eq!(SessionMeta::read(&store), Some(SessionMeta::default()));

    for written in [
        r#"{"session":"1-1","before":["main",3],"text":"try again"}"#,
        r#"{"session":"1-1","before":null}"#,
    ] {
        assert!(store.write_file("parent.json", written.as_bytes()).is_ok());
        let parent = ForkedFrom::read(&store);
        assert_eq!(
            parent.as_ref().map(|parent| parent.session.as_str()),
            Some("1-1")
        );
        assert!(parent.is_some_and(|parent| parent.write(&store).is_ok()));
        let read = store.read_file("parent.json").ok();
        assert_eq!(read.as_deref(), Some(written.as_bytes()));
    }
}
