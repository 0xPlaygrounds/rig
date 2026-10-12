use std::borrow::Cow;

use rig_core::completion::Message;
use rig_core::message::{
    CallId, DocumentSourceKind, ImageMediaType, ToolName, ToolResultContent, UserContent,
};

use super::{JournalStore, LogLines, MemoryStore, fork, load_images, store_images};

#[test]
fn images_are_logged_as_blobs_and_read_back() {
    let store = MemoryStore::default();
    let text = Message::user("no images");
    assert!(matches!(store_images(&text, &store), Ok(Cow::Borrowed(_))));
    // The same bytes as a user image and in a tool result.
    let png = DocumentSourceKind::Base64("cG5n".to_owned());
    let image = UserContent::image_raw(b"png".to_vec(), Some(ImageMediaType::PNG), None);
    let UserContent::Image(shot) = image.clone() else {
        panic!("an image");
    };
    let (call, name) = (
        CallId::from_wire("c"),
        ToolName::new("shot").expect("a name"),
    );
    let result = UserContent::tool_result(call, name, vec![ToolResultContent::Image(shot)]);
    let message = Message::User {
        content: vec![image, result],
    };

    let logged = store_images(&message, &store).expect("stored").into_owned();
    let names: Vec<_> = logged.images().map(|image| image.data.clone()).collect();
    let Some(DocumentSourceKind::Url(url)) = names.first() else {
        panic!("a blob name, got {names:?}");
    };
    let blob = url.strip_prefix("blob:").expect("a blob");
    assert!(names.len() == 2 && names.iter().all(|name| Some(name) == names.first()));
    assert!(blob.ends_with(".png") && store.blob(blob).expect("the blob") == b"png");

    let mut read = logged.clone();
    load_images(&mut read, &store).expect("loaded");
    assert!(read.images().all(|image| image.data == png));
    // Without its blob, an image has no data.
    let mut gone = logged;
    assert!(load_images(&mut gone, &MemoryStore::default()).is_err());
    assert!(
        gone.images()
            .all(|image| image.data == DocumentSourceKind::Unknown)
    );
}

/// The lines of a log are read without writing to it: a torn tail is left
/// out of the lines and of the length, and blank lines are skipped.
#[test]
fn a_log_is_read_without_its_torn_tail_and_never_written() {
    let store = MemoryStore::default();
    store
        .append("a", b"{\"n\":1}\n\n{\"n\":2}\n{\"n\":")
        .expect("appended");
    let log = LogLines::read(&store, "a").expect("read");
    let lines: Vec<(u64, &[u8])> = log.iter().collect();
    assert_eq!(lines, [(8, &b"{\"n\":1}"[..]), (17, &b"{\"n\":2}"[..])]);
    assert_eq!((log.len(), log.torn(), log.is_empty()), (17, true, false));
    assert_eq!(log.prefix(8), b"{\"n\":1}\n");
    assert_eq!(log.prefix(1000).len(), 17);
    assert_eq!(store.read("a").expect("read").len(), 22, "nothing was cut");

    // A complete last line that is not JSON is torn too.
    let log = LogLines::parse(b"{\"n\":1}\n{\"n\"\n".to_vec());
    assert_eq!((log.len(), log.torn()), (8, true));
    let empty = LogLines::parse(Vec::new());
    assert!(empty.is_empty() && !empty.torn() && empty.iter().next().is_none());
    assert!(LogLines::read(&store, "nobody").is_err());
}

/// A fork copies every blob and each log as far as asked, leaving out the
/// agents it is told to and every torn tail; a removed log is gone.
#[test]
fn a_fork_copies_blobs_and_each_log_up_to_a_point() {
    let from = MemoryStore::default();
    from.append("a", b"{\"n\":1}\n{\"n\":2}\n{")
        .expect("appended");
    from.append("b", b"{\"n\":1}\n").expect("appended");
    from.append("gone", b"{\"n\":1}\n").expect("appended");
    from.put_blob("x.png", b"png").expect("stored");
    let to = MemoryStore::default();
    fork(&from, &to, |agent, log| match agent {
        "a" => log.iter().next().map(|(end, _)| end),
        "gone" => None,
        _ => Some(log.len()),
    })
    .expect("forked");
    assert_eq!(to.read("a").expect("a"), b"{\"n\":1}\n");
    assert_eq!(to.read("b").expect("b"), b"{\"n\":1}\n");
    assert!(to.read("gone").is_err());
    assert_eq!(to.blobs().expect("listed"), ["x.png"]);
    assert_eq!(to.blob("x.png").expect("copied"), b"png");

    from.remove("a").expect("removed");
    let mut agents = from.agents().expect("listed");
    agents.sort();
    assert_eq!(agents, ["b", "gone"]);
}

/// A store that cannot list its blobs or remove a log says so.
#[test]
fn a_store_without_listing_or_removal_says_it_is_unsupported() {
    struct Bare;

    impl JournalStore for Bare {
        fn agents(&self) -> std::io::Result<Vec<String>> {
            Ok(Vec::new())
        }

        fn read(&self, _agent: &str) -> std::io::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn truncate(&self, _agent: &str, _len: u64) -> std::io::Result<()> {
            Ok(())
        }

        fn append(&self, _agent: &str, _bytes: &[u8]) -> std::io::Result<()> {
            Ok(())
        }

        fn put_blob(&self, _name: &str, _bytes: &[u8]) -> std::io::Result<()> {
            Ok(())
        }

        fn blob(&self, _name: &str) -> std::io::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn write_file(&self, _name: &str, _bytes: &[u8]) -> std::io::Result<()> {
            Ok(())
        }

        fn append_file(&self, _name: &str, _bytes: &[u8]) -> std::io::Result<()> {
            Ok(())
        }

        fn read_file(&self, _name: &str) -> std::io::Result<Vec<u8>> {
            Err(std::io::ErrorKind::NotFound.into())
        }

        fn remove_file(&self, _name: &str) -> std::io::Result<()> {
            Ok(())
        }
    }

    let unsupported = |result: std::io::Result<()>| {
        result.is_err_and(|error| error.kind() == std::io::ErrorKind::Unsupported)
    };
    assert!(unsupported(Bare.blobs().map(drop)));
    assert!(unsupported(Bare.files().map(drop)));
    assert!(unsupported(Bare.remove("a")));
    assert!(unsupported(fork(&Bare, &MemoryStore::default(), |_, _| {
        None
    })));
}

/// A directory store lists the blobs it holds, not one being written, and
/// removes a log for good.
#[cfg(feature = "jsonl")]
#[test]
fn a_directory_store_lists_blobs_and_removes_logs() {
    use super::JsonlDirStore;

    let dir = std::env::temp_dir().join(format!("rig-cassette-journal-{}", std::process::id()));
    let store = JsonlDirStore::new(&dir);
    assert_eq!(store.blobs().expect("none yet"), Vec::<String>::new());
    store.put_blob("x.png", b"png").expect("stored");
    std::fs::write(dir.join("blobs").join("y.tmp"), b"half").expect("a blob being written");
    store.append("a", b"{}\n").expect("appended");
    assert_eq!(store.blobs().expect("listed"), ["x.png"]);
    store.remove("a").expect("removed");
    store.remove("a").expect("removing nothing is fine");
    assert_eq!(store.agents().expect("listed"), Vec::<String>::new());
    std::fs::remove_dir_all(dir).ok();
}

/// A directory store keeps a named file at its name, so the effect log and
/// a plugin's files stay where they were, and refuses names that leave the
/// session or that it reads as a log or a blob.
#[cfg(feature = "jsonl")]
#[test]
fn a_directory_store_keeps_named_files_beside_its_logs() {
    use super::{EFFECT_LOG, JsonlDirStore};
    use std::io::ErrorKind::{InvalidInput, NotFound};

    let dir = std::env::temp_dir().join(format!("rig-cassette-files-{}", std::process::id()));
    let store = JsonlDirStore::new(&dir);
    store.write_file("meta.json", b"{}").expect("written");
    store
        .write_file("meta.json", b"{\"title\":1}")
        .expect("replaced");
    store.append_file(EFFECT_LOG, b"a\n").expect("appended");
    store.append_file(EFFECT_LOG, b"b\n").expect("appended");
    store
        .append_file("tui/notices.jsonl", b"x")
        .expect("appended");
    let on_disk = |name: &str| std::fs::read(dir.join(name)).expect("on disk");
    assert_eq!(on_disk("meta.json"), b"{\"title\":1}");
    assert_eq!(on_disk("tui/notices.jsonl"), b"x");
    assert_eq!(store.read_file(EFFECT_LOG).expect("read"), b"a\nb\n");
    assert_eq!(store.agents().expect("listed"), Vec::<String>::new());

    for refused in ["notes.jsonl", "blobs/x", "../x", "/etc/x", "a//b", "a\\b"] {
        let kind = store.write_file(refused, b"").map_err(|error| error.kind());
        assert_eq!(kind, Err(InvalidInput), "{refused}");
    }
    store.remove_file("meta.json").expect("removed");
    store
        .remove_file("meta.json")
        .expect("removing nothing is fine");
    let kind = store.read_file("meta.json").map_err(|error| error.kind());
    assert_eq!(kind, Err(NotFound));
    std::fs::remove_dir_all(dir).ok();
}

/// A directory store lists its named files, not its logs, its blobs,
/// a file being written or the plain files the directory keeps beside it,
/// which it refuses to write.
#[cfg(feature = "jsonl")]
#[test]
fn a_directory_store_lists_its_files_and_not_its_layout() {
    use super::JsonlDirStore;

    let dir = std::env::temp_dir().join(format!("rig-cassette-list-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let store = JsonlDirStore::new(&dir).with_plain(&["saved", "spill"]);
    assert_eq!(store.files().expect("none yet"), Vec::<String>::new());
    store.write_file("meta.json", b"{}").expect("written");
    store
        .append_file("tui/notices.jsonl", b"x")
        .expect("appended");
    store.append("a1", b"{}\n").expect("appended");
    store.put_blob("x.png", b"png").expect("stored");
    std::fs::write(dir.join("saved"), b"").expect("a plain marker");
    std::fs::create_dir_all(dir.join("spill")).expect("a plain directory");
    std::fs::write(dir.join("spill/out.txt"), b"").expect("a plain file");
    std::fs::write(dir.join("tui/half.json.tmp"), b"").expect("a file being written");
    let mut files = store.files().expect("listed");
    files.sort();
    let refused = store.write_file("saved", b"").is_err();
    std::fs::remove_dir_all(dir).ok();
    assert_eq!(files, ["meta.json", "tui/notices.jsonl"]);
    assert!(refused);
}

/// An effect log written to a store's file reads back, and a writer opened
/// on it later continues after its records.
#[cfg(feature = "jsonl")]
#[test]
fn an_effect_log_is_kept_in_a_store_file() {
    use super::EFFECT_LOG;
    use crate::effect_log::{EffectLog, jsonl};

    let fixture = include_str!("../../fixtures/effects/anthropic_completion_smoke.effects.json");
    let log: EffectLog = serde_json::from_str(fixture).expect("a fixture");
    let store = std::sync::Arc::new(MemoryStore::default());
    let mut writer = jsonl::Writer::on(store.clone(), EFFECT_LOG);
    writer.append(&log).expect("appended");
    let bytes = store.read_file(EFFECT_LOG).expect("written");
    let read = jsonl::read_from(&bytes[..]).expect("read back");
    let ids = |log: &EffectLog| {
        log.records
            .iter()
            .map(|record| record.id)
            .collect::<Vec<_>>()
    };
    assert!(!log.records.is_empty());
    assert_eq!(ids(&read), ids(&log));
    let last = jsonl::Writer::on(store, EFFECT_LOG)
        .last_id()
        .expect("read");
    assert_eq!(last, ids(&log).into_iter().max());
}
