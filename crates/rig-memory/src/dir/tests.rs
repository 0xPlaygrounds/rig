use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use rig_core::completion::Message;
use rig_core::conversation::Conversation;

use super::DirStore;
use crate::{Files, LoadError, Session, Store, save};

/// A fresh directory for test `name`.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-memory-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a directory");
    dir
}

/// Every file under `dir` with its bytes.
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).expect("listed") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                let bytes = fs::read(&path).expect("read");
                files.insert(path, bytes);
            }
        }
    }
    files
}

#[test]
fn a_session_saved_by_an_older_rig_is_refused_unchanged() {
    // The layout and logs rig wrote before this format: `<agent>.jsonl`
    // at the top, blobs, and the text log beside them.
    let dir = scratch("baseline");
    let mut log = None;
    for line in include_str!("baseline-session.txt").lines() {
        if let Some(name) = line.strip_prefix("=== ") {
            log = Some(name);
        } else if let Some(blob) = log.and_then(|name| name.strip_prefix("blob ")) {
            fs::create_dir_all(dir.join("blobs")).expect("blobs");
            fs::write(dir.join("blobs").join(blob), line).expect("a blob");
        } else if let Some(agent) = log {
            let path = dir.join(format!("{agent}.jsonl"));
            let mut bytes = fs::read(&path).unwrap_or_default();
            bytes.extend_from_slice(format!("{line}\n").as_bytes());
            fs::write(path, bytes).expect("a log");
        }
    }
    fs::write(dir.join("agent.log"), "what the agent did").expect("the text log");
    let before = snapshot(&dir);
    let refused = DirStore::open(&dir);
    assert!(matches!(refused, Err(LoadError::OldLayout)));
    let message = LoadError::OldLayout.to_string();
    assert!(message.contains("older version of rig"), "{message}");
    assert_eq!(snapshot(&dir), before);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_session_keeps_its_three_folders_and_destroy_finishes_a_crashed_one() {
    let dir = scratch("destroy");
    fs::write(dir.join("agent.log"), "kept").expect("a plain file");
    let store = DirStore::open(&dir).expect("opened");
    let mut conversation = Conversation::from_messages([Message::user("hello")]);
    save(&store, "main", &mut conversation).expect("saved");
    store.put_blob("abc.png", b"png").expect("a blob");
    store
        .write_file("plugins/notes.json", b"{}")
        .expect("a file");
    assert_eq!(store.logs().expect("logs"), ["main"]);
    assert_eq!(store.blobs().expect("blobs"), ["abc.png"]);
    assert_eq!(store.files().expect("files"), ["plugins/notes.json"]);
    assert!(store.write_file("../escape", b"").is_err());
    let read = Session::read(&DirStore::open(&dir).expect("reopened")).expect("read");
    assert_eq!(read.agents.len(), 1);

    // A crash after the logs were renamed away, before they were removed.
    fs::rename(dir.join("logs"), dir.join("logs.rig-trash")).expect("renamed");
    store.destroy().expect("destroyed");
    store.destroy().expect("destroyed again");
    let left: Vec<PathBuf> = snapshot(&dir).into_keys().collect();
    assert_eq!(left, [dir.join("agent.log")]);
    assert!(fs::read_dir(&dir).expect("listed").count() == 1);
    let _ = fs::remove_dir_all(&dir);
}
