use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::Conversation;

use super::{DirFiles, DirStore};
use crate::{Files, Session, SessionError, Store, Writer, as_of, history, load, save};

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
    assert!(matches!(refused, Err(SessionError::OldLayout)));
    let message = SessionError::OldLayout.to_string();
    assert!(message.contains("older version of rig"), "{message}");
    assert_eq!(snapshot(&dir), before);
    let _ = fs::remove_dir_all(&dir);
}

/// Writes the session `fixture` describes into `dir`: each `=== <path>`
/// starts a file, each line after it is a line of that file.
fn write_fixture(dir: &Path, fixture: &str) {
    let mut file = None;
    for line in fixture.lines() {
        if let Some(path) = line.strip_prefix("=== ") {
            let path = dir.join(path);
            fs::create_dir_all(path.parent().expect("a folder")).expect("a folder");
            fs::write(&path, b"").expect("a file");
            file = Some(path);
        } else if let Some(path) = &file {
            let mut bytes = fs::read(path).expect("read");
            bytes.extend_from_slice(format!("{line}\n").as_bytes());
            fs::write(path, bytes).expect("written");
        }
    }
}

/// A session saved in format `rig-session/1` is refused by every reader
/// and writer before anything is written, with a message that says so;
/// with only its format changed, the same logs read.
#[test]
fn a_session_saved_in_format_1_is_refused_before_anything_is_written() {
    let dir = scratch("format-1");
    write_fixture(&dir, include_str!("session-1.txt"));
    fs::write(dir.join("agent.log"), "what the agent did").expect("the text log");
    let before = snapshot(&dir);
    let store = Arc::new(DirStore::open(&dir).expect("the layout is this rig's"));
    let agent = "cd9a3280-91f0-4fb4-aa5c-c1c3629a0f32";
    let old = |refused: SessionError| {
        let message = refused.to_string();
        let format = matches!(&refused, SessionError::Format { found: Some(found), .. } if found == "rig-session/1");
        assert!(format, "{refused:?}");
        assert!(
            message.contains("older version of rig") && message.contains("rig-session/3"),
            "{message}"
        );
        // The log's name is quoted, so `session` does not read as a word.
        assert!(message.starts_with("the log `"), "{message}");
    };
    old(Session::read(&*store).expect_err("refused"));
    old(Writer::open(store.clone())
        .map(|_| ())
        .expect_err("refused"));
    old(load(&*store, agent).expect_err("refused"));
    old(history(&*store, agent).map(|_| ()).expect_err("refused"));
    old(as_of(&*store, agent, 4).map(|_| ()).expect_err("refused"));
    let mut conversation = Conversation::from_messages([Message::user("more")]);
    old(save(&*store, agent, &mut conversation).expect_err("refused"));
    let mut session = Session::default();
    old(session.save(&*store).expect_err("refused"));
    drop(store);
    assert_eq!(snapshot(&dir), before);
    let _ = fs::remove_dir_all(&dir);

    // A newer rig's format, and one that is not rig's, are refused too,
    // each saying which it is.
    for (format, says) in [
        ("rig-session/4", "a newer version of rig"),
        ("other/1", "is in format other/1, not rig-session/3"),
    ] {
        let dir = scratch(&format.replace('/', "-"));
        write_fixture(
            &dir,
            &include_str!("session-1.txt").replace("rig-session/1", format),
        );
        let store = DirStore::open(&dir).expect("opened");
        let refused = Session::read(&store).map(|_| ()).expect_err("refused");
        assert!(refused.to_string().contains(says), "{refused}");
        let _ = fs::remove_dir_all(&dir);
    }

    let dir = scratch("format-1-as-2");
    write_fixture(
        &dir,
        &include_str!("session-1.txt").replace("rig-session/1", "rig-session/2"),
    );
    let store = DirStore::open(&dir).expect("opened");
    let session = Session::read(&store).expect("read");
    let main = session.agent(agent).expect("the agent");
    assert_eq!(main.conversation.messages().len(), 2);
    assert_eq!(main.conversation.asides().count(), 1);
    assert_eq!(
        main.conversation.entries().map(|entry| entry.made).max(),
        Some(1_791_779_409_182)
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_session_keeps_its_two_folders_and_destroy_finishes_a_crashed_one() {
    let dir = scratch("destroy");
    fs::write(dir.join("agent.log"), "kept").expect("a plain file");
    let store = DirStore::open(&dir).expect("opened");
    let mut conversation = Conversation::from_messages([Message::user("hello")]);
    save(&store, "main", &mut conversation).expect("saved");
    store.write_file("blobs/abc.png", b"png").expect("a blob");
    store
        .write_file("plugins/notes.json", b"{}")
        .expect("a file");
    assert_eq!(store.logs().expect("logs"), ["main"]);
    let mut files = store.files().expect("files");
    files.sort();
    assert_eq!(files, ["blobs/abc.png", "plugins/notes.json"]);
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

#[test]
fn a_replaced_log_takes_the_next_append_and_a_crashed_replace_is_not_a_log() {
    let dir = scratch("replace");
    let store = DirStore::open(&dir).expect("opened");
    // The first append opens the handle later appends reuse.
    store.append_log("main", b"old 1\n").expect("appended");
    store.append_log("main", b"old 2\n").expect("appended");
    store.replace_log("main", b"new\n").expect("replaced");
    store.append_log("main", b"after\n").expect("appended");
    assert_eq!(store.read_log("main").expect("read"), b"new\nafter\n");

    // A replace that crashed before its rename leaves a part file beside
    // the log: the old log stands and no new log is listed.
    fs::write(dir.join("logs/main.jsonl.rig-part"), b"half").expect("a part");
    let store = DirStore::open(&dir).expect("reopened");
    assert_eq!(store.logs().expect("logs"), ["main"]);
    assert_eq!(store.read_log("main").expect("read"), b"new\nafter\n");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_part_file_is_never_listed_or_named() {
    let dir = scratch("parts");
    let files = DirFiles::new(dir.join("plugins"));
    files.write_file("notes/a.json", b"{}").expect("a file");
    // A write that crashed before its rename leaves a part file beside it.
    fs::write(dir.join("plugins/notes/b.json.rig-part"), b"half").expect("a part");
    assert_eq!(files.files().expect("files"), ["notes/a.json"]);
    assert!(files.write_file("c.json.rig-part", b"{}").is_err());
    let _ = fs::remove_dir_all(&dir);
}
