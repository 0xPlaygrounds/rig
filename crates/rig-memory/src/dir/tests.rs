use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::Conversation;

use super::{DirFiles, DirStore};
use crate::{
    FORMAT, Files, LogWriteError, Session, SessionError, Store, Writer, as_of, history, load, save,
};

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
    assert!(message.contains("another version of rig"), "{message}");
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

/// A session the rig before this format saved (`rig-session/3`) is refused
/// by every reader and writer before anything is written, its bytes
/// unchanged, with a message that says to start a new one; so is any other
/// format. With only its format changed to [`FORMAT`], the same logs read.
#[test]
fn an_old_format_session_is_refused_untouched() {
    let dir = scratch("format-3");
    write_fixture(&dir, include_str!("session-3.txt"));
    fs::write(dir.join("agent.log"), "what the agent did").expect("the text log");
    let before = snapshot(&dir);
    let store = Arc::new(DirStore::open(&dir).expect("the layout is this rig's"));
    let agent = "cd9a3280-91f0-4fb4-aa5c-c1c3629a0f32";
    let old = |refused: SessionError| {
        let message = refused.to_string();
        let format = matches!(&refused, SessionError::Format { found: Some(found), .. } if found == "rig-session/3");
        assert!(format, "{refused:?}");
        assert!(
            message.contains("made by another version of rig; start a new session"),
            "{message}"
        );
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

    for format in ["rig-session/5", "other/1"] {
        let dir = scratch(&format.replace('/', "-"));
        write_fixture(
            &dir,
            &include_str!("session-3.txt").replace("rig-session/3", format),
        );
        let store = DirStore::open(&dir).expect("opened");
        let refused = Session::read(&store).map(|_| ()).expect_err("refused");
        assert!(refused.to_string().contains(format), "{refused}");
        let _ = fs::remove_dir_all(&dir);
    }

    let dir = scratch("format-3-as-this");
    write_fixture(
        &dir,
        &include_str!("session-3.txt").replace("rig-session/3", FORMAT),
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
    // The lock file stays, so every store on the directory locks one file.
    let left: Vec<PathBuf> = snapshot(&dir).into_keys().collect();
    assert_eq!(left, [dir.join(".rig-memory.lock"), dir.join("agent.log")]);
    assert!(fs::read_dir(&dir).expect("listed").count() == 2);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_replaced_log_takes_the_next_append_and_a_crashed_replace_is_not_a_log() {
    let dir = scratch("replace");
    let store = DirStore::open(&dir).expect("opened");
    store.append_log("main", 0, b"old 1\n").expect("appended");
    store.append_log("main", 6, b"old 2\n").expect("appended");
    store.replace_log("main", 12, b"new\n").expect("replaced");
    store.append_log("main", 4, b"after\n").expect("appended");
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

/// Two stores open on one directory, as two processes are: each sees what
/// the other wrote, a replace included.
#[test]
fn a_second_dir_store_on_the_directory_sees_every_write() {
    let dir = scratch("second");
    let (a, b) = (
        DirStore::open(&dir).expect("opened"),
        DirStore::open(&dir).expect("opened"),
    );
    a.append_log("a", 0, b"a 1\n").expect("appended");
    b.replace_log("a", 4, b"b\n").expect("replaced");
    a.append_log("a", 2, b"a 2\n").expect("appended");
    assert_eq!(a.read_log("a").expect("read"), b"b\na 2\n");
    // Two writes at one length: the second is refused and writes nothing.
    b.append_log("a", 6, b"b 2\n").expect("appended");
    let refused = a.append_log("a", 6, b"a 3\n");
    assert!(
        matches!(refused, Err(LogWriteError::Changed)),
        "{refused:?}"
    );
    assert_eq!(a.read_log("a").expect("read"), b"b\na 2\nb 2\n");
    let _ = fs::remove_dir_all(&dir);
}
