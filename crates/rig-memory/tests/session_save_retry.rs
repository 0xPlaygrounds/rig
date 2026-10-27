#![cfg(feature = "fs")]

use std::io;

use rig_core::conversation::Conversation;
use rig_core::message::{ImageMediaType, Message, UserContent};
use rig_memory::{Session, Store, load, save};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn image_message() -> Message {
    Message::User {
        content: vec![
            UserContent::text("This new message must survive the failed save"),
            UserContent::image_base64(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRz8AAAAASUVORK5CYII=",
                Some(ImageMediaType::PNG),
                None,
            ),
        ],
    }
}

#[cfg(feature = "fs")]
#[test]
fn audit_real_directory_store_retry_must_persist_the_message() -> TestResult {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "rig-session-save-audit-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir(&path)?;
    let store = rig_memory::DirStore::open(&path)?;
    let mut conversation = Conversation::from_messages([Message::user("persisted base")]);
    save(&store, "main", &mut conversation)?;
    let mut session = Session::read(&store)?;
    session
        .agent_mut("main")
        .ok_or_else(|| io::Error::other("missing main"))?
        .conversation
        .push(image_message());

    // A file at the blob-directory path makes the real backend's write fail.
    std::fs::create_dir_all(path.join("files"))?;
    std::fs::write(path.join("files/blobs"), b"temporary obstruction")?;
    let before = store.read_log("main")?;
    let failed = session.save(&store);
    println!("real DirStore first save: {failed:?}");
    if failed.is_ok() || before != store.read_log("main")? {
        return Err(io::Error::other("filesystem failure did not leave log untouched").into());
    }

    // Restore a writable blob path, then retry the exact same session.
    std::fs::remove_file(path.join("files/blobs"))?;
    let retry = session.save(&store);
    println!("real DirStore retry after removing obstruction: {retry:?}");
    retry?;
    let disk = load(&store, "main")?;
    println!(
        "real DirStore reloaded messages={}, expected=2",
        disk.messages().len()
    );
    std::fs::remove_dir_all(&path)?;
    if disk.messages().len() != 2 {
        return Err(io::Error::other("real filesystem retry silently lost the new message").into());
    }
    Ok(())
}

#[derive(Default)]
struct FailAppend {
    inner: rig_memory::MemoryStore,
    fail: std::sync::atomic::AtomicBool,
}
impl rig_memory::Files for FailAppend {
    fn write_file(&self, n: &str, b: &[u8]) -> io::Result<()> {
        self.inner.write_file(n, b)
    }
    fn append_file(&self, n: &str, b: &[u8]) -> io::Result<()> {
        self.inner.append_file(n, b)
    }
    fn read_file(&self, n: &str) -> io::Result<Vec<u8>> {
        self.inner.read_file(n)
    }
    fn remove_file(&self, n: &str) -> io::Result<()> {
        self.inner.remove_file(n)
    }
    fn files(&self) -> io::Result<Vec<String>> {
        self.inner.files()
    }
}
impl Store for FailAppend {
    fn logs(&self) -> io::Result<Vec<String>> {
        self.inner.logs()
    }
    fn read_log(&self, n: &str) -> io::Result<Vec<u8>> {
        self.inner.read_log(n)
    }
    fn append_log(&self, n: &str, len: u64, b: &[u8]) -> Result<(), rig_memory::LogWriteError> {
        if n == "main" && self.fail.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(io::Error::other("one rejected append").into());
        }
        self.inner.append_log(n, len, b)
    }
    fn replace_log(&self, n: &str, len: u64, b: &[u8]) -> Result<(), rig_memory::LogWriteError> {
        self.inner.replace_log(n, len, b)
    }
    fn destroy(&self) -> io::Result<()> {
        self.inner.destroy()
    }
}

#[test]
fn failed_append_and_partial_flush_can_be_retried() -> TestResult {
    use rig_memory::SavedAgent;
    for with_child in [false, true] {
        let store = FailAppend::default();
        let mut session = Session::default();
        session.agents.push(SavedAgent {
            id: "main".into(),
            conversation: Conversation::from_messages([Message::user("base")]),
            ..SavedAgent::default()
        });
        if with_child {
            session.agents.push(SavedAgent {
                id: "child".into(),
                parent: Some("main".into()),
                conversation: Conversation::from_messages([Message::user("child base")]),
                ..SavedAgent::default()
            });
        }
        session.save(&store)?;
        for agent in &mut session.agents {
            agent.conversation.push(Message::user("pending"));
        }
        store.fail.store(true, std::sync::atomic::Ordering::SeqCst);
        if session.save(&store).is_ok() {
            return Err(io::Error::other("append should fail").into());
        }
        let child_before = with_child.then(|| store.read_log("child")).transpose()?;
        for agent in &session.agents {
            if (agent.id == "main") == agent.conversation.unsaved().is_empty() {
                return Err(io::Error::other("pending changes do not match committed logs").into());
            }
        }
        session.save(&store)?;
        for agent in &session.agents {
            if load(&store, &agent.id)? != agent.conversation
                || !agent.conversation.unsaved().is_empty()
            {
                return Err(
                    io::Error::other("retry did not persist exactly the pending changes").into(),
                );
            }
        }
        if let Some(before) = child_before
            && store.read_log("child")? != before
        {
            return Err(io::Error::other("retry duplicated a committed child log").into());
        }
    }
    Ok(())
}
