use std::fmt::Debug;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rig_core::conversation::Conversation;
use rig_core::message::{ImageMediaType, Message, UserContent};
use serde_json::Value;

use super::testing::{self, CHAIN, SYNTHETIC, SYNTHETIC_BLOBS, UNINSTALLED, UNINSTALLED_FILE};
use super::*;
use crate::{BLOBS, Files, LogWriteError, load_images};

/// A check that knows no saved type.
fn any(_: &str, _: &Value) -> io::Result<()> {
    Ok(())
}

/// An error saying why, when `holds` does not.
fn ensure(holds: bool, why: impl Debug) -> Result<(), SessionError> {
    match holds {
        true => Ok(()),
        false => Err(io::Error::other(format!("{why:?}")).into()),
    }
}

fn fixture() -> Result<MemoryStore, SessionError> {
    let store = MemoryStore::default();
    testing::write_fixture(&store)?;
    Ok(store)
}

/// The kept copy of the session in `store` that a migration from `from`
/// made.
fn kept(store: &dyn Store, from: &str) -> io::Result<Snapshot> {
    let prefix = format!("{KEPT}{from}/");
    let mut copy = Snapshot::default();
    for name in store.files()? {
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let bytes = store.read_file(&name)?;
        if let Some(log) = rest
            .strip_prefix("logs/")
            .and_then(|log| log.strip_suffix(".jsonl"))
        {
            copy.logs.insert(log.to_owned(), bytes);
        } else if let Some(file) = rest.strip_prefix("files/") {
            copy.files.insert(file.to_owned(), bytes);
        }
    }
    Ok(copy)
}

/// Every format change adds a step: the shipped chain goes from the first
/// format rig released to [`FORMAT`], and each step's generated fixture
/// is in its format and migrates through the rest of the chain.
#[test]
fn the_shipped_chain_reaches_the_format_and_each_step_has_a_fixture() -> Result<(), SessionError> {
    let mut at = FIRST;
    while at != FORMAT {
        let step = SHIPPED.steps.iter().find(|step| step.from == at);
        let Some(step) = step else {
            return Err(io::Error::other(format!("no step from {at} toward {FORMAT}")).into());
        };
        at = step.to;
    }
    for (chain, step) in [&SHIPPED, &CHAIN]
        .into_iter()
        .flat_map(|chain| chain.steps.iter().map(move |step| (chain, step)))
    {
        let store = (step.fixture)()?;
        let format = Snapshot::read(&store)?.format()?;
        let found = format.and_then(|(_, found)| found);
        ensure(found.as_deref() == Some(step.from), (step.from, &found))?;
        chain.migrate(&store, &any)?;
        Session::read(&store)?;
    }
    Ok(())
}

/// A migration takes the whole session to the format, each log's header
/// and its blobs; it keeps byte for byte what it does not know (a value,
/// an aside and a file of a plugin that is not installed), and keeps a
/// copy of the session as it was.
#[test]
fn a_synthetic_session_migrates_whole_and_keeps_what_it_was() -> Result<(), SessionError> {
    let store = fixture()?;
    let before = Snapshot::read(&store)?;
    let refused = Session::read(&store);
    ensure(
        matches!(refused, Err(SessionError::Format { .. })),
        &refused,
    )?;
    let checked = Mutex::new(Vec::new());
    let check = |key: &str, _: &Value| {
        if let Ok(mut checked) = checked.lock() {
            checked.push(key.to_owned());
        }
        Ok(())
    };
    CHAIN.migrate(&store, &check)?;
    let checked = checked.into_inner().unwrap_or_default();
    ensure(checked.iter().any(|key| key == UNINSTALLED), &checked)?;

    let session = Session::read(&store)?;
    let lost = |repair: &Repair| matches!(repair, Repair::LostImage { .. });
    ensure(!session.repairs.iter().any(lost), &session.repairs)?;
    let Some(agent) = session.agent("agent") else {
        return Err(io::Error::other("no agent").into());
    };
    for message in agent.conversation.messages() {
        load_images(&mut (**message).clone(), &store)?;
    }
    ensure(agent.conversation.messages().len() == 2, "two messages")?;
    let value = serde_json::json!({"kept": [false, "as is"]});
    ensure(
        session.saved.get(UNINSTALLED) == Some(&value),
        &session.saved,
    )?;
    let mut asides = agent.conversation.asides();
    ensure(
        asides.any(|(_, aside)| aside.kind == "uninstalled.aside"),
        "the uninstalled aside",
    )?;

    let after = Snapshot::read(&store)?;
    for file in [UNINSTALLED_FILE, META] {
        ensure(after.files.get(file) == before.files.get(file), file)?;
    }
    let names = || after.files.keys();
    ensure(
        !names().any(|name| name.starts_with(SYNTHETIC_BLOBS)),
        "synthetic blobs left",
    )?;
    ensure(names().any(|name| name.starts_with(BLOBS)), "no blob")?;
    ensure(after.logs.keys().eq(before.logs.keys()), "the logs")?;
    ensure(kept(&store, SYNTHETIC)? == before, "the kept copy")?;

    // In the format now, it is left alone.
    CHAIN.migrate(&store, &any)?;
    ensure(Snapshot::read(&store)? == after, "migrated again")
}

/// A session in the format, or with no log, gets no write and no check.
#[test]
fn a_session_in_the_format_is_left_alone() -> Result<(), SessionError> {
    let store = MemoryStore::default();
    SHIPPED.migrate(&store, &any)?;
    let mut conversation = crate::load(&store, "agent")?;
    conversation.push(Message::user("hi"));
    crate::save(&store, "agent", &mut conversation)?;
    let before = Snapshot::read(&store)?;
    let refuse = |_: &str, _: &Value| Err(io::Error::other("no value is checked"));
    CHAIN.migrate(&store, &refuse)?;
    ensure(Snapshot::read(&store)? == before, "a write")?;
    ensure(store.files()?.is_empty(), "a file")
}

/// Each refusal comes before any write: no step from the format, a step
/// that cannot do its work (such as one that needs a schema it lacks), a
/// step that loses a blob, one that saves a whole value naming a blob
/// that is not there, a saved value that does not decode, and logs in
/// different formats.
#[test]
fn a_refused_migration_writes_nothing() -> Result<(), SessionError> {
    fn refusing(_: &mut Snapshot) -> Result<(), SessionError> {
        Err(io::Error::other("this step needs a schema it does not have").into())
    }
    fn migrated(session: &mut Snapshot) -> Result<(), SessionError> {
        CHAIN
            .steps
            .iter()
            .try_for_each(|step| (step.apply)(session))
    }
    fn losing(session: &mut Snapshot) -> Result<(), SessionError> {
        session
            .files
            .retain(|name, _| !name.starts_with(SYNTHETIC_BLOBS));
        migrated(session)
    }
    fn resetting(session: &mut Snapshot) -> Result<(), SessionError> {
        migrated(session)?;
        let store = Arc::new(MemoryStore::default());
        session.write_to(&*store)?;
        let (writer, _) = crate::Writer::open(store.clone())?;
        writer.start();
        let image = UserContent::image_url("blob:missing.png", Some(ImageMediaType::PNG), None);
        let mut replaced = Conversation::from_messages([Message::User {
            content: vec![image],
        }]);
        writer.save("agent", &mut replaced);
        writer.flush()?;
        *session = Snapshot::read(&*store)?;
        Ok(())
    }
    const fn step(apply: fn(&mut Snapshot) -> Result<(), SessionError>) -> Step {
        Step {
            from: SYNTHETIC,
            to: FORMAT,
            apply,
            fixture,
        }
    }
    const REFUSED: &[Step] = &[step(refusing)];
    const LOST: &[Step] = &[step(losing)];
    const RESET: &[Step] = &[step(resetting)];
    let decode = |key: &str, _: &Value| match key == UNINSTALLED {
        true => Err(io::Error::other("not a value of its type")),
        false => Ok(()),
    };
    let tries: [(&str, Chain, &Verify<'_>); 5] = [
        ("no step", SHIPPED, &any),
        ("refusing step", Chain { steps: REFUSED }, &any),
        ("lost blob", Chain { steps: LOST }, &any),
        ("reset to a lost blob", Chain { steps: RESET }, &any),
        ("undecodable value", CHAIN, &decode),
    ];
    for (test, chain, verify) in tries {
        let store = fixture()?;
        let before = Snapshot::read(&store)?;
        let files = store.files()?.len();
        let migrated = chain.migrate(&store, verify);
        ensure(migrated.is_err(), (test, &migrated))?;
        ensure(Snapshot::read(&store)? == before, (test, "a write"))?;
        ensure(store.files()?.len() == files, (test, "a file"))?;
    }

    let store = fixture()?;
    let mut conversation = crate::load(&store, "other")?;
    conversation.push(Message::user("in the format"));
    crate::save(&store, "other", &mut conversation)?;
    let before = Snapshot::read(&store)?;
    let mixed = CHAIN.migrate(&store, &any);
    ensure(matches!(mixed, Err(SessionError::Format { .. })), &mixed)?;
    ensure(Snapshot::read(&store)? == before, "a write")
}

/// A store over a [`MemoryStore`] whose next log replace fails while
/// `fail` is set.
#[derive(Default)]
struct Failing {
    store: MemoryStore,
    fail: AtomicBool,
}

impl Files for Failing {
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.store.write_file(name, bytes)
    }
    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.store.append_file(name, bytes)
    }
    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        self.store.read_file(name)
    }
    fn remove_file(&self, name: &str) -> io::Result<()> {
        self.store.remove_file(name)
    }
    fn files(&self) -> io::Result<Vec<String>> {
        self.store.files()
    }
}

impl Store for Failing {
    fn logs(&self) -> io::Result<Vec<String>> {
        self.store.logs()
    }
    fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
        self.store.read_log(name)
    }
    fn append_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError> {
        self.store.append_log(name, len, bytes)
    }
    fn replace_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError> {
        if self.fail.swap(false, Ordering::SeqCst) {
            return Err(io::Error::other("the disk is full").into());
        }
        self.store.replace_log(name, len, bytes)
    }
    fn destroy(&self) -> io::Result<()> {
        self.store.destroy()
    }
}

/// A migration that stopped after its kept copy is not run again over
/// what it half wrote: the next is refused, writing nothing, and the
/// kept copy stays the session as it first was.
#[test]
fn a_migration_that_did_not_finish_keeps_the_first_copy() -> Result<(), SessionError> {
    let store = Failing::default();
    testing::write_fixture(&store)?;
    let original = Snapshot::read(&store)?;
    store.fail.store(true, Ordering::SeqCst);
    let failed = CHAIN.migrate(&store, &any);
    ensure(failed.is_err(), &failed)?;
    let half = Snapshot::read(&store)?;
    let files = store.files()?.len();
    let again = CHAIN.migrate(&store, &any);
    let refused = matches!(&again, Err(SessionError::Io(error)) if error.kind() == io::ErrorKind::AlreadyExists);
    ensure(refused, &again)?;
    ensure(Snapshot::read(&store)? == half, "a write")?;
    ensure(store.files()?.len() == files, "a file")?;
    ensure(kept(&store, SYNTHETIC)? == original, "the kept copy")
}

/// A directory store migrates through its own files: the kept copy is no
/// top-level log, so the directory opens again in this layout.
#[cfg(feature = "fs")]
#[test]
fn a_directory_store_migrates_and_opens_again() -> Result<(), SessionError> {
    let dir = std::env::temp_dir().join(format!("rig-memory-migrate-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let migrated = (|| {
        let store = crate::DirStore::open(&dir)?;
        testing::write_fixture(&store)?;
        CHAIN.migrate(&store, &any)?;
        let reopened = crate::DirStore::open(&dir)?;
        Session::read(&reopened)?;
        let names = reopened.files()?;
        ensure(names.iter().any(|name| name.starts_with(KEPT)), &names)
    })();
    std::fs::remove_dir_all(&dir).ok();
    migrated
}
