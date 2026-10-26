use std::io;

use serde_json::Value;

use super::testing::{self, CHAIN, SYNTHETIC, SYNTHETIC_BLOBS, UNINSTALLED, UNINSTALLED_FILE};
use super::*;
use crate::{Files, Repair, load_images};

/// A check that knows no saved type.
fn any(_: &str, _: &Value) -> io::Result<()> {
    Ok(())
}

fn fixture() -> MemoryStore {
    let store = MemoryStore::default();
    testing::write_fixture(&store).unwrap();
    store
}

/// The kept copy of the session in `store` that a migration from `from`
/// made.
fn kept(store: &MemoryStore, from: &str) -> Snapshot {
    let prefix = format!("{KEPT}{from}/");
    let mut copy = Snapshot::default();
    for name in store.files().unwrap() {
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let bytes = store.read_file(&name).unwrap();
        if let Some(log) = rest
            .strip_prefix("logs/")
            .and_then(|log| log.strip_suffix(".jsonl"))
        {
            copy.logs.insert(log.to_owned(), bytes);
        } else if let Some(file) = rest.strip_prefix("files/") {
            copy.files.insert(file.to_owned(), bytes);
        }
    }
    copy
}

/// Every format change adds a step: the shipped chain goes from the first
/// format rig released to [`FORMAT`], and each step's generated fixture
/// is in its format and migrates through the rest of the chain.
#[test]
fn the_shipped_chain_reaches_the_format_and_each_step_has_a_fixture() {
    let mut at = FIRST;
    while at != FORMAT {
        let step = SHIPPED.steps.iter().find(|step| step.from == at);
        at = step
            .unwrap_or_else(|| panic!("no step from {at} toward {FORMAT}"))
            .to;
    }
    for (chain, step) in [&SHIPPED, &CHAIN]
        .into_iter()
        .flat_map(|chain| chain.steps.iter().map(move |step| (chain, step)))
    {
        let store = (step.fixture)().unwrap();
        let format = Snapshot::read(&store).unwrap().format().unwrap();
        assert_eq!(
            format.and_then(|(_, found)| found).as_deref(),
            Some(step.from)
        );
        chain.migrate(&store, &any).unwrap();
        Session::read(&store).unwrap();
    }
}

/// A migration takes the whole session to the format, each log's header
/// and its blobs; it keeps byte for byte what it does not know (a value,
/// an aside and a file of a plugin that is not installed), and keeps a
/// copy of the session as it was.
#[test]
fn a_synthetic_session_migrates_whole_and_keeps_what_it_was() {
    let store = fixture();
    let before = Snapshot::read(&store).unwrap();
    assert!(matches!(
        Session::read(&store),
        Err(SessionError::Format { .. })
    ));
    let checked = std::sync::Mutex::new(Vec::new());
    let check = |key: &str, _: &Value| {
        checked.lock().unwrap().push(key.to_owned());
        Ok(())
    };
    CHAIN.migrate(&store, &check).unwrap();
    assert!(checked.lock().unwrap().iter().any(|key| key == UNINSTALLED));

    let session = Session::read(&store).unwrap();
    let lost = |repair: &Repair| matches!(repair, Repair::LostImage { .. });
    assert!(!session.repairs.iter().any(lost), "{:?}", session.repairs);
    let agent = session.agent("agent").unwrap();
    for message in agent.conversation.messages() {
        load_images(&mut (**message).clone(), &store).unwrap();
    }
    assert_eq!(agent.conversation.messages().len(), 2);
    let value = serde_json::json!({"kept": [false, "as is"]});
    assert_eq!(session.saved.get(UNINSTALLED), Some(&value));
    let mut asides = agent.conversation.asides();
    assert!(asides.any(|(_, aside)| aside.kind == "uninstalled.aside"));

    let after = Snapshot::read(&store).unwrap();
    for file in [UNINSTALLED_FILE, META] {
        assert_eq!(after.files.get(file), before.files.get(file), "{file}");
    }
    assert!(
        !after
            .files
            .keys()
            .any(|name| name.starts_with(SYNTHETIC_BLOBS))
    );
    assert!(after.files.keys().any(|name| name.starts_with(BLOBS)));
    assert!(after.logs.keys().eq(before.logs.keys()));
    assert_eq!(kept(&store, SYNTHETIC), before);

    // In the format now, it is left alone.
    CHAIN.migrate(&store, &any).unwrap();
    assert_eq!(Snapshot::read(&store).unwrap(), after);
}

/// A session in the format, or with no log, gets no write and no check.
#[test]
fn a_session_in_the_format_is_left_alone() {
    let store = MemoryStore::default();
    SHIPPED.migrate(&store, &any).unwrap();
    let mut conversation = crate::load(&store, "agent").unwrap();
    conversation.push(rig_core::message::Message::user("hi"));
    crate::save(&store, "agent", &mut conversation).unwrap();
    let before = Snapshot::read(&store).unwrap();
    let refuse = |_: &str, _: &Value| Err(io::Error::other("no value is checked"));
    CHAIN.migrate(&store, &refuse).unwrap();
    assert_eq!(Snapshot::read(&store).unwrap(), before);
    assert!(store.files().unwrap().is_empty());
}

/// Each refusal comes before any write: no step from the format, a step
/// that cannot do its work (such as one that needs a schema it lacks), a
/// step that loses a blob, a saved value that does not decode, and logs
/// in different formats.
#[test]
fn a_refused_migration_writes_nothing() {
    fn refusing(_: &mut Snapshot) -> Result<(), SessionError> {
        Err(io::Error::other("this step needs a schema it does not have").into())
    }
    fn losing(session: &mut Snapshot) -> Result<(), SessionError> {
        session
            .files
            .retain(|name, _| !name.starts_with(SYNTHETIC_BLOBS));
        CHAIN
            .steps
            .iter()
            .try_for_each(|step| (step.apply)(session))
    }
    const fn step(apply: fn(&mut Snapshot) -> Result<(), SessionError>) -> Step {
        Step {
            from: SYNTHETIC,
            to: FORMAT,
            apply,
            fixture: || Ok(fixture()),
        }
    }
    const REFUSED: &[Step] = &[step(refusing)];
    const LOST: &[Step] = &[step(losing)];
    let decode = |key: &str, _: &Value| match key == UNINSTALLED {
        true => Err(io::Error::other("not a value of its type")),
        false => Ok(()),
    };
    let tries: [(&str, Chain, &Verify<'_>); 4] = [
        ("no step", SHIPPED, &any),
        ("refusing step", Chain { steps: REFUSED }, &any),
        ("lost blob", Chain { steps: LOST }, &any),
        ("undecodable value", CHAIN, &decode),
    ];
    for (test, chain, verify) in tries {
        let store = fixture();
        let before = Snapshot::read(&store).unwrap();
        let files = store.files().unwrap().len();
        assert!(chain.migrate(&store, verify).is_err(), "{test}");
        assert_eq!(Snapshot::read(&store).unwrap(), before, "{test}");
        assert_eq!(store.files().unwrap().len(), files, "{test}");
    }

    let store = fixture();
    let mut conversation = crate::load(&store, "other").unwrap();
    conversation.push(rig_core::message::Message::user("in the format"));
    crate::save(&store, "other", &mut conversation).unwrap();
    let before = Snapshot::read(&store).unwrap();
    let mixed = CHAIN.migrate(&store, &any);
    assert!(
        matches!(mixed, Err(SessionError::Format { .. })),
        "{mixed:?}"
    );
    assert_eq!(Snapshot::read(&store).unwrap(), before);
}

/// A directory store migrates through its own files: the kept copy is no
/// top-level log, so the directory opens again in this layout.
#[cfg(feature = "fs")]
#[test]
fn a_directory_store_migrates_and_opens_again() {
    let dir = std::env::temp_dir().join(format!("rig-memory-migrate-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let store = crate::DirStore::open(&dir).unwrap();
    testing::write_fixture(&store).unwrap();
    CHAIN.migrate(&store, &any).unwrap();
    let reopened = crate::DirStore::open(&dir);
    let read = reopened.as_ref().map(|store| Session::read(store).is_ok());
    let names = reopened.as_ref().map(|store| store.files().unwrap());
    std::fs::remove_dir_all(&dir).ok();
    assert!(matches!(read, Ok(true)), "{read:?}");
    assert!(names.unwrap().iter().any(|name| name.starts_with(KEPT)));
}
