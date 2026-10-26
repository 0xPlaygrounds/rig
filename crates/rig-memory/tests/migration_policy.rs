//! The chain rig ships, linked as a normal build of the library: it has no
//! step from a format rig never released, or never wrote.

use std::io;

use rig_core::message::Message;
use rig_memory::{FORMAT, Files, MemoryStore, SessionError, Store, load, migration, save};

#[test]
fn the_shipped_chain_refuses_every_unreleased_format_and_writes_nothing() -> Result<(), SessionError>
{
    let formats = [
        "rig-session/1",
        "rig-session/2",
        "rig-session/3",
        "rig-session/synthetic",
        "rig-session/5",
    ];
    for format in formats {
        let store = MemoryStore::default();
        let mut conversation = load(&store, "agent")?;
        conversation.push(Message::user("a generated session"));
        save(&store, "agent", &mut conversation)?;
        let bytes = store.read_log("agent")?;
        let text = String::from_utf8(bytes.clone()).map_err(io::Error::other)?;
        let old = text.replacen(FORMAT, format, 1).into_bytes();
        store
            .replace_log("agent", bytes.len() as u64, &old)
            .map_err(io::Error::from)?;
        let migrated = migration::SHIPPED.migrate(&store, &|_, _| Ok(()));
        if !matches!(migrated, Err(SessionError::Format { .. }))
            || store.read_log("agent")? != old
            || !store.files()?.is_empty()
        {
            let why = format!("{format}: {migrated:?}");
            return Err(io::Error::other(why).into());
        }
    }
    Ok(())
}
