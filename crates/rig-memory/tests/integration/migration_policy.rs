//! The chain rig ships, linked as a normal build of the library: it has no
//! step from a format rig never released, or never wrote, nor from a log
//! that names none.

use std::io;

use rig_core::message::Message;
use rig_memory::{FORMAT, Files, MemoryStore, SessionError, Store, load, migration, save};

#[test]
fn the_shipped_chain_refuses_every_unreleased_format_and_writes_nothing() -> Result<(), SessionError>
{
    let later = format!("{FORMAT}.later");
    let formats = [
        "rig-session/1",
        "rig-session/2",
        "rig-session/3",
        "rig-session/synthetic",
        &later,
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

/// A log with lines but no format, or a first line that does not read, is
/// refused, writing nothing.
#[test]
fn the_shipped_chain_refuses_a_log_without_a_format_and_writes_nothing() -> Result<(), SessionError>
{
    for first in [&b"{}\n"[..], b"not json\n{}\n"] {
        let store = MemoryStore::default();
        store
            .replace_log("agent", 0, first)
            .map_err(io::Error::from)?;
        let migrated = migration::SHIPPED.migrate(&store, &|_, _| Ok(()));
        if !matches!(migrated, Err(SessionError::Format { found: None, .. }))
            || store.read_log("agent")? != first
            || !store.files()?.is_empty()
        {
            let why = format!("{:?}: {migrated:?}", String::from_utf8_lossy(first));
            return Err(io::Error::other(why).into());
        }
    }
    Ok(())
}
