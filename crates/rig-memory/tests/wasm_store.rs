//! On browser wasm a store, and its futures, need not be `Send` or `Sync`:
//! one over a JavaScript promise holds `!Send` values. This compiles only
//! there (`cargo check -p rig-memory --target wasm32-unknown-unknown --test
//! wasm_store`); natively it is empty.
#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::rc::Rc;

use rig_core::conversation::Conversation;
use rig_memory::{AsyncStore, LogWriteError, SessionError, load_async, save_async};

/// Logs and files in `Rc<RefCell<…>>`, as a store over IndexedDB keeps
/// its handles; every call holds the `Rc` across an await.
#[derive(Clone, Default)]
struct Browser {
    logs: Rc<RefCell<BTreeMap<String, Vec<u8>>>>,
    files: Rc<RefCell<BTreeMap<String, Vec<u8>>>>,
}

async fn tick() {}

fn missing() -> io::Error {
    io::ErrorKind::NotFound.into()
}

impl AsyncStore for Browser {
    async fn read_log_async(&self, name: &str) -> io::Result<Vec<u8>> {
        let logs = Rc::clone(&self.logs);
        tick().await;
        let bytes = logs.borrow().get(name).cloned();
        bytes.ok_or_else(missing)
    }

    async fn append_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        let logs = Rc::clone(&self.logs);
        tick().await;
        // The check and the append in one borrow, as in one transaction.
        let mut logs = logs.borrow_mut();
        let log = logs.entry(name.to_owned()).or_default();
        if log.len() as u64 != len {
            return Err(LogWriteError::Changed);
        }
        log.extend_from_slice(bytes);
        Ok(())
    }

    async fn replace_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        let logs = Rc::clone(&self.logs);
        tick().await;
        let mut logs = logs.borrow_mut();
        let log = logs.entry(name.to_owned()).or_default();
        if log.len() as u64 != len {
            return Err(LogWriteError::Changed);
        }
        *log = bytes.to_vec();
        Ok(())
    }

    async fn read_file_async(&self, name: &str) -> io::Result<Vec<u8>> {
        let files = Rc::clone(&self.files);
        tick().await;
        let bytes = files.borrow().get(name).cloned();
        bytes.ok_or_else(missing)
    }

    async fn write_file_async(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let files = Rc::clone(&self.files);
        tick().await;
        files.borrow_mut().insert(name.to_owned(), bytes.to_vec());
        Ok(())
    }
}

/// A turn saved and read back through the `!Send` store.
pub async fn keeps_a_turn(conversation: &mut Conversation) -> Result<Conversation, SessionError> {
    let store = Browser::default();
    save_async(&store, "agent", conversation).await?;
    load_async(&store, "agent").await
}
