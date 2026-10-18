//! A token store outside rig, such as one over a database or a keychain,
//! returns a lock guard of its own from `TokenStore::lock`, and dropping the
//! returned `StoreLock` runs that guard's release.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_core::error::BoxError;
use rig_core::providers::chatgpt::auth::{MemoryTokenStore, StoreLock, TokenStore};
use rig_core::wasm_compat::WasmBoxedFuture;

/// Counts the locks taken and released, as a lock table in a database would.
#[derive(Default)]
struct Leases {
    taken: AtomicUsize,
    released: AtomicUsize,
}

/// The guard a lock returns: releasing the lease is its drop.
struct Lease(Arc<Leases>);

impl Drop for Lease {
    fn drop(&mut self) {
        self.0.released.fetch_add(1, Ordering::SeqCst);
    }
}

/// Keeps records in memory and its locks in [`Leases`].
#[derive(Default)]
struct LeasedStore {
    records: MemoryTokenStore,
    leases: Arc<Leases>,
}

impl TokenStore for LeasedStore {
    fn load<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        self.records.load(key)
    }

    fn save<'a>(
        &'a self,
        key: &'a str,
        record: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        self.records.save(key, record)
    }

    fn delete<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        self.records.delete(key)
    }

    fn lock<'a>(&'a self, _key: &'a str) -> WasmBoxedFuture<'a, Result<StoreLock, BoxError>> {
        Box::pin(async move {
            self.leases.taken.fetch_add(1, Ordering::SeqCst);
            Ok(StoreLock::new(Lease(self.leases.clone())))
        })
    }
}

#[tokio::test]
async fn dropping_the_lock_runs_the_custom_release() -> Result<(), BoxError> {
    let store = LeasedStore::default();
    let leases = store.leases.clone();
    let shared: Arc<dyn TokenStore> = Arc::new(store);

    let held = shared.lock("chatgpt").await?;
    let (taken, released) = (
        leases.taken.load(Ordering::SeqCst),
        leases.released.load(Ordering::SeqCst),
    );
    if (taken, released) != (1, 0) {
        return Err(format!("while held: taken {taken}, released {released}").into());
    }

    drop(held);
    let released = leases.released.load(Ordering::SeqCst);
    if released != 1 {
        return Err(format!("after drop: released {released}").into());
    }
    Ok(())
}
