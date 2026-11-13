use super::{MemoryTokenStore, TokenStore, load_record, save_record};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Every store answers a missing record with `None`, keeps what is saved,
/// and forgets it on delete, twice without error.
async fn round_trips(store: &dyn TokenStore) -> anyhow::Result<()> {
    anyhow::ensure!(
        store
            .load("chatgpt")
            .await
            .map_err(anyhow::Error::from_boxed)?
            .is_none()
    );
    store
        .save("chatgpt", "{\"a\":1}")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    store
        .save("chatgpt", "{\"a\":2}")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let kept = store
        .load("chatgpt")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    anyhow::ensure!(kept.as_deref() == Some("{\"a\":2}"), "kept {kept:?}");
    anyhow::ensure!(
        store
            .load("copilot")
            .await
            .map_err(anyhow::Error::from_boxed)?
            .is_none()
    );
    store
        .delete("chatgpt")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    store
        .delete("chatgpt")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    anyhow::ensure!(
        store
            .load("chatgpt")
            .await
            .map_err(anyhow::Error::from_boxed)?
            .is_none()
    );
    Ok(())
}

#[tokio::test]
async fn memory_store_round_trips() -> anyhow::Result<()> {
    round_trips(&MemoryTokenStore::default()).await
}

#[tokio::test]
async fn memory_store_debug_shows_no_record() -> anyhow::Result<()> {
    let store = MemoryTokenStore::default();
    store
        .save("chatgpt", "synthetic-secret")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    anyhow::ensure!(!format!("{store:?}").contains("synthetic-secret"));
    Ok(())
}

#[tokio::test]
async fn file_store_round_trips() -> anyhow::Result<()> {
    let dir = assert_fs::TempDir::new()?;
    round_trips(&super::FileTokenStore::new(dir.path().join("auth"))).await
}

/// A shared store is one store: a record saved through one handle loads
/// through another, as JSON.
#[tokio::test]
async fn records_are_json_through_a_shared_store() -> anyhow::Result<()> {
    let store = Arc::new(MemoryTokenStore::default());
    let shared: Arc<dyn TokenStore> = store.clone();
    let record = BTreeMap::from([("access_token", "synthetic")]);
    save_record(&shared, "chatgpt", &record).await?;
    let text = store
        .load("chatgpt")
        .await
        .map_err(anyhow::Error::from_boxed)?
        .unwrap_or_default();
    anyhow::ensure!(
        serde_json::from_str::<BTreeMap<String, String>>(&text)?
            == BTreeMap::from([("access_token".to_owned(), "synthetic".to_owned())])
    );
    let loaded: BTreeMap<String, String> = load_record(&shared, "chatgpt").await?;
    anyhow::ensure!(loaded.get("access_token").map(String::as_str) == Some("synthetic"));
    let missing: BTreeMap<String, String> = load_record(&shared, "copilot").await?;
    anyhow::ensure!(missing.is_empty());
    Ok(())
}

/// The credential file and the directory created for it are the owner's alone.
#[cfg(unix)]
#[tokio::test]
async fn file_store_writes_private_files() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let dir = assert_fs::TempDir::new()?;
    let store = super::FileTokenStore::new(dir.path().join("auth"));
    store
        .save("chatgpt", "{}")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let path = store.path("chatgpt");
    anyhow::ensure!(path == dir.path().join("auth").join("chatgpt.json"));
    let file_mode = std::fs::metadata(&path)?.permissions().mode() & 0o777;
    anyhow::ensure!(file_mode == 0o600, "file mode {file_mode:o}");
    let dir_mode = std::fs::metadata(dir.path().join("auth"))?
        .permissions()
        .mode()
        & 0o777;
    anyhow::ensure!(dir_mode == 0o700, "directory mode {dir_mode:o}");
    Ok(())
}

/// A save replaces the credential whole, through a temporary file renamed
/// over it: none is left behind, and a file another tool widened is 0600
/// again.
#[cfg(unix)]
#[tokio::test]
async fn file_store_replaces_the_file_whole_and_keeps_it_private() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let dir = assert_fs::TempDir::new()?;
    let store = super::FileTokenStore::new(dir.path().join("auth"));
    store
        .save("chatgpt", "{\"a\":1}")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let path = store.path("chatgpt");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
    store
        .save("chatgpt", "{\"a\":2}")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    anyhow::ensure!(std::fs::read_to_string(&path)? == "{\"a\":2}");
    let mode = std::fs::metadata(&path)?.permissions().mode() & 0o777;
    anyhow::ensure!(mode == 0o600, "file mode {mode:o}");
    let names: Vec<String> = std::fs::read_dir(dir.path().join("auth"))?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    anyhow::ensure!(names == ["chatgpt.json"], "{names:?}");
    Ok(())
}

/// The file lock is held against every other holder, as another process's
/// would be: a second lock waits until the first is dropped.
#[tokio::test]
async fn file_store_lock_waits_for_the_holder() -> anyhow::Result<()> {
    let dir = assert_fs::TempDir::new()?;
    let store = super::FileTokenStore::new(dir.path().join("auth"));
    let other = super::FileTokenStore::new(dir.path().join("auth"));
    let held = store
        .lock("chatgpt")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let mut waiting = other.lock("chatgpt");
    let early = tokio::time::timeout(std::time::Duration::from_millis(200), &mut waiting).await;
    anyhow::ensure!(early.is_err(), "the second lock did not wait");
    drop(held);
    let taken = tokio::time::timeout(std::time::Duration::from_secs(5), waiting).await;
    anyhow::ensure!(matches!(taken, Ok(Ok(_))), "the second lock was not taken");
    Ok(())
}

/// A credential file another tool left readable by others is read, and is
/// the owner's alone afterwards.
#[cfg(unix)]
#[tokio::test]
async fn file_store_narrows_a_file_readable_by_others() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let dir = assert_fs::TempDir::new()?;
    let store = super::FileTokenStore::new(dir.path());
    let path = store.path("chatgpt");
    std::fs::write(&path, "{\"a\":1}")?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
    let loaded = store
        .load("chatgpt")
        .await
        .map_err(anyhow::Error::from_boxed)?;
    anyhow::ensure!(loaded.as_deref() == Some("{\"a\":1}"), "loaded {loaded:?}");
    let mode = std::fs::metadata(&path)?.permissions().mode() & 0o777;
    anyhow::ensure!(mode == 0o600, "file mode {mode:o}");
    Ok(())
}

/// The refusal `load("chatgpt")` answers with; an error when it loads.
async fn refusal(store: &super::FileTokenStore) -> anyhow::Result<crate::error::BoxError> {
    match store.load("chatgpt").await {
        Ok(record) => Err(anyhow::anyhow!(
            "loaded {:?} bytes",
            record.map(|record| record.len())
        )),
        Err(error) => Ok(error),
    }
}

/// A credential file behind a symlink is refused, naming the path, and the
/// file it points at is left as it was.
#[cfg(unix)]
#[tokio::test]
async fn file_store_refuses_a_symlink() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let dir = assert_fs::TempDir::new()?;
    let store = super::FileTokenStore::new(dir.path());
    let target = dir.path().join("elsewhere.json");
    std::fs::write(&target, "{\"a\":1}")?;
    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644))?;
    let path = store.path("chatgpt");
    std::os::unix::fs::symlink(&target, &path)?;
    let error = refusal(&store).await?;
    anyhow::ensure!(
        matches!(
            error.downcast_ref::<super::FileStoreError>(),
            Some(super::FileStoreError::NotAFile { path: named }) if *named == path
        ),
        "{error:?}"
    );
    anyhow::ensure!(error.to_string().contains(&path.display().to_string()));
    let mode = std::fs::metadata(&target)?.permissions().mode() & 0o777;
    anyhow::ensure!(mode == 0o644, "target mode {mode:o}");
    Ok(())
}

/// A credential file over 64 KiB is refused, naming the path.
#[tokio::test]
async fn file_store_refuses_a_file_over_64_kib() -> anyhow::Result<()> {
    let dir = assert_fs::TempDir::new()?;
    let store = super::FileTokenStore::new(dir.path());
    let path = store.path("chatgpt");
    std::fs::write(&path, "x".repeat(70 * 1024))?;
    let error = refusal(&store).await?;
    anyhow::ensure!(
        matches!(
            error.downcast_ref::<super::FileStoreError>(),
            Some(super::FileStoreError::TooLarge { path: named, limit: 65_536 }) if *named == path
        ),
        "{error:?}"
    );
    Ok(())
}
