//! Per-invocation artifact links retained across errors, unwinding and cancellation.

use serde_json::{Value, json};
use std::{cell::RefCell, future::Future, path::Path};

tokio::task_local! {
    static ARTIFACTS: RefCell<Vec<Value>>;
}

pub(super) async fn capture<T>(future: impl Future<Output = T>) -> (T, Vec<Value>) {
    ARTIFACTS
        .scope(RefCell::default(), async move {
            let result = future.await;
            let artifacts = ARTIFACTS.with(|items| std::mem::take(&mut *items.borrow_mut()));
            (result, artifacts)
        })
        .await
}

pub(super) fn record(kind: &str, path: &Path) {
    let _ = ARTIFACTS.try_with(|items| {
        items.borrow_mut().push(json!({
            "kind":kind,"status":"preserved","path":path.strip_prefix(super::artifacts::root()).unwrap_or(path)
        }))
    });
}

pub(super) fn unavailable(kind: &str, reason: &str) {
    let _ = ARTIFACTS.try_with(|items| {
        items.borrow_mut().push(json!({
            "kind":kind,"status":"unavailable","reason":crate::cassettes::scrub_artifact(&json!(reason))
        }))
    });
}

#[cfg(test)]
mod tests;
