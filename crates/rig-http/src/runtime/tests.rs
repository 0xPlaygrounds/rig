use super::handle;

/// Outside a runtime, every caller gets the one fallback runtime, which
/// runs spawned work; inside one, the current runtime.
#[test]
fn the_runtime_outside_tokio_is_one_shared_fallback() {
    let first = handle().expect("the fallback starts");
    let second = handle().expect("the same fallback");
    let task = first.spawn(async {
        tokio::task::yield_now().await;
        7
    });
    assert_eq!(second.block_on(task).ok(), Some(7));
    assert_eq!(
        tokio::runtime::Handle::try_current()
            .ok()
            .map(|handle| handle.id()),
        None,
        "this test runs outside any runtime"
    );
    let inside = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");
    let current = inside.block_on(async { handle().map(|handle| handle.id()) });
    assert_eq!(current.ok(), Some(inside.handle().id()));
    assert_ne!(Some(first.id()), Some(inside.handle().id()));
}
