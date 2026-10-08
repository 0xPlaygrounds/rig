//! Runs blocking tool work on its own thread.

use futures::channel::oneshot;
use rig_core::tool::ToolExecutionError;

/// Run `work` on a new thread and await its answer. A panic in `work`
/// becomes an error.
pub(super) async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, ToolExecutionError> + Send + 'static,
) -> Result<T, ToolExecutionError> {
    let (sender, receiver) = oneshot::channel();
    std::thread::Builder::new()
        .name("rig-code-tool".to_owned())
        .spawn(move || {
            sender.send(work()).ok();
        })
        .map_err(|error| ToolExecutionError::other(format!("could not start the tool: {error}")))?;
    receiver
        .await
        .map_err(|_| ToolExecutionError::other("the tool crashed before it answered"))?
}
