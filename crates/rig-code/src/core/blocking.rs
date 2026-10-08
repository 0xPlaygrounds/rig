//! Runs blocking tool work on its own thread, so no task pool thread
//! blocks. Plugin tools that do file, process or long CPU work use it too.

use futures::channel::oneshot;
use rig_core::tool::ToolExecutionError;

/// Run `work` on a new thread and await its answer. A panic in `work`
/// becomes an error. The thread runs in the calling tool call's
/// [working directory](super::workdir).
pub async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, ToolExecutionError> + Send + 'static,
) -> Result<T, ToolExecutionError> {
    let (sender, receiver) = oneshot::channel();
    let dir = super::workdir::current();
    std::thread::Builder::new()
        .name("rig-code-tool".to_owned())
        .spawn(move || {
            let _entered = super::workdir::enter(dir);
            sender.send(work()).ok();
        })
        .map_err(|error| ToolExecutionError::other(format!("could not start the tool: {error}")))?;
    receiver
        .await
        .map_err(|_| ToolExecutionError::other("the tool crashed before it answered"))?
}
