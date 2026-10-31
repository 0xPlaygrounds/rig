//! Runs blocking tool work on its own thread, so no async executor thread
//! blocks. Tools that do file, process or long CPU work use it.

use futures::channel::oneshot;
use rig_core::tool::ToolExecutionError;

use crate::process::Scope;

/// Run `work` on a new thread and await its answer. A panic in `work`
/// becomes an error.
///
/// The work is a [`Scope`]: dropping the future before the answer, as an
/// interrupted call does, stops every process the work started and refuses
/// those it would start after, while its thread runs on to its end.
pub async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, ToolExecutionError> + Send + 'static,
) -> Result<T, ToolExecutionError> {
    let scope = Scope::default();
    let (sender, receiver) = oneshot::channel();
    let inside = scope.clone();
    std::thread::Builder::new()
        .name("rig-tool".to_owned())
        .spawn(move || {
            sender.send(inside.enter(work)).ok();
        })
        .map_err(|error| ToolExecutionError::other(format!("could not start the tool: {error}")))?;
    let mut stop = StopOnDrop(Some(scope));
    let answer = receiver.await;
    stop.0 = None;
    answer.map_err(|_| ToolExecutionError::other("the tool crashed before it answered"))?
}

/// Stops the scope it holds when dropped.
struct StopOnDrop(Option<Scope>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        if let Some(scope) = &self.0 {
            scope.stop();
        }
    }
}
