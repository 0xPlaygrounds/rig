//! A working directory of an agent's own. An agent with a [`WorkDir`] runs
//! its tool calls there instead of in the process's working directory, so
//! agents of one process can each work in a directory of their own, as the
//! trials of an eval do. Its subagents inherit it.
//!
//! The directory travels with the call rather than with the process: each
//! tool call's future runs scoped to the agent's directory, which sets
//! it for the thread while the future is polled, and
//! [`blocking`](super::blocking::blocking) carries it to the thread it
//! starts. A tool reads it with [`current`] and makes its paths absolute
//! with [`resolve`]; an agent without one resolves against the process's
//! working directory, as before.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use serde::{Deserialize, Serialize};

use super::agent::Agent;
use super::save::ReflectSaved;
use super::subagents::SubagentOf;

/// The directory an agent's tool calls run in, absolute.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Clone, Debug, Serialize, Deserialize, Saved)]
pub struct WorkDir(pub PathBuf);

thread_local! {
    static CURRENT: RefCell<Option<Arc<Path>>> = const { RefCell::new(None) };
}

/// The working directory of the tool call running on this thread, when its
/// agent has a [`WorkDir`].
pub fn current() -> Option<Arc<Path>> {
    CURRENT.with(|current| current.borrow().clone())
}

/// `path` as a tool should open it: joined to the [`current`] directory
/// when it is relative and there is one, else unchanged.
pub fn resolve(path: &str) -> String {
    match current() {
        Some(dir) if Path::new(path).is_relative() => dir.join(path).to_string_lossy().into_owned(),
        _ => path.to_owned(),
    }
}

/// Sets `dir` as this thread's [`current`] directory until the guard drops,
/// which puts back the one before.
pub fn enter(dir: Option<Arc<Path>>) -> Entered {
    Entered(CURRENT.with(|current| current.replace(dir)))
}

/// Restores the thread's previous directory when dropped; from [`enter`].
pub struct Entered(Option<Arc<Path>>);

impl Drop for Entered {
    fn drop(&mut self) {
        let previous = self.0.take();
        CURRENT.with(|current| *current.borrow_mut() = previous);
    }
}

/// `work` with `dir` as the [`current`] directory whenever it is polled.
pub(crate) fn scoped<F: Future>(dir: Option<Arc<Path>>, work: F) -> Scoped<F> {
    Scoped {
        dir,
        work: Box::pin(work),
    }
}

/// A future polled in a working directory; from [`scoped`].
pub(crate) struct Scoped<F> {
    dir: Option<Arc<Path>>,
    work: Pin<Box<F>>,
}

impl<F: Future> Future for Scoped<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let _entered = enter(self.dir.clone());
        self.work.as_mut().poll(cx)
    }
}

/// A subagent works where the agent that started it works.
pub(crate) fn inherit_workdir(
    added: On<Add<Agent>>,
    agents: Query<(Option<&SubagentOf>, Has<WorkDir>)>,
    dirs: Query<&WorkDir>,
    mut commands: Commands,
) {
    let Ok((Some(&SubagentOf(parent)), false)) = agents.get(added.entity) else {
        return;
    };
    if let Ok(dir) = dirs.get(parent) {
        commands.entity(added.entity).insert(dir.clone());
    }
}
