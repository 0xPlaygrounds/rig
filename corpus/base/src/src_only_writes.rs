//! Writes only under `src/`. A Gate system refuses every `write` call whose `path` does not start
//! with `src/` (a text check: `src/../a` passes, which a real guard resolves first), before the
//! tool runs, with a reason the model reads. Its test shows how a plugin is
//! tested without a model: a small app of the kernel and the plugin, a call spawned for the
//! agent as its model would ask for it, and a check that the plugin's stage systems are ordered.

use rig_harness::prelude::*;

/// The tool whose calls are checked.
pub const TOOL: &str = "write";

/// The directory writes stay in.
pub const ROOT: &str = "src/";

/// Refuses a `write` outside `src/`.
#[derive(Default)]
pub struct SrcOnlyWritesPlugin;

impl Plugin for SrcOnlyWritesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, src_only.in_set(Stage::Gate));
    }
}

/// Refuses each `write` call about to start whose `path` is not under [`ROOT`].
fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let function = &request.call.function;
        let path = function
            .arguments
            .get("path")
            .and_then(|path| path.as_str());
        if function.name.as_str() == TOOL && !path.is_some_and(|path| path.starts_with(ROOT)) {
            let why = format!("write only under {ROOT}");
            decision.decide(Decider::plugin("src-only-writes"), Outcome::refused(why));
        }
    }
}

#[cfg(test)]
mod tests;
