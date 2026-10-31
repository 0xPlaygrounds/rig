//! An ACP (Agent Client Protocol) front for the rig agent: `rig acp` serves
//! ACP v1 on stdin and stdout, so an editor such as Zed uses rig's models
//! and sign-in, tools, sessions and slash commands. Each session runs in a
//! rig process of its own, in its directory; the MCP servers the editor
//! names go through rig-mcp-client for that session only.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_acp::AcpPlugin);
//! ```

#[cfg(target_family = "wasm")]
compile_error!("rig-acp runs sessions as processes: it needs a native target");

#[cfg(not(target_family = "wasm"))]
mod reply;
#[cfg(not(target_family = "wasm"))]
mod router;
#[cfg(not(target_family = "wasm"))]
mod session;
#[cfg(not(target_family = "wasm"))]
mod translate;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

#[cfg(not(target_family = "wasm"))]
use reply::{Reply, error, initialized};
#[cfg(not(target_family = "wasm"))]
use rig_harness::prelude::*;
use rig_harness::run::RunPlugin;
#[cfg(not(target_family = "wasm"))]
use rig_mcp_client::McpCorePlugin;

#[cfg(not(target_family = "wasm"))]
/// `rig acp`: ACP v1 on stdin and stdout, each session in a process of
/// its own (`rig acp-session`).
#[derive(Default)]
pub struct AcpPlugin;

#[cfg(not(target_family = "wasm"))]
impl Plugin for AcpPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<McpCorePlugin>() {
            app.add_plugins(McpCorePlugin);
        }
        if !app.is_plugin_added::<RunPlugin>() {
            app.add_plugins(RunPlugin);
        }
        app.add_front(Front::new("acp").sessionless().on_command(
            "acp",
            "",
            "Serve the Agent Client Protocol (ACP) on stdin and stdout, for an editor",
        ))
        .add_front(Front::new(crate::session::FRONT).on_command(
            crate::session::FRONT,
            "",
            "Run one ACP session; rig acp starts it",
        ));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front("acp") {
            crate::router::install(app);
        } else if app.runs_front(crate::session::FRONT) {
            crate::session::install(app);
        }
    }
}
