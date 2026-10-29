//! An ACP front for the Rig agent, with a process for each session.
//!
//! ```no_run
//! use rig_acp::AcpPlugin;
//! use rig_harness::prelude::*;
//! let mut app = App::new();
//! app.add_plugins(AcpPlugin);
//! ```

#[cfg(target_family = "wasm")]
compile_error!("rig-acp requires a native target");

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(not(target_family = "wasm"))]
pub use native::AcpPlugin;
