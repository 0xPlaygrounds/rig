//! The default plugins, one module each, as Bevy's `DefaultPlugins` are
//! modules of their crates. Each is listed in the default `plugins.toml`
//! and can be removed from it, and each uses only public kernel and
//! harness API, as a plugin in `RIG_HOME/plugins/` would: of this crate,
//! only the prelude, [`front`](crate::front) and its own modules.

pub mod activity;
pub mod basics;
pub mod compaction;
pub mod defaults;
pub mod diagnostics;
pub mod effect_log;
pub mod inspect;
pub mod login_chatgpt;
pub mod models;
pub mod print;
pub mod project_context;
pub mod reload_tool;
pub mod sessions;
pub mod subagents;
#[cfg(test)]
pub(crate) mod testing;
pub mod tools;
pub mod usage;
