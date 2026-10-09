//! The built-in tools and slash commands and sign-in, registered through
//! the same `App` extension methods and core primitives a third-party
//! plugin uses.

pub mod commands;
pub mod login;
pub mod tools;

pub use commands::BuiltinCommandsPlugin;
pub use login::LoginPlugin;
pub use rig_ecs::subagents::SubagentsPlugin;
pub use tools::BuiltinToolsPlugin;
