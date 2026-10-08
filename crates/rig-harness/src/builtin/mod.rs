//! The built-in tools, slash commands and subagents, registered through
//! the same `App` extension methods and core primitives a third-party
//! plugin uses.

pub mod commands;
pub mod login;
pub mod subagents;
pub mod tools;

pub use commands::BuiltinCommandsPlugin;
pub use login::LoginPlugin;
pub use subagents::SubagentsPlugin;
pub use tools::BuiltinToolsPlugin;
