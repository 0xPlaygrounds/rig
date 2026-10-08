//! The built-in tools and slash commands, registered through the same
//! `App` extension methods a third-party plugin uses.

pub mod commands;
pub mod tools;

pub use commands::BuiltinCommandsPlugin;
pub use tools::BuiltinToolsPlugin;
