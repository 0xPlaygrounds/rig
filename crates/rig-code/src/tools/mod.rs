//! The built-in tools: read, write, edit, shell and search. Each is a
//! rig-core `PortableTool` registered through [`AgentAppExt::add_tool`],
//! the same call a third-party plugin makes.

pub(crate) mod child;
mod files;
mod search;
mod shell;

use bevy::prelude::*;

use crate::core::{AgentAppExt, DataDir};

/// Adds the built-in tools.
#[derive(Default)]
pub struct BuiltinTools;

impl Plugin for BuiltinTools {
    fn build(&self, app: &mut App) {
        let scratch = app
            .world()
            .get_resource::<DataDir>()
            .map(|data| data.0.join("tmp"))
            .unwrap_or_else(std::env::temp_dir);
        app.add_tool(files::Read)
            .add_tool(files::Write)
            .add_tool(files::Edit)
            .add_tool(shell::Shell { scratch })
            .add_tool(search::Search);
    }
}

/// Model-visible output is cut to this many bytes.
const OUTPUT_LIMIT: usize = 50 * 1024;

/// `text`, cut at a character boundary to its first `OUTPUT_LIMIT` bytes,
/// with a note saying so.
fn truncate(text: &str) -> String {
    if text.len() <= OUTPUT_LIMIT {
        return text.to_owned();
    }
    let mut cut = OUTPUT_LIMIT;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let (head, _) = text.split_at(cut);
    format!("{head}\n[output cut at {OUTPUT_LIMIT} bytes]")
}
