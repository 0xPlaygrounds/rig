//! The built-in tools: read, write, edit, shell and search. Each is a
//! rig-core `PortableTool` registered through [`AgentAppExt::add_tool`],
//! the same call a third-party plugin makes.

mod child;
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

/// `text`, cut at a character boundary to its last `OUTPUT_LIMIT` bytes
/// when `keep_end`, or its first ones otherwise, with a note saying so.
fn truncate(text: &str, keep_end: bool) -> String {
    if text.len() <= OUTPUT_LIMIT {
        return text.to_owned();
    }
    let mut cut = if keep_end {
        text.len() - OUTPUT_LIMIT
    } else {
        OUTPUT_LIMIT
    };
    while !text.is_char_boundary(cut) {
        cut += 1;
    }
    let (head, tail) = text.split_at(cut);
    if keep_end {
        format!("[output cut to its last {OUTPUT_LIMIT} bytes]\n{tail}")
    } else {
        format!("{head}\n[output cut at {OUTPUT_LIMIT} bytes]")
    }
}
