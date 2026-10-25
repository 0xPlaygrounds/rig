//! FH-19: rich tool call rendering in the terminal view (`rig-tui`).
//!
//! The built-in tools' calls get looks of their own, in place of the
//! default plugins' ([`TOOLS`]): `edit` and `write` as a coloured,
//! highlighted diff, `shell` as `$ command` with its output as it comes,
//! then all of it and `exit N · 4.2s`, `read` as a line with the lines
//! read and the content highlighted, `search` as a count with the
//! matches. The terminal view draws a call through its look while the
//! model still streams its arguments. Any other tool keeps its own
//! renderer or the default look; the looks never change what is saved or
//! sent.

mod highlight;
mod looks;

use rig_harness::prelude::*;
use rig_harness::rig_tools::{Edit, Read, Search, Shell, Write};
use rig_tui::{AppToolRenderersExt, ToolRenderer};

/// The tools this plugin draws: the default plugins' looks of them are
/// turned off.
pub const TOOLS: [&str; 5] = [
    Edit::NAME,
    Write::NAME,
    Shell::NAME,
    Read::NAME,
    Search::NAME,
];

/// Draws the built-in tools' calls the way they read best.
#[derive(Default)]
pub struct RichToolRenderingPlugin;

impl Plugin for RichToolRenderingPlugin {
    fn build(&self, _: &mut App) {}

    fn finish(&self, app: &mut App) {
        // Every plugin's `build` ran: the default looks exist and are
        // turned off here, so their names are free.
        let world = app.world_mut();
        let mut renderers = world.query::<(Entity, &ToolRenderer)>();
        let taken: Vec<Entity> = renderers
            .iter(world)
            .filter(|(_, renderer)| TOOLS.contains(&renderer.tool.as_str()))
            .map(|(entity, _)| entity)
            .collect();
        for renderer in taken {
            world.entity_mut(renderer).insert(Disabled);
        }
        for tool in TOOLS {
            app.add_tool_renderer(tool, looks::render);
        }
    }
}

#[cfg(test)]
mod tests;
