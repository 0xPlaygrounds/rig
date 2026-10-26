//! FH-19: rich tool call rendering in the terminal view (`rig-tui`).
//!
//! The built-in tools' calls get looks of their own, in place of the
//! default plugins' ([`TOOLS`]): `edit` and `write` as a coloured,
//! highlighted diff, `shell` as `$ command` with its output as it comes,
//! then all of it and `exit N · 4.2s`, `read` as a line with the lines
//! read and the content highlighted, `search` as a count with the
//! matches. Its system runs after the default plugins' looks
//! ([`CodingToolLooks`]), so its lines are the ones drawn. The terminal view
//! draws a call through its look while the model still streams its
//! arguments. Any other tool keeps its own look or the default one; the
//! looks never change what is saved or sent.

mod highlight;
mod looks;

use rig_basics::coding_tools::CodingToolLooks;
use rig_harness::prelude::*;
use rig_harness::rig_tools::{Edit, Read, Search, Shell, Write};
use rig_tui::{RenderBatch, RenderSystems, TranscriptRender};

/// The tools this plugin draws, in place of the default plugins' looks.
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
    fn build(&self, app: &mut App) {
        app.add_systems(
            TranscriptRender,
            draw.after(CodingToolLooks).in_set(RenderSystems::Render),
        );
    }
}

/// Draws the calls of [`TOOLS`].
fn draw(mut batch: ResMut<RenderBatch>) {
    for tool in TOOLS {
        for (call, output) in batch.tools_mut(tool) {
            output.lines = looks::render(&call.view());
        }
    }
}

#[cfg(test)]
mod tests;
