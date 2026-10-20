//! The project's instructions ([`AgentsMdPlugin`]): the `AGENTS.md` files
//! of the working directory and each directory above it up to the
//! repository root, and the user's own `RIG_HOME/AGENTS.md`, as one
//! section of every agent's system prompt.
//!
//! The files are read once, when the agent starts, so the prompt stays the
//! same for the whole session and the provider's prompt cache keeps it. An
//! edit takes effect on `/reload` or `/new`. Without this plugin the system
//! prompt names no instruction file.

use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_ecs::prompt::PromptSection;
use rig_harness::services::{ProjectRoot, RigHome};

/// The file name read in each directory.
pub const FILE_NAME: &str = "AGENTS.md";

/// Adds the `AGENTS.md` files to every agent's system prompt.
#[derive(Default)]
pub struct AgentsMdPlugin;

impl Plugin for AgentsMdPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>();
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let home = RigHome::of(world);
        let root = world.resource::<ProjectRoot>().0.clone();
        let text = section(&instruction_files(&root, home.root()));
        world.spawn((
            Name::new("prompt:project_instructions"),
            PromptSection::new(PromptSection::ORDER_PROJECT, "project_instructions", text),
        ));
    }
}

/// The `AGENTS.md` files that apply in `dir`, in the order the model reads
/// them: the user's in `home` first, then from the repository root (the
/// nearest directory above `dir` holding `.git`) down to `dir`, so the
/// nearest comes last. Outside a repository only `dir`'s counts. A file
/// reached twice, such as through a symlink, is listed once.
pub fn instruction_files(dir: &Path, home: &Path) -> Vec<PathBuf> {
    let repository = dir.ancestors().find(|place| place.join(".git").exists());
    let mut places: Vec<&Path> = match repository {
        Some(top) => dir
            .ancestors()
            .take_while(|place| *place != top)
            .chain([top])
            .collect(),
        None => vec![dir],
    };
    places.push(home);
    places.reverse();
    let mut files: Vec<PathBuf> = Vec::new();
    let mut seen: Vec<PathBuf> = Vec::new();
    for file in places.into_iter().map(|place| place.join(FILE_NAME)) {
        let Ok(real) = file.canonicalize() else {
            continue;
        };
        if real.is_file() && !seen.contains(&real) {
            seen.push(real);
            files.push(file);
        }
    }
    files
}

/// The prompt section's text: each readable file, with its path. Empty,
/// which leaves the section out, when there is none.
fn section(files: &[PathBuf]) -> String {
    let texts: Vec<String> = files
        .iter()
        .filter_map(|file| match std::fs::read_to_string(file) {
            Ok(text) if !text.trim().is_empty() => Some(format!(
                "<file path=\"{}\">\n{}\n</file>",
                file.display(),
                text.trim()
            )),
            Ok(_) => None,
            Err(error) => {
                warn!("{} not read: {error}", file.display());
                None
            }
        })
        .collect();
    if texts.is_empty() {
        return String::new();
    }
    format!(
        "Instructions from the user and the project, in AGENTS.md files: the user's own \
         first, then from the repository root down to the working directory. Follow them; \
         where two disagree, the later file wins.\n\n{}",
        texts.join("\n\n")
    )
}

#[cfg(test)]
mod tests;
