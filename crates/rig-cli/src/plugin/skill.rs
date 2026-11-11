//! The `writing-rig-plugins` skill, which rig writes to `RIG_HOME/skills`
//! once, when it sets up a new home: one with no build yet. After that the
//! file is the user's, and rig never writes it again, so an edited or
//! deleted skill stays as the user left it.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::PathBuf;

use rig_harness_protocol::Home;

/// The skill's text.
const SKILL: &str = include_str!("skill/SKILL.md");

/// Writes the skill when `home` has no build yet and no file is there, and
/// returns a warning naming the file and why when it could not: the skill
/// is a help, so failing to write it never fails the build it comes with.
pub fn set_up(home: &Home) -> Option<String> {
    if home.good().exists() || home.staged().exists() {
        return None;
    }
    let file = file(home);
    let written = (file.parent())
        .map_or(Ok(()), fs::create_dir_all)
        .and_then(|()| fs::File::create_new(&file))
        .and_then(|mut new| new.write_all(SKILL.as_bytes()));
    match written {
        Err(failure) if failure.kind() != ErrorKind::AlreadyExists => Some(format!(
            "warning: could not write {}: {failure}",
            file.display()
        )),
        _ => None,
    }
}

/// Where the skill is written.
fn file(home: &Home) -> PathBuf {
    home.root().join("skills/writing-rig-plugins/SKILL.md")
}

#[cfg(test)]
mod tests;
