//! The `writing-rig-plugins` skill, which rig writes to `RIG_HOME/skills`
//! once, when it sets up a new home: with the agent project that its first
//! build writes. After that the file is the user's, and rig never writes
//! it again, so an edited or deleted skill stays as the user left it.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::PathBuf;

use rig_harness_protocol::Home;

use super::super::Result;

/// The skill's text.
const SKILL: &str = include_str!("skill/SKILL.md");

/// Writes the agent project with `generate` and, when that made a new
/// home's first one, the skill unless a file is there. A skill that could
/// not be written is the warning returned, naming the file and why: the
/// skill is a help, so it never fails the build it comes with.
pub fn set_up(home: &Home, generate: impl FnOnce() -> Result<()>) -> Result<Option<String>> {
    let new = !home.project().exists();
    let generated = generate();
    let warning = (new && home.project().exists())
        .then(|| write(home))
        .flatten();
    generated.map(|()| warning)
}

/// Writes the skill unless a file is there, or the warning saying why it
/// could not.
fn write(home: &Home) -> Option<String> {
    let file = file(home);
    let written = (file.parent())
        .map_or(Ok(()), fs::create_dir_all)
        .and_then(|()| match fs::File::create_new(&file) {
            Err(there) if there.kind() == ErrorKind::AlreadyExists => Ok(()),
            new => new.and_then(|mut new| new.write_all(SKILL.as_bytes())),
        });
    let failure = written.err()?;
    Some(format!(
        "warning: could not write {}: {failure}",
        file.display()
    ))
}

/// Where the skill is written.
fn file(home: &Home) -> PathBuf {
    home.root().join("skills/writing-rig-plugins/SKILL.md")
}

#[cfg(test)]
mod tests;
