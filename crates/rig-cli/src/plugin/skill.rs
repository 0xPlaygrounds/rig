//! The `writing-rig-plugins` skill, which each build of the agent writes to
//! `RIG_HOME/skills`, where rig-basics' skills plugin lists it. It points
//! at the plugin guide and the plugin corpus of the rig crates this build
//! uses, which only the launcher knows, so it is rewritten whenever they
//! change and a stale copy never points at another version's guide.

use std::fs;
use std::path::{Path, PathBuf};

use rig_harness_protocol::Home;

use super::super::project::{RigSource, rig_version};
use super::super::{Result, VERSION};

/// The skill's text, with `__VERSION__`, `__GUIDE__` and `__CORPUS__` for
/// the rig version, the guide and the corpus.
const SKILL: &str = include_str!("skill/SKILL.md");

/// Writes the skill as [`write`] does, and returns a warning naming the
/// file and why when it could not: the skill is a help, so failing to
/// write it never fails the build it comes with.
pub fn refresh(home: &Home, source: &RigSource) -> Option<String> {
    let failure = write(home, source).err()?;
    Some(format!(
        "warning: could not write {}: {failure}",
        file(home).display()
    ))
}

/// Where the skill is written.
fn file(home: &Home) -> PathBuf {
    home.root().join("skills/writing-rig-plugins/SKILL.md")
}

/// Writes the skill to `RIG_HOME/skills/writing-rig-plugins/SKILL.md` for
/// the rig crates of `source`, unless it holds that text already.
fn write(home: &Home, source: &RigSource) -> Result<()> {
    let file = file(home);
    let text = text(source, cargo_home().as_deref());
    if fs::read_to_string(&file).is_ok_and(|old| old == text) {
        return Ok(());
    }
    if let Some(directory) = file.parent() {
        fs::create_dir_all(directory)?;
    }
    fs::write(&file, text)?;
    Ok(())
}

/// The skill for the rig crates of `source`. A checkout's guide and corpus
/// are its own files. A crates.io build's guide is rig-harness's
/// `PLUGINS.md` in cargo's registry sources under `cargo_home`, or on
/// docs.rs before cargo has fetched it; its corpus, which is not
/// published, is the release's on GitHub.
fn text(source: &RigSource, cargo_home: Option<&Path>) -> String {
    let (guide, corpus) = match source {
        RigSource::Local(checkout) => (
            shown(&checkout.join("crates/rig-harness/PLUGINS.md")),
            shown(&checkout.join("corpus")),
        ),
        RigSource::Registry => (
            cargo_home.and_then(registry_guide).map_or_else(
                || format!("https://docs.rs/rig-harness/{VERSION}/rig_harness/plugin_guide/"),
                |guide| shown(&guide),
            ),
            format!("https://github.com/0xPlaygrounds/rig/tree/v{VERSION}/corpus"),
        ),
    };
    SKILL
        .replace("__VERSION__", &rig_version(source))
        .replace("__GUIDE__", &guide)
        .replace("__CORPUS__", &corpus)
}

/// `path` without its `..` parts when it exists.
fn shown(path: &Path) -> String {
    let path = fs::canonicalize(path).unwrap_or_else(|_| path.to_owned());
    path.display().to_string()
}

/// Cargo's home, where it keeps the crates it downloads: `CARGO_HOME`, else
/// `~/.cargo`.
fn cargo_home() -> Option<PathBuf> {
    std::env::var_os("CARGO_HOME")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::home_dir().map(|home| home.join(".cargo")))
}

/// rig-harness's `PLUGINS.md` at this launcher's version in the registry
/// sources under `cargo_home`, when cargo has fetched it.
fn registry_guide(cargo_home: &Path) -> Option<PathBuf> {
    fs::read_dir(cargo_home.join("registry/src"))
        .ok()?
        .flatten()
        .map(|index| {
            index
                .path()
                .join(format!("rig-harness-{VERSION}/PLUGINS.md"))
        })
        .find(|guide| guide.is_file())
}

#[cfg(test)]
mod tests;
