//! `rig hook install [--chain]` and `rig hook uninstall`: the
//! `prepare-commit-msg` hook that runs `rig commit-msg`, in the directory
//! git takes hooks from. Rig's hook is known by its [`MARKER`] line; a
//! chained hook is kept beside it as `prepare-commit-msg.rig-chained` and
//! put back when rig's goes.

use std::fs;
use std::path::{Path, PathBuf};

use rig_harness::rig_tools::process::Launcher;

use crate::commit_msg_hook::draft::git;

/// The line that says a hook is rig's.
pub const MARKER: &str = "# rig-commit-msg-hook: written by `rig hook install`";

/// The hook git runs.
const HOOK: &str = "prepare-commit-msg";

/// The name a chained hook is kept under.
const CHAINED: &str = "prepare-commit-msg.rig-chained";

/// Where the hooks of the repository at `root` go, and its
/// `prepare-commit-msg` hook and chained one.
struct Hooks {
    hook: PathBuf,
    chained: PathBuf,
}

impl Hooks {
    fn of(root: &Path, launcher: &Launcher) -> Result<Self, String> {
        let dir = git(root, launcher, &["rev-parse", "--git-path", "hooks"])?;
        let dir = root.join(dir.trim());
        Ok(Self {
            hook: dir.join(HOOK),
            chained: dir.join(CHAINED),
        })
    }
}

/// Runs `rig hook` with `words` in the repository at `root`, and says what
/// it did.
///
/// # Errors
///
/// Why it did nothing: a bad command line, no repository, a hook that is
/// not rig's, or a file it could not write.
pub fn run(words: &[String], root: &Path, launcher: &Launcher) -> Result<String, String> {
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["install"] => install(&Hooks::of(root, launcher)?, false),
        ["install", "--chain"] | ["--chain", "install"] => {
            install(&Hooks::of(root, launcher)?, true)
        }
        ["uninstall"] => uninstall(&Hooks::of(root, launcher)?),
        _ => Err("usage: rig hook install [--chain] | uninstall".to_owned()),
    }
}

/// Whether the file at `path` is rig's hook; `None` when there is none.
fn is_ours(path: &Path) -> Option<bool> {
    let text = fs::read(path).ok()?;
    Some(
        String::from_utf8_lossy(&text)
            .lines()
            .any(|line| line == MARKER),
    )
}

/// The hook's script: a chained hook first, whose failure stops the
/// commit as it would have alone, then `rig commit-msg`, whose failure
/// never does, and which never waits on the terminal.
fn script(chain: bool) -> String {
    let chained = match chain {
        true => format!(
            "chained=\"$(dirname \"$0\")/{CHAINED}\"\n\
             if [ -x \"$chained\" ]; then \"$chained\" \"$@\" || exit $?; fi\n"
        ),
        false => String::new(),
    };
    format!(
        "#!/bin/sh\n{MARKER}\n# `rig hook uninstall` removes it.\n{chained}\
         command -v rig >/dev/null 2>&1 || exit 0\n\
         rig commit-msg \"$@\" </dev/null || true\n\
         exit 0\n"
    )
}

/// Writes rig's hook, keeping a hook that is there with `chain`.
fn install(hooks: &Hooks, chain: bool) -> Result<String, String> {
    let chained = match is_ours(&hooks.hook) {
        None => false,
        // Written again, with the hook it chains if it chains one.
        Some(true) => hooks.chained.exists(),
        Some(false) if !chain => {
            return Err(format!(
                "{} is not rig's; `rig hook install --chain` keeps it and runs it first",
                hooks.hook.display()
            ));
        }
        Some(false) if hooks.chained.exists() => {
            return Err(format!(
                "{} is in the way of chaining the hook there",
                hooks.chained.display()
            ));
        }
        Some(false) => {
            fs::rename(&hooks.hook, &hooks.chained).map_err(|why| why.to_string())?;
            true
        }
    };
    if let Some(dir) = hooks.hook.parent() {
        fs::create_dir_all(dir).map_err(|why| why.to_string())?;
    }
    fs::write(&hooks.hook, script(chained)).map_err(|why| why.to_string())?;
    executable(&hooks.hook).map_err(|why| why.to_string())?;
    Ok(match chained {
        true => format!(
            "installed {}, running the old hook first",
            hooks.hook.display()
        ),
        false => format!("installed {}", hooks.hook.display()),
    })
}

/// Removes rig's hook, putting a chained one back; another hook stays.
fn uninstall(hooks: &Hooks) -> Result<String, String> {
    match is_ours(&hooks.hook) {
        None => Ok("no prepare-commit-msg hook to remove".to_owned()),
        Some(false) => Err(format!(
            "{} is not rig's, so it stays",
            hooks.hook.display()
        )),
        Some(true) => {
            fs::remove_file(&hooks.hook).map_err(|why| why.to_string())?;
            if hooks.chained.exists() {
                fs::rename(&hooks.chained, &hooks.hook).map_err(|why| why.to_string())?;
                return Ok(format!(
                    "removed rig's hook; {} is back",
                    hooks.hook.display()
                ));
            }
            Ok(format!("removed {}", hooks.hook.display()))
        }
    }
}

/// Makes the file at `path` executable by everyone who can read it.
#[cfg(unix)]
fn executable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
}

/// Hooks need no mode where there are no modes.
#[cfg(not(unix))]
fn executable(_: &Path) -> std::io::Result<()> {
    Ok(())
}
