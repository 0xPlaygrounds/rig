//! The rc snippets `rig shell-init` prints. Each keeps the shell's state
//! in a directory of its own (`RIG_SHELL_STATE_DIR`, else one per shell
//! process): `command`, `exit`, `cwd`, and for `rig-run` commands
//! `output`; `??` runs `rig shell-ask --state DIR` (`RIG_BIN` names
//! another `rig`) and puts `DIR/suggested` in the edit buffer.

/// The bash snippet.
pub const BASH: &str = include_str!("snippet/bash.sh");

/// The zsh snippet.
pub const ZSH: &str = include_str!("snippet/zsh.sh");

/// The fish snippet.
pub const FISH: &str = include_str!("snippet/fish.fish");

/// The snippet for `shell`, if it is one this plugin knows.
pub fn for_shell(shell: &str) -> Option<&'static str> {
    match shell {
        "bash" => Some(BASH),
        "zsh" => Some(ZSH),
        "fish" => Some(FISH),
        _ => None,
    }
}
