use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;
use std::process::Command;

use super::command;

/// A fresh directory for one test, named `name`, beside the test binary.
#[cfg(unix)]
fn scratch(name: &str) -> PathBuf {
    let dir = rig_harness_test_support::scratch_home(&format!("corpus-editor-{name}"));
    let dir = dir.root().to_path_buf();
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Runs the editor `line` on `files` and says whether it exited with
/// success.
#[cfg(unix)]
fn edits(line: &str, files: &[&Path]) -> bool {
    command(Some(line), files)
        .status()
        .is_ok_and(|status| status.success())
}

/// The arguments `command` runs its program with.
fn args(command: &Command) -> Vec<String> {
    let args = command.get_args();
    args.map(|arg| arg.to_string_lossy().into_owned()).collect()
}

#[cfg(unix)]
#[test]
fn a_quoted_shell_command_edits_the_file() {
    let dir = scratch("quoted");
    let file = dir.join("draft.md");
    assert!(edits("sh -c 'echo MARKER >> \"$1\"' sh", &[&file]));
    let text = std::fs::read_to_string(&file).unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(text, "MARKER\n");
}

#[cfg(unix)]
#[test]
fn a_quoted_script_path_with_spaces_runs() {
    let root = scratch("spaces");
    let dir = root.join("my editor");
    std::fs::create_dir_all(&dir).ok();
    let script = dir.join("edit.sh");
    let body = "for file; do echo MARKER >> \"$file\"; done\n";
    std::fs::write(&script, body).ok();
    let line = format!("sh '{}'", script.display());
    assert!(edits(&line, &[&dir.join("a.md")]));
    let text = std::fs::read_to_string(dir.join("a.md")).unwrap_or_default();
    std::fs::remove_dir_all(&root).ok();
    assert_eq!(text, "MARKER\n");
}

#[cfg(unix)]
#[test]
fn every_file_reaches_the_editor_as_it_is() {
    let dir = scratch("files");
    let odd = dir.join("a $HOME `x` 'b'.md");
    let plain = dir.join("plain.md");
    assert!(edits("touch", &[&odd, &plain]));
    let made = odd.exists() && plain.exists();
    std::fs::remove_dir_all(&dir).ok();
    assert!(made);
}

#[cfg(unix)]
#[test]
fn no_editor_is_vi() {
    let file = Path::new("draft.md");
    for blank in [None, Some(""), Some("  ")] {
        let run = command(blank, &[file]);
        assert_eq!(run.get_program(), "sh");
        assert_eq!(
            args(&run),
            ["-c", "vi \"$@\"", "vi", "draft.md"],
            "{blank:?}"
        );
    }
}

#[cfg(windows)]
#[test]
fn windows_runs_the_line_through_cmd() {
    let file = Path::new(r"C:\my notes\draft.md");
    let run = command(Some(r#""C:\Program Files\ed.exe" --wait"#), &[file]);
    assert_eq!(run.get_program(), "cmd.exe");
    assert_eq!(
        args(&run),
        [
            "/d",
            "/v:off",
            "/s",
            "/c",
            r#"""C:\Program Files\ed.exe" --wait "C:\my notes\draft.md"""#
        ]
    );
}
