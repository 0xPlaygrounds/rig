use std::path::PathBuf;
use std::process::Command;

use super::*;
use crate::tests::view_app;

/// A directory of its own for this test, beside the test binary, so
/// nothing lands in the source tree.
fn scratch() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe
        .parent()
        .map(|deps| deps.join("rig-tui-run-in-terminal"))
        .unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

#[cfg(unix)]
#[test]
fn a_program_runs_with_the_terminal_and_its_exit_comes_back() {
    let (mut app, _) = view_app();
    let draft = scratch().join("draft.md");
    let mut editor = Command::new("sh");
    editor
        .arg("-c")
        .arg("printf 'from the editor' > \"$0\"")
        .arg(&draft);
    let edited = app.world_mut().spawn(RunInTerminal(editor)).id();
    let missing = app
        .world_mut()
        .spawn(RunInTerminal(Command::new("rig-tui-no-such-program")))
        .id();
    app.update();
    let exit = |entity| app.world().get::<TerminalExit>(entity).cloned();
    assert!(exit(edited).is_some_and(|exit| exit.success()));
    assert!(exit(missing).is_some_and(|exit| exit.0.is_err()));
    assert_eq!(
        std::fs::read_to_string(&draft).ok().as_deref(),
        Some("from the editor")
    );
    // Run once: another frame runs nothing again.
    std::fs::remove_file(&draft).ok();
    app.update();
    assert!(!draft.exists());
}
