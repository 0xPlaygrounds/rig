use super::*;
use rig_harness_protocol::SessionId;

/// The sweep removes the session a killed launcher left without a saved
/// conversation, and keeps a saved one, one a live launcher holds, and
/// one with no launcher lock, such as a fork being written.
#[test]
fn the_sweep_removes_the_unsaved_sessions_of_gone_launchers() {
    let root = std::env::temp_dir().join(format!("rig-sweep-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let left = (|| -> Result<_> {
        let session = || -> Result<(SessionId, File)> {
            let (id, _) = home.new_session()?;
            let lock = home.session(&id).hold()?.ok_or("held elsewhere")?;
            Ok((id, lock))
        };
        let (killed, lock) = session()?;
        drop(lock);
        let (saved, lock) = session()?;
        home.session(&saved).mark_saved()?;
        drop(lock);
        let (running, _held) = session()?;
        let (forking, _) = home.new_session()?;
        sweep(&home)?;
        Ok([killed, saved, running, forking].map(|id| home.session(&id).path().exists()))
    })();
    fs::remove_dir_all(&root).ok();
    assert_eq!(left.ok(), Some([false, true, true, true]));
}

/// The launcher's run directories live in `run/`, beside rigd's socket
/// and token, which rigd refuses to serve from unless `run/` is ours
/// alone: the launcher makes it 0700, and narrows one it finds wider.
#[cfg(unix)]
#[test]
fn the_run_directory_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("rig-run-mode-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let mode = |home: &Home| {
        fs::metadata(home.runs())
            .map(|metadata| metadata.permissions().mode() & 0o777)
            .ok()
    };
    let fresh = hold_run(&home).map(|_run| mode(&home)).ok().flatten();
    fs::set_permissions(home.runs(), fs::Permissions::from_mode(0o755)).ok();
    let narrowed = hold_run(&home).map(|_run| mode(&home)).ok().flatten();
    fs::remove_dir_all(&root).ok();
    assert_eq!((fresh, narrowed), (Some(0o700), Some(0o700)));
}
