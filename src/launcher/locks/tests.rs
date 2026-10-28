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

/// A wait for the build lock shorter than its quiet time says nothing; a
/// longer one says so once.
#[test]
fn a_short_wait_for_the_build_lock_is_quiet() {
    let path = std::env::temp_dir().join(format!("rig-build-lock-{}", std::process::id()));
    let held = || -> Result<File> {
        let held = lock_file(&path, true)?;
        held.lock()?;
        Ok(held)
    };
    let quiet = std::time::Duration::from_secs(30);
    let short = (|| -> Result<bool> {
        let other = held()?;
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(other);
        });
        let mut said = false;
        take(&lock_file(&path, true)?, quiet, || said = true)?;
        release.join().ok();
        Ok(said)
    })();
    // Released only once the wait said so.
    let long = (|| -> Result<bool> {
        let other = held()?;
        let mut said = false;
        take(&lock_file(&path, true)?, std::time::Duration::ZERO, || {
            said = true;
            drop(other);
        })?;
        Ok(said)
    })();
    fs::remove_file(&path).ok();
    assert_eq!((short.ok(), long.ok()), (Some(false), Some(true)));
}
