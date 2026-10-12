use std::os::unix::fs::PermissionsExt;

use super::*;

/// The secret this test runs a copy of itself with.
const PROBE: &str = "RIG_SECRET_PROBE";

#[test]
fn the_rebuild_gets_no_secret() {
    if std::env::var_os(PROBE).is_none() {
        let name = "rebuild::tests::the_rebuild_gets_no_secret";
        let output = std::env::current_exe().and_then(|test| {
            Command::new(test)
                .args(["--exact", name, "--test-threads", "1"])
                .env(PROBE, "hunter2")
                .output()
        });
        let ran = output.is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed")
        });
        assert!(ran, "the run with a secret failed");
        return;
    }
    // A launcher that writes the secret it sees, if any.
    let dir = std::env::temp_dir().join(format!("rig-rebuild-secret-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let launcher = dir.join("rig");
    let script = "#!/bin/sh\nprintf %s \"${RIG_SECRET_PROBE-none}\" > \"$0.seen\"\n";
    std::fs::write(&launcher, script).ok();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).ok();
    let build = ReloadBuild::start(launcher.as_os_str(), Entity::PLACEHOLDER, Wake::default());
    let exited = build.map(|mut build| {
        build.exited = build.child.wait().is_ok();
        build.exited
    });
    let seen = std::fs::read_to_string(dir.join("rig.seen")).ok();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!((exited.ok(), seen.as_deref()), (Some(true), Some("none")));
}
