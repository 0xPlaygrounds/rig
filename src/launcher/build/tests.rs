use std::os::unix::fs::PermissionsExt;

use super::*;

/// A home whose last build is the shell script `body`, and whose good
/// build is an older one.
fn built(name: &str, body: &str) -> Home {
    let root = std::env::temp_dir().join(format!("rig-describe-{name}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let artifact = home
        .target()
        .join("debug")
        .join(format!("{PACKAGE}{}", std::env::consts::EXE_SUFFIX));
    for (path, script) in [(&artifact, body), (&home.good(), "exit 0")] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(path, format!("#!/bin/sh\n{script}\n")).ok();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).ok();
    }
    home
}

#[test]
fn a_build_is_staged_with_its_report_and_one_that_cannot_describe_itself_is_rejected() {
    let report = "printf 'rig-describe 1\\ncommand\\tstats\\t-\\theadless\\tsession\\tCount\\n' \
                  > \"$RIG_DESCRIBE\"";
    let home = built("staged", report);
    let staged = stage(&home, &home.staged(), Staging::Always);
    let kept = good_report(&home, &home.staged());
    fs::remove_dir_all(home.root()).ok();
    assert!(staged.is_ok(), "{staged:?}");
    assert!(kept.is_some_and(|report| report.command("stats").is_some()));
    // Two plugins with one command: the describe exits 2, saying why.
    let clash = "echo 'rig: plugins `a` and `b` both add the command `stats`' >&2; exit 2";
    let home = built("clash", clash);
    let refused = stage(&home, &home.staged(), Staging::Always).map_err(|why| why.to_string());
    let rejected = fs::read_to_string(home.rejected()).ok();
    let (good, staged) = (home.good().exists(), home.staged().exists());
    let artifact = home.target().join("debug").join(PACKAGE);
    let stamp = stamp(&artifact).ok();
    fs::remove_dir_all(home.root()).ok();
    let refused = refused.err().unwrap_or_default();
    assert!(
        refused.contains("both add the command `stats`"),
        "{refused}"
    );
    assert!(good && !staged);
    assert_eq!(rejected, stamp);
}

#[test]
fn a_build_that_does_not_describe_itself_in_time_is_stopped() {
    let home = built("slow", "exec sleep 5");
    let artifact = home.target().join("debug").join(PACKAGE);
    let started = Instant::now();
    let child = child(&artifact).stderr(Stdio::piped()).spawn();
    let waited = child.map_err(Into::into).and_then(|child| {
        wait_described(
            child,
            &home.root().join("report"),
            Duration::from_millis(200),
        )
    });
    fs::remove_dir_all(home.root()).ok();
    let why = waited
        .map_err(|why| why.to_string())
        .err()
        .unwrap_or_default();
    assert!(why.contains("did not describe itself"), "{why}");
    assert!(started.elapsed() < Duration::from_secs(4));
}

/// The secret this file's tests run a copy of themselves with.
const PROBE: &str = "RIG_SECRET_PROBE";

#[test]
fn no_launcher_child_gets_a_secret() {
    if std::env::var_os(PROBE).is_none() {
        let name = "launcher::build::tests::no_launcher_child_gets_a_secret";
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
    let home = Home::at("home");
    for command in [cargo(&home.project()), child("stty")] {
        let removed = command
            .get_envs()
            .any(|(name, value)| name == PROBE && value.is_none());
        assert!(removed, "{command:?}");
    }
}
