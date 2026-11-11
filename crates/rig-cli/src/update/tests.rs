use std::os::unix::fs::PermissionsExt;

use super::*;

/// A fresh directory of the test's own named `name`.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-update-{name}-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(&dir).ok();
    dir
}

/// Writes the shell script `body` to `path`, executable.
fn script(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/sh\n{body}\n")).ok();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).ok();
}

fn words(command: &Command) -> Vec<String> {
    command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

#[test]
fn the_install_command_follows_the_detected_source() {
    let root = Path::new("/opt/rig");
    let registry = install_command(&RigSource::Registry, root);
    assert_eq!(registry.get_program(), "cargo");
    assert_eq!(
        words(&registry),
        ["install", "--force", "rig-cli", "--root", "/opt/rig"]
    );
    let checkout = RigSource::Local(PathBuf::from("/src/my rig"));
    let local = install_command(&checkout, root);
    assert_eq!(
        words(&local),
        [
            "install",
            "--force",
            "--locked",
            "--path",
            "/src/my rig/crates/rig-cli",
            "--root",
            "/opt/rig"
        ]
    );
    assert_eq!(
        shell_words(&local),
        "cargo install --force --locked --path '/src/my rig/crates/rig-cli' --root /opt/rig"
    );
    assert_eq!(install_root(Path::new("/opt/rig/bin/rig")).ok(), Some(root));
    assert!(install_root(Path::new("/src/rig/target/debug/rig")).is_err());
}

/// A failed install fails `rig update` and leaves the launcher as it was,
/// unbuilt; one that replaces the launcher builds with the new one.
#[test]
fn a_failed_install_fails_and_leaves_the_launcher() {
    let dir = scratch("install");
    let bin = dir.join("bin");
    fs::create_dir_all(&bin).ok();
    let launcher = bin.join("rig");
    let old = "echo old > \"$0.built\"";
    script(&launcher, old);
    let cargo = |body: &str| {
        script(&dir.join("cargo"), body);
        let mut install = install_command(&RigSource::Registry, &dir);
        let path = std::env::var_os("PATH").unwrap_or_default();
        let paths = std::iter::once(dir.clone()).chain(std::env::split_paths(&path));
        install.env("PATH", std::env::join_paths(paths).unwrap_or_default());
        install
    };
    let failed = install_and_build(cargo("echo 'error: offline' >&2; exit 101"), &launcher);
    let kept = fs::read_to_string(&launcher).ok();
    let built = bin.join("rig.built");
    let built_after_failure = built.exists();
    // As cargo does: the new binary renamed over the old one.
    let replace = "printf '#!/bin/sh\\necho new > \"$0.built\"\\n' > \"$5/bin/.rig\"; \
                   chmod +x \"$5/bin/.rig\"; mv \"$5/bin/.rig\" \"$5/bin/rig\"";
    let updated = install_and_build(cargo(replace), &launcher);
    let new_build = fs::read_to_string(&built).ok();
    fs::remove_dir_all(&dir).ok();
    assert_eq!(failed, ExitCode::FAILURE);
    assert_eq!(kept, Some(format!("#!/bin/sh\n{old}\n")));
    assert!(!built_after_failure, "a failed install builds nothing");
    assert_eq!(updated, ExitCode::SUCCESS);
    assert_eq!(new_build.as_deref(), Some("new\n"));
}

#[test]
fn a_reexec_resumes_a_saved_session_with_the_same_arguments() {
    let args = vec![OsString::from("--model"), OsString::from("ollama/qwen3")];
    let launcher = Path::new("/opt/rig/bin/rig");
    let saved = Reexec::new(launcher, Some("1-2"), args.clone(), None, None);
    assert_eq!(saved.args, ["--resume", "1-2", "--model", "ollama/qwen3"]);
    assert_eq!(
        Reexec::new(launcher, None, args.clone(), None, None).args,
        args
    );
}

/// The new launcher reads the session left from the re-exec's
/// environment; one with no switch drops a value an earlier re-exec left,
/// and no other process the launcher starts gets it.
#[test]
fn a_reexec_tells_the_new_launcher_the_session_it_left() {
    let left: Option<SessionId> = "1-2".parse().ok();
    let passed = |reexec: Reexec| {
        let command = reexec.command();
        let value = command
            .get_envs()
            .find(|(name, _)| *name == LEFT)
            .map(|(_, value)| value.map(OsStr::to_os_string));
        value.map(|value| parse_left(value.as_deref()).map_err(|failure| failure.to_string()))
    };
    let launcher = Path::new("/opt/rig/bin/rig");
    let switched = Reexec::new(launcher, Some("3-4"), Vec::new(), None, left.clone());
    assert_eq!(passed(switched), Some(Ok(left)));
    let plain = Reexec::new(launcher, Some("3-4"), Vec::new(), None, None);
    assert_eq!(passed(plain), Some(Ok(None)), "removed, not inherited");
    let cargo = super::super::build::child("cargo");
    assert!(
        cargo
            .get_envs()
            .any(|(name, value)| name == LEFT && value.is_none())
    );
    assert!(parse_left(Some(OsStr::new("not an id"))).is_err());
}
