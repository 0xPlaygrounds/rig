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
        "cd '/src/my rig' && cargo install --force --locked --path '/src/my rig/crates/rig-cli' \
         --root /opt/rig"
    );
    assert_eq!(registry.get_current_dir(), Some(root));
    assert_eq!(
        install_root(Path::new("/opt/rig/bin/rig")).ok().as_deref(),
        Some(root)
    );
    assert!(install_root(Path::new("/src/rig/target/debug/rig")).is_err());
}

/// A launcher path relative to where rig started, as macOS may give it,
/// still names its own install root once cargo starts somewhere else.
#[test]
fn a_relative_launcher_installs_into_its_own_root() {
    let started_in = std::env::current_dir().unwrap_or_default();
    assert!(started_in.is_absolute());
    for (launcher, root) in [
        (".cargo/bin/rig", started_in.join(".cargo")),
        ("bin/rig", started_in.clone()),
    ] {
        let found = install_root(Path::new(launcher)).unwrap_or_default();
        assert_eq!(found, root, "{launcher}");
        let install = install_command(&RigSource::Registry, &found);
        assert_eq!(
            install.get_current_dir(),
            Some(root.as_path()),
            "{launcher}"
        );
        let given = words(&install)
            .last()
            .map(PathBuf::from)
            .unwrap_or_default();
        assert_eq!(given, root, "{launcher}: --root");
    }
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

/// A directory rig starts in may pin an older toolchain in its
/// `rust-toolchain.toml`, which rustup applies to a cargo started there, so
/// the install never runs where rig was started: a checkout's own install
/// runs in the checkout, under its own pin, and crates.io's in the install
/// root. The command shown says where it runs.
#[test]
fn the_install_runs_away_from_the_directory_rig_started_in() {
    let dir = scratch("directory");
    // As `pwd -P` says it, through any symlink in the temporary directory.
    let dir = fs::canonicalize(&dir).unwrap_or(dir);
    let bin = dir.join("bin");
    let checkout = dir.join("my rig");
    fs::create_dir_all(&bin).ok();
    fs::create_dir_all(&checkout).ok();
    let launcher = bin.join("rig");
    script(&launcher, "true");
    script(&dir.join("cargo"), "pwd -P > \"$0.ran-in\"");
    let path = std::env::var_os("PATH").unwrap_or_default();
    let paths = std::iter::once(dir.clone()).chain(std::env::split_paths(&path));
    let path = std::env::join_paths(paths).unwrap_or_default();
    let ran_in = |source: &RigSource| {
        let mut install = install_command(source, &dir);
        install.env("PATH", &path);
        let status = install_and_build(install, &launcher);
        let ran_in = fs::read_to_string(dir.join("cargo.ran-in")).ok();
        (status, ran_in.map(|line| PathBuf::from(line.trim_end())))
    };
    let registry = ran_in(&RigSource::Registry);
    let local = ran_in(&RigSource::Local(checkout.clone()));
    let shown = shell_words(&install_command(&RigSource::Local(checkout.clone()), &dir));
    let started_in = std::env::current_dir().ok();
    fs::remove_dir_all(&dir).ok();
    assert_eq!(registry, (ExitCode::SUCCESS, Some(dir.clone())));
    assert_eq!(local, (ExitCode::SUCCESS, Some(checkout.clone())));
    assert_ne!(started_in, Some(dir.clone()));
    let checkout = shell_word(&checkout.to_string_lossy());
    assert!(
        shown.starts_with(&format!("cd {checkout} && cargo install ")),
        "{shown}"
    );
}

/// The secret this test runs a copy of itself with.
const PROBE: &str = "RIG_SECRET_PROBE";

/// A secret exported where `rig update` runs reaches neither its
/// `cargo install` nor the new launcher's `rig build`.
#[test]
fn an_update_passes_no_secret_on() {
    if std::env::var_os(PROBE).is_none() {
        let name = "update::tests::an_update_passes_no_secret_on";
        let output = std::env::current_exe().and_then(|test| {
            Command::new(test)
                .args(["--exact", name, "--test-threads", "1"])
                .env(PROBE, "hunter2")
                .output()
        });
        let stdout = output.map(|output| output.stdout).unwrap_or_default();
        let stdout = String::from_utf8_lossy(&stdout);
        assert!(
            stdout.contains("1 passed"),
            "the run with a secret failed: {stdout}"
        );
        return;
    }
    let dir = scratch("secret");
    let bin = dir.join("bin");
    fs::create_dir_all(&bin).ok();
    let launcher = bin.join("rig");
    let seen = format!("echo \"${{{PROBE}-unset}}\" > \"$0.seen\"");
    script(&launcher, &seen);
    script(&dir.join("cargo"), &seen);
    let mut install = install_command(&RigSource::Registry, &dir);
    let path = std::env::var_os("PATH").unwrap_or_default();
    let paths = std::iter::once(dir.clone()).chain(std::env::split_paths(&path));
    install.env("PATH", std::env::join_paths(paths).unwrap_or_default());
    let status = install_and_build(install, &launcher);
    let cargo = fs::read_to_string(dir.join("cargo.seen")).ok();
    let build = fs::read_to_string(bin.join("rig.seen")).ok();
    fs::remove_dir_all(&dir).ok();
    assert_eq!(status, ExitCode::SUCCESS);
    assert_eq!(cargo.as_deref(), Some("unset\n"), "cargo install");
    assert_eq!(build.as_deref(), Some("unset\n"), "rig build");
}

#[test]
fn a_reexec_resumes_a_saved_session_with_the_same_arguments() {
    let args = vec![OsString::from("--model"), OsString::from("ollama/qwen3")];
    let launcher = Path::new("/opt/rig/bin/rig");
    let home = Home::at("/home/user/.rig");
    let saved = Reexec::new(&home, launcher, Some("1-2"), args.clone(), None, None);
    assert_eq!(saved.args, ["--resume", "1-2", "--model", "ollama/qwen3"]);
    assert_eq!(
        Reexec::new(&home, launcher, None, args.clone(), None, None).args,
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
        let Ok(command) = reexec.command() else {
            return Some(Err("no command".to_owned()));
        };
        let value = command
            .get_envs()
            .find(|(name, _)| *name == LEFT)
            .map(|(_, value)| value.map(OsStr::to_os_string));
        value.map(|value| parse_left(value.as_deref()).map_err(|failure| failure.to_string()))
    };
    let launcher = Path::new("/opt/rig/bin/rig");
    let home = Home::at("/home/user/.rig");
    let switched = Reexec::new(&home, launcher, Some("3-4"), Vec::new(), None, left.clone());
    assert_eq!(passed(switched), Some(Ok(left)));
    let plain = Reexec::new(&home, launcher, Some("3-4"), Vec::new(), None, None);
    assert_eq!(passed(plain), Some(Ok(None)), "removed, not inherited");
    let cargo = super::super::build::child("cargo");
    assert!(
        cargo
            .get_envs()
            .any(|(name, value)| name == LEFT && value.is_none())
    );
    assert!(parse_left(Some(OsStr::new("not an id"))).is_err());
}

/// The new launcher starts in the session's directory, so a home given
/// as a relative path reaches it as the same absolute one.
#[test]
fn a_reexec_in_another_directory_keeps_the_home() {
    let home = Home::at("relative/.rig");
    let elsewhere = PathBuf::from("/elsewhere/project");
    let reexec = Reexec::new(
        &home,
        Path::new("/opt/rig/bin/rig"),
        None,
        Vec::new(),
        Some(elsewhere.clone()),
        None,
    );
    let command = reexec.command().map_err(|failure| failure.to_string());
    let passed = command.as_ref().map(|command| {
        let home = command
            .get_envs()
            .find(|(name, _)| *name == env::HOME)
            .and_then(|(_, value)| value.map(PathBuf::from));
        (home, command.get_current_dir().map(Path::to_path_buf))
    });
    let expected = (Some(home.root().to_path_buf()), Some(elsewhere));
    assert_eq!(passed, Ok(expected));
    assert!(home.root().is_absolute());
}

/// A relative `RIG_SOURCE` reaches the new launcher made absolute against
/// this launcher's directory. It runs a copy of itself with one set.
#[test]
fn a_reexec_in_another_directory_keeps_a_relative_source() {
    const SOURCE: &str = "relative/rig";
    if std::env::var_os(env::SOURCE).as_deref() != Some(std::ffi::OsStr::new(SOURCE)) {
        let name = "update::tests::a_reexec_in_another_directory_keeps_a_relative_source";
        let output = std::env::current_exe().and_then(|test| {
            Command::new(test)
                .args(["--exact", name, "--test-threads", "1"])
                .env(env::SOURCE, SOURCE)
                .output()
        });
        let ran = output.is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed")
        });
        assert!(ran, "the run with a relative RIG_SOURCE failed");
        return;
    }
    let reexec = Reexec::new(
        &Home::at("/home/user/.rig"),
        Path::new("/opt/rig/bin/rig"),
        None,
        Vec::new(),
        Some(PathBuf::from("/elsewhere/project")),
        None,
    );
    let command = reexec.command().map_err(|failure| failure.to_string());
    let source = command.as_ref().ok().and_then(|command| {
        command
            .get_envs()
            .find(|(name, _)| *name == env::SOURCE)
            .and_then(|(_, value)| value.map(PathBuf::from))
    });
    assert_eq!(
        source,
        std::env::current_dir().ok().map(|here| here.join(SOURCE))
    );
}
