#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures run programs and touch files of their own"
)]

use std::os::unix::fs::PermissionsExt;

use rig_harness_protocol::Home;
use rig_harness_protocol::launch::AbsPath;

use super::*;

/// The launch of the launcher at `launcher`, on a home of its own.
fn at(launcher: &Path) -> Result<Launch> {
    crate::test_launch(&Home::at("/home/user/.rig"), launcher)
}

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
    fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).ok();
}

fn words(command: &Command) -> Vec<String> {
    command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

#[test]
fn the_install_command_follows_the_detected_source() {
    let launch = at(Path::new("/opt/rig/bin/rig"));
    assert!(launch.is_ok());
    let Ok(launch) = launch else { return };
    let root = Path::new("/opt/rig");
    let registry = install_command(&launch, &RigSource::Registry, root);
    assert_eq!(registry.get_program(), "cargo");
    assert_eq!(
        words(&registry),
        [
            "install", "--force", "--locked", "rig-cli", "--root", "/opt/rig"
        ]
    );
    let checkout = RigSource::Local(PathBuf::from("/src/my rig"));
    let local = install_command(&launch, &checkout, root);
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
    let started_in = Path::new("/home/user/project");
    for (launcher, root) in [
        (".cargo/bin/rig", started_in.join(".cargo")),
        ("bin/rig", started_in.to_path_buf()),
    ] {
        let launch = Launch::from_parts(Path::new(launcher), false, [], Some(started_in), []);
        assert!(launch.is_ok(), "{launcher}");
        let Ok(launch) = launch else { return };
        let found = install_root(&launch.launcher).unwrap_or_default();
        assert_eq!(found, root, "{launcher}");
        let install = install_command(&launch, &RigSource::Registry, &found);
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
    let launch = at(&launcher);
    assert!(launch.is_ok());
    let Ok(launch) = launch else { return };
    let cargo = |body: &str| {
        script(&dir.join("cargo"), body);
        let mut install = install_command(&launch, &RigSource::Registry, &dir);
        let path = std::env::var_os("PATH").unwrap_or_default();
        let paths = std::iter::once(dir.clone()).chain(std::env::split_paths(&path));
        install.env("PATH", std::env::join_paths(paths).unwrap_or_default());
        install
    };
    let failed = install_and_build(&launch, cargo("echo 'error: offline' >&2; exit 101"));
    let kept = fs::read_to_string(&launcher).ok();
    let built = bin.join("rig.built");
    let built_after_failure = built.exists();
    // As cargo does: the new binary renamed over the old one.
    let replace = "printf '#!/bin/sh\\necho new > \"$0.built\"\\n' > \"$6/bin/.rig\"; \
                   chmod +x \"$6/bin/.rig\"; mv \"$6/bin/.rig\" \"$6/bin/rig\"";
    let updated = install_and_build(&launch, cargo(replace));
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
    let launch = at(&launcher);
    assert!(launch.is_ok());
    let Ok(launch) = launch else { return };
    let ran_in = |source: &RigSource| {
        let mut install = install_command(&launch, source, &dir);
        install.env("PATH", &path);
        let status = install_and_build(&launch, install);
        let ran_in = fs::read_to_string(dir.join("cargo.ran-in")).ok();
        (status, ran_in.map(|line| PathBuf::from(line.trim_end())))
    };
    let registry = ran_in(&RigSource::Registry);
    let local = ran_in(&RigSource::Local(checkout.clone()));
    let local_install = install_command(&launch, &RigSource::Local(checkout.clone()), &dir);
    let shown = shell_words(&local_install);
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
    // This process's launch, as `rig update` reads it, of the launcher in
    // `bin`.
    let launch = Launch::from_process().map_err(|failure| failure.to_string());
    let held = AbsPath::new(&launcher).map_err(|failure| failure.to_string());
    assert!(launch.is_ok() && held.is_ok(), "{launch:?} {held:?}");
    let (Ok(mut launch), Ok(held)) = (launch, held) else {
        return;
    };
    assert!(launch.secrets.contains_key(std::ffi::OsStr::new(PROBE)));
    launch.launcher = held;
    let mut install = install_command(&launch, &RigSource::Registry, &dir);
    let path = std::env::var_os("PATH").unwrap_or_default();
    let paths = std::iter::once(dir.clone()).chain(std::env::split_paths(&path));
    install.env("PATH", std::env::join_paths(paths).unwrap_or_default());
    let status = install_and_build(&launch, install);
    let cargo = fs::read_to_string(dir.join("cargo.seen")).ok();
    let build = fs::read_to_string(bin.join("rig.seen")).ok();
    fs::remove_dir_all(&dir).ok();
    assert_eq!(status, ExitCode::SUCCESS);
    assert_eq!(cargo.as_deref(), Some("unset\n"), "cargo install");
    assert_eq!(build.as_deref(), Some("unset\n"), "rig build");
}
