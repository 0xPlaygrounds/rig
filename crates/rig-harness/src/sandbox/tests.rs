use std::path::{Path, PathBuf};
use std::process::Command;

use rig_protocol::{DirGrant, Grant};

use super::*;

fn dirs(path: &Path, write: bool) -> Grant {
    Grant {
        dirs: Some(vec![DirGrant {
            path: path.to_path_buf(),
            write,
        }]),
        net: true,
        ..Grant::default()
    }
}

#[test]
fn a_grant_over_rigs_run_directory_is_refused() {
    let home = Path::new("/home/u/.rig");
    let over = plan(
        &dirs(Path::new("/home/u"), true),
        &home.join("sessions/s"),
        home,
    );
    assert!(
        matches!(over, Err(SandboxError::Unavailable(_))),
        "{over:?}"
    );
    let beside = plan(
        &dirs(Path::new("/home/u/src"), true),
        &home.join("sessions/s"),
        home,
    );
    assert!(matches!(beside, Ok(Some(_))), "{beside:?}");
    assert!(matches!(
        plan(
            &Grant {
                net: true,
                ..Grant::default()
            },
            home,
            home
        ),
        Ok(None)
    ));
}

#[test]
fn a_confined_worker_gets_a_private_temporary_folder_not_the_shared_one() {
    // The shared temporary folder holds other sessions' scratch, and may hold
    // rig's home: a confined worker never reaches it.
    let home = Path::new("/home/u/.rig");
    let session = home.join("sessions/s");
    let planned = plan(&dirs(Path::new("/home/u/src"), true), &session, home)
        .ok()
        .flatten()
        .unwrap_or_default();
    assert!(planned.write.contains(&session.join("tmp")), "{planned:?}");
    let shared = std::env::temp_dir();
    assert!(
        !planned
            .read
            .iter()
            .chain(&planned.write)
            .any(|p| shared.starts_with(p) || p.starts_with(&shared)),
        "{planned:?}"
    );
}

#[test]
fn the_macos_profile_writes_only_where_planned_and_never_reads_run() {
    let plan = Plan {
        read: vec!["/usr".into()],
        write: vec!["/w".into()],
        connect: Vec::new(),
    };
    let profile = macos_profile(Some(&plan), Path::new("/h/.rig"));
    assert!(
        profile.contains("(deny file-write*)\n(allow file-write* (subpath \"/w\"))"),
        "{profile}"
    );
    assert!(
        profile.ends_with("(deny file-read* file-write* (subpath \"/h/.rig/run\"))\n"),
        "{profile}"
    );
}

/// The probe a confined child runs: what it could read and write, one
/// word per path, on stdout.
const PROBE: &str = "RIG_SANDBOX_PROBE";

#[test]
fn confined_probe() {
    let Some(root) = std::env::var_os(PROBE).map(PathBuf::from) else {
        return;
    };
    let grant = dirs(&root.join("work"), true);
    // A thread that was already running when the worker confined itself,
    // as Bevy's task pools are.
    let (go, wait) = std::sync::mpsc::channel::<()>();
    let early_root = root.clone();
    let early = std::thread::spawn(move || {
        wait.recv().ok();
        std::fs::read_to_string(early_root.join("outside/secret")).is_ok()
    });
    let applied = apply(&grant, &root.join("home/sessions/s"));
    go.send(()).ok();
    println!("thread outside/secret {}", early.join().unwrap_or(true));
    println!("applied {}", applied.is_ok());
    let read = |p: &str| std::fs::read_to_string(root.join(p)).is_ok();
    for p in [
        "work/own.txt",
        "outside/secret",
        "work/link",
        "home/run/token",
        "work/token",
    ] {
        println!("read {p} {}", read(p));
    }
    println!(
        "write work/new {}",
        std::fs::write(root.join("work/new"), "x").is_ok()
    );
    println!(
        "write outside/new {}",
        std::fs::write(root.join("outside/new"), "x").is_ok()
    );
    let connects = |p: &str| std::os::unix::net::UnixStream::connect(root.join(p)).is_ok();
    println!(
        "connect home/run/rig.sock {}",
        connects("home/run/rig.sock")
    );
    println!(
        "connect outside/other.sock {}",
        connects("outside/other.sock")
    );
    println!("read home/auth/key {}", read("home/auth/key"));
    println!("read home/defaults.json {}", read("home/defaults.json"));
    println!(
        "write home/defaults.json {}",
        std::fs::write(root.join("home/defaults.json"), "{}").is_ok()
    );
    // A command the confined worker runs, through its confine step (this
    // test binary standing in for the agent's).
    let exe = std::env::current_exe().unwrap_or_default();
    let session = root.join("home/sessions/s");
    let (prefix, env) = shell_launch(&grant, &session, &exe).unwrap_or_default();
    let command = |line: &str| {
        let mut argv: Vec<std::ffi::OsString> = prefix.clone();
        argv.extend(["sh".into(), "-c".into(), line.into()]);
        let mut cmd = Command::new(&exe);
        cmd.args([
            "--exact",
            "sandbox::tests::confine_step",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .envs(env.iter().cloned())
        .env(
            STEP_ARGV,
            argv.iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("\u{1f}"),
        )
        .current_dir(&root);
        cmd.output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("STEP-OK"))
            .unwrap_or(false)
    };
    println!(
        "command work/own.txt {}",
        command("cat work/own.txt >/dev/null && echo STEP-OK")
    );
    println!(
        "command home/auth/key {}",
        command("cat home/auth/key >/dev/null && echo STEP-OK")
    );
    println!(
        "command home/run/rig.sock {}",
        command(
            "python3 -c 'import socket; s=socket.socket(socket.AF_UNIX); s.connect(\"home/run/rig.sock\")' && echo STEP-OK"
        )
    );
    let shell = Command::new("sh")
        .args(["-c", "cat outside/secret"])
        .current_dir(&root)
        .output()
        .is_ok_and(|o| o.status.success());
    println!("shell outside/secret {shell}");
}

/// The command a confine step runs, its words joined by U+001F.
const STEP_ARGV: &str = "RIG_SANDBOX_STEP_ARGV";

/// The confine step, as the agent binary's `confine` role runs it.
#[test]
fn confine_step() {
    let Ok(argv) = std::env::var(STEP_ARGV) else {
        return;
    };
    let argv: Vec<std::ffi::OsString> = argv.split('\u{1f}').map(Into::into).collect();
    // The prefix ends with the agent binary; this test binary is it.
    let exe = std::env::current_exe().unwrap_or_default();
    let argv = argv
        .into_iter()
        .skip_while(|a| *a != exe.as_os_str())
        .skip(1)
        .collect();
    std::process::exit(i32::from(run_confined(argv)));
}

#[cfg(target_os = "linux")]
#[test]
fn a_landlocked_child_reads_only_its_folders_even_through_a_symlink() {
    let base = std::env::temp_dir().join(format!("rig-sandbox-{}", std::process::id()));
    // The confined child's temporary folder is writable, so the files it
    // must not reach are beside it, not in it.
    let root = base.join("root");
    for dir in ["work", "outside", "home/run", "home/sessions/s", "tmp"] {
        std::fs::create_dir_all(root.join(dir)).ok();
    }
    std::fs::write(root.join("work/own.txt"), "mine").ok();
    std::fs::write(root.join("outside/secret"), "secret").ok();
    std::fs::write(root.join("home/run/token"), "token").ok();
    std::os::unix::fs::symlink(root.join("outside/secret"), root.join("work/link")).ok();
    // The worker's own way to rigd, and a socket it must not reach.
    let _rigd = std::os::unix::net::UnixListener::bind(root.join("home/run/rig.sock"));
    let _other = std::os::unix::net::UnixListener::bind(root.join("outside/other.sock"));
    // Its sign-in, which its model calls read and refresh.
    std::fs::create_dir_all(root.join("home/auth")).ok();
    std::fs::write(root.join("home/auth/key"), "key").ok();
    // The model and reasoning a session starts with, read at its start.
    std::fs::write(root.join("home/defaults.json"), "{\"model\":\"vendor/model\"}").ok();
    std::os::unix::fs::symlink(root.join("home/run/token"), root.join("work/token")).ok();

    let exe = std::env::current_exe().unwrap_or_default();
    let out = Command::new(exe)
        .args([
            "--exact",
            "sandbox::tests::confined_probe",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(PROBE, &root)
        .env(crate::harness_protocol::env::HOME, root.join("home"))
        .env("TMPDIR", root.join("tmp"))
        .output();
    std::fs::remove_dir_all(&base).ok();
    let stdout = out
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    for line in [
        "applied true",
        "thread outside/secret false",
        "read work/own.txt true",
        "read outside/secret false",
        "read work/link false",
        "read home/run/token false",
        "read work/token false",
        "write work/new true",
        "write outside/new false",
        "shell outside/secret false",
        "connect home/run/rig.sock true",
        "connect outside/other.sock false",
        "read home/auth/key true",
        "read home/defaults.json true",
        "write home/defaults.json false",
        "command work/own.txt true",
        "command home/auth/key false",
        "command home/run/rig.sock false",
    ] {
        assert!(stdout.contains(line), "missing `{line}` in:\n{stdout}");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn a_shell_without_network_sees_only_loopback() {
    let Ok(prefix) = shell_prefix(&Grant {
        net: false,
        ..Grant::default()
    }) else {
        return; // no user namespaces on this machine: the grant is refused, as it should be
    };
    let mut command = Command::new(prefix.first().cloned().unwrap_or_default());
    command
        .args(prefix.iter().skip(1))
        .args(["sh", "-c", "cat /proc/net/dev"]);
    let out = command
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let devices: Vec<&str> = out
        .lines()
        .skip(2)
        .filter_map(|l| l.split(':').next())
        .map(str::trim)
        .collect();
    assert_eq!(devices, ["lo"], "{out}");
}

#[cfg(unix)]
#[test]
fn a_confined_workers_commands_lose_whatever_names_a_socket_or_a_credential() {
    use std::ffi::OsStr;
    let out = |name: &str, value: &str| reaches_out(OsStr::new(name), OsStr::new(value));
    for (name, value) in [
        ("ANY_TOOL_SOCKET", "/x"),
        ("AGENT_SOCK", "/x"),
        ("SOME_API_KEY", "k"),
        ("GH_TOKEN", "t"),
        ("DB_PASSWORD", "p"),
        ("XAUTHORITY", "/x"),
        ("REGISTRY_AUTH", "x"),
        ("NPM_AUTH_TOKEN", "x"),
        ("SESSION_BUS_ADDRESS", "unix:path=/run/bus"),
    ] {
        assert!(out(name, value), "{name}");
    }
    for (name, value) in [
        ("PATH", "/usr/bin:/bin"),
        ("HOME", "/home/u"),
        ("TERM", "xterm"),
        ("LANG", "C.UTF-8"),
        ("AUTHOR", "me"),
        ("GIT_AUTHOR_NAME", "me"),
        ("GIT_AUTHOR_EMAIL", "me@example.com"),
    ] {
        assert!(!out(name, value), "{name}");
    }
    // A variable whose value is a live socket's path, whatever its name.
    let dir = std::env::temp_dir().join(format!("rig-scrub-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let socket = dir.join("s");
    let _listener = std::os::unix::net::UnixListener::bind(&socket);
    let value = socket.to_string_lossy().into_owned();
    assert!(out("WHERE_IT_LISTENS", &value));
    assert!(out("WHERE_IT_LISTENS", &format!("{value}:0:1")));
    std::fs::remove_dir_all(&dir).ok();
}
