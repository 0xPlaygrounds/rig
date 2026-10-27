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
    for command in [cargo(&home), child("stty")] {
        let removed = command
            .get_envs()
            .any(|(name, value)| name == PROBE && value.is_none());
        assert!(removed, "{command:?}");
    }
}

/// What a [`BuildLog`] shows, kept for the test to read.
#[derive(Clone, Default)]
struct Shown(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);

impl Write for Shown {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Shown {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.borrow()).into_owned()
    }
}

/// A build as `output` shows it, on a terminal when `erase`, of a fake
/// cargo that writes `stderr`: what it showed, what it logged, and its
/// failure's report.
fn fake_build(
    name: &str,
    (output, erase): (BuildOutput, bool),
    stderr: &str,
) -> (String, String, String) {
    let root = std::env::temp_dir().join(format!("rig-build-log-{name}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let shown = Shown::default();
    let mut log = BuildLog::create(&home, output, Box::new(shown.clone()));
    log.erase = erase;
    let mut cargo = Command::new("sh");
    cargo
        .args(["-c", "printf \"$0\" >&2", stderr])
        .stderr(Stdio::piped());
    let ran = log.run(cargo).map(|status| status.success());
    let told = log.fail("cargo could not build the agent".to_owned());
    let logged = fs::read_to_string(home.build_log()).unwrap_or_default();
    fs::remove_dir_all(&root).ok();
    assert!(matches!(ran, Ok(true)), "{ran:?}");
    (shown.text(), logged, told.told(output))
}

#[test]
fn a_log_only_build_shows_one_line_while_cargo_works_and_its_errors_once_it_fails() {
    let cargo = "   Compiling a v1.0.0\n   Compiling b v1.0.0\nerror: boom\n";
    let (shown, logged, told) = fake_build("quiet", (BuildOutput::LogOnly, false), cargo);
    let lines: Vec<&str> = shown.lines().collect();
    let [line] = lines.as_slice() else {
        return assert_eq!(lines.len(), 1, "{shown:?}");
    };
    assert!(
        line.starts_with("rig: building the agent (output in "),
        "{line}"
    );
    for line in cargo.lines() {
        assert!(logged.contains(line), "{logged:?}");
    }
    // Nothing showed the errors, so the failure tells them, and the log.
    assert!(
        told.contains("error: boom") && told.contains("build.log"),
        "{told}"
    );
    // A build with nothing to do says nothing at all.
    let finished = "    Finished `dev` profile [unoptimized] target(s) in 0.20s\n";
    let (shown, logged, _) = fake_build("noop", (BuildOutput::LogOnly, false), finished);
    assert_eq!(shown, "");
    assert!(logged.contains("Finished"), "{logged:?}");
}

#[test]
fn a_live_build_shows_cargo_as_it_writes_erasing_each_redrawn_line_on_a_terminal() {
    // The bar's tail outlived a shorter line after it: `… in 4.57s)`. A
    // line ended by `\r\n` stays.
    let cargo = "error: boom\r\n    Building [=> ] 1/2: a(bin)\r    Finished in 1s\n";
    let erased = "error: boom\r\n    Building [=> ] 1/2: a(bin)\r\x1b[K    Finished in 1s\n";
    let (shown, _, told) = fake_build("live", (BuildOutput::Live, true), cargo);
    assert_eq!(shown, erased);
    // The terminal showed the errors; the failure is one line.
    assert_eq!(told.lines().count(), 1, "{told}");
    // Redirected, it is passed on as it came.
    let (shown, _, _) = fake_build("redirected", (BuildOutput::Live, false), cargo);
    assert_eq!(shown, cargo);
}

#[test]
fn a_crlf_split_between_two_reads_is_not_erased() {
    let home = Home::at(std::env::temp_dir().join("rig-build-log-split"));
    let shown = Shown::default();
    let mut log = BuildLog::create(&home, BuildOutput::Live, Box::new(shown.clone()));
    log.erase = true;
    for chunk in [&b"one\r"[..], b"\ntwo\r", b"three\n"] {
        log.show(chunk);
    }
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(shown.text(), "one\r\ntwo\r\x1b[Kthree\n");
}
