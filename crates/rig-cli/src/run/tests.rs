use super::*;

/// The launcher path the tests' runs pass on: no file, so never replaced.
const NO_LAUNCHER: &str = "/nonexistent/rig";

#[cfg(unix)]
#[test]
fn only_a_failure_says_the_agent_stopped_and_only_a_saved_session_is_offered_to_resume() {
    use std::os::unix::process::ExitStatusExt;

    let failed = ExitStatus::from_raw(1 << 8);
    let log = Some(Path::new("/home/.rig/sessions/s/agent.log"));
    let saved = stopped(failed, Some("rig --resume 1-2"), None, false, log);
    assert!(
        saved.is_some_and(|saved| saved.contains("`rig --resume 1-2` resumes")),
        "a failure offers to resume"
    );
    let expected = "The agent stopped (exit status: 1). Log: /home/.rig/sessions/s/agent.log";
    assert_eq!(
        stopped(failed, None, None, false, log).as_deref(),
        Some(expected)
    );
    let no_log = stopped(failed, None, None, false, None);
    assert_eq!(
        no_log.as_deref(),
        Some("The agent stopped (exit status: 1).")
    );
    let crashed = stopped(ExitStatus::from_raw(9), None, None, false, log);
    assert!(crashed.is_some_and(|said| said.starts_with("The agent stopped (signal: 9")));
    // A refusal is said once: again only where a restarted view kept the
    // screen, and never as the agent stopping.
    let refused = ExitStatus::from_raw(i32::from(REFUSED_EXIT_CODE) << 8);
    let refusal = Some("rig: plain.\n");
    assert_eq!(stopped(refused, None, refusal, false, log), None);
    assert_eq!(
        stopped(refused, None, refusal, true, log).as_deref(),
        Some("rig: plain.")
    );
    // A signal the agent was told of ends the run as a quit does.
    let signalled = ExitStatus::from_raw(i32::from(SIGNAL_EXIT_CODE) << 8);
    assert_eq!(
        stopped(signalled, Some("rig --resume 1-2"), None, false, log).as_deref(),
        Some("To resume this session: rig --resume 1-2")
    );
    assert_eq!(stopped(signalled, None, None, false, log), None);
}

/// The resume command for a session named `name`, beside a second saved
/// session named `other`, in a home of the test's own named `test`; and
/// the session's id.
fn resume_named(test: &str, name: &str, other: &str) -> Option<(Option<String>, SessionId)> {
    let root = std::env::temp_dir().join(format!("rig-resume-{test}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let named = |name: &str| -> Result<SessionId> {
        let (id, session) = home.new_session()?;
        session.mark_saved()?;
        // Written raw, as a name file from anywhere may hold it.
        fs::write(session.path().join("name"), name)?;
        Ok(id)
    };
    let made = named(name).and_then(|id| named(other).map(|_| id));
    let resumed = made.ok().map(|id| (resume_command(&home, &id, false), id));
    fs::remove_dir_all(&root).ok();
    resumed
}

#[test]
fn a_resume_command_names_the_session_quoted_or_by_its_id() {
    let quoted = [
        ("plain", "fix tests", "'fix tests'"),
        ("apostrophe", "it's done", r"'it'\''s done'"),
        (
            "metacharacters",
            "a $(b) `c` | d; e & *",
            "'a $(b) `c` | d; e & *'",
        ),
        ("trimmed", "  padded\n", "'padded'"),
    ];
    for (test, name, shown) in quoted {
        let resumed = resume_named(test, name, "other");
        let command = resumed.and_then(|(command, _)| command);
        assert_eq!(command, Some(format!("rig --resume {shown}")), "{name:?}");
    }
    let by_id = [
        ("duplicate", "same", "same"),
        ("dash", "-p", "other"),
        ("control", "a\tb", "other"),
        ("newline", "a\nb", "other"),
        ("bidi", "a\u{202e}b", "other"),
        ("tag", "a\u{e0041}b", "other"),
        ("line-separator", "a\u{2028}b", "other"),
        ("variation-selector", "a\u{fe0f}b", "other"),
        ("id-shaped", "2026-10-25", "other"),
    ];
    for (test, name, other) in by_id {
        let resumed = resume_named(test, name, other);
        let id = resumed.map(|(command, id)| (command, format!("rig --resume {id}")));
        assert!(
            id.as_ref()
                .is_some_and(|(command, id)| command.as_ref() == Some(id)),
            "{name:?}: {id:?}"
        );
    }
}

/// How an interactive run of a new session ends whose agent runs `agent`
/// and exits, and describes itself as adding nothing; and the session the
/// run ended in.
#[cfg(unix)]
fn interactive(test: &str, agent: &str, headless: bool) -> Option<(Ended, SessionId)> {
    let home = builds(&format!("interactive-{test}"), "", "");
    let ended = locks::hold_run(&home).and_then(|run| {
        // The good build only: a trial that never says it started is rolled back.
        fs::remove_file(home.staged())?;
        let agent = format!("#!/bin/sh\n[ -n \"$RIG_DESCRIBE\" ] && exit 0\n{agent}\n");
        fs::write(home.good(), agent)?;
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        let ended = launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &Invocation::default(),
            headless,
            claimed,
            running,
            None,
            None,
        )?;
        let last = fs::read_to_string(home.root().join("last"))?;
        Ok((ended, last.trim().parse()?))
    });
    fs::remove_dir_all(home.root()).ok();
    ended.ok()
}

/// A clean exit of a saved session, after `/new` switched to it, says how
/// to resume the session it ended in; an unsaved session, a headless run
/// and an agent that never said it started say nothing.
#[cfg(unix)]
#[test]
fn a_clean_exit_says_how_to_resume_the_session_it_ended_in() {
    let saved = "echo \"$RIG_SESSION\" > \"$RIG_HOME/last\"
touch \"$RIG_RUN/ready\"
if [ ! -e \"$RIG_HOME/switched\" ]; then
  touch \"$RIG_HOME/switched\"
  touch \"$RIG_HOME/sessions/$RIG_SESSION/saved\"
  : > \"$RIG_HOME/sessions/$RIG_SESSION/switch\"
  exit 75
fi
touch \"$RIG_HOME/sessions/$RIG_SESSION/saved\"";
    let ended = interactive("clean", saved, false);
    let said =
        ended.map(|(ended, id)| (ended, format!("To resume this session: rig --resume {id}")));
    assert!(
        said.as_ref()
            .is_some_and(|(ended, said)| ended.code == ExitCode::SUCCESS
                && ended.said.as_ref() == Some(said)),
        "{said:?}"
    );

    let quiet = |test: &str, agent: &str, headless: bool| {
        let ended = interactive(test, agent, headless).map(|(ended, _)| ended);
        assert_eq!(ended, Some(Ended::quiet(ExitCode::SUCCESS)), "{test}");
    };
    let mark = "echo \"$RIG_SESSION\" > \"$RIG_HOME/last\"";
    quiet(
        "unsaved",
        &format!("{mark}\ntouch \"$RIG_RUN/ready\""),
        false,
    );
    let saved_once =
        format!("{mark}\ntouch \"$RIG_RUN/ready\" \"$RIG_HOME/sessions/$RIG_SESSION/saved\"");
    quiet("headless", &saved_once, true);
    let never_started = format!("{mark}\ntouch \"$RIG_HOME/sessions/$RIG_SESSION/saved\"");
    quiet("never-started", &never_started, false);
}

/// A failed exit of a saved session says how to resume it only when its
/// agent said it started.
#[cfg(unix)]
#[test]
fn a_failed_exit_offers_the_resume_only_after_the_agent_started() {
    let mark = "echo \"$RIG_SESSION\" > \"$RIG_HOME/last\"
touch \"$RIG_HOME/sessions/$RIG_SESSION/saved\"";
    let offers = |test: &str, agent: &str| {
        let ended = interactive(test, &format!("{mark}\n{agent}\nexit 3"), false);
        ended.map(|(ended, id)| {
            let offered = ended
                .said
                .is_some_and(|said| said.contains(&format!("`rig --resume {id}`")));
            (ended.code, offered)
        })
    };
    let failed = ExitCode::from(3);
    assert_eq!(
        offers("failed-started", "touch \"$RIG_RUN/ready\""),
        Some((failed, true))
    );
    assert_eq!(offers("failed-never-started", ":"), Some((failed, false)));
}

/// The process groups an agent leaves running when it dies without stopping
/// them, such as on `SIGKILL`, are killed once it is gone; one it leaves
/// for a reload is not, nor a listed one whose leader lacks the mark of a
/// process the agent started (an id since given to another process).
/// Sessions start through Perl's `setsid`, which macOS has too.
#[cfg(unix)]
#[test]
fn groups_the_agent_left_are_killed_after_it_dies_and_kept_across_a_reload() {
    let notes = std::env::temp_dir().join(format!("rig-left-groups-{}", std::process::id()));
    fs::remove_dir_all(&notes).ok();
    fs::create_dir_all(&notes).ok();
    let agent = format!(
        "echo \"$RIG_SESSION\" > \"$RIG_HOME/last\"
touch \"$RIG_RUN/ready\"
cd '{}'
if [ ! -e marked ]; then
  perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' env {}=\"$RIG_RUN/groups\" \\
    sh -c 'sleep 30 & echo $! > child; wait' &
  echo $! > marked
  while [ ! -s child ]; do sleep 0.01; done
  perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' sleep 30 &
  echo $! > unmarked
  cat marked unmarked > \"$RIG_RUN/groups\"
  exit 75
fi
kill -0 \"$(cat marked)\" && touch kept
cat marked unmarked > \"$RIG_RUN/groups\"
kill -KILL $$",
        notes.display(),
        env::PROCESS_LEDGER,
    );
    let ended = interactive("left-groups", &agent, false);
    assert!(ended.is_some());
    let id = |name: &str| {
        let id = fs::read_to_string(notes.join(name)).ok();
        id.and_then(|id| id.trim().parse::<u32>().ok())
    };
    let (marked, started, unmarked) = (id("marked"), id("child"), id("unmarked"));
    assert!(marked.is_some() && started.is_some() && unmarked.is_some());
    assert!(notes.join("kept").exists(), "killed on the reload");
    // Dead once it is gone or a zombie waiting for its new parent.
    let alive = |pid: u32| {
        let state = child("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output();
        state.is_ok_and(|state| {
            let state = String::from_utf8_lossy(&state.stdout);
            !state.trim().is_empty() && !state.trim_start().starts_with('Z')
        })
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    // The whole group: the leader and the child it started.
    let any_alive = || marked.is_some_and(alive) || started.is_some_and(alive);
    while any_alive() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!any_alive());
    assert!(unmarked.is_some_and(alive));
    if let Some(unmarked) = unmarked {
        child("kill").arg(unmarked.to_string()).status().ok();
    }
    fs::remove_dir_all(&notes).ok();
}

/// The launcher puts the terminal back only after a front that took its
/// screen, even when the agent that took it reloaded into one that did
/// not and failed; not after one that never took it, such as `--json`
/// typed at a terminal.
#[cfg(unix)]
#[test]
fn the_terminal_is_put_back_only_after_a_front_took_its_screen() {
    let mark = "echo \"$RIG_SESSION\" > \"$RIG_HOME/last\"
touch \"$RIG_RUN/ready\" \"$RIG_HOME/sessions/$RIG_SESSION/saved\"";
    let reloaded = format!(
        "{mark}
if [ ! -e \"$RIG_HOME/reloaded\" ]; then
  touch \"$RIG_HOME/reloaded\" \"$RIG_RUN/screen\"
  exit 75
fi
exit 2"
    );
    let restored = |test: &str, agent: &str| {
        interactive(test, agent, false).map(|(ended, _)| (ended.code, ended.restore))
    };
    assert_eq!(
        restored("screen-reloaded", &reloaded),
        Some((ExitCode::from(2), true))
    );
    assert_eq!(
        restored("screen-lines", mark),
        Some((ExitCode::SUCCESS, false))
    );
    let quit = format!("{mark}\ntouch \"$RIG_RUN/screen\"");
    assert_eq!(
        restored("screen-quit", &quit),
        Some((ExitCode::SUCCESS, true))
    );
}

/// A home whose staged build is `trial` and whose good build is `good`:
/// shell scripts that add their name to the home's `ran` file and then run
/// their body, which may use the launcher's environment.
#[cfg(unix)]
fn builds(name: &str, trial: &str, good: &str) -> Home {
    use std::os::unix::fs::PermissionsExt;

    let root = std::env::temp_dir().join(format!("rig-launch-{name}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    fs::create_dir_all(home.bin()).ok();
    for (path, which, body) in [(home.staged(), "trial", trial), (home.good(), "good", good)] {
        let script = format!("#!/bin/sh\necho {which} >> \"$RIG_HOME/ran\"\n{body}\n");
        fs::write(&path, script).ok();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).ok();
    }
    home
}

/// Launches a new session of `home` as a one-shot run, and returns its exit
/// code, the builds that ran, in order, and the one that is now the good
/// build.
#[cfg(unix)]
fn one_shot(home: &Home) -> (Option<ExitCode>, String, String) {
    let print = Invocation {
        print: Some("hi".to_owned()),
        ..Invocation::default()
    };
    let code = locks::hold_run(home)
        .and_then(|run| {
            let (claimed, _) = claim(home, Start::New, None)?;
            let running = select(home, &claimed.id, run.dir(), None)?;
            launch(
                home,
                Path::new(NO_LAUNCHER),
                run.dir(),
                &print,
                true,
                claimed,
                running,
                None,
                None,
            )
        })
        .map(|ended| ended.code)
        .ok();
    let ran = fs::read_to_string(home.root().join("ran")).unwrap_or_default();
    let good = fs::read_to_string(home.good()).unwrap_or_default();
    let good = ["trial", "good"]
        .into_iter()
        .find(|which| good.contains(&format!("echo {which}")))
        .unwrap_or_default();
    fs::remove_dir_all(home.root()).ok();
    (code, ran, good.to_owned())
}

#[cfg(unix)]
#[test]
fn a_one_shot_run_is_never_run_again_on_the_previous_build() {
    let home = builds("early", "exit 3", "exit 0");
    let (code, ran, good) = one_shot(&home);
    assert_eq!(ran, "trial\n", "the previous build ran too");
    assert_eq!((code, good.as_str()), (Some(ExitCode::from(3)), "good"));
    // A build that said it started is kept, whatever its exit.
    let home = builds("ready", "touch \"$RIG_RUN/ready\"; exit 1", "exit 0");
    let (code, ran, good) = one_shot(&home);
    assert_eq!(ran, "trial\n");
    assert_eq!((code, good.as_str()), (Some(ExitCode::from(1)), "trial"));
}

/// A start whose settings name a plugin rig does not know, or a local
/// crate that is not there or whose `Cargo.toml` is not TOML, refuses,
/// naming the mistake and the fix, and runs no build: the previous one was
/// built from other settings.
#[cfg(unix)]
#[test]
fn an_unknown_plugin_refuses_the_start_instead_of_running_the_previous_build() {
    let plugin = "rig_basics::gone::GonePlugin";
    for (name, settings, said) in [
        (
            "unknown-plugin",
            format!("[plugins]\nenable = [\"{plugin}\"]\n"),
            format!("unknown plugin `{plugin}`"),
        ),
        (
            "missing-crate",
            "[[plugins.extra]]\ncrate = \"gone\"\npath = \"plugins/gone\"\nplugin = \"gone::GonePlugin\"\n"
                .to_owned(),
            "plugins/gone/Cargo.toml: No such file or directory".to_owned(),
        ),
        (
            "malformed-crate",
            "[[plugins.extra]]\ncrate = \"bad\"\npath = \"plugins/bad\"\nplugin = \"bad::BadPlugin\"\n"
                .to_owned(),
            "is not TOML: TOML parse error".to_owned(),
        ),
    ] {
        let home = builds(name, "exit 0", "exit 0");
        fs::write(home.settings(), settings).ok();
        fs::create_dir_all(home.root().join("plugins/bad")).ok();
        fs::write(home.root().join("plugins/bad/Cargo.toml"), "[package\n").ok();
        let started =
            run(&home, &["-p".to_owned(), "hi".to_owned()]).map_err(|why| why.to_string());
        let ran = fs::read_to_string(home.root().join("ran")).unwrap_or_default();
        fs::remove_dir_all(home.root()).ok();
        assert_eq!(ran, "", "a build ran for {name}");
        assert!(
            matches!(&started, Err(why) if why.contains(&said) && why.contains("take")),
            "{started:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn a_build_without_a_report_is_described_once_and_the_command_line_read_against_it() {
    // The scratch home a describe runs in is `run/<launcher>/describe/home`.
    let describe = "[ -n \"$RIG_DESCRIBE\" ] || exit 0
echo described >> \"$RIG_HOME/../../../../described\"
printf 'rig-describe 1\\ncommand\\tstats\\t-\\theadless\\tsession\\tCount\\n' > \"$RIG_DESCRIBE\"";
    let home = builds("report", describe, "exit 0");
    let run = RunDir::at(home.runs().join("report"));
    let session = SessionId::generate();
    let read = (|| -> Result<_> {
        let staged = select(&home, &session, &run, None)?;
        let again = build::described(&home, &staged.current.executable)?;
        // A build from before reports ignores the request: it has none.
        let old = select(&home, &session, &run, None)?;
        let first = staged.current.description.clone();
        Ok((first, again, old.current.description.clone()))
    })();
    let described = fs::read_to_string(home.root().join("described")).unwrap_or_default();
    fs::remove_dir_all(home.root()).ok();
    let (first, again, old) = read.unwrap_or_default();
    assert!(first.command("stats").is_some(), "{first:?}");
    assert_eq!((first.clone(), described.as_str()), (again, "described\n"));
    assert_eq!(old, AgentDescription::default());
    let unknown = first.invocation(&["statz"]).err().unwrap_or_default();
    assert!(
        unknown.ends_with("commands: build, update, plugin, help, help-agents, stats"),
        "{unknown}"
    );
}

/// A run that failed before its session saved anything, such as one
/// refused, leaves no session directory; its text log stays in the
/// launcher's run directory, where the launcher points.
#[cfg(unix)]
#[test]
fn a_failed_run_leaves_no_unsaved_session_but_its_log() {
    let failing = "touch \"$RIG_RUN/ready\"
echo refused > \"$RIG_HOME/sessions/$RIG_SESSION/agent.log\"
exit 2";
    let home = builds("unsaved", failing, "exit 0");
    let print = Invocation {
        print: Some("hi".to_owned()),
        ..Invocation::default()
    };
    let left = locks::hold_run(&home).and_then(|run| {
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        let session = home.session(&claimed.id);
        let code = launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &print,
            true,
            claimed,
            running,
            None,
            None,
        )?
        .code;
        let log = fs::read_to_string(run.dir().log()).ok();
        Ok((code, session.path().exists(), log))
    });
    fs::remove_dir_all(home.root()).ok();
    let left = left.ok();
    let log = Some("refused\n".to_owned());
    assert_eq!(left, Some((ExitCode::from(2), false, log)));
}

/// A refused run of a saved session, such as a `/resume` of one an older
/// rig saved, logged into the launcher's run directory: the launcher
/// names that log, not the session's, which earlier runs wrote.
#[cfg(unix)]
#[test]
fn a_refused_run_of_a_saved_session_names_the_log_of_the_run() {
    let root = std::env::temp_dir().join(format!("rig-refused-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let named = locks::hold_run(&home).and_then(|run| {
        let (id, session) = home.new_session()?;
        session.mark_saved()?;
        fs::write(session.log(), "earlier runs\n")?;
        let (claimed, _) = claim(&home, Start::Resume(id), None)?;
        fs::write(run.dir().refusal(), "rig: refused.\n")?;
        let named = claimed.leave(&home, run.dir(), false);
        let kept = fs::read_to_string(session.log())?;
        Ok((named == run.dir().log(), kept))
    });
    fs::remove_dir_all(&root).ok();
    assert_eq!(named.ok(), Some((true, "earlier runs\n".to_owned())));
}

/// A resumed session whose saved directory is gone runs where rig was
/// started, and its start notice names both.
#[cfg(unix)]
#[test]
fn a_resumed_session_whose_directory_is_gone_says_where_it_runs() {
    let root = std::env::temp_dir().join(format!("rig-gone-dir-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let (gone, here) = (root.join("gone"), root.join("here"));
    let said = locks::hold_run(&home).and_then(|_run| {
        let (id, session) = home.new_session()?;
        session.mark_saved()?;
        session.set_working_directory(&gone)?;
        fs::create_dir_all(&here)?;
        let (claimed, _) = claim(&home, Start::Resume(id), Some(&here))?;
        Ok((claimed.directory, claimed.moved))
    });
    fs::remove_dir_all(&root).ok();
    let moved = format!(
        "The session's directory {} is gone, so it runs in {}.",
        gone.display(),
        here.display()
    );
    assert_eq!(said.ok(), Some((Some(here), Some(moved))));
}

/// A `/resume` of a session whose run is refused, such as one an older rig
/// saved, goes back to the session it left, told why, instead of ending.
#[cfg(unix)]
#[test]
fn a_refused_switch_goes_back_to_the_session_it_left() {
    let home = builds("switch-back", "", "");
    let back = locks::hold_run(&home).and_then(|run| {
        let saved = || -> Result<SessionId> {
            let (id, session) = home.new_session()?;
            session.mark_saved()?;
            Ok(id)
        };
        let (left, refused) = (saved()?, saved()?);
        let agent = format!(
            "echo \"$RIG_SESSION $RIG_START $RIG_NOTICE\" >> \"$RIG_HOME/started\"
touch \"$RIG_RUN/ready\"
[ \"$RIG_SESSION\" = {refused} ] && echo 'rig: too old.' > \"$RIG_RUN/refusal\" && exit 2
[ -e \"$RIG_HOME/switched\" ] && exit 0
touch \"$RIG_HOME/switched\"
echo {refused} > \"$RIG_HOME/sessions/$RIG_SESSION/switch\"
exit 75"
        );
        for build in [home.staged(), home.good()] {
            fs::write(build, format!("#!/bin/sh\n{agent}\n"))?;
        }
        let (claimed, _) = claim(&home, Start::Resume(left.clone()), None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        let code = launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
            None,
            None,
        )?
        .code;
        let started = fs::read_to_string(home.root().join("started"))?;
        Ok((code, started, left, refused))
    });
    fs::remove_dir_all(home.root()).ok();
    let back = back.map_err(|failure| failure.to_string());
    assert!(back.is_ok(), "{back:?}");
    if let Ok((code, started, left, refused)) = back {
        assert_eq!(code, ExitCode::SUCCESS);
        let expected = format!(
            "{left} resumed \n{refused} resumed Resumed the session.\n\
             {left} resumed Could not switch sessions: too old.\n"
        );
        assert_eq!(started, expected);
    }
}

/// A refused switch that a signal to the launcher came with does not go
/// back: nothing starts after the signal. It signals its own process, so it
/// needs one of its own, as nextest gives each test.
#[cfg(unix)]
#[test]
fn a_refused_switch_after_a_signal_starts_nothing() {
    let home = builds("switch-signal", "", "");
    let back = locks::hold_run(&home).and_then(|run| {
        let saved = || -> Result<SessionId> {
            let (id, session) = home.new_session()?;
            session.mark_saved()?;
            Ok(id)
        };
        let (left, refused) = (saved()?, saved()?);
        let agent = format!(
            "echo \"$RIG_SESSION $RIG_START\" >> \"$RIG_HOME/started\"
touch \"$RIG_RUN/ready\"
[ \"$RIG_SESSION\" = {refused} ] && trap '' TERM && echo 'rig: too old.' > \"$RIG_RUN/refusal\" && kill -TERM $PPID && exit 2
echo {refused} > \"$RIG_HOME/sessions/$RIG_SESSION/switch\"
exit 75"
        );
        for build in [home.staged(), home.good()] {
            fs::write(build, format!("#!/bin/sh\n{agent}\n"))?;
        }
        let (claimed, _) = claim(&home, Start::Resume(left.clone()), None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        let ended = launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
            None,
            None,
        )?;
        let started = fs::read_to_string(home.root().join("started"))?;
        Ok((ended.code, started, left, refused))
    });
    fs::remove_dir_all(home.root()).ok();
    let back = back.map_err(|failure| failure.to_string());
    assert!(back.is_ok(), "{back:?}");
    if let Ok((code, started, left, refused)) = back {
        // The refused agent's end is the launcher's: no agent ran after it.
        assert_eq!(code, ExitCode::from(REFUSED_EXIT_CODE));
        assert_eq!(started, format!("{left} resumed\n{refused} resumed\n"));
    }
}

/// Each start says why it started: a new session, a `/reload` of it, a
/// `/new`, then the previous build after the new one stopped during
/// startup.
#[cfg(unix)]
#[test]
fn each_start_says_why_it_started() {
    let home = builds("reasons", "", "");
    let starts = locks::hold_run(&home).and_then(|run| {
        let agent = format!(
            "#!/bin/sh
echo \"$RIG_START\" >> \"$RIG_HOME/starts\"
case $(( $(wc -l < \"$RIG_HOME/starts\") )) in
1) touch \"$RIG_RUN/ready\"; exit 75 ;;
2) touch \"$RIG_RUN/ready\"; cp \"$0\" {staged}; : > \"$RIG_HOME/sessions/$RIG_SESSION/switch\"; exit 75 ;;
3) exit 1 ;;
esac
touch \"$RIG_RUN/ready\"",
            staged = home.staged().display(),
        );
        for build in [home.staged(), home.good()] {
            fs::write(build, &agent)?;
        }
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        let code = launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
            None,
            None,
        )?
        .code;
        Ok((code, fs::read_to_string(home.root().join("starts"))?))
    });
    fs::remove_dir_all(home.root()).ok();
    let starts = starts.map_err(|failure| failure.to_string());
    assert_eq!(
        starts,
        Ok((
            ExitCode::SUCCESS,
            "new\nreloaded\nnew\nrestarted\n".to_owned()
        ))
    );
}

/// A one-shot run whose agent reloads is restarted without the prompt it
/// answered, and run again on the previous build when the reload's new
/// build stops during startup, so the agent hears of the rollback.
#[cfg(unix)]
#[test]
fn a_one_shot_run_that_reloads_carries_on_without_its_prompt() {
    let home = builds("headless-reload", "", "");
    let starts = locks::hold_run(&home).and_then(|run| {
        let agent = format!(
            "#!/bin/sh
echo \"$RIG_START $*\" >> \"$RIG_HOME/starts\"
case $(( $(wc -l < \"$RIG_HOME/starts\") )) in
1) touch \"$RIG_RUN/ready\"; cp \"$0\" {staged}; exit 75 ;;
2) exit 1 ;;
esac
touch \"$RIG_RUN/ready\"",
            staged = home.staged().display(),
        );
        for build in [home.staged(), home.good()] {
            fs::write(build, &agent)?;
        }
        let print = Invocation {
            print: Some("hi".to_owned()),
            ..Invocation::default()
        };
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        let code = launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &print,
            true,
            claimed,
            running,
            None,
            None,
        )?
        .code;
        Ok((code, fs::read_to_string(home.root().join("starts"))?))
    });
    fs::remove_dir_all(home.root()).ok();
    let starts = starts.map_err(|failure| failure.to_string());
    let said = "new --print -- hi\nreloaded --print\nrestarted --print\n";
    assert_eq!(starts, Ok((ExitCode::SUCCESS, said.to_owned())));
}

/// A build that describes itself as adding the command `name`.
#[cfg(unix)]
fn describing(name: &str) -> String {
    format!(
        "[ -n \"$RIG_DESCRIBE\" ] || exit 0
printf 'rig-describe 1\\ncommand\\t{name}\\t-\\theadless\\tsession\\tCount\\n' > \"$RIG_DESCRIBE\""
    )
}

/// `rig help` lists what the build a new session starts with adds: the
/// staged one, newer than the good one that ran before.
#[cfg(unix)]
#[test]
fn help_reads_the_build_that_starts_next() {
    let home = builds("help", &describing("new"), &describing("old"));
    let kept = [home.staged(), home.good()].map(|binary| build::described(&home, &binary));
    let report = last_report(&home);
    fs::remove_dir_all(home.root()).ok();
    assert!(kept.iter().all(Result::is_ok));
    let report = report.unwrap_or_default();
    assert!(report.command("new").is_some(), "{report:?}");
}

#[test]
fn a_secret_is_read_byte_by_byte_and_ctrl_c_gives_none() {
    let read = |typed: &[u8]| read_secret(&mut &typed[..]).ok().flatten();
    assert_eq!(read(b"ab\x7fc\rleft over"), Some(b"ac".to_vec()));
    assert_eq!(read("pé\x7f".as_bytes()), Some(b"p".to_vec()));
    assert_eq!(read(b"ab\x03\r"), None);
}

/// A trial build stopped by a Ctrl-C the launcher heard before the trial's
/// first frame is not rejected: the build never got its chance. The trial
/// sends SIGINT to this test's process by its pid, which nextest gives the
/// test alone, and never says it is ready.
#[cfg(unix)]
#[test]
fn a_trial_stopped_by_a_heard_ctrl_c_is_not_rejected() {
    let interrupt = format!(
        "[ -n \"$RIG_DESCRIBE\" ] && exit 0; kill -INT {}; sleep 1; exit 130",
        std::process::id()
    );
    let home = builds("heard-ctrl-c", &interrupt, "exit 0");
    let print = Invocation {
        print: Some("hi".to_owned()),
        ..Invocation::default()
    };
    let ended = locks::hold_run(&home).and_then(|run| {
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &print,
            true,
            claimed,
            running,
            None,
            None,
        )
    });
    let rejected = home.rejected().exists();
    fs::remove_dir_all(home.root()).ok();
    assert!(ended.is_ok(), "{:?}", ended.err());
    assert_eq!(stop::heard(), Some(libc::SIGINT), "the launcher heard it");
    assert!(
        !rejected,
        "the interrupted trial is recorded in bin/rejected"
    );
}

/// Writes an agent's log holding a user message where the default store
/// keeps it, `logs/<agent>.jsonl`, as a session whose saved marker could
/// not be written has it.
fn log_a_conversation(session: &SessionDir) -> Result<()> {
    let logs = session.path().join("logs");
    fs::create_dir_all(&logs)?;
    let log = r#"{"seq":0,"n":3,"t":1700000000000,"type":"header","format":"rig-session/2","log":"0a589d65-24d0-42bc-ae72-da7e134a17a6","lineage":1}
{"seq":1,"n":4,"t":1700000000001,"type":"push","entries":[{"message":{"role":"user","content":[{"type":"text","text":"hi"}]},"id":0}]}
"#;
    fs::write(logs.join("0a589d65-24d0-42bc-ae72-da7e134a17a6.jsonl"), log)?;
    Ok(())
}

/// Neither the launcher's leave nor its startup sweep removes a session
/// whose agent logs hold a conversation, though it has no saved marker.
#[test]
fn a_session_with_a_logged_conversation_outlives_the_leave_and_the_sweep() {
    let root = std::env::temp_dir().join(format!("rig-logged-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let kept = locks::hold_run(&home).and_then(|run| {
        let (claimed, _) = claim(&home, Start::New, None)?;
        let left = home.session(&claimed.id);
        log_a_conversation(&left)?;
        claimed.leave(&home, run.dir(), true);
        let (id, _) = home.new_session()?;
        let swept = home.session(&id);
        drop(swept.hold()?.ok_or("held elsewhere")?);
        log_a_conversation(&swept)?;
        locks::sweep(&home)?;
        Ok((left.path().exists(), swept.path().exists()))
    });
    fs::remove_dir_all(&root).ok();
    assert_eq!(
        kept.ok(),
        Some((true, true)),
        "(kept by the leave, kept by the sweep)"
    );
}

/// The agent gets the version `rig --version` prints, so its screens show
/// the same.
#[cfg(unix)]
#[test]
fn the_agent_gets_the_launchers_version() {
    let home = builds(
        "version",
        "printf %s \"$RIG_VERSION\" > \"$RIG_HOME-seen\"",
        "exit 0",
    );
    let seen = PathBuf::from(format!("{}-seen", home.root().display()));
    one_shot(&home);
    let version = fs::read_to_string(&seen).unwrap_or_default();
    fs::remove_file(&seen).ok();
    assert_eq!(version, crate::LONG_VERSION);
    assert!(version.starts_with(crate::VERSION), "{version}");
}

/// A run whose agent reloads once, first asking to switch to `switch`
/// (`Some("")` for a new session, else the second session) and, when
/// `replace`, renaming a new file over its launcher.
#[cfg(unix)]
#[derive(Clone, Copy, Default)]
struct Reload<'a> {
    replace: bool,
    switch: Option<&'a str>,
    headless: bool,
    /// Whether the session it runs is saved.
    unsaved: bool,
    /// Whether another rig holds the second session.
    busy: bool,
}

/// How a run as `reload` says ends, with its launcher, its session, the
/// second saved session, and whether its session is still there.
#[cfg(unix)]
fn reloaded(test: &str, reload: Reload) -> Option<(Ended, PathBuf, SessionId, SessionId, bool)> {
    let home = builds(&format!("replaced-{test}"), "", "");
    let ran = locks::hold_run(&home).and_then(|run| {
        let session = |saved: bool| -> Result<SessionId> {
            let (id, session) = home.new_session()?;
            if saved {
                session.mark_saved()?;
            }
            Ok(id)
        };
        let (current, other) = (session(!reload.unsaved)?, session(true)?);
        let _busy = if reload.busy {
            home.session(&other).hold()?
        } else {
            None
        };
        let launcher = home.root().join("launcher").join("rig");
        fs::create_dir_all(home.root().join("launcher"))?;
        fs::write(&launcher, "old")?;
        let switch = reload.switch.map_or_else(String::new, |target| {
            let target = if target.is_empty() {
                String::new()
            } else {
                other.to_string()
            };
            format!("printf '{target}' > \"$RIG_HOME/sessions/$RIG_SESSION/switch\"")
        });
        let replace = if reload.replace {
            "echo new > \"$RIG_LAUNCHER.new\"; mv \"$RIG_LAUNCHER.new\" \"$RIG_LAUNCHER\""
        } else {
            ""
        };
        let agent = format!(
            "[ -e \"$RIG_HOME/reloaded\" ] && exit 0
touch \"$RIG_HOME/reloaded\"
{switch}
{replace}
exit 75"
        );
        fs::remove_file(home.staged())?;
        fs::write(home.good(), format!("#!/bin/sh\n{agent}\n"))?;
        // The directory the session runs in, which a new one runs in too.
        let here = Some(home.root());
        let (claimed, _) = claim(&home, Start::Resume(current.clone()), here)
            .or_else(|_| claim(&home, Start::New, here))?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        let current = claimed.id.clone();
        let ended = launch(
            &home,
            &launcher,
            run.dir(),
            &Invocation::default(),
            reload.headless,
            claimed,
            running,
            None,
            None,
        )?;
        let kept = home.session(&current).path().exists();
        Ok((ended, launcher, current, other, kept))
    });
    fs::remove_dir_all(home.root()).ok();
    ran.ok()
}

/// A view's reload after the launcher's file was replaced ends this
/// launcher's runs for the new launcher: on the session the agent switched
/// to, when it can run, or its own. A plain reload, and a headless run's,
/// restart the agent as before.
#[cfg(unix)]
#[test]
fn a_reload_after_the_launcher_was_replaced_restarts_on_the_new_one() {
    let replace = Reload {
        replace: true,
        ..Reload::default()
    };
    let reexec = |launcher: &Path, session: Option<&SessionId>, left: Option<SessionId>| {
        let here = launcher
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        let home = Home::at(here.clone().unwrap_or_default());
        Some(Reexec::new(
            &home,
            launcher,
            session.map(SessionId::as_str),
            Vec::new(),
            here,
            left,
        ))
    };
    let plain = reloaded("plain", Reload::default());
    assert!(
        plain.is_some_and(|(ended, ..)| ended.reexec.is_none() && ended.code == ExitCode::SUCCESS),
        "a plain reload restarts the agent"
    );
    let own = reloaded("own", replace);
    assert!(own.is_some_and(|(ended, launcher, current, ..)| {
        ended.reexec == reexec(&launcher, Some(&current), None) && ended.said.is_none()
    }));
    let other = Reload {
        switch: Some("other"),
        ..replace
    };
    let switched = reloaded("switch", other);
    assert!(
        switched.is_some_and(|(ended, launcher, current, other, _)| {
            ended.reexec == reexec(&launcher, Some(&other), Some(current))
        })
    );
    // The switch file outlives the unsaved session it is in.
    let unsaved = reloaded(
        "unsaved",
        Reload {
            unsaved: true,
            ..other
        },
    );
    assert!(
        unsaved.is_some_and(|(ended, launcher, current, other, kept)| {
            ended.reexec == reexec(&launcher, Some(&other), Some(current)) && !kept
        })
    );
    let busy = reloaded(
        "busy",
        Reload {
            busy: true,
            ..other
        },
    );
    assert!(busy.is_some_and(|(ended, launcher, current, other, _)| {
        ended.reexec == reexec(&launcher, Some(&current), None)
            && ended.said
                == Some(format!(
                    "Could not switch sessions: session {other} is open in another rig."
                ))
    }));
    let new = reloaded(
        "new",
        Reload {
            switch: Some(""),
            ..replace
        },
    );
    assert!(new.is_some_and(|(ended, launcher, current, ..)| {
        ended.reexec == reexec(&launcher, None, Some(current))
    }));
    let headless = reloaded(
        "headless",
        Reload {
            headless: true,
            ..replace
        },
    );
    assert!(headless.is_some_and(|(ended, _, _, _, kept)| {
        ended.reexec.is_none() && ended.code == ExitCode::SUCCESS && kept
    }));
}

/// A `/resume` after the launcher's file was replaced, of a session the
/// new launcher's agent refuses, such as one an older rig saved, goes back
/// to the session it left, told why, as it does without a new launcher.
#[cfg(unix)]
#[test]
fn a_refused_switch_on_the_new_launcher_goes_back_to_the_session_it_left() {
    let home = builds("replaced-switch-back", "", "");
    let back = locks::hold_run(&home).and_then(|run| {
        let saved = || -> Result<SessionId> {
            let (id, session) = home.new_session()?;
            session.mark_saved()?;
            Ok(id)
        };
        let (left, refused) = (saved()?, saved()?);
        let launcher = home.root().join("launcher").join("rig");
        fs::create_dir_all(home.root().join("launcher"))?;
        fs::write(&launcher, "old")?;
        let agent = format!(
            "echo \"$RIG_SESSION $RIG_START $RIG_NOTICE\" >> \"$RIG_HOME/started\"
touch \"$RIG_RUN/ready\"
[ \"$RIG_SESSION\" = {refused} ] && echo 'rig: too old.' > \"$RIG_RUN/refusal\" && exit 2
[ -e \"$RIG_HOME/switched\" ] && exit 0
touch \"$RIG_HOME/switched\"
echo {refused} > \"$RIG_HOME/sessions/$RIG_SESSION/switch\"
echo new > \"$RIG_LAUNCHER.new\"; mv \"$RIG_LAUNCHER.new\" \"$RIG_LAUNCHER\"
exit 75"
        );
        for build in [home.staged(), home.good()] {
            fs::write(build, format!("#!/bin/sh\n{agent}\n"))?;
        }
        let here = Some(home.root());
        let (claimed, _) = claim(&home, Start::Resume(left.clone()), here)?;
        let start = |claimed: Claimed, notice| {
            let running = select(&home, &claimed.id, run.dir(), None)?;
            launch(
                &home,
                &launcher,
                run.dir(),
                &Invocation::default(),
                false,
                claimed,
                running,
                notice,
                None,
            )
        };
        let reexec = start(claimed, None)?.reexec;
        let told = Reexec::new(
            &home,
            &launcher,
            Some(refused.as_str()),
            Vec::new(),
            here.map(Path::to_path_buf),
            Some(left.clone()),
        );
        // The new launcher, as `reexec` starts it: resuming the refused
        // session, told the one it left.
        let (mut claimed, notice) = claim(&home, Start::Resume(refused.clone()), here)?;
        claimed.left = Some(left.clone());
        let code = start(claimed, notice)?.code;
        let started = fs::read_to_string(home.root().join("started"))?;
        Ok((code, started, reexec == Some(told), left, refused))
    });
    fs::remove_dir_all(home.root()).ok();
    let back = back.map_err(|failure| failure.to_string());
    assert!(back.is_ok(), "{back:?}");
    if let Ok((code, started, told, left, refused)) = back {
        assert!(told, "the new launcher is told the session it left");
        assert_eq!(code, ExitCode::SUCCESS);
        let expected = format!(
            "{left} resumed \n{refused} resumed Resumed the session.\n\
             {left} resumed Could not switch sessions: too old.\n"
        );
        assert_eq!(started, expected);
    }
}

/// The warnings each start of a run of `home` got, one start per item,
/// from an agent that reloads once and then exits.
#[cfg(unix)]
fn warned(home: &Home) -> Result<Vec<Vec<String>>> {
    let agent = "#!/bin/sh
printf '%s\\n---\\n' \"$RIG_WARNING\" >> \"$RIG_HOME/warned\"
touch \"$RIG_RUN/ready\"
[ $(grep -c -- --- \"$RIG_HOME/warned\") -lt 2 ] && exit 75
exit 0";
    for build in [home.staged(), home.good()] {
        fs::write(build, agent)?;
    }
    let run = locks::hold_run(home)?;
    let (claimed, _) = claim(home, Start::New, None)?;
    let running = select(home, &claimed.id, run.dir(), None)?;
    launch(
        home,
        Path::new(NO_LAUNCHER),
        run.dir(),
        &Invocation::default(),
        false,
        claimed,
        running,
        None,
        None,
    )?;
    let warned = fs::read_to_string(home.root().join("warned"))?;
    Ok(warned
        .split_terminator("---\n")
        .map(|start| rig_harness_protocol::warnings(start.trim_end()))
        .collect())
}

/// A home rig 0.46 left, with `plugins.toml` and `defaults.json` that
/// `settings.toml` has no `[plugins]` or `[model]` for, is warned of once
/// per run, a warning per file naming the file, the table its settings
/// move to and MIGRATING.md, even when the home's path holds a line break,
/// and its files are left as they are. A home whose settings cover them,
/// even with empty tables, and a fresh home, are not warned of.
#[cfg(unix)]
#[test]
fn a_0_46_home_is_warned_of_once_per_run_and_left_as_it_is() {
    let plugins = "[[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_basics::prompt::DefaultPromptPlugin\"\n";
    let defaults = "{\n  \"model\": \"ollama/qwen3\",\n  \"effort\": null\n}";
    for name in ["legacy-old", "legacy-old\nline"] {
        let old = builds(name, "", "");
        let wrote = fs::write(old.root().join("plugins.toml"), plugins)
            .and_then(|()| fs::write(old.root().join("defaults.json"), defaults));
        let starts = wrote.map_err(Into::into).and_then(|()| warned(&old));
        let left = (
            fs::read_to_string(old.root().join("plugins.toml")).ok(),
            fs::read_to_string(old.root().join("defaults.json")).ok(),
        );
        fs::remove_dir_all(old.root()).ok();
        let starts = starts.map_err(|failure| failure.to_string());
        assert_eq!(starts.as_ref().map(Vec::len), Ok(2), "{starts:?}");
        let first = starts.as_ref().ok().and_then(|starts| starts.first());
        let warnings = first.cloned().unwrap_or_default();
        assert_eq!(warnings.len(), 2, "{first:?}");
        for (file, table) in [("plugins.toml", "[plugins]"), ("defaults.json", "[model]")] {
            let path = old.root().join(file).display().to_string();
            assert!(
                warnings.iter().any(|warning| {
                    warning.starts_with("rig no longer reads ")
                        && warning.contains(&path)
                        && warning.contains(table)
                        && warning.contains("MIGRATING.md")
                }),
                "no warning names {path}, {table} and MIGRATING.md: {warnings:?}"
            );
        }
        let again = starts.as_ref().ok().and_then(|starts| starts.get(1));
        assert_eq!(again, Some(&Vec::new()), "a reload warned again");
        assert_eq!(left, (Some(plugins.to_owned()), Some(defaults.to_owned())));
    }

    let covered = builds("legacy-covered", "", "");
    let settings = "[plugins]\ndisable = []\n\n[model]\ndefault = \"ollama/qwen3\"\n";
    let wrote = fs::write(covered.root().join("plugins.toml"), plugins)
        .and_then(|()| fs::write(covered.root().join("defaults.json"), defaults))
        .and_then(|()| fs::write(covered.settings(), settings));
    let starts = wrote.map_err(Into::into).and_then(|()| warned(&covered));
    fs::remove_dir_all(covered.root()).ok();
    let starts = starts.map_err(|failure| failure.to_string());
    assert_eq!(starts, Ok(vec![Vec::new(), Vec::new()]));

    let empty = builds("legacy-empty", "", "");
    let wrote = fs::write(empty.root().join("plugins.toml"), plugins)
        .and_then(|()| fs::write(empty.root().join("defaults.json"), defaults))
        .and_then(|()| fs::write(empty.settings(), "[plugins]\n\n[model]\n"));
    let starts = wrote.map_err(Into::into).and_then(|()| warned(&empty));
    fs::remove_dir_all(empty.root()).ok();
    let starts = starts.map_err(|failure| failure.to_string());
    assert_eq!(starts, Ok(vec![Vec::new(), Vec::new()]));

    let fresh = builds("legacy-fresh", "", "");
    let starts = warned(&fresh).map_err(|failure| failure.to_string());
    fs::remove_dir_all(fresh.root()).ok();
    assert_eq!(starts, Ok(vec![Vec::new(), Vec::new()]));
}

/// A run whose new build stops before it is ready still warns of a 0.46
/// home: the previous build it rolls back to gets the warning.
#[cfg(unix)]
#[test]
fn a_rolled_back_first_start_still_warns_of_a_0_46_home() {
    let home = builds(
        "legacy-rollback",
        "exit 1",
        "printf %s \"$RIG_WARNING\" > \"$RIG_HOME/warned\"; touch \"$RIG_RUN/ready\"",
    );
    let warned = fs::write(home.root().join("plugins.toml"), "[[plugin]]\n")
        .map_err(Into::into)
        .and_then(|()| {
            let run = locks::hold_run(&home)?;
            let (claimed, _) = claim(&home, Start::New, None)?;
            let running = select(&home, &claimed.id, run.dir(), None)?;
            launch(
                &home,
                Path::new(NO_LAUNCHER),
                run.dir(),
                &Invocation::default(),
                false,
                claimed,
                running,
                None,
                None,
            )?;
            Ok(fs::read_to_string(home.root().join("warned"))?)
        });
    let ran = fs::read_to_string(home.root().join("ran")).unwrap_or_default();
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(ran, "trial\ngood\n", "no rollback");
    let warned = warned.map_err(|failure: Box<dyn std::error::Error>| failure.to_string());
    assert!(
        warned
            .as_ref()
            .is_ok_and(|warned| warned.contains("plugins.toml")),
        "{warned:?}"
    );
}

/// Writes `body` as an executable shell script at `path`.
#[cfg(unix)]
fn script(path: &Path, body: &str) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    fs::write(path, format!("#!/bin/sh\n{body}\n"))?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

/// Two launchers on one home, their steps called in order: launcher A runs
/// the build whose description it read its command line and secrets
/// against, when B claims the shared staged build in between, and when B
/// makes another build the good one in between.
#[cfg(unix)]
#[test]
fn a_launcher_runs_the_build_it_read_its_command_line_against() {
    let home = builds("one-build", &describing("staged"), &describing("good"));
    let (a, b) = (SessionId::generate(), SessionId::generate());
    let [run_a, run_b] = ["a", "b"].map(|run| RunDir::at(home.runs().join(run)));
    let ran = (|| -> Result<_> {
        // B selects after A read the staged build.
        let a_staged = select(&home, &a, &run_a, None)?;
        let _b = select(&home, &b, &run_b, None)?;
        let read_staged = a_staged.current.description.clone();
        let ran_staged = build::described(&home, &a_staged.current.executable)?;
        // B's trial, which describes itself otherwise, becomes the good build.
        drop(a_staged);
        let a_good = select(&home, &a, &run_a, None)?;
        let newer = home.bin().join("newer");
        script(&newer, &describing("newer"))?;
        fs::rename(&newer, home.good())?;
        let read_good = a_good.current.description.clone();
        let ran_good = build::described(&home, &a_good.current.executable)?;
        Ok([(read_staged, ran_staged), (read_good, ran_good)])
    })();
    fs::remove_dir_all(home.root()).ok();
    let ran = ran.map_err(|failure| failure.to_string());
    let names = ran.map(|pairs| {
        pairs.map(|(read, ran)| {
            let name = |report: &AgentDescription| {
                ["staged", "good", "newer"]
                    .into_iter()
                    .find(|name| report.command(name).is_some())
            };
            (name(&read), name(&ran))
        })
    });
    assert_eq!(
        names,
        Ok([
            (Some("staged"), Some("staged")),
            (Some("good"), Some("good"))
        ])
    );
}

/// A reload whose new build stops during startup rolls back to the build
/// the session ran before it, though another session made a third build
/// the good one meanwhile.
#[cfg(unix)]
#[test]
fn a_rolled_back_reload_runs_the_sessions_own_previous_build() {
    let home = builds("own-rollback", "", "");
    let ran = locks::hold_run(&home).and_then(|run| {
        fs::remove_file(home.staged())?;
        let other = home.root().join("other");
        script(&other, "echo other >> \"$RIG_HOME/ran\"")?;
        let failing = home.root().join("failing");
        script(&failing, "echo failing >> \"$RIG_HOME/ran\"; exit 1")?;
        // The first start reloads onto a build that fails; meanwhile
        // another session's build becomes the good one.
        script(
            &home.good(),
            &format!(
                "[ -n \"$RIG_DESCRIBE\" ] && exit 0
echo own >> \"$RIG_HOME/ran\"
[ \"$(wc -l < \"$RIG_HOME/ran\")\" -gt 1 ] && exit 0
touch \"$RIG_RUN/ready\"
cp {other} \"$RIG_HOME/bin/other\" && mv \"$RIG_HOME/bin/other\" \"$RIG_HOME/bin/good\"
cp {failing} \"$RIG_HOME/bin/staged-$RIG_SESSION\"
exit 75",
                other = other.display(),
                failing = failing.display(),
            ),
        )?;
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
            None,
            None,
        )?;
        Ok(fs::read_to_string(home.root().join("ran"))?)
    });
    fs::remove_dir_all(home.root()).ok();
    let ran = ran.map_err(|failure| failure.to_string());
    assert_eq!(ran, Ok("own\nfailing\nown\n".to_owned()));
}

/// A trial that starts up after another session made a newer build the
/// good one leaves that build good.
#[cfg(unix)]
#[test]
fn a_trial_ready_after_a_newer_good_build_leaves_it_good() {
    let home = builds("stale-trial", "", "");
    let good = locks::hold_run(&home).and_then(|run| {
        let newer = home.root().join("newer");
        script(&newer, "echo newer")?;
        script(
            &home.staged(),
            &format!(
                "[ -n \"$RIG_DESCRIBE\" ] && exit 0
cp {newer} \"$RIG_HOME/bin/newer\" && mv \"$RIG_HOME/bin/newer\" \"$RIG_HOME/bin/good\"
touch \"$RIG_RUN/ready\"",
                newer = newer.display(),
            ),
        )?;
        let print = Invocation {
            print: Some("hi".to_owned()),
            ..Invocation::default()
        };
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(&home, &claimed.id, run.dir(), None)?;
        launch(
            &home,
            Path::new(NO_LAUNCHER),
            run.dir(),
            &print,
            true,
            claimed,
            running,
            None,
            None,
        )?;
        Ok(fs::read_to_string(home.good())?)
    });
    fs::remove_dir_all(home.root()).ok();
    let good = good.map_err(|failure| failure.to_string());
    assert_eq!(good, Ok("#!/bin/sh\necho newer\n".to_owned()));
}
