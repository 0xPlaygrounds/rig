#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures run programs and touch files of their own"
)]

use rig_harness_protocol::fake_agent::{self, Step};

use super::*;
use crate::test_launch as at;

/// A step of the test's own shell commands.
fn sh(commands: &str) -> Step {
    Step::Shell(commands.to_owned())
}

/// Notes the session the agent runs in `RIG_HOME/last`.
const MARK: &str = "echo \"$RIG_SESSION\" > \"$RIG_HOME/last\"";

/// Marks the agent's session saved.
const SAVE: &str = "touch \"$RIG_HOME/sessions/$RIG_SESSION/saved\"";

/// The launcher path the tests' runs pass on: no file, so never replaced.
const NO_LAUNCHER: &str = "/nonexistent/rig";

#[cfg(unix)]
#[test]
fn only_a_failure_says_the_agent_stopped_and_only_a_saved_session_is_offered_to_resume() {
    use std::os::unix::process::ExitStatusExt;

    let status = |code: i32| ExitStatus::from_raw(code << 8);
    let failed = AgentEnd::of_status(status(1));
    let log = Some(Path::new("/home/.rig/sessions/s/agent.log"));
    let saved = stopped(&failed, status(1), Some("rig --resume 1-2"), false, log);
    assert!(
        saved.is_some_and(|saved| saved.contains("`rig --resume 1-2` resumes")),
        "a failure offers to resume"
    );
    let expected = "The agent stopped (exit status: 1). Log: /home/.rig/sessions/s/agent.log";
    assert_eq!(
        stopped(&failed, status(1), None, false, log).as_deref(),
        Some(expected)
    );
    let no_log = stopped(&failed, status(1), None, false, None);
    assert_eq!(
        no_log.as_deref(),
        Some("The agent stopped (exit status: 1).")
    );
    let killed = ExitStatus::from_raw(9);
    let crashed = stopped(&AgentEnd::of_status(killed), killed, None, false, log);
    assert!(crashed.is_some_and(|said| said.starts_with("The agent stopped (signal: 9")));
    // A refusal is said once: again only where a restarted view kept the
    // screen, and never as the agent stopping.
    let refused = AgentEnd::Refused("plain.".to_owned());
    let code = status(i32::from(REFUSED_EXIT_CODE));
    assert_eq!(stopped(&refused, code, None, false, log), None);
    assert_eq!(
        stopped(&refused, code, None, true, log).as_deref(),
        Some("rig: plain.")
    );
    // A signal the agent was told of ends the run as a quit does.
    let signalled = status(i32::from(SIGNAL_EXIT_CODE));
    assert_eq!(
        stopped(
            &AgentEnd::Signal,
            signalled,
            Some("rig --resume 1-2"),
            false,
            log
        )
        .as_deref(),
        Some("To resume this session: rig --resume 1-2")
    );
    assert_eq!(
        stopped(&AgentEnd::Signal, signalled, None, false, log),
        None
    );
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

/// How an interactive run of a new session ends whose agent takes the
/// steps `starts` (one list per start, [`fake_agent::script`]); and the
/// session the run ended in.
#[cfg(unix)]
fn interactive(test: &str, starts: &[&[Step]], headless: bool) -> Option<(Ended, SessionId)> {
    let home = builds(&format!("interactive-{test}"), "", "");
    let ended = locks::hold_run(&home).and_then(|run| {
        // The good build only: a trial that never says it started is rolled back.
        fs::remove_file(home.staged())?;
        fs::write(home.good(), fake_agent::script(starts))?;
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        let ended = run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &Invocation::default(),
            headless,
            claimed,
            running,
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
    let saved: &[&[Step]] = &[
        &[
            sh(MARK),
            Step::Ready,
            sh(SAVE),
            Step::End(AgentEnd::Switch(None)),
        ],
        &[sh(MARK), Step::Ready, sh(SAVE)],
    ];
    let ended = interactive("clean", saved, false);
    let said =
        ended.map(|(ended, id)| (ended, format!("To resume this session: rig --resume {id}")));
    assert!(
        said.as_ref()
            .is_some_and(|(ended, said)| ended.code == ExitCode::SUCCESS
                && ended.said.as_ref() == Some(said)),
        "{said:?}"
    );

    let quiet = |test: &str, agent: &[Step], headless: bool| {
        let ended = interactive(test, &[agent], headless).map(|(ended, _)| ended);
        assert_eq!(ended, Some(Ended::quiet(ExitCode::SUCCESS)), "{test}");
    };
    quiet("unsaved", &[sh(MARK), Step::Ready], false);
    quiet("headless", &[sh(MARK), Step::Ready, sh(SAVE)], true);
    quiet("never-started", &[sh(MARK), sh(SAVE)], false);
}

/// A failed exit of a saved session says how to resume it only when its
/// agent said it started.
#[cfg(unix)]
#[test]
fn a_failed_exit_offers_the_resume_only_after_the_agent_started() {
    let offers = |test: &str, ready: bool| {
        let agent = [sh(MARK), sh(SAVE)]
            .into_iter()
            .chain(ready.then_some(Step::Ready))
            .chain([Step::Exit(3)])
            .collect::<Vec<_>>();
        let ended = interactive(test, &[&agent], false);
        ended.map(|(ended, id)| {
            let offered = ended
                .said
                .is_some_and(|said| said.contains(&format!("`rig --resume {id}`")));
            (ended.code, offered)
        })
    };
    let failed = ExitCode::from(3);
    assert_eq!(offers("failed-started", true), Some((failed, true)));
    assert_eq!(offers("failed-never-started", false), Some((failed, false)));
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
    let start = format!(
        "cd '{}'
perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' env {}=\"$RIG_RUN/groups\" \\
  sh -c 'sleep 30 & echo $! > child; wait' &
echo $! > marked
while [ ! -s child ]; do sleep 0.01; done
perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' sleep 30 &
echo $! > unmarked
cat marked unmarked > \"$RIG_RUN/groups\"",
        notes.display(),
        rig_harness_protocol::env::PROCESS_LEDGER,
    );
    let killed = format!(
        "cd '{}'
kill -0 \"$(cat marked)\" && touch kept
cat marked unmarked > \"$RIG_RUN/groups\"
kill -KILL $$",
        notes.display(),
    );
    let starts: &[&[Step]] = &[
        &[
            sh(MARK),
            Step::Ready,
            sh(&start),
            Step::End(AgentEnd::Reload),
        ],
        &[sh(MARK), Step::Ready, sh(&killed)],
    ];
    let ended = interactive("left-groups", starts, false);
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
        let state = Command::new("ps")
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
        Command::new("kill").arg(unmarked.to_string()).status().ok();
    }
    fs::remove_dir_all(&notes).ok();
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
        fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).ok();
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
            let running = select(
                &at(home, Path::new(NO_LAUNCHER))?,
                &claimed.id,
                run.dir(),
                None,
            )?;
            run_agent(
                &mut at(home, Path::new(NO_LAUNCHER))?,
                run.dir(),
                &print,
                true,
                claimed,
                running,
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
    let ready = fake_agent::commands(&[Step::Ready, Step::Exit(1)]);
    let home = builds("ready", &ready, "exit 0");
    let (code, ran, good) = one_shot(&home);
    assert_eq!(ran, "trial\n");
    assert_eq!((code, good.as_str()), (Some(ExitCode::from(1)), "trial"));
}

/// A staged build that is a directory, a symlink loop or a file that cannot
/// be read is no build: the start runs the good build instead of failing on it.
#[cfg(unix)]
#[test]
fn a_staged_file_that_is_no_build_starts_the_good_build() {
    use std::os::unix::fs::PermissionsExt;
    for broken in ["directory", "loop", "unreadable"] {
        let home = builds(&format!("staged-{broken}"), "exit 3", "exit 0");
        let staged = home.staged();
        match broken {
            "directory" => fs::remove_file(&staged).and_then(|()| fs::create_dir_all(&staged)),
            "loop" => {
                fs::remove_file(&staged).and_then(|()| std::os::unix::fs::symlink(&staged, &staged))
            }
            _ => fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o000)),
        }
        .ok();
        let (code, ran, _) = one_shot(&home);
        assert_eq!(
            (code, ran.as_str()),
            (Some(ExitCode::SUCCESS), "good\n"),
            "{broken}"
        );
    }
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
        let started = at(&home, Path::new(NO_LAUNCHER))
            .and_then(|launch| run(launch, &["-p".to_owned(), "hi".to_owned()]))
            .map_err(|why| why.to_string());
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
        let staged = select(&at(&home, Path::new(NO_LAUNCHER))?, &session, &run, None)?;
        let again = build::described(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &staged.current.executable,
        )?;
        // A build from before reports ignores the request: it has none.
        let old = select(&at(&home, Path::new(NO_LAUNCHER))?, &session, &run, None)?;
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
    let failing = fake_agent::commands(&[
        Step::Ready,
        sh("echo refused > \"$RIG_HOME/sessions/$RIG_SESSION/agent.log\""),
        Step::Exit(2),
    ]);
    let home = builds("unsaved", &failing, "exit 0");
    let print = Invocation {
        print: Some("hi".to_owned()),
        ..Invocation::default()
    };
    let left = locks::hold_run(&home).and_then(|run| {
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        let session = home.session(&claimed.id);
        let code = run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &print,
            true,
            claimed,
            running,
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
        let refused = AgentEnd::Refused("refused.".to_owned());
        let named = claimed.leave(&home, run.dir(), &refused);
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

/// Notes each start's session, why it started and its notice in
/// `RIG_HOME/started`.
const LOG_START: &str = "echo \"$RIG_SESSION $RIG_START $RIG_NOTICE\" >> \"$RIG_HOME/started\"";

/// The refusal of a session an older rig saved.
fn too_old() -> AgentEnd {
    AgentEnd::Refused("too old.".to_owned())
}

/// A resumed session whose saved directory file cannot be read, here as it
/// is not text, is refused, naming the file, and the file stays: rig does
/// not take the read as no directory and move the session to where it was
/// started.
#[cfg(unix)]
#[test]
fn a_resumed_session_whose_directory_cannot_be_read_is_refused() {
    let root = std::env::temp_dir().join(format!("rig-unread-dir-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let here = root.join("here");
    let unreadable = [0xff, 0xfe, b'\n'];
    let said = locks::hold_run(&home).and_then(|_run| {
        let (id, session) = home.new_session()?;
        session.mark_saved()?;
        fs::write(session.directory(), unreadable)?;
        fs::create_dir_all(&here)?;
        let claimed = claim(&home, Start::Resume(id), Some(&here)).map(|_| ());
        Ok((
            claimed.map_err(|failure| failure.to_string()),
            fs::read(session.directory())?,
        ))
    });
    fs::remove_dir_all(&root).ok();
    let (claimed, kept) = match said {
        Ok(said) => said,
        Err(failure) => (Err(format!("setup: {failure}")), Vec::new()),
    };
    assert!(
        claimed
            .as_ref()
            .is_err_and(|failure| failure.contains("/directory: ")),
        "{claimed:?}"
    );
    assert_eq!(kept, unreadable);
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
        let log = || sh(LOG_START);
        let agent = fake_agent::script(&[
            &[
                log(),
                Step::Ready,
                Step::End(AgentEnd::Switch(Some(refused.clone()))),
            ],
            &[log(), Step::Ready, Step::End(too_old())],
            &[log(), Step::Ready],
        ]);
        for build in [home.staged(), home.good()] {
            fs::write(build, &agent)?;
        }
        let (claimed, _) = claim(&home, Start::Resume(left.clone()), None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        let code = run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
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

/// Each start says why it started: a new session, a `/reload` of it, a
/// `/new`, then the previous build after the new one stopped during
/// startup.
#[cfg(unix)]
#[test]
fn each_start_says_why_it_started() {
    let home = builds("reasons", "", "");
    let starts = locks::hold_run(&home).and_then(|run| {
        let log = || sh("echo \"$RIG_START\" >> \"$RIG_HOME/starts\"");
        let stage = sh(&format!("cp \"$0\" {}", home.staged().display()));
        let agent = fake_agent::script(&[
            &[log(), Step::Ready, Step::End(AgentEnd::Reload)],
            &[log(), Step::Ready, stage, Step::End(AgentEnd::Switch(None))],
            &[log(), Step::Exit(1)],
            &[log(), Step::Ready],
        ]);
        for build in [home.staged(), home.good()] {
            fs::write(build, &agent)?;
        }
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        let code = run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
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
        let log = || sh("echo \"$RIG_START $*\" >> \"$RIG_HOME/starts\"");
        let stage = sh(&format!("cp \"$0\" {}", home.staged().display()));
        let agent = fake_agent::script(&[
            &[log(), Step::Ready, stage, Step::End(AgentEnd::Reload)],
            &[log(), Step::Exit(1)],
            &[log(), Step::Ready],
        ]);
        for build in [home.staged(), home.good()] {
            fs::write(build, &agent)?;
        }
        let print = Invocation {
            print: Some("hi".to_owned()),
            ..Invocation::default()
        };
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        let code = run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &print,
            true,
            claimed,
            running,
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
    let kept = [home.staged(), home.good()]
        .map(|binary| build::described(&at(&home, Path::new(NO_LAUNCHER))?, &binary));
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
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &print,
            true,
            claimed,
            running,
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
fn log_a_conversation(session: &rig_harness_protocol::SessionDir) -> Result<()> {
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
        claimed.leave(&home, run.dir(), &AgentEnd::Quit);
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

/// Renames a new file over the agent's launcher, as `rig update` does.
const REPLACE_LAUNCHER: &str =
    "echo new > \"$RIG_LAUNCHER.new\"; mv \"$RIG_LAUNCHER.new\" \"$RIG_LAUNCHER\"";

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
        let end = reload.switch.map_or(AgentEnd::Reload, |target| {
            AgentEnd::Switch((!target.is_empty()).then(|| other.clone()))
        });
        let replace = sh(if reload.replace {
            REPLACE_LAUNCHER
        } else {
            ":"
        });
        fs::remove_file(home.staged())?;
        fs::write(
            home.good(),
            fake_agent::script(&[&[replace, Step::End(end)], &[]]),
        )?;
        // The directory the session runs in, which a new one runs in too.
        let here = Some(home.root());
        let (claimed, _) = claim(&home, Start::Resume(current.clone()), here)
            .or_else(|_| claim(&home, Start::New, here))?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        let current = claimed.id.clone();
        let ended = run_agent(
            &mut at(&home, &launcher)?,
            run.dir(),
            &Invocation::default(),
            reload.headless,
            claimed,
            running,
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
    // The launch the new launcher gets: resuming `session`, or a new one,
    // in the session's directory, told what this launcher hands on.
    let reexec = |launcher: &Path, session: Option<&SessionId>, carry: Carry| {
        let here = launcher.parent().and_then(Path::parent)?;
        let mut launch = at(&Home::at(here), launcher).ok()?;
        launch.here = AbsPath::new(here).ok();
        launch.args = (session.into_iter())
            .flat_map(|id| ["--resume".into(), OsString::from(id.as_str())])
            .collect();
        launch.carry = carry;
        Some(launch)
    };
    let reloaded_on = Carry {
        reason: Some(StartReason::Reloaded),
        ..Carry::default()
    };
    let plain = reloaded("plain", Reload::default());
    assert!(
        plain.is_some_and(|(ended, ..)| ended.reexec.is_none() && ended.code == ExitCode::SUCCESS),
        "a plain reload restarts the agent"
    );
    let own = reloaded("own", replace);
    assert!(own.is_some_and(|(ended, launcher, current, ..)| {
        ended.reexec == reexec(&launcher, Some(&current), reloaded_on.clone())
            && ended.said.is_none()
    }));
    let resumed = |left| Carry {
        reason: Some(StartReason::Resumed),
        left: Some(left),
        notice: Some("Resumed the session.".to_owned()),
        ..Carry::default()
    };
    let other = Reload {
        switch: Some("other"),
        ..replace
    };
    let switched = reloaded("switch", other);
    assert!(
        switched.is_some_and(|(ended, launcher, current, other, _)| {
            ended.reexec == reexec(&launcher, Some(&other), resumed(current))
        })
    );
    // The switch outlives the unsaved session it left.
    let unsaved = reloaded(
        "unsaved",
        Reload {
            unsaved: true,
            ..other
        },
    );
    assert!(
        unsaved.is_some_and(|(ended, launcher, current, other, kept)| {
            ended.reexec == reexec(&launcher, Some(&other), resumed(current)) && !kept
        })
    );
    let busy = reloaded(
        "busy",
        Reload {
            busy: true,
            ..other
        },
    );
    // Told by the new launcher's agent, as a switch refused without a new
    // launcher is.
    assert!(busy.is_some_and(|(ended, launcher, current, other, _)| {
        let notice = format!("Could not switch sessions: session {other} is open in another rig.");
        let carry = Carry {
            notice: Some(notice),
            ..reloaded_on.clone()
        };
        ended.reexec == reexec(&launcher, Some(&current), carry) && ended.said.is_none()
    }));
    let new = reloaded(
        "new",
        Reload {
            switch: Some(""),
            ..replace
        },
    );
    assert!(new.is_some_and(|(ended, launcher, current, ..)| {
        let carry = Carry {
            reason: Some(StartReason::New),
            left: Some(current),
            ..Carry::default()
        };
        ended.reexec == reexec(&launcher, None, carry)
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

/// Runs the launcher `next`'s re-exec starts, past its build, as its
/// command carries it ([`Launch::from_parts`]): it resumes the session
/// its command line names, and its first start goes on with what the
/// launcher before it handed on ([`carry_on`]).
#[cfg(unix)]
fn as_reexecuted(home: &Home, next: &Launch, run: &RunDir) -> Result<Ended> {
    let command = next.reexec();
    let mut launch = Launch::from_parts(
        Path::new(command.get_program()),
        false,
        command.get_envs(),
        command.get_current_dir(),
        command.get_args(),
    )?;
    let args: Vec<String> = (launch.args.iter())
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let start = start_of(home, AgentDescription::default().resume(&args).as_deref())?;
    let carried = std::mem::take(&mut launch.carry);
    let (mut claimed, notice) = claim(home, start, launch.here.as_deref())?;
    launch.carry.notice = carry_on(&mut launch, carried, &mut claimed, notice);
    let running = select(&launch, &claimed.id, run, None)?;
    run_agent(
        &mut launch,
        run,
        &Invocation::default(),
        false,
        claimed,
        running,
    )
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
        let log = || sh(LOG_START);
        let agent = fake_agent::script(&[
            &[
                log(),
                Step::Ready,
                sh(REPLACE_LAUNCHER),
                Step::End(AgentEnd::Switch(Some(refused.clone()))),
            ],
            &[log(), Step::Ready, Step::End(too_old())],
            &[log(), Step::Ready],
        ]);
        for build in [home.staged(), home.good()] {
            fs::write(build, &agent)?;
        }
        let here = Some(home.root());
        let (claimed, _) = claim(&home, Start::Resume(left.clone()), here)?;
        let mut launch = at(&home, &launcher)?;
        let running = select(&launch, &claimed.id, run.dir(), None)?;
        let ended = run_agent(
            &mut launch,
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
        )?;
        let Some(next) = ended.reexec else {
            return Err("no re-exec".into());
        };
        let told = next.carry.left.as_ref() == Some(&left);
        let code = as_reexecuted(&home, &next, run.dir())?.code;
        let started = fs::read_to_string(home.root().join("started"))?;
        Ok((code, started, told, left, refused))
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
    let log = || sh("printf '%s\\n---\\n' \"$RIG_WARNING\" >> \"$RIG_HOME/warned\"");
    let agent = fake_agent::script(&[
        &[log(), Step::Ready, Step::End(AgentEnd::Reload)],
        &[log(), Step::Ready],
    ]);
    for build in [home.staged(), home.good()] {
        fs::write(build, &agent)?;
    }
    let run = locks::hold_run(home)?;
    let (claimed, _) = claim(home, Start::New, None)?;
    let running = select(
        &at(home, Path::new(NO_LAUNCHER))?,
        &claimed.id,
        run.dir(),
        None,
    )?;
    run_agent(
        &mut at(home, Path::new(NO_LAUNCHER))?,
        run.dir(),
        &Invocation::default(),
        false,
        claimed,
        running,
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
    let good = fake_agent::commands(&[
        sh("printf %s \"$RIG_WARNING\" > \"$RIG_HOME/warned\""),
        Step::Ready,
    ]);
    let home = builds("legacy-rollback", "exit 1", &good);
    let warned = fs::write(home.root().join("plugins.toml"), "[[plugin]]\n")
        .map_err(Into::into)
        .and_then(|()| {
            let run = locks::hold_run(&home)?;
            let (claimed, _) = claim(&home, Start::New, None)?;
            let running = select(
                &at(&home, Path::new(NO_LAUNCHER))?,
                &claimed.id,
                run.dir(),
                None,
            )?;
            run_agent(
                &mut at(&home, Path::new(NO_LAUNCHER))?,
                run.dir(),
                &Invocation::default(),
                false,
                claimed,
                running,
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
    fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))?;
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
        let a_staged = select(&at(&home, Path::new(NO_LAUNCHER))?, &a, &run_a, None)?;
        let _b = select(&at(&home, Path::new(NO_LAUNCHER))?, &b, &run_b, None)?;
        let read_staged = a_staged.current.description.clone();
        let ran_staged = build::described(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &a_staged.current.executable,
        )?;
        // B's trial, which describes itself otherwise, becomes the good build.
        drop(a_staged);
        let a_good = select(&at(&home, Path::new(NO_LAUNCHER))?, &a, &run_a, None)?;
        let newer = home.bin().join("newer");
        script(&newer, &describing("newer"))?;
        fs::rename(&newer, home.good())?;
        let read_good = a_good.current.description.clone();
        let ran_good = build::described(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &a_good.current.executable,
        )?;
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
        let swap = format!(
            "cp {other} \"$RIG_HOME/bin/other\" && mv \"$RIG_HOME/bin/other\" \"$RIG_HOME/bin/good\"
cp {failing} \"$RIG_HOME/bin/staged-$RIG_SESSION\"",
            other = other.display(),
            failing = failing.display(),
        );
        let own = || sh("echo own >> \"$RIG_HOME/ran\"");
        let agent = fake_agent::script(&[
            &[own(), Step::Ready, sh(&swap), Step::End(AgentEnd::Reload)],
            &[own()],
        ]);
        fs::write(home.good(), agent)?;
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
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
{ready}",
                newer = newer.display(),
                ready = fake_agent::commands(&[Step::Ready]),
            ),
        )?;
        let print = Invocation {
            print: Some("hi".to_owned()),
            ..Invocation::default()
        };
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &print,
            true,
            claimed,
            running,
        )?;
        Ok(fs::read_to_string(home.good())?)
    });
    fs::remove_dir_all(home.root()).ok();
    let good = good.map_err(|failure| failure.to_string());
    assert_eq!(good, Ok("#!/bin/sh\necho newer\n".to_owned()));
}

/// A launcher whose ready trial ended while another launcher holds the
/// build lock ends on a signal heard as it waits to make the trial good.
#[cfg(unix)]
#[test]
fn a_signal_ends_the_wait_to_promote_a_ready_trial() {
    let home = builds("promote-wait", "", "exit 0");
    let waited = locks::hold_run(&home).and_then(|run| {
        script(
            &home.staged(),
            &format!(
                "[ -n \"$RIG_DESCRIBE\" ] && exit 0
{}(sleep 1; kill -TERM {}) &",
                fake_agent::commands(&[Step::Ready]),
                std::process::id()
            ),
        )?;
        let print = Invocation {
            print: Some("hi".to_owned()),
            ..Invocation::default()
        };
        let (claimed, _) = claim(&home, Start::New, None)?;
        let running = select(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &claimed.id,
            run.dir(),
            None,
        )?;
        // Another launcher's build holds the lock for longer than the test.
        let other = locks::try_build_lock(&home)?.ok_or("the build lock is held")?;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(10));
            drop(other);
        });
        let started = std::time::Instant::now();
        run_agent(
            &mut at(&home, Path::new(NO_LAUNCHER))?,
            run.dir(),
            &print,
            true,
            claimed,
            running,
        )?;
        Ok(started.elapsed())
    });
    fs::remove_dir_all(home.root()).ok();
    let waited = waited.map_err(|failure| failure.to_string());
    assert_eq!(stop::heard(), Some(libc::SIGTERM), "the launcher heard it");
    assert!(
        waited
            .as_ref()
            .is_ok_and(|waited| *waited < Duration::from_secs(5)),
        "{waited:?}"
    );
}

/// Two launchers that start after `rig build` staged a build for any
/// launcher, A then B, each staging as a start's build does (only a build
/// not already known) and selecting in one hold of the build lock: both
/// run the new build, B from its own copy once A claimed the shared one.
#[cfg(unix)]
#[test]
fn two_launchers_starting_after_rig_build_both_run_the_new_build() {
    let home = builds("both-new", "", &describing("old"));
    let ran = (|| -> Result<_> {
        fs::remove_file(home.staged())?;
        let artifact = build::artifact(&home);
        if let Some(target) = artifact.parent() {
            fs::create_dir_all(target)?;
        }
        script(&artifact, &describing("new"))?;
        build::stage(
            &at(&home, Path::new(NO_LAUNCHER))?,
            &home.staged(),
            Staging::Always,
        )?;
        let mut ran = Vec::new();
        for name in ["a", "b"] {
            let session = SessionId::generate();
            let run = RunDir::at(home.runs().join(name));
            let _lock = locks::hold_build_lock(&home, BuildOutput::LogOnly)?;
            build::stage(
                &at(&home, Path::new(NO_LAUNCHER))?,
                &home.staged_for(&session),
                Staging::OnlyNew,
            )?;
            let running = select(&at(&home, Path::new(NO_LAUNCHER))?, &session, &run, None)?;
            let new = running.current.description.command("new").is_some();
            ran.push((new, running.current.is_trial()));
        }
        Ok(ran)
    })();
    fs::remove_dir_all(home.root()).ok();
    let ran = ran.map_err(|failure| failure.to_string());
    assert_eq!(ran, Ok(vec![(true, true), (true, true)]));
}

/// A reload after the launcher's file was replaced goes on, on the new
/// launcher, as the same reload would on this one: its agent is told it
/// reloaded, not that the session was resumed, is not warned again of a
/// 0.46 home the run warned of already, and gets the secrets typed at the
/// old launcher's prompt.
#[cfg(unix)]
#[test]
fn a_reload_on_the_new_launcher_starts_as_a_reload() {
    let home = builds("replaced-carry", "", "");
    let started = locks::hold_run(&home).and_then(|run| {
        fs::write(home.root().join("plugins.toml"), "")?;
        let (session, saved) = home.new_session()?;
        saved.mark_saved()?;
        let launcher = home.root().join("launcher").join("rig");
        fs::create_dir_all(home.root().join("launcher"))?;
        fs::write(&launcher, "old")?;
        let log = || {
            sh("echo \"$RIG_START|$RIG_NOTICE|$RIG_WARNING|$RIG_SECRET_TYPED\" >> \"$RIG_HOME/started\"")
        };
        let agent = fake_agent::script(&[
            &[
                log(),
                Step::Ready,
                sh(REPLACE_LAUNCHER),
                Step::End(AgentEnd::Reload),
            ],
            &[log(), Step::Ready],
        ]);
        for build in [home.staged(), home.good()] {
            fs::write(build, &agent)?;
        }
        let here = Some(home.root());
        let (claimed, notice) = claim(&home, Start::Resume(session), here)?;
        let mut launch = at(&home, &launcher)?;
        launch.carry.notice = notice;
        // As the prompt keeps a secret typed at it.
        launch
            .secrets
            .insert("RIG_SECRET_TYPED".into(), "typed".into());
        let running = select(&launch, &claimed.id, run.dir(), None)?;
        let ended = run_agent(
            &mut launch,
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
            running,
        )?;
        let Some(next) = ended.reexec else {
            return Err("no re-exec".into());
        };
        as_reexecuted(&home, &next, run.dir())?;
        Ok(fs::read_to_string(home.root().join("started"))?)
    });
    fs::remove_dir_all(home.root()).ok();
    let started = started.map_err(|failure| failure.to_string());
    let lines: Vec<String> = started
        .iter()
        .flat_map(|started| started.lines().map(str::to_owned))
        .collect();
    assert_eq!(lines.len(), 2, "{started:?}");
    assert!(
        lines.first().is_some_and(
            |first| first.starts_with("resumed|Resumed the session.|[\"rig no longer reads ")
        ),
        "{lines:?}"
    );
    assert!(
        lines.first().is_some_and(|first| first.ends_with("|typed")),
        "{lines:?}"
    );
    assert_eq!(
        lines.get(1).map(String::as_str),
        Some("reloaded|||typed"),
        "{lines:?}"
    );
}

/// Runs an agent that takes the steps `starts` (one list per start,
/// [`fake_agent::script`]) as the only build of `home` for `start`, as
/// `invocation` says, and returns how the run ended and what the agent
/// wrote to the home's `started` file.
#[cfg(unix)]
fn run_once(
    home: &Home,
    starts: &[&[Step]],
    start: Start,
    invocation: &Invocation,
    headless: bool,
) -> Result<(Ended, String)> {
    let run = locks::hold_run(home)?;
    fs::optional(fs::remove_file(home.staged()))?;
    fs::optional(fs::remove_file(home.root().join("fake-agent-starts")))?;
    fs::write(home.good(), fake_agent::script(starts))?;
    let (claimed, _) = claim(home, start, None)?;
    let running = select(
        &at(home, Path::new(NO_LAUNCHER))?,
        &claimed.id,
        run.dir(),
        None,
    )?;
    let ended = run_agent(
        &mut at(home, Path::new(NO_LAUNCHER))?,
        run.dir(),
        invocation,
        headless,
        claimed,
        running,
    )?;
    let started = fs::optional(fs::read_to_string(home.root().join("started")))?;
    Ok((ended, started.unwrap_or_default()))
}

/// Notes each start's session in `RIG_HOME/started`.
const LOG_SESSION: &str = "echo \"$RIG_SESSION\" >> \"$RIG_HOME/started\"";

/// A one-shot run whose prompt is `/new` or `/resume` ends with the codes
/// `rig help-agents` promises: the agent's switch is not followed, and the
/// script is not told the reload code.
#[cfg(unix)]
#[test]
fn a_one_shot_switch_ends_with_a_code_scripts_are_promised() {
    let home = builds("one-shot-switch", "", "");
    let print = Invocation {
        print: Some("/new".to_owned()),
        ..Invocation::default()
    };
    let agent: &[&[Step]] = &[&[
        sh(LOG_SESSION),
        Step::Ready,
        Step::End(AgentEnd::Switch(None)),
    ]];
    let ended = run_once(&home, agent, Start::New, &print, true);
    fs::remove_dir_all(home.root()).ok();
    let ended = ended.map_err(|failure| failure.to_string());
    let code = ended
        .as_ref()
        .map(|(ended, started)| (ended.code, started.lines().count()));
    assert_eq!(code, Ok((ExitCode::SUCCESS, 1)), "{ended:?}");
}

/// An agent that exits with the reload code without asking to reload, as
/// a plugin's front failing with its child's code 75 did, ended: it is
/// not restarted.
#[cfg(unix)]
#[test]
fn an_exit_code_of_75_alone_is_a_failure_not_a_reload() {
    let home = builds("raw-75", "", "");
    let agent: &[&[Step]] = &[
        &[sh(LOG_SESSION), Step::Ready, Step::Exit(75)],
        &[sh(LOG_SESSION), Step::Ready],
    ];
    let ended = run_once(&home, agent, Start::New, &Invocation::default(), false);
    fs::remove_dir_all(home.root()).ok();
    let ended = ended.map_err(|failure| failure.to_string());
    let starts = ended
        .as_ref()
        .map(|(ended, started)| (ended.code, started.lines().count()));
    assert_eq!(starts, Ok((ExitCode::FAILURE, 1)), "{ended:?}");
}

/// A reload's record is cleared before the next start: an agent that
/// reloaded, then exits with the reload code alone, ended.
#[cfg(unix)]
#[test]
fn an_earlier_start_s_record_is_not_read_for_the_next() {
    let home = builds("stale-end", "", "");
    let agent: &[&[Step]] = &[
        &[sh(LOG_SESSION), Step::Ready, Step::End(AgentEnd::Reload)],
        &[sh(LOG_SESSION), Step::Ready, Step::Exit(75)],
        &[sh(LOG_SESSION), Step::Ready],
    ];
    let ended = run_once(&home, agent, Start::New, &Invocation::default(), false);
    fs::remove_dir_all(home.root()).ok();
    let ended = ended.map_err(|failure| failure.to_string());
    let starts = ended
        .as_ref()
        .map(|(ended, started)| (ended.code, started.lines().count()));
    assert_eq!(starts, Ok((ExitCode::FAILURE, 2)), "{ended:?}");
}

/// An agent killed after it asked to switch sessions, before it exited,
/// switches nothing: neither then nor on the session's next reload.
#[cfg(unix)]
#[test]
fn an_agent_killed_after_asking_to_switch_switches_nothing() {
    let home = builds("killed-switch", "", "");
    let ran = (|| -> Result<_> {
        let saved = || -> Result<SessionId> {
            let (id, session) = home.new_session()?;
            session.mark_saved()?;
            Ok(id)
        };
        let (current, other) = (saved()?, saved()?);
        let killed: &[&[Step]] = &[&[
            sh(LOG_SESSION),
            Step::Ready,
            Step::Record(AgentEnd::Switch(Some(other))),
            sh("kill -9 $$"),
        ]];
        let resume = || Start::Resume(current.clone());
        let none = Invocation::default();
        let (ended, _) = run_once(&home, killed, resume(), &none, false)?;
        let reloads: &[&[Step]] = &[
            &[sh(LOG_SESSION), Step::Ready, Step::End(AgentEnd::Reload)],
            &[sh(LOG_SESSION), Step::Ready],
        ];
        let (_, started) = run_once(&home, reloads, resume(), &none, false)?;
        Ok((ended.code, started, current))
    })();
    fs::remove_dir_all(home.root()).ok();
    let ran = ran.map_err(|failure| failure.to_string());
    let stayed = ran.as_ref().is_ok_and(|(code, started, current)| {
        *code == ExitCode::FAILURE && *started == format!("{current}\n").repeat(3)
    });
    assert!(stayed, "{ran:?}");
}

/// An error in the launcher's loop after a front took the terminal's
/// screen, such as one at the start after a `/new`, puts the terminal back
/// before the error is said. The test runs itself again as the launcher,
/// to read what it wrote to its stderr.
#[cfg(unix)]
#[test]
fn an_error_after_a_front_took_the_screen_puts_the_terminal_back() {
    const PROBE: &str = "RIG_SCREEN_ERROR_PROBE";
    if std::env::var_os(PROBE).is_some() {
        // The screen file as a directory: the agent took the screen, and
        // the next start cannot clear it.
        let taken = sh("mkdir -p \"$RIG_RUN/screen/kept\"");
        let starts: &[&[Step]] = &[
            &[Step::Ready, taken, Step::End(AgentEnd::Switch(None))],
            &[Step::Ready],
        ];
        let home = builds("screen-error", "", "");
        let failed = run_once(&home, starts, Start::New, &Invocation::default(), false);
        fs::remove_dir_all(home.root()).ok();
        if let Err(failure) = failed {
            eprintln!("error: {failure}");
        }
        return;
    }
    let name = "run::tests::an_error_after_a_front_took_the_screen_puts_the_terminal_back";
    let launcher = std::env::current_exe().and_then(|test| {
        Command::new(test)
            .args(["--exact", name, "--test-threads", "1", "--nocapture"])
            .env(PROBE, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .output()
    });
    let said = launcher.map(|output| String::from_utf8_lossy(&output.stderr).into_owned());
    let said = said.map_err(|failure| failure.to_string());
    let restored = said
        .as_deref()
        .ok()
        .and_then(|said| said.find("\x1b[?1049l"));
    let error = said.as_deref().ok().and_then(|said| said.find("error: "));
    assert!(
        matches!((restored, error), (Some(restored), Some(error)) if restored < error),
        "{said:?}"
    );
}

/// The step after each end an agent can leave, as a table: the decision
/// the fake-agent runs above make, without running an agent.
#[test]
fn each_end_decides_the_next_step() {
    use super::Step as Next;
    use rig_harness_protocol::end::FailCode;

    let id = SessionId::generate();
    let view = Runs {
        headless: false,
        answers_prompt: false,
        reason: StartReason::New,
        kept_screen: false,
        back: None,
        trial: false,
        previous: false,
    };
    let prompt = Runs {
        headless: true,
        answers_prompt: true,
        ..view.clone()
    };
    let command = Runs {
        headless: true,
        ..view.clone()
    };
    let trial = Runs {
        trial: true,
        previous: true,
        ..view.clone()
    };
    let quit = Left {
        end: AgentEnd::Quit,
        ready: true,
        screen: false,
        heard: false,
        replaced: false,
        saved: true,
    };
    let ended = |end: AgentEnd| Left {
        end,
        ..quit.clone()
    };
    let reload = ended(AgentEnd::Reload);
    let new = ended(AgentEnd::Switch(None));
    let resume = ended(AgentEnd::Switch(Some(id.clone())));
    let refused = ended(AgentEnd::Refused("too old.".to_owned()));
    let failed = ended(AgentEnd::Failed(FailCode::or_failure(3)));
    let end = |said, restore| Next::Stop(Stop::End { said, restore });
    let stopped = |hidden| end(Said::Stopped { hidden }, hidden);
    let replaced = |left: &Left| Left {
        replaced: true,
        ..left.clone()
    };
    let screen = |left: &Left| Left {
        screen: true,
        ..left.clone()
    };
    let heard = |left: &Left| Left {
        heard: true,
        ..left.clone()
    };
    let unready = |left: &Left| Left {
        ready: false,
        ..left.clone()
    };
    let back = |runs: &Runs| Runs {
        back: Some(id.clone()),
        ..runs.clone()
    };
    let rows: Vec<(&str, Runs, Left, Next)> = vec![
        // A quit offers the resume of a saved session the agent started.
        ("quit", view.clone(), quit.clone(), end(Said::Resume, false)),
        (
            "quit unsaved",
            view.clone(),
            Left {
                saved: false,
                ..quit.clone()
            },
            end(Said::Nothing, false),
        ),
        (
            "quit never ready",
            view.clone(),
            unready(&quit),
            end(Said::Nothing, false),
        ),
        (
            "quit headless",
            prompt.clone(),
            quit.clone(),
            end(Said::Nothing, false),
        ),
        // The terminal is put back after a front that took the screen,
        // only when the launcher says something after it.
        (
            "quit screen",
            view.clone(),
            screen(&quit),
            end(Said::Resume, true),
        ),
        (
            "quit screen unsaved",
            view.clone(),
            Left {
                saved: false,
                ..screen(&quit)
            },
            end(Said::Nothing, false),
        ),
        ("failed", view.clone(), failed.clone(), stopped(false)),
        (
            "failed after a view kept the screen",
            Runs {
                kept_screen: true,
                ..view.clone()
            },
            screen(&failed),
            stopped(true),
        ),
        (
            "failed headless",
            prompt.clone(),
            failed.clone(),
            end(Said::Nothing, false),
        ),
        // A reload or switch restarts a view, on the new launcher once the
        // launcher's file was replaced.
        (
            "reload",
            view.clone(),
            reload.clone(),
            Next::Restart { to: None },
        ),
        (
            "new",
            view.clone(),
            new.clone(),
            Next::Restart {
                to: Some(Start::New),
            },
        ),
        (
            "resume",
            view.clone(),
            resume.clone(),
            Next::Restart {
                to: Some(Start::Resume(id.clone())),
            },
        ),
        (
            "reload replaced",
            view.clone(),
            replaced(&reload),
            Next::Reexec { to: None },
        ),
        (
            "resume replaced",
            view.clone(),
            replaced(&resume),
            Next::Reexec {
                to: Some(Start::Resume(id.clone())),
            },
        ),
        ("reload heard", view.clone(), heard(&reload), stopped(false)),
        // A headless run follows only a reload, of a prompt, under this
        // launcher.
        (
            "reload headless replaced",
            prompt.clone(),
            replaced(&reload),
            Next::Restart { to: None },
        ),
        (
            "new headless",
            prompt.clone(),
            new.clone(),
            end(Said::Nothing, false),
        ),
        (
            "reload of a command",
            command.clone(),
            reload.clone(),
            end(Said::Nothing, false),
        ),
        // A refused first start after a switch goes back, but for a signal.
        (
            "refused after a switch",
            back(&view),
            refused.clone(),
            Next::GoBack {
                to: id.clone(),
                why: "too old.".to_owned(),
                otherwise: Stop::End {
                    said: Said::Stopped { hidden: false },
                    restore: false,
                },
            },
        ),
        (
            "refused after a switch, heard",
            back(&view),
            heard(&refused),
            stopped(false),
        ),
        (
            "refused after a switch, headless",
            back(&prompt),
            refused.clone(),
            end(Said::Nothing, false),
        ),
        ("refused", view.clone(), refused.clone(), stopped(false)),
        // A trial that stops during startup is rolled back, and run again
        // for a view or a prompt's reload.
        (
            "trial stopped",
            trial.clone(),
            unready(&failed),
            Next::Stop(Stop::RollBack),
        ),
        (
            "trial stopped headless",
            Runs {
                headless: true,
                answers_prompt: true,
                ..trial.clone()
            },
            unready(&failed),
            end(Said::RolledBack, false),
        ),
        (
            "trial stopped after a headless reload",
            Runs {
                headless: true,
                answers_prompt: true,
                reason: StartReason::Reloaded,
                ..trial.clone()
            },
            unready(&failed),
            Next::Stop(Stop::RollBack),
        ),
        (
            "trial stopped, no previous",
            Runs {
                previous: false,
                ..trial.clone()
            },
            screen(&unready(&failed)),
            end(Said::NoPrevious, true),
        ),
        (
            "trial stopped, heard",
            Runs {
                headless: true,
                ..trial.clone()
            },
            heard(&unready(&failed)),
            end(Said::Nothing, false),
        ),
        (
            "trial reloaded unready",
            trial.clone(),
            unready(&reload),
            Next::Restart { to: None },
        ),
        (
            "trial refused after a switch",
            back(&trial),
            unready(&refused),
            Next::Stop(Stop::RollBack),
        ),
    ];
    for (row, runs, left, step) in rows {
        assert_eq!(next(&runs, &left), step, "{row}");
    }
    let rejected = |runs: &Runs, left: &Left| runs.rejected(left);
    assert!(rejected(&trial, &unready(&failed)));
    assert!(!rejected(&trial, &failed), "a ready trial is kept");
    assert!(!rejected(&trial, &heard(&unready(&failed))), "heard");
    assert!(!rejected(&view, &unready(&failed)), "a proven build");
}

/// A screen a view took stays taken across a restart whose agent never
/// takes it, and is handed over once.
#[test]
fn a_screen_taken_before_a_restart_stays_taken_until_handed_over() {
    let mut screen = Screen {
        taken: false,
        sane: None,
    };
    assert!(!screen.took(false));
    assert!(screen.took(true));
    assert!(screen.took(false));
    assert!(screen.hand_over());
    assert!(!screen.hand_over());
    assert!(!screen.took(false));
}
