use super::*;

#[cfg(unix)]
#[test]
fn only_a_session_with_a_conversation_is_offered_to_resume() {
    use std::os::unix::process::ExitStatusExt;

    let status = ExitStatus::from_raw(2 << 8);
    let log = Some(Path::new("/home/.rig/sessions/s/agent.log"));
    let saved = stopped(status, Some("rig --resume 1-2"), None, log);
    assert!(saved.contains("`rig --resume 1-2` resumes"), "{saved}");
    let empty = stopped(status, None, None, log);
    let expected = "The agent stopped (exit status: 2). Log: /home/.rig/sessions/s/agent.log";
    assert_eq!(empty, expected);
    // A refusal said where a restarted view kept the screen comes again.
    let refused = stopped(status, None, Some("rig: plain.\n"), log);
    assert_eq!(refused, format!("rig: plain.\n{expected}"));
    let no_log = stopped(status, None, None, None);
    assert_eq!(no_log, "The agent stopped (exit status: 2).");
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
/// and exits; and the session the run ended in.
#[cfg(unix)]
fn interactive(test: &str, agent: &str, headless: bool) -> Option<(Ended, SessionId)> {
    let home = builds(&format!("interactive-{test}"), "", "");
    let ended = locks::hold_run(&home).and_then(|run| {
        // The good build only: a trial that never says it started is rolled back.
        fs::remove_file(home.staged())?;
        fs::write(home.good(), format!("#!/bin/sh\n{agent}\n"))?;
        let (claimed, _) = claim(&home, Start::New, None)?;
        let ended = launch(
            &home,
            run.dir(),
            &Invocation::default(),
            headless,
            claimed,
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
            launch(home, run.dir(), &print, true, claimed, None, None)
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

#[cfg(unix)]
#[test]
fn a_build_without_a_report_is_described_once_and_the_command_line_read_against_it() {
    // The scratch home a describe runs in is `run/<launcher>/describe/home`.
    let describe = "[ -n \"$RIG_DESCRIBE\" ] || exit 0
echo described >> \"$RIG_HOME/../../../../described\"
printf 'rig-describe 1\\ncommand\\tstats\\t-\\theadless\\tsession\\tCount\\n' > \"$RIG_DESCRIBE\"";
    let home = builds("report", describe, "exit 0");
    let session = SessionId::generate();
    let first = report(&home, &session);
    let again = report(&home, &session);
    let described = fs::read_to_string(home.root().join("described")).unwrap_or_default();
    // A build from before reports ignores the request: it has none.
    fs::remove_file(home.staged()).ok();
    let old = report(&home, &session);
    fs::remove_dir_all(home.root()).ok();
    assert!(first.command("stats").is_some(), "{first:?}");
    assert_eq!((first.clone(), described.as_str()), (again, "described\n"));
    assert_eq!(old, AgentDescription::default());
    let unknown = first.invocation(&["statz"]).err().unwrap_or_default();
    assert!(
        unknown.ends_with("commands: build, plugin, help, help-agents, stats"),
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
        let session = home.session(&claimed.id);
        let code = launch(&home, run.dir(), &print, true, claimed, None, None)?.code;
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
        let code = launch(
            &home,
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
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
case $(wc -l < \"$RIG_HOME/starts\") in
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
        let code = launch(
            &home,
            run.dir(),
            &Invocation::default(),
            false,
            claimed,
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
case $(wc -l < \"$RIG_HOME/starts\") in
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
        let code = launch(&home, run.dir(), &print, true, claimed, None, None)?.code;
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
