#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures write and read files of their own"
)]

use super::{
    AgentDescription, DescribedCommand, DescribedFlag, DescribedSecret, Home, Invocation, Stamp,
    is_secret_variable, secret_variable, warnings,
};

/// Blank text is no warning, and any other text that is not an array of
/// strings is one warning, as it is.
#[test]
fn warnings_read_plain_text_whole() {
    assert!(warnings("").is_empty() && warnings(" \n").is_empty());
    assert_eq!(warnings("not an array"), ["not an array"]);
    assert_eq!(warnings("[1, \"a\"]"), ["[1, \"a\"]"]);
}

#[test]
fn plugin_options_pass_through_and_read_back() {
    let args = [
        "--json",
        "-m",
        "deepseek/deepseek-flash",
        "--serve=127.0.0.1:8080",
        "-p",
        "hello",
        "world",
    ];
    let invocation = Invocation::parse(&args);
    let expected = Invocation {
        print: Some("hello world".to_owned()),
        model: Some("deepseek/deepseek-flash".to_owned()),
        flags: vec![
            ("json".to_owned(), None),
            ("serve".to_owned(), Some("127.0.0.1:8080".to_owned())),
        ],
        ..Invocation::default()
    };
    assert_eq!(invocation.as_ref(), Ok(&expected));
    let again: Vec<String> = expected
        .to_args()
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert_eq!(Invocation::parse(&again), Ok(expected));
    assert_eq!(
        Invocation::parse(&["--model=openai/gpt-5"]).map(|args| args.model),
        Ok(Some("openai/gpt-5".to_owned()))
    );
}

#[test]
fn malformed_options_are_refused_and_a_command_takes_the_words_after_it() {
    for args in [&["-x"][..], &["--Json"], &["--=1"], &["--9lives"]] {
        assert!(Invocation::parse(args).is_err(), "{args:?}");
    }
    let command = Invocation {
        flags: vec![("remote".to_owned(), Some("h:/p".to_owned()))],
        command: Some("stats".to_owned()),
        arguments: vec!["--all".to_owned(), "-p".to_owned(), "a b".to_owned()],
        ..Invocation::default()
    };
    let args: Vec<String> = command
        .to_args()
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert_eq!(args, ["--remote=h:/p", "stats", "--all", "-p", "a b"]);
    assert_eq!(Invocation::parse(&args), Ok(command));
}

/// A build with the commands `stats` and `import`, the options `--remote
/// HOST:PATH` and `--yes`, the secret `crypt`, and a stage conflict.
fn described() -> AgentDescription {
    AgentDescription {
        commands: vec![
            DescribedCommand {
                name: "stats".to_owned(),
                usage: String::new(),
                terminal: false,
                sessionless: true,
                help: "Count the saved sessions".to_owned(),
            },
            DescribedCommand {
                name: "import".to_owned(),
                usage: "<tool> [path]".to_owned(),
                terminal: true,
                sessionless: false,
                help: "Import a\tsession".to_owned(),
            },
        ],
        flags: vec![
            DescribedFlag {
                name: "remote".to_owned(),
                value: Some("HOST:PATH".to_owned()),
                help: "Work over SSH".to_owned(),
            },
            DescribedFlag {
                name: "yes".to_owned(),
                value: None,
                help: "Publish without asking".to_owned(),
            },
        ],
        secrets: vec![DescribedSecret {
            name: "crypt".to_owned(),
            help: "Passphrase".to_owned(),
        }],
        conflicts: vec!["`a::APlugin` and `b::BPlugin`: a::count and b::reset".to_owned()],
    }
}

#[test]
fn a_report_reads_back_and_one_without_its_header_is_empty() {
    let mut report = described();
    let read = AgentDescription::parse(&report.write());
    // A tab or line break in a field becomes a space.
    if let Some(import) = report.commands.last_mut() {
        import.help = "Import a session".to_owned();
    }
    assert_eq!(read, report);
    let headless = report.write().replacen("rig-describe 1", "", 1);
    assert_eq!(
        AgentDescription::parse(&headless),
        AgentDescription::default()
    );
    assert_eq!(
        AgentDescription::parse("Usage: rig-agent"),
        AgentDescription::default()
    );
    assert_eq!(secret_variable("my-crypt"), "RIG_SECRET_MY_CRYPT");
}

#[test]
fn a_report_of_a_later_version_reads_the_fields_it_knows() {
    let mut report = described();
    if let Some(import) = report.commands.last_mut() {
        import.help = "Import a session".to_owned();
    }
    let later: String = report
        .write()
        .lines()
        .map(|line| match line {
            "rig-describe 1" => format!("{line}\nhook\tpre-run\tA record of a later kind."),
            line => format!("{line}\tlater"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(AgentDescription::parse(&later), report);
}

#[test]
fn the_command_line_is_read_against_the_report() {
    let report = described();
    let words = |args: &[&str]| report.invocation(args);
    // An option's value after `=` or as the next word, alike.
    let spaced = words(&["--remote", "h:/p", "-p", "hi"]);
    assert_eq!(spaced, words(&["--remote=h:/p", "-p", "hi"]));
    assert_eq!(
        spaced.map(|invocation| (invocation.flags, invocation.print)),
        Ok((
            vec![("remote".to_owned(), Some("h:/p".to_owned()))],
            Some("hi".to_owned())
        ))
    );
    // The first bare word is the command; every later word is its own.
    let command = words(&["--remote", "h:/p", "stats", "a", "--b"]);
    assert_eq!(
        command.map(|invocation| (invocation.command, invocation.arguments)),
        Ok((
            Some("stats".to_owned()),
            vec!["a".to_owned(), "--b".to_owned()]
        ))
    );
    let refused = |args: &[&str]| words(args).err().unwrap_or_default();
    // A launcher's word says the exact line that runs it: options before
    // it dropped, and only the words it takes.
    for (args, line) in [
        (&["--yes", "build"][..], "rig build"),
        (&["-m", "ollama/qwen3", "update"], "rig update"),
        (&["update", "now"], "rig update"),
        (&["-r", "fix", "--yes", "help-agents"], "rig help-agents"),
        (
            &["--remote=h:/p", "help", "stats", "more"],
            "rig help stats",
        ),
        (
            &["--yes", "plugin", "new", "my plugin", "--path", "it's"],
            r"rig plugin new 'my plugin' --path 'it'\''s'",
        ),
        (
            &["--yes", "plugin", "", "$HOME", "`x`", "a\\b", "a\nb"],
            "rig plugin '' '$HOME' '`x`' 'a\\b' 'a\nb'",
        ),
    ] {
        let why = refused(args);
        let command = line.split(' ').nth(1).unwrap_or_default();
        assert_eq!(
            why,
            format!("`{command}` is the launcher's own command; run it as `{line}`"),
            "{args:?}"
        );
    }
    let unknown = refused(&["statz"]);
    assert!(
        unknown.contains(
            "unknown command `statz`; commands: build, update, plugin, help, help-agents, stats, import"
        ),
        "{unknown}"
    );
    assert!(refused(&["--nope"]).contains("options: -p, -m, --remote HOST:PATH, --yes"));
    assert!(refused(&["--remote"]).contains("needs a value"));
    assert!(refused(&["--yes=1"]).contains("takes no value"));
}

#[test]
fn a_home_at_a_root_lays_out_its_files_under_it() {
    let home = Home::at("some/root");
    assert!(home.root().is_absolute());
    assert!(home.root().ends_with("some/root"));
    assert_eq!(home.settings(), home.root().join("settings.toml"));
    assert_eq!(
        home.described("1700000000000000000 4096\n"),
        home.root().join("bin/described/1700000000000000000-4096")
    );
}

#[test]
fn ids_generated_in_one_second_by_one_process_differ() {
    let (first, second) = (super::SessionId::generate(), super::SessionId::generate());
    assert_ne!(first, second);
    assert!(second.as_str().parse::<super::SessionId>().is_ok());
}

#[test]
fn a_session_directory_knows_its_id() {
    let id = "1700000000-42".parse::<super::SessionId>().ok();
    assert!(id.is_some());
    let read = id.as_ref().and_then(|id| Home::at("root").session(id).id());
    assert_eq!(read, id);
}

#[test]
fn new_sessions_of_one_second_take_the_next_free_id() {
    let root = std::env::temp_dir().join(format!("rig-new-session-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let made: Vec<_> = (0..2).filter_map(|_| home.new_session().ok()).collect();
    let created = made
        .iter()
        .all(|(id, dir)| dir.id().as_ref() == Some(id) && dir.path().is_dir());
    std::fs::remove_dir_all(&root).ok();
    assert!(created && made.len() == 2, "{made:?}");
    let ids: Vec<_> = made.into_iter().map(|(id, _)| id).collect();
    assert_ne!(ids.first(), ids.last());
}

#[test]
fn a_session_is_saved_once_marked_and_not_for_a_log_file_alone_nor_while_deleting() {
    let root = std::env::temp_dir().join(format!("rig-saved-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let made = Home::at(&root).new_session();
    assert!(made.is_ok(), "{made:?}");
    let Ok((_, dir)) = made else { return };
    std::fs::create_dir_all(dir.path().join("logs")).ok();
    std::fs::write(dir.path().join("logs/a1.jsonl"), "").ok();
    let before = dir.is_saved();
    let marked = dir.mark_saved();
    let after = dir.is_saved();
    let deleting = dir.mark_deleting();
    let hidden = !dir.is_saved() && dir.is_deleting();
    std::fs::remove_dir_all(&root).ok();
    assert!(
        marked.is_ok() && deleting.is_ok(),
        "{marked:?} {deleting:?}"
    );
    assert!(!before && after && hidden);
}

/// A session holds logs once a log in `logs` is not empty, and counts as
/// holding them when `logs` cannot be read, so no deleter removes it then.
#[test]
fn a_session_holds_logs_once_one_is_not_empty_or_when_they_cannot_be_read() {
    let root = std::env::temp_dir().join(format!("rig-holds-logs-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let dir = super::SessionDir::at(&root);
    let logs = root.join("logs");
    let none = dir.holds_logs();
    std::fs::create_dir_all(&logs).ok();
    std::fs::write(logs.join("a1.jsonl"), "").ok();
    let empty = dir.holds_logs();
    std::fs::write(logs.join("a1.jsonl"), "{}\n").ok();
    let written = dir.holds_logs();
    std::fs::remove_dir_all(&logs).ok();
    std::fs::write(&logs, "").ok();
    let unreadable = dir.holds_logs();
    std::fs::remove_dir_all(&root).ok();
    assert_eq!(
        (none, empty, written, unreadable),
        (false, false, true, true),
        "(no logs, an empty log, a written log, logs unreadable)"
    );
}

#[test]
fn help_lists_the_secrets_a_build_asks_for_by_their_variable() {
    let usage = described().usage();
    assert!(usage.contains("rig stats"), "{usage}");
    assert!(usage.contains("--remote HOST:PATH"), "{usage}");
    let secret = usage.lines().find(|line| line.contains("RIG_SECRET_CRYPT"));
    assert!(
        secret.is_some_and(|line| line.contains("Passphrase")),
        "{usage}"
    );
    assert!(!AgentDescription::default().usage().contains("secrets"));
}

/// A name too long for the column keeps a line of its own, and its help
/// starts the next one at the column.
#[test]
fn help_puts_a_long_name_on_a_line_of_its_own() {
    let mut report = described();
    if let Some(command) = report.commands.first_mut() {
        command.usage = "list|migrate|rekey".to_owned();
    }
    let usage = report.usage();
    let column = " ".repeat(25);
    assert!(
        usage.contains(&format!("  rig stats list|migrate|rekey\n{column}Count")),
        "{usage}"
    );
    assert!(
        usage.contains("  --remote HOST:PATH     Work over SSH\n"),
        "{usage}"
    );
}

#[test]
fn a_session_whose_lock_another_file_holds_runs_elsewhere() {
    let root = std::env::temp_dir().join(format!("rig-runs-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let made = Home::at(&root).new_session();
    assert!(made.is_ok(), "{made:?}");
    let Ok((_, dir)) = made else { return };
    let idle = dir.runs_elsewhere();
    let lock = std::fs::File::create(dir.launcher_lock());
    let held = lock.as_ref().is_ok_and(|lock| lock.try_lock().is_ok());
    let running = dir.runs_elsewhere();
    drop(lock);
    let after = dir.runs_elsewhere();
    std::fs::remove_dir_all(&root).ok();
    assert!(held && !idle && running && !after);
}

/// A saved session is found by its id, or by the name in its directory
/// when no other saved session has it; an unsaved session, a name nobody
/// has and a name two share are refused, the last with both ids.
#[test]
fn a_saved_session_is_found_by_its_id_or_its_own_name() {
    let root = std::env::temp_dir().join(format!("rig-find-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let made: Vec<_> = (0..3).filter_map(|_| home.new_session().ok()).collect();
    let [(parser, first), (tests, second), (unsaved, third)] = made.as_slice() else {
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(made.len(), 3, "{made:?}");
        return;
    };
    let written = [
        first.mark_saved(),
        second.mark_saved(),
        first.set_name(Some(" parser\n")),
        second.set_name(Some("tests")),
        third.set_name(Some("tests")),
    ];
    let name = first.name();
    let by_id = home.find(parser.as_str());
    let by_name = home.find("parser");
    let only_saved = home.find(" tests ");
    let not_saved = home.find(unsaved.as_str());
    let nobody = home.find("nobody");
    let marked = third.mark_saved();
    let shared = home.find("tests");
    let unnamed = (
        second.set_name(None),
        second.set_name(Some("  ")),
        second.name(),
    );
    let left = home.find("tests");
    std::fs::remove_dir_all(&root).ok();

    assert!(written.iter().all(Result::is_ok) && marked.is_ok());
    assert_eq!(name.ok().flatten().as_deref(), Some("parser"));
    assert_eq!((by_id, by_name), (Ok(parser.clone()), Ok(parser.clone())));
    assert_eq!(only_saved, Ok(tests.clone()));
    assert!(not_saved.is_err() && nobody.is_err());
    assert!(
        shared
            .as_ref()
            .is_err_and(|why| why.contains(tests.as_str()) && why.contains(unsaved.as_str())),
        "{shared:?}"
    );
    assert!(unnamed.0.is_ok() && unnamed.1.is_ok() && unnamed.2.is_ok_and(|name| name.is_none()));
    assert_eq!(left, Ok(unsaved.clone()));
}

/// A saved session whose name cannot be read may have the wanted name, so
/// finding by name is refused naming its file and saying that resuming by
/// id still works; finding by id is not affected.
#[test]
fn a_name_that_cannot_be_read_refuses_finding_by_name_but_not_by_id() {
    let root = std::env::temp_dir().join(format!("rig-find-unreadable-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let made: Vec<_> = (0..2).filter_map(|_| home.new_session().ok()).collect();
    let [(wanted, first), (_, second)] = made.as_slice() else {
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(made.len(), 2, "{made:?}");
        return;
    };
    let unreadable = second.path().join("name");
    let written = [
        first.mark_saved(),
        second.mark_saved(),
        first.set_name(Some("mysess")),
        std::fs::create_dir_all(&unreadable),
    ];
    let by_name = home.find("mysess");
    let by_id = home.find(wanted.as_str());
    std::fs::remove_dir_all(&root).ok();

    assert!(written.iter().all(Result::is_ok), "{written:?}");
    let read = format!("read {}: ", unreadable.display());
    assert!(
        by_name
            .as_ref()
            .is_err_and(|why| why.starts_with(&read) && why.ends_with("resume it by its id")),
        "{by_name:?}"
    );
    assert_eq!(by_id, Ok(wanted.clone()));
}

/// `-r` goes wherever an option goes, so `rig -m a/b -r fix-login -p hi`
/// resumes `fix-login`; never among a command's words or after `--`.
#[test]
fn resume_goes_wherever_an_option_goes() {
    let report = described();
    let args = ["-m", "a/b", "-r", "fix-login", "-p", "hi"];
    let read = report.invocation(&args);
    assert_eq!(
        read.map(|invocation| invocation.print),
        Ok(Some("hi".to_owned()))
    );
    assert_eq!(report.resume(&args).as_deref(), Some("fix-login"));
    assert_eq!(report.resume(&["--resume=x"]).as_deref(), Some("x"));
    // Only a build that says `--remote` takes a value reads `-r` after it.
    let spaced = ["--remote", "h:/p", "-r", "x"];
    assert_eq!(report.resume(&spaced).as_deref(), Some("x"));
    assert_eq!(AgentDescription::default().resume(&spaced), None);
    for args in [&["stats", "-r", "x"][..], &["-p", "--", "-r", "x"]] {
        assert_eq!(report.resume(args), None, "{args:?}");
    }
    assert!(
        Invocation::parse(&["-r", "x"]).is_err(),
        "the agent never gets it"
    );
}

/// A group whose leader exited is killed while a process it left carries
/// the ledger's mark: an MCP server a wrapper started, say.
#[cfg(unix)]
#[test]
fn a_group_whose_leader_left_is_killed_for_the_process_it_left() {
    use std::os::unix::process::CommandExt;

    let root = std::env::temp_dir().join(format!("rig-protocol-ledger-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let run = super::RunDir::at(&root);
    let left = root.join("left");
    // The leader starts a sleep in its own group, writes its id, and exits.
    let script = format!("sleep 60 & echo $! > '{}'", left.display());
    let leader = std::process::Command::new("sh")
        .args(["-c", &script])
        .env(super::env::PROCESS_LEDGER, run.groups())
        .process_group(0)
        .spawn();
    let leader = leader.and_then(|mut leader| leader.wait().map(|_| leader.id()));
    assert!(leader.is_ok());
    let Ok(leader) = leader else { return };
    let sleeper = std::fs::read_to_string(&left).unwrap_or_default();
    // Gone once killed, even before whoever adopted it reaps it.
    let alive = || {
        let state = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", sleeper.trim()])
            .output();
        state.is_ok_and(|state| {
            let state = String::from_utf8_lossy(&state.stdout);
            !state.trim().is_empty() && !state.trim_start().starts_with('Z')
        })
    };
    assert!(!sleeper.trim().is_empty() && alive());
    std::fs::write(run.groups(), format!("{leader}\n")).ok();
    assert!(run.kill_left_groups().is_ok());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while alive() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(!alive());
    std::fs::remove_dir_all(root).ok();
}

/// The groups an agent left are asked to stop before they are killed, so
/// one that ends on its own when asked, such as an agent that ends what it
/// started, gets to; one that ignores the request is killed after the
/// grace.
#[cfg(unix)]
#[test]
fn groups_left_are_asked_to_stop_before_they_are_killed() {
    use std::os::unix::process::CommandExt;

    let root = std::env::temp_dir().join(format!("rig-protocol-term-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let run = super::RunDir::at(&root);
    let asked = root.join("asked");
    let ready = [root.join("heeds"), root.join("ignores")];
    let [heeding, ignoring] = ready.each_ref().map(|ready| ready.display());
    // Each says it is ready once its trap is set.
    let heeds = format!(
        "trap 'touch \"{}\"; exit' TERM; touch '{heeding}'; while :; do sleep 0.05; done",
        asked.display()
    );
    let ignores = format!("trap '' TERM; touch '{ignoring}'; exec sleep 60");
    let mut leaders = Vec::new();
    for script in [heeds, ignores] {
        let leader = std::process::Command::new("sh")
            .args(["-c", &script])
            .env(super::env::PROCESS_LEDGER, run.groups())
            .process_group(0)
            .spawn();
        assert!(leader.is_ok());
        leaders.extend(leader.ok());
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !ready.iter().all(|ready| ready.exists()) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let ids: String = leaders
        .iter()
        .map(|leader| format!("{}\n", leader.id()))
        .collect();
    std::fs::write(run.groups(), ids).ok();
    let started = std::time::Instant::now();
    assert!(run.kill_left_groups().is_ok());
    let took = started.elapsed();
    let ended: Vec<bool> = leaders
        .iter_mut()
        .map(|leader| leader.wait().is_ok())
        .collect();
    assert_eq!(ended, [true, true]);
    assert!(asked.exists(), "the group was killed without being asked");
    assert!(took >= super::RunDir::STOP_GRACE, "{took:?}");
    std::fs::remove_dir_all(root).ok();
}

/// A group whose marked processes end when asked is still killed while
/// another of its processes, without the mark, ignores the request.
#[cfg(target_os = "linux")]
#[test]
fn a_group_is_killed_while_it_holds_processes_once_asked() {
    use std::os::unix::process::CommandExt;

    let root = std::env::temp_dir().join(format!("rig-protocol-unmarked-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let run = super::RunDir::at(&root);
    let pid = root.join("stubborn");
    // The marked leader starts an unmarked member that ignores SIGTERM,
    // then waits, ending on SIGTERM itself.
    let script = format!(
        "env -i sh -c \"trap '' TERM; echo \\$\\$ > '{}'; exec sleep 60\" & wait",
        pid.display()
    );
    let leader = std::process::Command::new("sh")
        .args(["-c", &script])
        .env(super::env::PROCESS_LEDGER, run.groups())
        .process_group(0)
        .spawn();
    let Ok(mut leader) = leader else {
        return assert!(leader.is_ok());
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !pid.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
    let stubborn = std::fs::read_to_string(&pid).unwrap_or_default();
    std::fs::write(run.groups(), format!("{}\n", leader.id())).ok();
    assert!(run.kill_left_groups().is_ok());
    leader.wait().ok();
    let gone = || {
        std::fs::read_to_string(format!("/proc/{}/stat", stubborn.trim())).map_or(true, |stat| {
            stat.rsplit(')')
                .next()
                .unwrap_or_default()
                .trim_start()
                .starts_with('Z')
        })
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !gone() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::fs::remove_dir_all(&root).ok();
    assert!(
        !stubborn.trim().is_empty() && gone(),
        "the unmarked member outlived the cleanup"
    );
}

/// A new session's directory and `sessions/`, which hold every
/// conversation, are the user's alone whatever the umask, and an existing
/// wider `sessions/` is tightened.
#[cfg(unix)]
#[test]
fn a_new_session_directory_is_private_to_the_user() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("rig-private-session-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    std::fs::create_dir_all(home.sessions()).ok();
    let wide = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(home.sessions(), wide).ok();
    let mode = |path: &std::path::Path| {
        std::fs::metadata(path)
            .map(|meta| meta.permissions().mode() & 0o777)
            .ok()
    };
    let made = home.new_session();
    let modes = made
        .as_ref()
        .map(|(_, dir)| (mode(&home.sessions()), mode(dir.path())));
    std::fs::remove_dir_all(&root).ok();
    let private = |mode: Option<u32>| mode.is_some_and(|mode| mode & 0o077 == 0);
    assert!(
        modes
            .as_ref()
            .is_ok_and(|(sessions, session)| private(*sessions) && private(*session)),
        "sessions/ and the session directory modes: {modes:?}"
    );
}

/// The four directories that hold what the user said are tightened in a
/// home of ordinary modes, whether old, missing or linked elsewhere. The
/// home keeps its mode, even through a link to it, and so does a file where
/// a directory belongs.
#[cfg(unix)]
#[test]
fn keep_private_tightens_the_four_directories_and_not_the_home() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let root = std::env::temp_dir().join(format!("rig-keep-private-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let wide = std::fs::Permissions::from_mode(0o755);
    let home = Home::at(root.join("home"));
    let elsewhere = root.join("elsewhere");
    for dir in [home.sessions(), home.projects(), elsewhere.clone()] {
        std::fs::create_dir_all(&dir).ok();
        std::fs::set_permissions(&dir, wide.clone()).ok();
    }
    symlink(&elsewhere, home.runs()).ok();
    std::fs::set_permissions(home.root(), wide.clone()).ok();
    let odd = Home::at(root.join("odd"));
    std::fs::create_dir_all(odd.root()).ok();
    std::fs::set_permissions(odd.root(), wide).ok();
    symlink(".", odd.projects()).ok();
    std::fs::write(odd.data(), "not a directory").ok();
    std::fs::set_permissions(odd.data(), std::fs::Permissions::from_mode(0o644)).ok();
    let mode = |path: &std::path::Path| {
        std::fs::metadata(path)
            .map(|meta| meta.permissions().mode() & 0o777)
            .ok()
    };
    home.keep_private();
    odd.keep_private();
    let dirs = [home.sessions(), home.projects(), home.data(), home.runs()];
    let modes = dirs.each_ref().map(|dir| mode(dir));
    let kept = [mode(home.root()), mode(odd.root()), mode(&odd.data())];
    std::fs::remove_dir_all(&root).ok();
    assert!(
        modes
            .iter()
            .all(|mode| mode.is_some_and(|mode| mode & 0o077 == 0)),
        "{modes:?}"
    );
    assert_eq!(kept, [Some(0o755), Some(0o755), Some(0o644)]);
}

/// A file renamed over the launcher's path replaces it; the same file read
/// again, as at a plain reload, and a file gone do not. A path that cannot
/// be looked at, here under a file, has no stamp: it is an error.
#[test]
fn a_stamp_tells_a_replaced_file_from_the_same_one_or_none() {
    let dir = std::env::temp_dir().join(format!("rig-stamp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let launcher = dir.join("rig");
    let stamps = (|| -> std::io::Result<_> {
        std::fs::write(&launcher, "old")?;
        let started = Stamp::of(&launcher)?;
        let again = Stamp::of(&launcher)?;
        let under_a_file = Stamp::of(&launcher.join("rig")).is_err();
        std::fs::write(dir.join(".rig"), "new")?;
        std::fs::rename(dir.join(".rig"), &launcher)?;
        let replaced = Stamp::of(&launcher)?;
        std::fs::remove_file(&launcher)?;
        let gone = Stamp::of(&launcher)?;
        Ok((started, again, under_a_file, replaced, gone))
    })();
    std::fs::remove_dir_all(&dir).ok();
    let stamps = stamps.map_err(|failure| failure.to_string());
    assert!(stamps.is_ok(), "{stamps:?}");
    let Ok((started, again, under_a_file, replaced, gone)) = stamps else {
        return;
    };
    assert!(under_a_file);
    assert!(!started.replaced(&again));
    assert!(started.replaced(&replaced));
    assert!(!started.replaced(&gone));
    assert!(!gone.replaced(&replaced));
    // As the launcher passes it on to its agent.
    assert_eq!(Stamp::parse(&started.text()), started);
    assert_eq!(Stamp::parse(&gone.text()), gone);
    assert_eq!(Stamp::parse("12:x"), gone);
}

/// A list holding bytes that are not text still names its groups, to the
/// agent that takes them over and to the launcher that kills them.
#[cfg(unix)]
#[test]
fn a_list_with_bytes_that_are_not_text_still_names_its_groups() {
    use std::os::unix::process::CommandExt;

    let root = std::env::temp_dir().join(format!("rig-protocol-bytes-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let run = super::RunDir::at(&root);
    let job = std::process::Command::new("sleep")
        .arg("30")
        .env(super::env::PROCESS_LEDGER, run.groups())
        .process_group(0)
        .spawn();
    assert!(job.is_ok());
    let Ok(mut job) = job else { return };
    let mut listed = b"\xff\n".to_vec();
    listed.extend_from_slice(format!("{}\n", job.id()).as_bytes());
    std::fs::write(run.groups(), listed).ok();
    let live = run.live_groups().ok();
    let killed = run.kill_left_groups().is_ok();
    // Reaped only once killed, so `try_wait` sees it ended.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut ended = job.try_wait().ok().flatten().is_some();
    while !ended && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
        ended = job.try_wait().ok().flatten().is_some();
    }
    job.kill().ok();
    job.wait().ok();
    std::fs::remove_dir_all(root).ok();
    let id = i32::try_from(job.id()).ok();
    assert_eq!(live, id.map(|id| vec![id]));
    assert!(killed && ended, "killed: {killed}, ended: {ended}");
}

/// A variable whose name starts with the secret prefix is a secret even
/// when the rest of its name is not UTF-8, so no child inherits it.
#[cfg(unix)]
#[test]
fn a_secret_name_that_is_not_utf8_is_still_a_secret() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    assert!(is_secret_variable(OsStr::from_bytes(b"RIG_SECRET_\xff")));
    assert!(is_secret_variable(OsStr::new("RIG_SECRET_CRYPT")));
    assert!(!is_secret_variable(OsStr::from_bytes(b"RIG_\xffSECRET_")));
}
