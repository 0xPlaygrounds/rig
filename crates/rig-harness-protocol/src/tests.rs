use super::{
    AgentDescription, DescribedCommand, DescribedFlag, DescribedSecret, Home, Invocation,
    secret_variable,
};

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
    assert!(refused(&["--yes", "build"]).contains("put it first"));
    let unknown = refused(&["statz"]);
    assert!(
        unknown.contains("unknown command `statz`; commands: build, plugin, help, help-agents, stats, import"),
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
    assert_eq!(name.as_deref(), Some("parser"));
    assert_eq!((by_id, by_name), (Ok(parser.clone()), Ok(parser.clone())));
    assert_eq!(only_saved, Ok(tests.clone()));
    assert!(not_saved.is_err() && nobody.is_err());
    assert!(
        shared
            .as_ref()
            .is_err_and(|why| why.contains(tests.as_str()) && why.contains(unsaved.as_str())),
        "{shared:?}"
    );
    assert!(unnamed.0.is_ok() && unnamed.1.is_ok() && unnamed.2.is_none());
    assert_eq!(left, Ok(unsaved.clone()));
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
