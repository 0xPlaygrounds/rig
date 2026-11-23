use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{Agent, Carry, Launch, LaunchError, Purpose};
use crate::{RunDir, Stamp, StartReason, env};

/// The secret every test launch holds.
const PROBE: &str = "RIG_SECRET_PROBE";

/// The launch of `/opt/rig/bin/rig` in `/work/project` given `vars`.
fn launch(vars: &[(&str, &str)]) -> Result<Launch, LaunchError> {
    Launch::from_parts(
        Path::new("/opt/rig/bin/rig"),
        true,
        vars.iter()
            .map(|(name, value)| (OsStr::new(name), Some(OsStr::new(value)))),
        Some(Path::new("/work/project")),
        ["--model", "ollama/qwen3"].map(OsStr::new),
    )
}

/// The launch the process `command` starts reads, as [`Launch::from_process`]
/// would in it.
fn read_back(command: &Command, terminal: bool) -> Result<Launch, LaunchError> {
    Launch::from_parts(
        Path::new(command.get_program()),
        terminal,
        command.get_envs(),
        command.get_current_dir(),
        command.get_args(),
    )
}

/// What `command` sets (`Some`) or removes (`None`), by name.
fn envs(command: &Command) -> BTreeMap<String, Option<String>> {
    command
        .get_envs()
        .map(|(name, value)| {
            let text = |value: &OsStr| value.to_string_lossy().into_owned();
            (text(name), value.map(text))
        })
        .collect()
}

/// A re-exec's command reads back as the launch it carries: relative
/// paths made absolute against the directory they were given in, the
/// carried state, every warning whole whatever it holds, and the secrets,
/// typed ones too.
#[test]
fn a_reexec_reads_back_as_the_launch_it_carries() {
    let given = launch(&[
        (env::HOME, "../home/.rig"),
        (env::SOURCE, "rig"),
        (PROBE, "hunter2"),
        ("PATH", "/usr/bin"),
    ]);
    assert!(given.is_ok(), "{given:?}");
    let Ok(mut given) = given else {
        return;
    };
    assert_eq!(given.home.as_os_str(), "/work/project/../home/.rig");
    assert_eq!(
        given.source.as_deref(),
        Some(Path::new("/work/project/rig"))
    );
    given
        .secrets
        .insert("RIG_SECRET_TYPED".into(), "typed at the prompt".into());
    let warnings = [
        "rig no longer reads /home/a\nb/.rig/plugins.toml.",
        "a\nb\r\nc",
        "quotes \" ' and \\ backslashes, \t tabs",
        "",
        "ünïcode ✓",
    ];
    for count in 0..=warnings.len() {
        let mut launch = given.clone();
        launch.here = Some(super::AbsPath(PathBuf::from("/srv/elsewhere")));
        launch.carry = Carry {
            session: "1-2".parse().ok(),
            left: "3-4".parse().ok(),
            reason: Some(StartReason::Reloaded),
            warned: true,
            warnings: warnings
                .get(..count)
                .unwrap_or_default()
                .iter()
                .map(|warning| (*warning).to_owned())
                .collect(),
            notice: Some("Could not switch sessions: too old.".to_owned()),
            build_failure: Some("cargo could not build the agent".to_owned()),
        };
        let read = read_back(&launch.reexec(), launch.terminal);
        assert_eq!(read.ok().as_ref(), Some(&launch));
    }
}

/// Every child of each purpose: as many as [`Purpose`] has variants.
fn children(launch: &Launch) -> Vec<Command> {
    let all = [
        Purpose::Cargo,
        Purpose::Install(Path::new("/opt/rig")),
        Purpose::Rebuild,
        Purpose::Stty,
    ];
    // A new purpose fails to compile here until it is listed above.
    for purpose in all {
        match purpose {
            Purpose::Cargo | Purpose::Install(_) | Purpose::Rebuild | Purpose::Stty => {}
        }
    }
    let describing = launch.describing(
        Path::new("/home/.rig/bin/good"),
        Path::new("/run/describe/home"),
        Path::new("/run/describe/report"),
        Some(Path::new("/home/.rig/settings.toml")),
    );
    all.into_iter()
        .map(|purpose| launch.build_child(purpose))
        .chain([describing])
        .collect()
}

/// A secret the launcher holds reaches its agent and the launcher in its
/// place, and no other process it starts.
#[test]
fn only_the_agent_and_the_next_launcher_get_a_secret() {
    let given = launch(&[(PROBE, "hunter2"), (env::SESSION, "1-2")]);
    assert!(given.is_ok(), "{given:?}");
    let Ok(launch) = given else {
        return;
    };
    for child in children(&launch) {
        assert_eq!(
            envs(&child).get(PROBE),
            Some(&None),
            "{:?}",
            child.get_program()
        );
    }
    let run = RunDir::at("/home/.rig/runs/1");
    let agent = launch.agent(&Agent {
        program: Path::new("/home/.rig/bin/good"),
        args: &[],
        dir: None,
        run: &run,
        version: "0.46.0",
        launcher: Stamp::of(Path::new("/nonexistent")).ok(),
    });
    for command in [agent, launch.reexec()] {
        assert_eq!(envs(&command).get(PROBE), Some(&Some("hunter2".to_owned())));
    }
}

/// Each process gets the protocol's variables it is told, and drops every
/// other one it would inherit: a build child none but the reload build's
/// home, source and session; an agent without a launcher no launcher,
/// nor the warnings of the launcher that started the agent starting it.
#[test]
fn each_process_gets_only_the_variables_it_is_told() {
    let inherited = [
        (env::HOME, "/home/.rig"),
        (env::SOURCE, "/src/rig"),
        (env::SESSION, "1-2"),
        (env::START, "resumed"),
        (env::WARNING, r#"["The launcher's own warning."]"#),
        (env::NOTICE, "Resumed the session."),
        (env::BUILD_FAILURE, "failed"),
        (env::RUN, "/home/.rig/runs/0"),
        (env::LAUNCHER, "/opt/rig/bin/rig"),
        (env::LAUNCHER_STAMP, "1.2:3"),
        (env::VERSION, "0.46.0 (abc)"),
        (env::SWITCHED_FROM, "5-6"),
        (env::WARNED, "1"),
    ];
    let given = launch(&inherited);
    assert!(given.is_ok(), "{given:?}");
    let Ok(launch) = given else {
        return;
    };
    let set = |command: &Command| -> BTreeMap<String, String> {
        envs(command)
            .into_iter()
            .filter_map(|(name, value)| Some((name, value?)))
            .collect()
    };
    let removed = |command: &Command| -> Vec<String> {
        envs(command)
            .into_iter()
            .filter(|(_, value)| value.is_none())
            .map(|(name, _)| name)
            .collect()
    };
    for (name, _) in inherited {
        for child in children(&launch) {
            let envs = envs(&child);
            assert!(envs.contains_key(name), "{name} {:?}", child.get_program());
        }
    }
    let rebuild = launch.build_child(Purpose::Rebuild);
    let expected: BTreeMap<String, String> = [
        (env::HOME, "/home/.rig"),
        (env::SOURCE, "/src/rig"),
        (env::SESSION, "1-2"),
    ]
    .map(|(name, value)| (name.to_owned(), value.to_owned()))
    .into();
    assert_eq!(set(&rebuild), expected);
    assert!(set(&launch.build_child(Purpose::Cargo)).is_empty());

    // An ACP session's agent, started by the router with a launch of its
    // own: its session, its reason, and nothing it was not told.
    let mut session = launch.clone();
    session.carry = Carry {
        session: "7-8".parse().ok(),
        reason: Some(StartReason::New),
        ..Carry::default()
    };
    let run = RunDir::at("/home/.rig/runs/acp/1");
    let args: Vec<OsString> = vec!["--front=acp-session".into()];
    let agent = session.agent(&Agent {
        program: Path::new("/home/.rig/runs/acp/rig"),
        args: &args,
        dir: Some(Path::new("/work/client")),
        run: &run,
        version: "0.46.0 (abc)",
        launcher: None,
    });
    let expected: BTreeMap<String, String> = [
        (env::HOME, "/home/.rig"),
        (env::SOURCE, "/src/rig"),
        (env::SESSION, "7-8"),
        (env::START, "new"),
        (env::RUN, "/home/.rig/runs/acp/1"),
        (env::VERSION, "0.46.0 (abc)"),
    ]
    .map(|(name, value)| (name.to_owned(), value.to_owned()))
    .into();
    assert_eq!(set(&agent), expected);
    let removed = removed(&agent);
    for name in [
        env::WARNING,
        env::NOTICE,
        env::BUILD_FAILURE,
        env::LAUNCHER,
        env::LAUNCHER_STAMP,
        env::SWITCHED_FROM,
        env::WARNED,
    ] {
        assert!(removed.iter().any(|removed| removed == name), "{name}");
    }
    assert_eq!(agent.get_current_dir(), Some(Path::new("/work/client")));
}

/// A variable that does not hold what the protocol writes is an error
/// naming it, never read as unset.
#[test]
fn a_value_that_does_not_read_names_its_variable() {
    for (name, value) in [
        (env::START, "rebooted"),
        (env::SESSION, "abc"),
        (env::SWITCHED_FROM, "../x"),
        (env::WARNED, "yes"),
    ] {
        let read = launch(&[(name, value)]).map_err(|failure| failure.to_string());
        assert!(
            read.as_ref()
                .is_err_and(|failure| failure.starts_with(name)),
            "{name}: {read:?}"
        );
    }
}
