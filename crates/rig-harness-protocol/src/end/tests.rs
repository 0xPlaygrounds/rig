use std::path::PathBuf;

use super::{AgentEnd, FailCode};
use crate::RunDir;

/// A run directory of the test's own named `name`, empty.
fn run_dir(name: &str) -> RunDir {
    let root: PathBuf =
        std::env::temp_dir().join(format!("rig-protocol-end-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).ok();
    RunDir::at(root)
}

/// Every end, each with what a record of it must carry.
fn ends() -> Vec<AgentEnd> {
    let session = "1700000000-19".parse().ok();
    vec![
        AgentEnd::Quit,
        AgentEnd::Reload,
        AgentEnd::Switch(None),
        AgentEnd::Switch(session),
        AgentEnd::Refused("a reason\nover two lines, 'quoted'.".to_owned()),
        AgentEnd::Signal,
        AgentEnd::Failed(FailCode::FAILURE),
        AgentEnd::Failed(FailCode::or_failure(2)),
    ]
}

/// The status of a process that exited with `code`.
#[cfg(unix)]
fn exited(code: u8) -> std::process::ExitStatus {
    use std::os::unix::process::ExitStatusExt;

    std::process::ExitStatus::from_raw(i32::from(code) << 8)
}

#[cfg(unix)]
#[test]
fn each_end_reads_back_as_written_when_the_agent_exited_with_its_code() {
    let run = run_dir("round-trip");
    let read: Vec<Option<AgentEnd>> = ends()
        .iter()
        .map(|end| {
            end.write(&run).ok()?;
            Some(AgentEnd::read(&run, exited(end.code())))
        })
        .collect();
    std::fs::remove_dir_all(run.path()).ok();
    assert_eq!(read, ends().into_iter().map(Some).collect::<Vec<_>>());
}

/// A record whose code is not the one the agent exited with, as when it
/// was killed after writing it, is not read; nor is a missing or garbled
/// one. The
/// status alone never asks to reload or switch.
#[cfg(unix)]
#[test]
fn without_a_record_for_its_exit_the_status_says_how_the_agent_ended() {
    use std::os::unix::process::ExitStatusExt;

    let run = run_dir("status");
    let killed = std::process::ExitStatus::from_raw(9);
    let without = [
        (exited(0), AgentEnd::Quit),
        (exited(130), AgentEnd::Signal),
        (exited(75), AgentEnd::Failed(FailCode::FAILURE)),
        (exited(3), AgentEnd::Failed(FailCode::or_failure(3))),
        (killed, AgentEnd::Failed(FailCode::FAILURE)),
    ];
    let absent: Vec<Option<AgentEnd>> = without
        .iter()
        .map(|(status, _)| Some(AgentEnd::read(&run, *status)))
        .collect();
    let switch = AgentEnd::Switch(None);
    let written = switch.write(&run);
    let after_kill = Some(AgentEnd::read(&run, killed));
    let after_quit = Some(AgentEnd::read(&run, exited(0)));
    let garbled = std::fs::write(run.path().join(super::FILE), "reload sometime");
    let after_garble = Some(AgentEnd::read(&run, exited(75)));
    std::fs::remove_dir_all(run.path()).ok();
    assert!(written.is_ok() && garbled.is_ok());
    let expected: Vec<Option<AgentEnd>> = without.into_iter().map(|(_, end)| Some(end)).collect();
    assert_eq!(absent, expected);
    assert_eq!(after_kill, Some(AgentEnd::Failed(FailCode::FAILURE)));
    assert_eq!(after_quit, Some(AgentEnd::Quit));
    assert_eq!(after_garble, Some(AgentEnd::Failed(FailCode::FAILURE)));
}

#[test]
fn a_failure_code_is_never_one_the_launcher_reads_as_another_end() {
    assert_eq!(FailCode::new(0), None);
    assert_eq!(FailCode::new(75), None);
    assert_eq!(FailCode::new(130), None);
    assert_eq!(FailCode::new(2).map(FailCode::get), Some(2));
    assert_eq!(FailCode::or_failure(75), FailCode::FAILURE);
    assert_eq!(FailCode::FAILURE.get(), 1);
    assert_eq!(AgentEnd::parse("failed 75"), None);
}

/// The stand-in agent's steps say the same messages: what its script
/// writes reads back as the end it was given, and as ready and the screen.
#[cfg(unix)]
#[test]
fn the_fake_agent_says_what_the_agent_says() {
    use crate::fake_agent::{self, Step};

    let run = run_dir("fake");
    let home = run.path().join("home");
    std::fs::create_dir_all(&home).ok();
    let agent = run.path().join("agent");
    let ran: Vec<Option<(AgentEnd, bool, bool)>> = ends()
        .into_iter()
        .map(|end| {
            std::fs::remove_file(home.join("fake-agent-starts")).ok();
            run.clear_messages().ok()?;
            let steps = [Step::Ready, Step::Screen, Step::End(end)];
            std::fs::write(&agent, fake_agent::script(&[&steps])).ok()?;
            let status = std::process::Command::new("sh")
                .arg(&agent)
                .env(crate::env::RUN, run.path())
                .env(crate::env::HOME, &home)
                .env_remove(crate::env::DESCRIBE)
                .status()
                .ok()?;
            let read = AgentEnd::read(&run, status);
            Some((read, run.heard_ready(), run.took_screen()))
        })
        .collect();
    std::fs::remove_dir_all(run.path()).ok();
    let expected: Vec<Option<(AgentEnd, bool, bool)>> = ends()
        .into_iter()
        .map(|end| Some((end, true, true)))
        .collect();
    assert_eq!(ran, expected);
}
