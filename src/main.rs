//! The `rig` launcher. It generates a Cargo project for the rig-harness agent
//! with the plugins listed in `$RIG_HOME/plugins.toml`, builds it, and runs it.
//! When the agent exits with the reload code it starts the new build, and a
//! build that crashes during startup is rolled back to the last one that
//! worked. It uses only std.

mod launcher;

use std::process::ExitCode;

use launcher::run::Start;
use rig::harness_protocol::{Home, INVOCATION_USAGE, Invocation, Mode};

const USAGE: &str = "\
Usage: rig [session] [mode] | rig build | rig help

  rig                    Build the agent if needed and run it. In a directory
                         whose last session did not quit cleanly, that session
                         resumes.
  rig build              Regenerate the agent project from plugins.toml, build
                         it, and stage the new binary for the next start.

Session (a print run starts a new one unless told otherwise):
  -c, --continue         Continue the last session run in this directory.
  -r, --resume <id>      Resume the session <id>, in the directory it ran in.
                         In the agent, /resume lists the sessions.
  -n, --new              Start a new session.

Mode:
";

const ENVIRONMENT: &str = "
Environment:
  RIG_HOME    Root of every rig directory (default: ~/.rig).
  RIG_SOURCE  A rig checkout to build the agent from instead of crates.io.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let home = Home::from_env();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["build"] => launcher::build::build(&home).map(|()| ExitCode::SUCCESS),
        ["help" | "--help" | "-h"] => {
            print!("{USAGE}{INVOCATION_USAGE}{ENVIRONMENT}");
            Ok(ExitCode::SUCCESS)
        }
        _ => match parse(&args) {
            Ok((start, invocation)) => launcher::run::run(&home, start, &invocation),
            Err(failure) => {
                eprint!("error: {failure}\n\n{USAGE}{INVOCATION_USAGE}{ENVIRONMENT}");
                return ExitCode::from(2);
            }
        },
    };
    result.unwrap_or_else(|failure| {
        eprintln!("error: {failure}");
        ExitCode::FAILURE
    })
}

/// The session to run and the agent's arguments: the session options come
/// first, and the rest is the agent's.
fn parse(args: &[String]) -> Result<(Start, Invocation), String> {
    let mut start = None;
    let mut rest = args;
    loop {
        let (chosen, taken) = match rest {
            [flag, ..] if flag == "-c" || flag == "--continue" => (Start::Continue, 1),
            [flag, ..] if flag == "-n" || flag == "--new" => (Start::New, 1),
            [flag, id, ..] if flag == "-r" || flag == "--resume" => (
                Start::Resume(id.parse().map_err(|failure| format!("{failure}"))?),
                2,
            ),
            _ => break,
        };
        if start.replace(chosen).is_some() {
            return Err("give at most one of --continue, --resume and --new".to_owned());
        }
        rest = rest.get(taken..).unwrap_or_default();
    }
    let invocation = Invocation::parse(rest)?;
    // A headless run is its own session unless one is named: it must not
    // pick up the session this directory's terminal view left behind.
    let start = start.unwrap_or(match invocation.mode {
        Mode::Interactive => Start::Default,
        _ => Start::New,
    });
    Ok((start, invocation))
}
