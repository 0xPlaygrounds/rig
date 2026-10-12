//! The `rig` launcher. It generates a Cargo project for the rig-harness agent
//! with the plugins listed in `$RIG_HOME/plugins.toml`, builds it, and runs it.
//! When the agent exits with the reload code it starts the new build, and a
//! build that crashes during startup is rolled back to the last one that
//! worked. It uses only std.
//!
//! Its own first words (`build`, `plugin`, `help`) and `--resume` are read
//! before the build; the rest of the command line once the agent is built,
//! against the commands and options its plugins add (`rig help` lists
//! them).

mod launcher;

use std::process::ExitCode;

use launcher::plugin::USAGE as PLUGIN_USAGE;
use launcher::run::Start;
use rig::harness_protocol::{Described, Home, INVOCATION_USAGE};

const USAGE: &str = "\
Usage: rig [-r <id>] [options] [-p prompt…] | rig [options] <command> [args…]
       | rig build | rig plugin <command> | rig help [command]

  rig                    Build the agent if needed and run it in a new session.
  rig <command> [args…]  Build the agent if needed and run a plugin's command.
  rig build              Regenerate the agent project from plugins.toml, build
                         it, and stage the new binary for the next start.
  rig help [command]     This text, or what a command does.
";

const SESSION: &str = "
Session:
  -r, --resume <id>      Resume the session <id>, in the directory it ran in.
                         In the agent, /resume lists the sessions.

Mode:
";

const ENVIRONMENT: &str = "
Environment:
  RIG_HOME    Root of every rig directory (default: ~/.rig).
  RIG_SOURCE  A rig checkout to build the agent from instead of crates.io.
  RIG_SECRET_<NAME>
              A secret a plugin asks for, given to the agent alone; the
              launcher asks for one the environment does not hold.

Exit codes:
  0           Success.
  1           A launcher error, or a build that failed or was rejected.
  2           A command line the launcher or the agent refuses.
  other       The agent's own: a plugin's failure (it may also be 1 or 2).
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
        ["plugin", command @ ..] => {
            launcher::plugin::run(&home, command).map(|()| ExitCode::SUCCESS)
        }
        ["help" | "--help" | "-h"] => {
            print!(
                "{USAGE}{PLUGIN_USAGE}{SESSION}{INVOCATION_USAGE}{}{ENVIRONMENT}",
                plugin_usage(&home)
            );
            Ok(ExitCode::SUCCESS)
        }
        ["help", command] => Ok(help(&home, command)),
        _ => match resume(&args) {
            Ok((start, rest)) => launcher::run::run(&home, start, rest),
            Err(failure) => {
                eprintln!("rig: {failure}\nRun `rig help` for usage.");
                return ExitCode::from(2);
            }
        },
    };
    result.unwrap_or_else(|failure| {
        eprintln!("error: {failure}");
        ExitCode::FAILURE
    })
}

/// The report of the build that would run next, `good` else `staged`, as
/// kept from its last build; `None` when it has none. It never builds.
fn last_report(home: &Home) -> Option<Described> {
    [home.good(), home.staged()]
        .into_iter()
        .find(|binary| binary.exists())
        .and_then(|binary| launcher::build::kept_report(home, &binary))
}

/// The plugins' commands, options and secrets for `rig help`.
fn plugin_usage(home: &Home) -> String {
    last_report(home).map_or_else(
        || "\nRun `rig build` to list plugin commands.\n".to_owned(),
        |report| report.usage(),
    )
}

/// `rig help <command>`: its usage and what it does; exit code 2 for a
/// command the last build has not.
fn help(home: &Home, command: &str) -> ExitCode {
    match command {
        "build" | "help" => print!("{USAGE}"),
        "plugin" => print!("{PLUGIN_USAGE}"),
        command => {
            let report = last_report(home).unwrap_or_default();
            let Some(info) = report.command(command) else {
                eprintln!(
                    "rig: unknown command `{command}`; commands: {}{}",
                    report.command_names().join(", "),
                    if report == Described::default() {
                        " (run `rig build` to list plugin commands)"
                    } else {
                        ""
                    }
                );
                return ExitCode::from(2);
            };
            println!("Usage: rig {} {}\n\n  {}", info.name, info.usage, info.help);
        }
    }
    ExitCode::SUCCESS
}

/// The session to run, from `--resume <id>` before everything else, and
/// the rest of the command line.
fn resume(args: &[String]) -> Result<(Start, &[String]), String> {
    Ok(match args {
        [flag, id, rest @ ..] if flag == "-r" || flag == "--resume" => {
            (Start::Resume(id.parse()?), rest)
        }
        [flag] if flag == "-r" || flag == "--resume" => {
            return Err(format!("{flag} needs a session id"));
        }
        rest => (Start::New, rest),
    })
}
