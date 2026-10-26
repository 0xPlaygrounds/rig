//! The `rig` launcher. It generates a Cargo project for the rig-harness agent
//! with the plugins listed in `$RIG_HOME/plugins.toml`, builds it, and runs it.
//! When the agent exits with the reload code it starts the new build, and a
//! build that crashes during startup is rolled back to the last one that
//! worked. It uses only std.
//!
//! Its own first words (`build`, `plugin`, `help`, `help-agents`,
//! `--version`) are read before the build; the rest of the command line,
//! `--resume` included, once the agent is built, against the commands and
//! options its plugins add (`rig help` lists them).

mod launcher;

use std::process::ExitCode;

use launcher::plugin::USAGE as PLUGIN_USAGE;
use launcher::run::last_report;
use rig_harness_protocol::{AgentDescription, Home, INVOCATION_USAGE};

const USAGE: &str = "\
Usage: rig [-r <id|name>] [options] [-p prompt…] | rig [options] <command> [args…]
       | rig build | rig plugin <command> | rig help [command]

Headless, for scripts and agents: `rig -p \"task\"` answers and exits,
`rig -p --json \"task\"` writes JSON lines, `rig --json` takes requests on
stdin; `rig help-agents` says how.

  rig                    Build the agent if needed and run it in a new session.
  rig <command> [args…]  Build the agent if needed and run a plugin's command.
  rig build              Regenerate the agent project from plugins.toml, build
                         it, and stage the new binary for the next start.
  rig help [command]     This text, or what a command does.
  rig help-agents        How a script or another agent drives rig headless,
                         over its stdin and stdout (-p, --json).
  rig --version          The launcher's version.
";

const SESSION: &str = "
Session:
  -r, --resume <id|name>
                         Resume the session with that id, or the one /name
                         gave that name, in the directory it ran in; it goes
                         wherever an option goes. In the agent, /resume
                         lists the sessions.

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

/// `rig help-agents`: the page a driver reads once.
const AGENTS: &str = include_str!("launcher/help-agents.txt");

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
        ["help-agents"] => {
            print!("{AGENTS}");
            Ok(ExitCode::SUCCESS)
        }
        ["--version" | "-V"] => {
            println!("rig {}", launcher::VERSION);
            Ok(ExitCode::SUCCESS)
        }
        ["help", command] => Ok(help(&home, command)),
        _ => launcher::run::run(&home, &args),
    };
    result.unwrap_or_else(|failure| {
        eprintln!("error: {failure}");
        ExitCode::FAILURE
    })
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
        "help-agents" => print!("{AGENTS}"),
        "plugin" => print!("{PLUGIN_USAGE}"),
        command => {
            let report = last_report(home).unwrap_or_default();
            let Some(info) = report.command(command) else {
                eprintln!(
                    "rig: unknown command `{command}`; commands: {}{}",
                    report.command_names().join(", "),
                    if report == AgentDescription::default() {
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
