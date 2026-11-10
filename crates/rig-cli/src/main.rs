//! The `rig` launcher. It generates a Cargo project for the rig-harness agent
//! with the plugins `$RIG_HOME/settings.toml` turns on, builds it, and runs
//! it.
//! When the agent exits with the reload code it starts the new build, and a
//! build that crashes during startup is rolled back to the last one that
//! worked. It uses std, and libc on unix (ctrlc elsewhere) to stop its build
//! and to pass a signal on to its agent and wait for it, when told to end.
//!
//! Its own first words (`build`, `update`, `plugin`, `help`, `help-agents`,
//! `--version`) are read before the build; the rest of the command line,
//! `--resume` included, once the agent is built, against the commands and
//! options its plugins add (`rig help` lists them).

mod build;
mod locks;
mod plugin;
mod project;
mod run;
mod update;

use std::process::ExitCode;

use plugin::USAGE as PLUGIN_USAGE;
use rig_harness_protocol::{AgentDescription, Home, INVOCATION_USAGE, REFUSED_EXIT_CODE};
use run::last_report;

pub use rig_harness_protocol::{BEVY_VERSION, VERSION};

/// What `rig --version` prints after `rig`: [`VERSION`], with the commit
/// when the launcher is built in a rig checkout (`build.rs`). The agent
/// gets it as [`env::VERSION`](rig_harness_protocol::env::VERSION).
pub const LONG_VERSION: &str = env!("RIG_LONG_VERSION");

/// The commit the launcher was built from, when built in a rig checkout;
/// else empty (`build.rs`).
pub const COMMIT: &str = env!("RIG_COMMIT");

/// Every launcher failure is reported to the user as its message.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "\
Usage: rig [-r <id|name>] [options] [-p prompt…] | rig [options] <command> [args…]
       | rig build | rig update | rig plugin <command> | rig help [command]

Headless, for scripts and agents: `rig -p \"task\"` answers and exits,
`rig -p --json \"task\"` writes JSON lines, `rig --json` takes requests on
stdin; `rig help-agents` says how.

  rig                    Build the agent if needed and run it in a new session.
  rig <command> [args…]  Build the agent if needed and run a plugin's command.
  rig build              Regenerate the agent project, build it, and stage the
                         new binary for the next start. The agent holds only
                         the plugins [plugins] in RIG_HOME/settings.toml turns
                         on, so turning one on or off rebuilds it: only the
                         generated crate compiles again, and a plugin's own
                         crate the first time it is on.
  rig update             Reinstall rig with `cargo install --force`, from
                         crates.io, or from the checkout it was installed
                         from as it is now, then build the agent with it.
                         It says the command first. In the agent, /update.
  rig help [command]     This text, or what a command does.
  rig help-agents        How a script or another agent drives rig headless,
                         over its stdin and stdout (-p, --json).
  rig --version          The launcher's version, with the commit when it was
                         built in a rig checkout.
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
  RIG_HOME    Root of every rig directory (default: ~/.rig); its settings.toml
              is the one file to edit.
  RIG_SOURCE  A rig checkout to build the agent from instead of crates.io.

Exit codes:
  0           Success.
  1           A launcher error, or a build that failed or was rejected.
  2           A command line the launcher or the agent refuses.
  other       The agent's own: a plugin's failure (it may also be 1 or 2).
";

/// `rig help-agents`: the page a driver reads once.
const AGENTS: &str = include_str!("help-agents.txt");

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let home = Home::from_env();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["build"] => build::build(&home).map(|()| ExitCode::SUCCESS),
        ["update"] => Ok(update::update()),
        ["plugin", command @ ..] => plugin::run(&home, command),
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
            println!("rig {LONG_VERSION}");
            Ok(ExitCode::SUCCESS)
        }
        ["help", command] => Ok(help(&home, command)),
        _ => run::run(&home, &args),
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
        "build" | "help" | "update" => print!("{USAGE}"),
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
                return ExitCode::from(REFUSED_EXIT_CODE);
            };
            println!("Usage: rig {} {}\n\n  {}", info.name, info.usage, info.help);
        }
    }
    ExitCode::SUCCESS
}
