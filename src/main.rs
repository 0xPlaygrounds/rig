//! The `rig` launcher. It generates a Cargo project for the rig-harness agent
//! with the plugins listed in `$RIG_HOME/plugins.toml`, builds it, and runs it.
//! When the agent exits with the reload code it starts the new build, and a
//! build that crashes during startup is rolled back to the last one that
//! worked. It uses only std.
//!
//! Its own first words (`build`, `plugin`, `help`, `help-agents`) are read
//! before the build; the session verbs (`start`, `wait`, ..., `-p`) run in
//! the build's `cli` role ([`launcher::cli`]); the rest of the command line, `--resume` included, once the
//! agent is built, against the commands and options its plugins add (`rig
//! help` lists them).

mod launcher;

use std::process::ExitCode;

use launcher::plugin::USAGE as PLUGIN_USAGE;
use launcher::run::last_report;
use rig_harness_protocol::{AgentDescription, Home, INVOCATION_USAGE, guide};

const USAGE: &str = "\
Agents and scripts: `rig help-agents` (start, send, wait, answer, stop, -p).
Usage: rig [-r <id|name>] [options] | rig [options] <command> [args…]
       | rig <verb> [args…] | rig -p [prompt…] | rig build | rig plugin <command>
       | rig help [command | verb | --option] | rig help-agents [topic]

  rig                    Build the agent if needed and run it in a new session.
  rig <verb> [args…]     A session verb: start, send, wait, answer, log, agents,
                         attach, stop, connect, acp, serve, server.
  rig -p [prompt…]       Answer one prompt in a new session and print the answer;
                         `rig help -p`.
  rig <command> [args…]  Build the agent if needed and run a plugin's command.
  rig build              Regenerate the agent project from plugins.toml, build
                         it, and stage the new binary for the next start.
                         --print-build-id instead installs it apart and
                         prints its id, for /reload to name; with
                         --plugins FILE, built from that plugin list.
  rig help [command]     This text, or what a command does.
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
        ["build", "--print-build-id"] => launcher::cli::print_build_id(&home, None),
        ["build", "--print-build-id", "--plugins", list]
        | ["build", "--plugins", list, "--print-build-id"] => {
            launcher::cli::print_build_id(&home, Some(std::path::Path::new(list)))
        }
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
            print!("{}", guide::FIRST_SCREEN);
            Ok(ExitCode::SUCCESS)
        }
        ["help-agents", topic] => Ok(help_agents(topic)),
        ["help", command] => Ok(help(&home, command)),
        _ if launcher::cli::is_cli(&args) => launcher::cli::run(&home, &args),
        // The terminal view is a client of the server: a new session, or
        // `-r` attaching to one.
        [] => launcher::cli::view(&home),
        ["-r" | "--resume", session] => {
            launcher::cli::run(&home, &["attach".to_owned(), (*session).to_owned()])
        }
        _ => launcher::run::run(&home, &args),
    };
    result.unwrap_or_else(|failure| {
        eprintln!("error: {failure}");
        ExitCode::FAILURE
    })
}

/// `rig help-agents <topic>`; exit code 2 for a topic it has not.
fn help_agents(topic: &str) -> ExitCode {
    let Some(text) = guide::topic(topic) else {
        let topics: Vec<&str> = guide::TOPICS.iter().map(|(name, _)| *name).collect();
        eprintln!(
            "rig: no help-agents topic `{topic}`; topics: {}\nnext: rig help-agents",
            topics.join(", ")
        );
        return ExitCode::from(2);
    };
    print!("{text}");
    ExitCode::SUCCESS
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
        command if guide::verb(command).is_some() => {
            print!(
                "{}",
                guide::verb(command)
                    .map(guide::verb_help)
                    .unwrap_or_default()
            );
        }
        option if guide::option(option).is_some() => {
            print!(
                "{}",
                guide::option(option)
                    .map(guide::option_help)
                    .unwrap_or_default()
            );
        }
        command => {
            let report = last_report(home).unwrap_or_default();
            let Some(info) = report.command(command) else {
                eprintln!(
                    "rig: unknown command, verb or option `{command}`; commands: {}{}\n\
                     next: rig help-agents",
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
