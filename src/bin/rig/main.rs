//! `rig`: the launcher of the rig-code agent.
//!
//! It generates a small Cargo project from the plugin list, builds it, runs
//! the binary, starts the freshly built one when the agent exits with
//! [`RELOAD`], and rolls back to the last binary that started when a new one
//! crashes before it is ready. `rig build` regenerates and builds only; the
//! agent's `/reload` runs it. Std only.

mod build;
mod home;
mod plugins;
mod project;
mod run;

use std::process::ExitCode;

/// The exit code with which the agent asks to be restarted on a new build.
const RELOAD: i32 = 75;

const USAGE: &str = "\
usage: rig            build the agent if needed and run it
       rig build      regenerate and build the agent, then stage it for the next start

environment:
  RIG_HOME     keep config, cache and data under this one directory
  RIG_SOURCE   a rig repository to build rig-code from (local-source mode)
  RIG_JOBS     parallel jobs for cargo (-j)";

fn main() -> ExitCode {
    let home = match home::Home::from_env() {
        Ok(home) => home,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut args = std::env::args().skip(1);
    match (args.next().as_deref(), args.next()) {
        (None, _) => run::run(&home),
        (Some("build"), None) => build::command(&home),
        (Some("-V" | "--version"), None) => {
            println!("rig {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        (Some("-h" | "--help" | "help"), None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}
