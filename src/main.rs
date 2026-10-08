//! The `rig` launcher. It generates a Cargo project for the rig-code agent
//! with the plugins listed in `$RIG_HOME/rig.toml`, builds it, and runs it.
//! When the agent exits with the reload code it starts the new build, and a
//! build that crashes during startup is rolled back to the last one that
//! worked. It uses only std.

mod launcher;

use std::process::ExitCode;

use launcher::home::Home;

const USAGE: &str = "\
Usage: rig [build | help]

  rig        Build the agent if needed and run it.
  rig build  Regenerate the agent project from rig.toml, build it, and stage
             the new binary for the next start.

Environment:
  RIG_HOME    Root of every rig directory (default: ~/.rig).
  RIG_SOURCE  A rig checkout to build the agent from instead of crates.io.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let home = Home::from_env();
    let result = match args.as_slice() {
        [] => launcher::run::run(&home),
        ["build"] => launcher::build::build(&home).map(|()| ExitCode::SUCCESS),
        ["help" | "--help" | "-h"] => {
            print!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        _ => {
            eprint!("{USAGE}");
            Ok(ExitCode::from(2))
        }
    };
    result.unwrap_or_else(|failure| {
        eprintln!("error: {failure}");
        ExitCode::FAILURE
    })
}
