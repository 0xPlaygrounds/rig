//! The `rig` launcher. It generates a Cargo project for the rig-code agent
//! with the plugins listed in `$RIG_HOME/plugins.toml`, builds it, and runs it.
//! When the agent exits with the reload code it starts the new build, and a
//! build that crashes during startup is rolled back to the last one that
//! worked. It uses only std.

mod launcher;

use std::process::ExitCode;

use launcher::run::Start;
use rig::code_protocol::Home;

const USAGE: &str = "\
Usage: rig [--continue | --resume <session> | --new] | rig build | rig help

  rig                    Build the agent if needed and run it. In a directory
                         whose last session did not quit cleanly, that session
                         resumes.
  rig -c, --continue     Continue the last session run in this directory.
  rig -r, --resume <id>  Resume the session <id>, in the directory it ran in.
                         In the agent, /resume lists the sessions.
  rig -n, --new          Start a new session.
  rig build              Regenerate the agent project from plugins.toml, build
                         it, and stage the new binary for the next start.

Environment:
  RIG_HOME    Root of every rig directory (default: ~/.rig).
  RIG_SOURCE  A rig checkout to build the agent from instead of crates.io.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let home = Home::from_env();
    let result = match args.as_slice() {
        [] => launcher::run::run(&home, Start::Default),
        ["-c" | "--continue"] => launcher::run::run(&home, Start::Continue),
        ["-n" | "--new"] => launcher::run::run(&home, Start::New),
        ["-r" | "--resume", id] => match id.parse() {
            Ok(id) => launcher::run::run(&home, Start::Resume(id)),
            Err(failure) => Err(Box::new(failure).into()),
        },
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
