//! `rig`: builds the rig-code agent from the plugin list and runs it.
//!
//! `rig` generates a small Cargo project from `plugins.toml`, builds it,
//! and supervises the agent: exit code 75 restarts it with the freshly
//! built binary, and a new binary that crashes before it is ready is rolled
//! back to the last one that worked. `rig build` only builds and stages the
//! binary; the agent's `/reload` runs it.
//!
//! ```text
//! RIG_HOME=/path/to/home rig -j 12
//! ```

mod build;
mod dirs;
mod plugins;
mod project;
mod supervise;

use std::process::ExitCode;

use dirs::Dirs;

/// Exit code of a `rig build` whose compile failed.
const BUILD_FAILED: u8 = 1;
/// Exit code for a configuration or Bevy-version problem.
const CONFIG_FAILED: u8 = 2;

const USAGE: &str = "\
usage: rig [-j N] [build]

  rig          build the agent if needed, then run it
  rig build    build the agent and stage it for the next start (used by /reload)
  -j, --jobs N build with N parallel jobs; remembered for later builds

Environment:
  RIG_HOME         keep config, cache and data under this directory
  RIG_CODE_SOURCE  build rig-code from this checkout instead of crates.io";

/// Why `rig` stops: the exit code and what to tell the user.
#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) code: u8,
    pub(crate) message: String,
}

impl Failure {
    /// A configuration problem: exit code 2.
    pub(crate) fn config(message: impl Into<String>) -> Self {
        Self {
            code: CONFIG_FAILED,
            message: message.into(),
        }
    }

    /// A failed build or an I/O problem: exit code 1.
    pub(crate) fn build(message: impl Into<String>) -> Self {
        Self {
            code: BUILD_FAILED,
            message: message.into(),
        }
    }

    /// An I/O error while doing `what`.
    pub(crate) fn io(what: impl std::fmt::Display, error: std::io::Error) -> Self {
        Self::build(format!("{what}: {error}"))
    }
}

/// What the command line asks for.
#[derive(PartialEq)]
enum Mode {
    Run,
    Build,
    Help,
}

/// The command line.
struct Arguments {
    jobs: Option<u32>,
    mode: Mode,
}

fn parse_arguments() -> Result<Arguments, Failure> {
    let mut arguments = Arguments {
        jobs: None,
        mode: Mode::Run,
    };
    let mut words = std::env::args().skip(1);
    while let Some(word) = words.next() {
        let jobs = match word.as_str() {
            "build" => {
                arguments.mode = Mode::Build;
                continue;
            }
            "-h" | "--help" => {
                arguments.mode = Mode::Help;
                continue;
            }
            "-j" | "--jobs" => words.next(),
            other => other
                .strip_prefix("-j")
                .or_else(|| other.strip_prefix("--jobs="))
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
        };
        let Some(jobs) = jobs else {
            return Err(Failure::config(format!(
                "unknown argument `{word}`\n{USAGE}"
            )));
        };
        match jobs.parse::<u32>() {
            Ok(jobs) if jobs > 0 => arguments.jobs = Some(jobs),
            _ => {
                return Err(Failure::config(format!(
                    "-j needs a positive number, got `{jobs}`"
                )));
            }
        }
    }
    Ok(arguments)
}

fn run() -> Result<u8, Failure> {
    let arguments = parse_arguments()?;
    if arguments.mode == Mode::Help {
        println!("{USAGE}");
        return Ok(0);
    }
    let dirs = Dirs::resolve()?;
    if arguments.mode == Mode::Build {
        build::build(&dirs, arguments.jobs)?;
        return Ok(0);
    }
    let notice = match build::build(&dirs, arguments.jobs) {
        Ok(()) => None,
        Err(failure) if dirs.current_bin().exists() => {
            eprintln!("rig: {}", failure.message);
            eprintln!("rig: the build failed; starting the last working build");
            Some(
                "the agent build failed, so this is the last working build; run `rig build` \
                 to see why"
                    .to_owned(),
            )
        }
        Err(failure) => return Err(failure),
    };
    supervise::supervise(&dirs, notice)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(failure) => {
            eprintln!("rig: {}", failure.message);
            ExitCode::from(failure.code)
        }
    }
}
