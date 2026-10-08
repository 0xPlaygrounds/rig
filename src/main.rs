//! `rig`: the launcher of the rig coding agent.
//!
//! It generates a small Cargo project from the plugin list in
//! `plugins.toml`, builds it, and runs the agent. When the agent exits with
//! the reload code it starts the freshly built binary on the same session;
//! when a new binary crashes during startup it rolls back to the last one
//! that worked. It uses only the standard library.
//!
//! ```text
//! rig [-j N] [--resume]   generate, build and run the agent
//! rig build [-j N]        regenerate the agent project and build it
//! ```

use std::process::ExitCode;

mod launcher {
    pub mod build;
    pub mod config;
    pub mod paths;
    pub mod project;
    pub mod run;

    use std::fmt;

    /// A launcher failure, explained in one message for the user.
    #[derive(Debug)]
    pub struct Error(pub String);

    impl fmt::Display for Error {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    impl From<std::io::Error> for Error {
        fn from(error: std::io::Error) -> Self {
            Self(error.to_string())
        }
    }

    /// The launcher's result.
    pub type Result<T> = std::result::Result<T, Error>;
}

const USAGE: &str = "\
usage: rig [-j N] [--resume]   generate, build and run the coding agent
       rig build [-j N]        regenerate the agent project and build it

  -j N        build with N parallel jobs (overrides [build] jobs in plugins.toml)
  --resume    continue the most recent session

Set RIG_HOME to keep every rig directory under one root, and RIG_SOURCE to
build the agent from a local rig checkout.";

/// What the command line asks for.
struct Args {
    build: bool,
    jobs: Option<u32>,
    resume: bool,
}

fn main() -> ExitCode {
    let args = match parse(std::env::args().skip(1)) {
        Ok(Some(args)) => args,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("rig: {error}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = launcher::paths::Paths::from_env().and_then(|paths| {
        if args.build {
            launcher::build::build(&paths, args.jobs).map(|()| ExitCode::SUCCESS)
        } else {
            launcher::run::run(&paths, args.jobs, args.resume)
        }
    });
    match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("rig: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Read the arguments; `None` asks for the usage.
fn parse(mut arguments: impl Iterator<Item = String>) -> launcher::Result<Option<Args>> {
    let mut args = Args {
        build: false,
        jobs: None,
        resume: false,
    };
    let mut first = true;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "build" if first => args.build = true,
            "--resume" => args.resume = true,
            "-h" | "--help" => return Ok(None),
            "-j" | "--jobs" => {
                let value = arguments.next().unwrap_or_default();
                let jobs = value.parse().ok().filter(|jobs| *jobs > 0).ok_or_else(|| {
                    launcher::Error(format!("`{argument}` takes a positive number"))
                })?;
                args.jobs = Some(jobs);
            }
            other => return Err(launcher::Error(format!("unknown argument `{other}`"))),
        }
        first = false;
    }
    if args.build && args.resume {
        return Err(launcher::Error("`rig build` does not take --resume".into()));
    }
    Ok(Some(args))
}
