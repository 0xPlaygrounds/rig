//! `session-edit`: see the library's documentation for its commands.

use std::process::ExitCode;

use corpus_library::session_edit_library::run;
use rig_harness_protocol::Home;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let mut out = std::io::stdout().lock();
    match run(&args, &Home::from_env(), &mut out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("session-edit: {why}");
            ExitCode::FAILURE
        }
    }
}
