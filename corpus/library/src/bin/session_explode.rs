//! `session-explode`: see the library's documentation.

use std::process::ExitCode;

use corpus_library::branch_explode_library::run;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let mut out = std::io::stdout().lock();
    match run(&args, &mut out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("session-explode: {why}");
            ExitCode::FAILURE
        }
    }
}
