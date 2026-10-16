//! `session-compare [--json] a b`: how two saved rig sessions differ.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match corpus_library::session_compare_library::run(&args) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(why) => {
            eprintln!("session-compare: {why}");
            ExitCode::FAILURE
        }
    }
}
