//! `session-compare [--json] a[#branch] b[#branch]`: how two saved rig
//! sessions, or two branches of one, differ.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match corpus_ct_session_compare_library::run(&args) {
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
