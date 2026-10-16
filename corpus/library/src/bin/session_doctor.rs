//! `session-doctor <sessions-dir> [--fix <out-dir>] [--json]`: whether every
//! saved rig session still loads, and repaired copies of those that need it.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match corpus_library::session_doctor_library::run(&args) {
        Ok((report, code)) => {
            print!("{report}");
            ExitCode::from(code)
        }
        Err(why) => {
            eprintln!("session-doctor: {why}");
            ExitCode::from(2)
        }
    }
}
