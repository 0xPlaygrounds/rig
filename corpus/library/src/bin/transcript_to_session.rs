//! `transcript-to-session`: turns a chat transcript saved as Markdown into
//! a rig session the harness resumes
//! ([`corpus_library::transcript_import_library`]).

use std::process::ExitCode;

use rig_harness_protocol::Home;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match corpus_library::transcript_import_library::run(&args, &Home::from_env()) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(why) => {
            eprintln!("transcript-to-session: {why}");
            ExitCode::FAILURE
        }
    }
}
