//! `tool-exec [--user <id>] [--timeout-ms <ms>]`: one tool call per stdin
//! line, one JSON result per stdout line; full errors go to stderr.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use corpus_rf_08::run_lines;
use corpus_rf_08::tools::{User, registry};
use rig_core::tool::ToolContext;

fn main() -> ExitCode {
    let mut context = ToolContext::new();
    let mut timeout = Duration::from_secs(30);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match (flag.as_str(), args.next()) {
            ("--user", Some(user)) => {
                if let Err(why) = context.insert(User(user)) {
                    eprintln!("tool-exec: {why}");
                    return ExitCode::from(2);
                }
            }
            ("--timeout-ms", Some(ms)) => match ms.parse() {
                Ok(ms) => timeout = Duration::from_millis(ms),
                Err(why) => {
                    eprintln!("tool-exec: --timeout-ms: {why}");
                    return ExitCode::from(2);
                }
            },
            _ => {
                eprintln!("usage: tool-exec [--user <id>] [--timeout-ms <ms>]");
                return ExitCode::from(2);
            }
        }
    }
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
    {
        Ok(runtime) => runtime,
        Err(why) => {
            eprintln!("tool-exec: {why}");
            return ExitCode::FAILURE;
        }
    };
    let tools = registry(Arc::default());
    let run = run_lines(
        &tools,
        &context,
        timeout,
        std::io::stdin().lock(),
        std::io::stdout().lock(),
        std::io::stderr().lock(),
    );
    match runtime.block_on(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("tool-exec: {why}");
            ExitCode::FAILURE
        }
    }
}
