//! The command-line verbs of the rig agent, each a client of the rig agent
//! server through `rig-client`. The generated agent binary calls [`run`] in
//! its `cli` role; the Bevy app is never built for it.
//!
//! The contract every verb keeps:
//! - stdout carries only the answer, the requested text, or with `--json`
//!   one JSON object per line, each with `"rig":1` and a `kind`;
//! - every failure prints `rig: <what happened>` and `next: <command>` on
//!   stderr, and exits with one of [`rig_protocol::Exit`]'s codes;
//! - stdin is read only for a prompt given as `-`, or by `rig -p` without
//!   one.
//!
//! `rig help`, `rig help-agents` and their texts are the launcher's
//! ([`rig_harness_protocol::guide`]), answered before any build; a verb
//! given `--help` prints its help here too.

mod acp;
mod args;
mod oneshot;
mod out;
mod serve;
mod verbs;

use args::Args;
use out::Failure;

/// Runs the verb in `args` (without the program name) and returns the exit
/// code: one of [`rig_protocol::Exit`]'s.
pub fn run(args: Vec<String>) -> u8 {
    let json = args.iter().any(|arg| arg == "--json");
    match dispatch(&args) {
        Ok(code) => code,
        Err(failure) => failure.report(json),
    }
}

fn dispatch(args: &[String]) -> Result<u8, Failure> {
    if let Some(help) = rig_harness_protocol::guide::asked_help(args) {
        out::text(&help);
        return Ok(0);
    }
    let (verb, rest) = match args.split_first() {
        Some((first, rest)) if !first.starts_with('-') => (first.as_str(), rest.to_vec()),
        _ => {
            // `rig -p ...`, `rig -r ID -p ...`, or a bare `rig` fed on stdin.
            return oneshot::run(&Args::parse("-p", &print_args(args))?);
        }
    };
    match verb {
        "start" => verbs::start(&Args::parse(verb, &rest)?),
        "send" => verbs::send(&Args::parse(verb, &rest)?),
        "wait" => verbs::wait(&Args::parse(verb, &rest)?),
        "answer" => verbs::answer(&Args::parse(verb, &rest)?),
        "log" => verbs::log(&Args::parse(verb, &rest)?),
        "agents" => verbs::agents(&Args::parse(verb, &rest)?),
        "stop" => verbs::stop(&Args::parse(verb, &rest)?),
        "attach" => serve::attach(&Args::parse(verb, &rest)?),
        "serve" => serve::serve(),
        "server" => serve::server(&Args::parse(verb, &rest)?),
        "connect" => serve::connect(&Args::parse(verb, &rest)?),
        "acp" => acp::run(&verbs::Ctx::new(&Args::parse(verb, &rest)?)?),
        other => Err(Failure::usage(
            format!("unknown verb `{other}`"),
            "help-agents",
        )),
    }
}

/// `rig -p`'s arguments without the `-p` (or `--print`) that chose it.
fn print_args(args: &[String]) -> Vec<String> {
    args.iter()
        .filter(|arg| !matches!(arg.as_str(), "-p" | "--print"))
        .cloned()
        .collect()
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests;
