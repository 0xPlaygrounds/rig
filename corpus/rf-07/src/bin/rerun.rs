//! `rerun record <dir> --model <label>`, `rerun replay <dir> --from <turn>`,
//! `rerun fork <dir> --from <turn> --model <label> --to <new-dir>`.
//!
//! `record` and `fork` run against an OpenAI model named `<label>`
//! (`OPENAI_API_KEY`); `replay` needs no network and no key.

use std::path::PathBuf;

use corpus_rf_07::{Program, Rows};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|at| args.get(at + 1))
        .cloned()
}

fn program() -> Program {
    Program {
        preamble: "You clean up the customer table one row at a time.".into(),
        prompt: "Clean up the customer table.".into(),
        max_turns: 12,
        streaming: true,
        observer: None,
    }
}

fn openai(label: &str) -> Result<rig_core::DynModel<rig_core::operation::Completion>, String> {
    let client = rig_core::providers::openai::OpenAI::from_env().map_err(|e| e.to_string())?;
    Ok(client.completion(label).into())
}

async fn main_inner(args: Vec<String>) -> Result<String, String> {
    let usage = "usage: rerun record <dir> --model <label> | replay <dir> --from <turn> | fork <dir> --from <turn> --model <label> --to <new-dir>";
    let (Some(command), Some(dir)) = (args.first(), args.get(1)) else {
        return Err(usage.into());
    };
    let dir = PathBuf::from(dir);
    let from = || -> Result<usize, String> {
        flag(&args, "--from")
            .ok_or(usage)?
            .parse()
            .map_err(|_| "--from takes a turn number".to_owned())
    };
    let label = || flag(&args, "--model").ok_or_else(|| usage.to_owned());
    match command.as_str() {
        "record" => {
            corpus_rf_07::record(&program(), openai(&label()?)?, Rows::default(), &dir, true)
                .await
                .map(|ending| format!("{ending:?}"))
                .map_err(|e| e.to_string())
        }
        "replay" => {
            let from = from()?;
            let verdict = corpus_rf_07::replay(&program(), &dir, from)
                .await
                .map_err(|e| e.to_string())?;
            let calls = corpus_rf_07::tool_calls(&dir, from).map_err(|e| e.to_string())?;
            Ok(format!(
                "turn {from} tool calls:\n{}\n{verdict}",
                calls.join("\n")
            ))
        }
        "fork" => {
            let to = PathBuf::from(flag(&args, "--to").ok_or(usage)?);
            corpus_rf_07::fork(
                &program(),
                &dir,
                from()?,
                openai(&label()?)?,
                Rows::default(),
                &to,
            )
            .await
            .map(|ending| format!("{ending:?}"))
            .map_err(|e| e.to_string())
        }
        _ => Err(usage.into()),
    }
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match main_inner(std::env::args().skip(1).collect()).await {
        Ok(out) => {
            println!("{out}");
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
