//! `rig-coder` runs one coding task to completion with Rig's default agent
//! runtime and six workspace tools, writing a live JSONL transcript and the
//! run's effect log. Harbor runs it inside benchmark task containers.
//!
//! ```text
//! rig-coder --cwd /app --task-file task.md --transcript t.jsonl --effect-log effects.json
//! ```

mod tools;
mod transcript;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, bail};
use rig::agent::AgentBuilder;
use rig::cassette::agent::AgentReplayExt;
use rig::cassette::effect_log::EffectLogRecorder;
use rig::prelude::*;
use rig::providers::{anthropic::Anthropic, gemini::Gemini, openai::OpenAI};
use serde_json::json;

use crate::transcript::{Transcript, TranscriptHook, usage_fields};

const PROMPT: &str = include_str!("prompt.md");
const USAGE: &str = "usage: rig-coder [--cwd DIR] [--task-file PATH | TASK] [--provider gemini|anthropic|openai] \
[--model NAME] [--max-turns N] [--timeout-secs N] [--max-tokens N] [--transcript PATH] [--effect-log PATH]";

struct Args {
    cwd: PathBuf,
    task: String,
    provider: String,
    model: String,
    max_turns: usize,
    max_tokens: u64,
    timeout_secs: u64,
    transcript: Option<PathBuf>,
    effect_log: Option<PathBuf>,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut cwd = None;
    let mut task = None;
    let mut task_file = None;
    let mut provider = std::env::var("RIG_CODER_PROVIDER").ok();
    let mut model = std::env::var("RIG_CODER_MODEL").ok();
    let mut max_turns = 200;
    let mut max_tokens = 32_000;
    let mut timeout_secs = 3_600;
    let mut transcript = None;
    let mut effect_log = None;
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value\n{USAGE}"));
        match arg.as_str() {
            "--cwd" => cwd = Some(PathBuf::from(value()?)),
            "--task-file" => task_file = Some(PathBuf::from(value()?)),
            "--provider" => provider = Some(value()?),
            "--model" => model = Some(value()?),
            "--max-turns" => max_turns = value()?.parse()?,
            "--max-tokens" => max_tokens = value()?.parse()?,
            "--timeout-secs" => timeout_secs = value()?.parse()?,
            "--transcript" => transcript = Some(PathBuf::from(value()?)),
            "--effect-log" => effect_log = Some(PathBuf::from(value()?)),
            "-h" | "--help" => bail!("{USAGE}"),
            flag if flag.starts_with("--") => bail!("unknown flag {flag}\n{USAGE}"),
            _ => task = Some(arg),
        }
    }
    let task = match (task, task_file) {
        (Some(task), None) => task,
        (None, Some(path)) => std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
        _ => bail!("give exactly one of TASK or --task-file\n{USAGE}"),
    };
    let provider = provider.unwrap_or_else(|| "gemini".into());
    let model = match (model, provider.as_str()) {
        (Some(model), _) => model,
        (None, "gemini") => "gemini-3.8-flash".into(),
        (None, other) => bail!("--model is required for provider {other}"),
    };
    Ok(Args {
        cwd: match cwd {
            Some(cwd) => cwd,
            None => std::env::current_dir()?,
        },
        task,
        provider,
        model,
        max_turns,
        max_tokens,
        timeout_secs,
        transcript,
        effect_log,
    })
}

fn agent_builder(provider: &str, model: &str) -> anyhow::Result<AgentBuilder> {
    Ok(match provider {
        "gemini" => Gemini::from_env()?.bound()?.agent(model),
        "anthropic" => Anthropic::from_env()?.bound()?.agent(model),
        "openai" => OpenAI::from_env()?.bound()?.agent(model),
        other => bail!("unknown provider {other}"),
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("rig-coder: {error:#}");
            ExitCode::from(3)
        }
    }
}

async fn run() -> anyhow::Result<ExitCode> {
    let args = parse_args()?;
    let transcript = Transcript::create(args.transcript.as_deref())?;
    transcript.event(
        "task",
        json!({ "provider": args.provider, "model": args.model, "cwd": args.cwd, "task": args.task }),
    );
    let recorder = EffectLogRecorder::new();
    let agent = agent_builder(&args.provider, &args.model)?
        .preamble(PROMPT)
        .max_tokens(args.max_tokens)
        .record_to(recorder.clone())
        .dynamic_tools(tools::all(args.cwd.clone()))
        .build();
    let run = agent
        .prompt(args.task.as_str())
        .max_turns(args.max_turns)
        .add_hook(TranscriptHook(transcript.clone()))
        .run();
    let code = match tokio::time::timeout(Duration::from_secs(args.timeout_secs), run).await {
        Ok(Ok(response)) => {
            transcript.event(
                "settled",
                json!({ "output": response.output, "usage": usage_fields(&response.usage) }),
            );
            println!("{}", response.output);
            ExitCode::SUCCESS
        }
        Ok(Err(error)) => {
            transcript.event("failed", json!({ "error": format!("{error:#}") }));
            eprintln!("rig-coder: run failed: {error:#}");
            ExitCode::FAILURE
        }
        Err(_) => {
            transcript.event("failed", json!({ "error": "timeout", "timeout_secs": args.timeout_secs }));
            eprintln!("rig-coder: timed out after {}s", args.timeout_secs);
            ExitCode::from(2)
        }
    };
    if let Some(path) = &args.effect_log {
        let log = agent.stamp(recorder.take());
        let json = serde_json::to_vec(&log)?;
        std::fs::write(path, json).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(code)
}
