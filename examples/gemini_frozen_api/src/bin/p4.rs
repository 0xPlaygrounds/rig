use std::process::Command;
use std::time::Duration;

use rig::agent::{AgentHook, CompletionCallAction, CompletionCallEvent, HookContext};
use rig::providers::gemini::{self, CacheExpiry, CachedPrefix, Gemini, NewCachedContent};
use rig::tool::{PortableTool, tool_definition};
use rig::{AgentBuilder, AgentRun};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// Gemini refuses caches under 1,024 tokens; the preamble is well above that.
const PREAMBLE: &str = include_str!("coding_agent_preamble.md");
const TASK: &str = "Make `cargo test` pass in this workspace without changing any test.";
const MAX_TURNS: usize = 80;
const CHECKPOINT: &str = "coding-agent.checkpoint.json";
const CHECKPOINT_TMP: &str = "coding-agent.checkpoint.tmp";

#[derive(Deserialize)]
struct PathArgs {
    path: String,
}

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    contents: String,
}

#[derive(Deserialize)]
struct TestArgs {
    filter: Option<String>,
}

struct ReadFile;

impl PortableTool for ReadFile {
    const NAME: &'static str = "read_file";
    type Args = PathArgs;
    type Output = String;
    type Error = std::io::Error;

    fn description(&self) -> String {
        "Read a UTF-8 file from the workspace.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] })
    }

    async fn call(&self, args: PathArgs) -> Result<String, std::io::Error> {
        std::fs::read_to_string(args.path)
    }
}

struct WriteFile;

impl PortableTool for WriteFile {
    const NAME: &'static str = "write_file";
    type Args = WriteArgs;
    type Output = String;
    type Error = std::io::Error;

    fn description(&self) -> String {
        "Replace a file's contents.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "contents": { "type": "string" } },
            "required": ["path", "contents"]
        })
    }

    async fn call(&self, args: WriteArgs) -> Result<String, std::io::Error> {
        std::fs::write(&args.path, args.contents)?;
        Ok(format!("wrote {}", args.path))
    }
}

struct RunTests;

impl PortableTool for RunTests {
    const NAME: &'static str = "run_tests";
    type Args = TestArgs;
    type Output = String;
    type Error = std::io::Error;

    fn description(&self) -> String {
        "Run `cargo test`, optionally filtered, and return its output.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "filter": { "type": "string" } } })
    }

    async fn call(&self, args: TestArgs) -> Result<String, std::io::Error> {
        let mut command = Command::new("cargo");
        command.args(["test", "--quiet"]);
        if let Some(filter) = args.filter {
            command.arg(filter);
        }
        let output = command.output()?;
        Ok(format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    run: AgentRun,
    cache: CachedPrefix,
}

fn load() -> Result<Option<Checkpoint>, Box<dyn std::error::Error>> {
    match std::fs::read(CHECKPOINT) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn save(checkpoint: &Checkpoint) -> std::io::Result<()> {
    // Write then rename, so a crash never leaves half a checkpoint.
    let bytes = serde_json::to_vec(checkpoint).map_err(std::io::Error::other)?;
    std::fs::write(CHECKPOINT_TMP, bytes)?;
    std::fs::rename(CHECKPOINT_TMP, CHECKPOINT)
}

/// Saves the run and its cache before every model call.
struct SaveRun {
    cache: CachedPrefix,
}

impl AgentHook for SaveRun {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let checkpoint = Checkpoint {
            run: AgentRun::new(event.prompt.clone())
                .with_history(event.history.to_vec())
                .max_turns(MAX_TURNS),
            cache: self.cache.clone(),
        };
        match save(&checkpoint) {
            Ok(()) => CompletionCallAction::continue_run(),
            Err(error) => CompletionCallAction::stop(format!("checkpoint not saved: {error}")),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gemini = Gemini::from_env()?;
    let saved = load()?;

    // A resumed run keeps its cache, or recreates it from the same request if it
    // expired meanwhile. For implicit caching, drop this and `.cached_content(..)`.
    let caches = gemini.cached_contents();
    let cache = match &saved {
        Some(checkpoint) => caches.ensure(&checkpoint.cache).await?,
        None => {
            caches
                .create(
                    NewCachedContent::new(gemini::GEMINI_3_8_FLASH)
                        .system_instruction(PREAMBLE)
                        .tools([
                            tool_definition(&ReadFile),
                            tool_definition(&WriteFile),
                            tool_definition(&RunTests),
                        ])
                        .expiry(CacheExpiry::ttl(Duration::from_secs(60 * 60))),
                )
                .await?
        }
    };

    let agent = AgentBuilder::new(
        gemini
            .completion(gemini::GEMINI_3_8_FLASH)
            .cached_content(cache.clone()),
    )
    .preamble(PREAMBLE)
    .tool(ReadFile)
    .tool(WriteFile)
    .tool(RunTests)
    .default_max_turns(MAX_TURNS)
    .add_hook(SaveRun { cache })
    .build();

    let runner = match saved {
        Some(checkpoint) => agent.resume(checkpoint.run),
        None => agent.prompt(TASK),
    };
    let response = runner.tool_concurrency(8).await?;
    std::fs::remove_file(CHECKPOINT)?;

    for call in response.completion_calls() {
        eprintln!(
            "input={:?} cached={:?} output={:?}",
            call.usage.input_tokens, call.usage.cached_input_tokens, call.usage.output_tokens
        );
    }
    println!("{}", response.output);
    Ok(())
}
