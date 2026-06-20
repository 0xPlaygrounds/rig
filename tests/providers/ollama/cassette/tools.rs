//! Ollama tool-handling coverage matching real consumers (e.g. repo-tagger):
//! a tool with an `Option<T>` argument, and a non-streaming multi-tool chain.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::CompletionClient;
use rig::completion::{Chat, Message, ToolDefinition};
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::super::support::with_ollama_cassette;

const MODEL: &str = "qwen3:4b";

#[derive(Debug, thiserror::Error)]
#[error("tool error")]
struct ToolError;

// --- a tool with an Option<T> arg (mirrors repo-tagger read_file's max_bytes) ---

#[derive(Deserialize, JsonSchema)]
struct RepeatArgs {
    /// The text to repeat.
    text: String,
    /// Number of repetitions; defaults to 2 when omitted.
    times: Option<u32>,
}

#[derive(Clone)]
struct RepeatTool {
    calls: Arc<AtomicUsize>,
}

impl Tool for RepeatTool {
    const NAME: &'static str = "repeat_text";
    type Error = ToolError;
    type Args = RepeatArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "repeat_text".to_string(),
            description: "Repeat `text`. `times` is optional and defaults to 2.".to_string(),
            // schemars schema, exactly as repo-tagger builds its tool parameters.
            parameters: serde_json::to_value(schemars::schema_for!(RepeatArgs)).unwrap_or_default(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let times = args.times.unwrap_or(2) as usize;
        Ok(vec![args.text.as_str(); times].join(" "))
    }
}

#[tokio::test]
async fn tool_with_optional_argument() {
    let calls = Arc::new(AtomicUsize::new(0));
    with_ollama_cassette("tools/optional_argument", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble("Use the repeat_text tool whenever asked to repeat text.")
            .tool(RepeatTool {
                calls: calls.clone(),
            })
            .additional_params(json!({ "think": false }))
            .default_max_turns(4)
            .build();

        let result = agent
            .chat(
                "Use the repeat_text tool to repeat the word \"banana\" 3 times, then show me the \
                 exact result.",
                &mut Vec::<Message>::new(),
            )
            .await
            .expect("chat with optional-arg tool should succeed");

        assert!(
            calls.load(Ordering::SeqCst) >= 1,
            "[ollama] repeat_text tool should be invoked"
        );
        assert!(
            result.to_ascii_lowercase().contains("banana"),
            "[ollama] result should reference the repeated text: {result}"
        );
    })
    .await;
}

// --- two distinct tools used in one non-streaming run (mirrors list+read) ---

#[derive(Deserialize, JsonSchema)]
struct BinOpArgs {
    a: i64,
    b: i64,
}

#[derive(Clone)]
struct AddTool {
    calls: Arc<AtomicUsize>,
}

impl Tool for AddTool {
    const NAME: &'static str = "add";
    type Error = ToolError;
    type Args = BinOpArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add two integers a and b.".to_string(),
            parameters: serde_json::to_value(schemars::schema_for!(BinOpArgs)).unwrap_or_default(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(args.a + args.b)
    }
}

#[derive(Clone)]
struct MultiplyTool {
    calls: Arc<AtomicUsize>,
}

impl Tool for MultiplyTool {
    const NAME: &'static str = "multiply";
    type Error = ToolError;
    type Args = BinOpArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "multiply".to_string(),
            description: "Multiply two integers a and b.".to_string(),
            parameters: serde_json::to_value(schemars::schema_for!(BinOpArgs)).unwrap_or_default(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(args.a * args.b)
    }
}

#[tokio::test]
async fn two_tools_nonstreaming_chain() {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let mul_calls = Arc::new(AtomicUsize::new(0));
    with_ollama_cassette("tools/two_tools_nonstreaming", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(
                "You are a calculator. Use the add and multiply tools for arithmetic; never \
                 compute by hand.",
            )
            .tool(AddTool {
                calls: add_calls.clone(),
            })
            .tool(MultiplyTool {
                calls: mul_calls.clone(),
            })
            .additional_params(json!({ "think": false }))
            .default_max_turns(6)
            .build();

        let result = agent
            .chat(
                "Compute (4 + 6) * 2. First call the add tool, then call the multiply tool on the \
                 result. Tell me the final number.",
                &mut Vec::<Message>::new(),
            )
            .await
            .expect("two-tool chat should succeed");

        let mut diag = String::new();
        let _ = write!(
            diag,
            "add={}, multiply={}",
            add_calls.load(Ordering::SeqCst),
            mul_calls.load(Ordering::SeqCst)
        );
        assert!(
            add_calls.load(Ordering::SeqCst) >= 1,
            "[ollama] add tool should be invoked ({diag})"
        );
        assert!(
            mul_calls.load(Ordering::SeqCst) >= 1,
            "[ollama] multiply tool should be invoked ({diag})"
        );
        assert!(
            result.contains("20"),
            "[ollama] final answer should be 20, got: {result}"
        );
    })
    .await;
}
