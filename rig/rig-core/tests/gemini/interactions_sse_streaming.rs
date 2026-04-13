//! Gemini Interactions live regressions for SSE deserialization warnings.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test gemini gemini::interactions_sse_streaming:: -- --ignored --test-threads=1 --nocapture`

use std::io;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::ToolDefinition;
use rig::message::{Message, ToolChoice};
use rig::providers::gemini;
use rig::providers::gemini::interactions_api::InteractionsCompletionModel;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;

type InteractionsAgent = Agent<InteractionsCompletionModel>;

const BASIC_PREAMBLE: &str =
    "You are a concise assistant. Answer in plain text using one or two short sentences.";
const BASIC_PROMPT: &str = "Write one short sentence about hummingbirds.";
const ORDER_SUPPORT_PREAMBLE: &str = "You are an order support assistant. If the conversation \
already contains a lookup_order tool result for the requested order, answer only from that \
existing tool result and do not call the tool again. Otherwise, you must call the lookup_order \
tool exactly once for the current order before answering. After the tool returns, answer in one \
concise plain-text sentence using only the tool JSON fields order_id, status, and eta_days.";
const ORDER_SUPPORT_PROMPT: &str =
    "Check order A-17. Use the tool exactly once, then answer from the tool result.";
const ORDER_SUPPORT_FOLLOWUP_PROMPT: &str = "Using only the existing conversation history, \
restate the order status in plain text without calling any tool.";
const INTERACTIONS_SSE_DESERIALIZE_ERROR: &str = "Failed to deserialize interactions SSE event";

#[derive(Debug, Deserialize, JsonSchema)]
struct OrderLookupArgs {
    order_id: String,
}

#[derive(Debug, Serialize)]
struct OrderLookupResult {
    order_id: String,
    status: String,
    eta_days: u64,
}

#[derive(Debug, Error)]
#[error("Order lookup unavailable")]
struct OrderLookupError;

#[derive(Clone)]
struct OrderLookupTool {
    call_count: Arc<AtomicUsize>,
}

impl OrderLookupTool {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

fn order_lookup_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: OrderLookupTool::NAME.to_string(),
        description: "Look up an order by id and return its status JSON.".to_string(),
        parameters: serde_json::to_value(schema_for!(OrderLookupArgs))
            .expect("tool schema should serialize"),
    }
}

fn build_basic_agent() -> InteractionsAgent {
    gemini::InteractionsClient::from_env()
        .agent("gemini-3-flash-preview")
        .preamble(BASIC_PREAMBLE)
        .build()
}

fn build_tool_agent(call_count: Arc<AtomicUsize>) -> InteractionsAgent {
    gemini::InteractionsClient::from_env()
        .agent("gemini-3-flash-preview")
        .preamble(ORDER_SUPPORT_PREAMBLE)
        .tool(OrderLookupTool::new(call_count))
        .tool_choice(ToolChoice::Auto)
        .build()
}

#[derive(Default)]
struct StreamObservation {
    tool_call_names: Vec<String>,
    tool_results: usize,
    final_turn_text: String,
    final_response_text: Option<String>,
    final_history: Option<Vec<Message>>,
}

impl StreamObservation {
    fn resolved_final_text(&self) -> &str {
        if self.final_turn_text.trim().is_empty() {
            self.final_response_text.as_deref().unwrap_or_default()
        } else {
            self.final_turn_text.as_str()
        }
    }
}

async fn collect_stream_observation<R, S>(stream: S) -> StreamObservation
where
    S: futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>,
{
    futures::pin_mut!(stream);

    let mut observation = StreamObservation::default();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    observation.tool_call_names.push(tool_call.function.name);
                }
                StreamedAssistantContent::Text(text) => {
                    observation.final_turn_text.push_str(&text.text);
                }
                _ => {}
            },
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                observation.tool_results += 1;
                observation.final_turn_text.clear();
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                observation.final_response_text = Some(response.response().to_owned());
                observation.final_history = response.history().map(|history| history.to_vec());
            }
            Ok(_) => {}
            Err(error) => panic!("stream should succeed: {error}"),
        }
    }

    observation
}

#[derive(Clone, Default)]
struct SharedLogBuffer {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl SharedLogBuffer {
    fn snapshot(&self) -> String {
        let bytes = self
            .bytes
            .lock()
            .expect("log buffer should not be poisoned")
            .clone();
        String::from_utf8(bytes).expect("captured logs should be valid UTF-8")
    }
}

struct SharedLogWriter {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl io::Write for SharedLogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.bytes
            .lock()
            .expect("log buffer should not be poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for SharedLogBuffer {
    type Writer = SharedLogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        SharedLogWriter {
            bytes: self.bytes.clone(),
        }
    }
}

fn install_interactions_streaming_log_capture()
-> (SharedLogBuffer, tracing::subscriber::DefaultGuard) {
    let log_buffer = SharedLogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(
            "rig::providers::gemini::interactions_api::streaming=debug",
        ))
        .with_writer(log_buffer.clone())
        .without_time()
        .with_ansi(false)
        .finish();
    let guard = tracing::subscriber::set_default(subscriber);
    (log_buffer, guard)
}

fn assert_nonempty_text(text: &str) {
    assert!(
        !text.trim().is_empty(),
        "expected non-empty final text, got {text:?}"
    );
}

fn assert_mentions_tool_output(text: &str) {
    assert_nonempty_text(text);
    let text_lower = text.to_ascii_lowercase();
    assert!(
        text.contains("A-17") || text_lower.contains("backordered") || text.contains("3"),
        "expected final text to reference tool output, got {text:?}"
    );
}

fn assert_no_sse_deserialize_warnings(logs: &str) {
    assert!(
        !logs.contains(INTERACTIONS_SSE_DESERIALIZE_ERROR),
        "unexpected interactions SSE deserialize warning in captured logs:\n{logs}"
    );
}

impl Tool for OrderLookupTool {
    const NAME: &'static str = "lookup_order";

    type Error = OrderLookupError;
    type Args = OrderLookupArgs;
    type Output = OrderLookupResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        order_lookup_tool_definition()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(OrderLookupResult {
            order_id: args.order_id,
            status: "backordered".to_string(),
            eta_days: 3,
        })
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires GEMINI_API_KEY"]
async fn interactions_streaming_basic_emits_no_sse_deserialize_errors() {
    let agent = build_basic_agent();
    let (log_buffer, _guard) = install_interactions_streaming_log_capture();

    let observation = collect_stream_observation(
        agent
            .stream_chat(BASIC_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await,
    )
    .await;

    assert_nonempty_text(observation.resolved_final_text());
    assert_no_sse_deserialize_warnings(&log_buffer.snapshot());
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires GEMINI_API_KEY"]
async fn interactions_streaming_single_tool_emits_no_sse_deserialize_errors() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = build_tool_agent(call_count.clone());
    let (log_buffer, _guard) = install_interactions_streaming_log_capture();

    let observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUPPORT_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await,
    )
    .await;

    assert!(
        observation
            .tool_call_names
            .iter()
            .any(|name| name == OrderLookupTool::NAME),
        "expected at least one lookup_order tool call, got {:?}",
        observation.tool_call_names
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "expected exactly one tool execution"
    );
    assert!(
        observation.tool_results >= 1,
        "expected at least one streamed tool result, got {}",
        observation.tool_results
    );
    assert_mentions_tool_output(observation.resolved_final_text());
    assert_no_sse_deserialize_warnings(&log_buffer.snapshot());
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires GEMINI_API_KEY"]
async fn interactions_streaming_followup_reuses_prior_tool_result_without_sse_deserialize_errors() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = build_tool_agent(call_count.clone());
    let (log_buffer, _guard) = install_interactions_streaming_log_capture();

    let first_observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUPPORT_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await,
    )
    .await;

    let history = first_observation
        .final_history
        .clone()
        .expect("expected final history from the initial Interactions run");

    let followup_observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUPPORT_FOLLOWUP_PROMPT, history)
            .multi_turn(3)
            .await,
    )
    .await;

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "expected exactly one tool execution across both turns"
    );
    assert!(
        followup_observation.tool_call_names.is_empty(),
        "follow-up run should not emit a tool call: {:?}",
        followup_observation.tool_call_names
    );
    assert_mentions_tool_output(followup_observation.resolved_final_text());
    assert_no_sse_deserialize_warnings(&log_buffer.snapshot());
}
