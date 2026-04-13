//! Gemini live regressions for tool + schema streaming.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test gemini gemini::tool_schema_streaming:: -- --ignored --test-threads=1 --nocapture`

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
use rig::providers::gemini::completion::CompletionModel as GeminiCompletionModel;
use rig::providers::gemini::interactions_api::InteractionsCompletionModel;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;

type GenerateContentAgent = Agent<GeminiCompletionModel>;
type InteractionsAgent = Agent<InteractionsCompletionModel>;

const ORDER_SUMMARY_PREAMBLE: &str = "You are an order support assistant. If the conversation \
already contains a lookup_order tool result for the requested order, answer only from that \
existing tool result and do not call the tool again. Otherwise, you must call the lookup_order \
tool exactly once for the current order before answering. Final answers must be valid JSON with \
exactly the fields order_id, status, and eta_days.";
const ORDER_SUMMARY_PROMPT: &str =
    "Check order A-17. Use the tool exactly once, then answer with JSON only.";
const ORDER_SUMMARY_FOLLOWUP_PROMPT: &str = "Using only the existing conversation history, \
return the same order summary as JSON only. Do not call any tool.";
const GENERATE_CONTENT_SCHEMA_ERROR: &str =
    "Function calling with a response mime type: 'application/json' is unsupported";
const INTERACTIONS_SSE_DESERIALIZE_ERROR: &str = "Failed to deserialize interactions SSE event";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
struct OrderSummary {
    order_id: String,
    status: String,
    eta_days: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct OrderLookupArgs {
    order_id: String,
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

fn expected_order_summary() -> OrderSummary {
    OrderSummary {
        order_id: "A-17".to_string(),
        status: "backordered".to_string(),
        eta_days: 3,
    }
}

fn prior_history() -> Vec<Message> {
    vec![
        Message::user("Before we begin, confirm the support session is ready."),
        Message::assistant("Support session ready."),
    ]
}

fn build_generate_content_agent(call_count: Arc<AtomicUsize>) -> GenerateContentAgent {
    gemini::Client::from_env()
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(ORDER_SUMMARY_PREAMBLE)
        .tool(OrderLookupTool::new(call_count))
        .tool_choice(ToolChoice::Auto)
        .output_schema::<OrderSummary>()
        .build()
}

fn build_interactions_agent(call_count: Arc<AtomicUsize>) -> InteractionsAgent {
    gemini::InteractionsClient::from_env()
        .agent("gemini-3-flash-preview")
        .preamble(ORDER_SUMMARY_PREAMBLE)
        .tool(OrderLookupTool::new(call_count))
        .tool_choice(ToolChoice::Auto)
        .output_schema::<OrderSummary>()
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

async fn collect_stream_error<R, S>(stream: S) -> StreamingError
where
    S: futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>,
{
    futures::pin_mut!(stream);

    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            return error;
        }
    }

    panic!("stream should fail");
}

fn parse_order_summary(response_text: &str) -> OrderSummary {
    serde_json::from_str(response_text.trim())
        .unwrap_or_else(|err| panic!("expected JSON OrderSummary, got {response_text:?}: {err}"))
}

impl Tool for OrderLookupTool {
    const NAME: &'static str = "lookup_order";

    type Error = OrderLookupError;
    type Args = OrderLookupArgs;
    type Output = OrderSummary;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Look up an order by id and return its structured status JSON."
                .to_string(),
            parameters: serde_json::to_value(schema_for!(OrderLookupArgs))
                .expect("tool schema should serialize"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let expected = expected_order_summary();
        assert_eq!(
            args.order_id, expected.order_id,
            "tool should be called for the deterministic order id"
        );

        Ok(expected)
    }
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

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn generate_content_streaming_tool_and_output_schema_provider_rejects_request() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = build_generate_content_agent(call_count.clone());

    let error = collect_stream_error(
        agent
            .stream_chat(ORDER_SUMMARY_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await,
    )
    .await;

    let error_text = error.to_string();
    assert!(
        error_text.contains(GENERATE_CONTENT_SCHEMA_ERROR),
        "expected Gemini provider rejection mentioning unsupported response mime type, got {error_text:?}"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        0,
        "generateContent request should fail before any tool call executes"
    );
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn interactions_streaming_tool_and_output_schema_roundtrips_structured_json() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = build_interactions_agent(call_count.clone());

    let observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUMMARY_PROMPT, Vec::<Message>::new())
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
    assert!(
        observation.tool_results >= 1,
        "expected at least one streamed tool result, got {}",
        observation.tool_results
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "expected exactly one tool invocation"
    );

    let structured = parse_order_summary(observation.resolved_final_text());
    assert_eq!(structured, expected_order_summary());
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires GEMINI_API_KEY"]
async fn interactions_streaming_tool_and_output_schema_emits_no_sse_deserialize_errors() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = build_interactions_agent(call_count.clone());

    let log_buffer = SharedLogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(
            "rig::providers::gemini::interactions_api::streaming=debug",
        ))
        .with_writer(log_buffer.clone())
        .without_time()
        .with_ansi(false)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let first_observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUMMARY_PROMPT, prior_history())
            .multi_turn(3)
            .await,
    )
    .await;

    assert!(
        first_observation
            .tool_call_names
            .iter()
            .any(|name| name == OrderLookupTool::NAME),
        "expected the initial Interactions run to call lookup_order, got {:?}",
        first_observation.tool_call_names
    );
    assert!(
        first_observation.tool_results >= 1,
        "expected the initial Interactions run to emit a tool result, got {}",
        first_observation.tool_results
    );

    let first_history = first_observation
        .final_history
        .clone()
        .expect("expected final history from the initial Interactions run");

    let followup_observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUMMARY_FOLLOWUP_PROMPT, first_history)
            .multi_turn(3)
            .await,
    )
    .await;

    let structured = parse_order_summary(followup_observation.resolved_final_text());
    let captured_logs = log_buffer.snapshot();

    assert_eq!(structured, expected_order_summary());
    assert!(
        followup_observation.tool_call_names.is_empty(),
        "follow-up run should reuse prior tool result without a new tool call: {:?}",
        followup_observation.tool_call_names
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "expected the tool to be called exactly once across the full exchange"
    );
    assert!(
        !captured_logs.contains(INTERACTIONS_SSE_DESERIALIZE_ERROR),
        "unexpected interactions SSE deserialize warning in captured logs:\n{captured_logs}"
    );
}
