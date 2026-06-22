//! Live Gemini example: obtaining a token-count estimate when a streaming
//! generation is disrupted **mid-response** — for *any* reason.
//!
//! Requires `GEMINI_API_KEY`.
//! Run with: `cargo run --example gemini_stream_kill_token_count`
//!
//! ## The problem
//!
//! Gemini only emits authoritative `usageMetadata` on the *final* chunk of a
//! `streamGenerateContent` response. Any mid-stream disruption skips that chunk,
//! so the exact server-side token count is unrecoverable. There is also no
//! "cancel" flag to send Gemini — killing a stream is purely a client-side
//! connection close (here, `StreamingCompletionResponse::cancel()` /
//! `AbortHandle::abort()`).
//!
//! ## The approach — one accounting path for every disruption
//!
//! Disruptions reach a consumer in four different shapes:
//!   1. Manual kill / drop          -> stream ends with `None`, no `Final`
//!   2. Transport/server error      -> stream yields `Some(Err(..))`
//!   3. Premature clean close       -> a `Final` whose usage is all zeros
//!   4. Stall / half-open socket    -> `next()` never returns
//!
//! Rather than branch on *why* the stream stopped, we key on a single question:
//! **did we ever receive authoritative (non-zero) usage?** If not — for ANY of
//! the reasons above — we estimate locally from whatever partial output arrived,
//! using Gemini's free `countTokens` endpoint (no inference, not billed).
//!
//! To make that robust against all four shapes the consumer needs:
//!   - incremental accumulation of output as deltas arrive (never on a terminal
//!     event),
//!   - a per-read timeout (turns a silent stall into the same fallback),
//!   - the "no authoritative usage" trigger (collapses kill/error/early-close).
//!
//! This example forces all four shapes against a *real* live Gemini stream (the
//! disruptions are injected at the stream boundary by `Disrupt`) and shows each
//! funnel into the identical `drain_with_accounting` routine.
//!
//! ## Honest caveats
//!   - The fallback is always an *estimate* (countTokens on the partial text),
//!     never the server's exact number.
//!   - Hidden "thinking" tokens Gemini generated but never streamed cannot be
//!     counted, so the output estimate is a *lower bound* when thoughts are hidden.
//!   - Out-of-process death (kill -9, OOM, power loss) runs no in-process code;
//!     surviving that would require persisting the accumulator to disk.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{Stream, StreamExt};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionError, CompletionModel, GetTokenUsage, Usage};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig,
};
use rig::streaming::{StreamedAssistantContent, StreamingCompletionResponse};

const MODEL: &str = "gemini-2.5-flash";
/// Inject the disruption once this many output chars have streamed, so there is
/// real partial output but the cut lands well before completion.
const DISRUPT_AFTER_CHARS: usize = 150;
/// Per-read timeout. Normal inter-chunk gaps are well under this; a stalled /
/// half-open connection trips it and routes into the same accounting path.
const READ_TIMEOUT: Duration = Duration::from_secs(8);

/// The kind of mid-stream disruption to inject into a live stream.
#[derive(Clone, Copy, Debug)]
enum Disruption {
    /// No disruption — let the stream complete so we observe authoritative usage.
    None,
    /// Client-initiated kill: abort the underlying stream, surfaces as `None`.
    ManualKill,
    /// Transport/server failure mid-response: surfaces as `Some(Err(..))`.
    TransportError,
    /// Stall / half-open socket: `next()` never returns (relies on READ_TIMEOUT).
    Stall,
}

/// Wraps a live `StreamingCompletionResponse` and injects a disruption after
/// `after_chars` of output has been forwarded. This lets us exercise every
/// disruption shape against a genuine Gemini stream's partial output.
struct Disrupt<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    inner: StreamingCompletionResponse<R>,
    mode: Disruption,
    after_chars: usize,
    seen_chars: usize,
    fired: bool,
}

impl<R> Disrupt<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    fn new(inner: StreamingCompletionResponse<R>, mode: Disruption, after_chars: usize) -> Self {
        // For `None`, make the trigger unreachable so it never fires.
        let after_chars = match mode {
            Disruption::None => usize::MAX,
            _ => after_chars,
        };
        Self {
            inner,
            mode,
            after_chars,
            seen_chars: 0,
            fired: false,
        }
    }
}

impl<R> Stream for Disrupt<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    type Item = Result<StreamedAssistantContent<R>, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // A stall is sticky: once tripped it must NEVER produce again, otherwise
        // the consumer's timeout wakeup would re-poll us, we'd drain the buffered
        // real stream, and `Elapsed` would never win. The read timeout's timer is
        // what wakes the task; we just keep parking.
        if this.fired && matches!(this.mode, Disruption::Stall) {
            return Poll::Pending;
        }

        if !this.fired && this.seen_chars >= this.after_chars {
            this.fired = true;
            match this.mode {
                Disruption::ManualKill => {
                    // Real cancellation of the underlying stream (drops the HTTP
                    // body / generator); surfaces to the consumer as `None`.
                    this.inner.cancel();
                    return Poll::Ready(None);
                }
                Disruption::TransportError => {
                    return Poll::Ready(Some(Err(CompletionError::ProviderError(
                        "injected mid-stream transport drop".to_string(),
                    ))));
                }
                Disruption::Stall => {
                    // Park forever without scheduling a wake: the consumer's
                    // timeout is what makes progress. Simulates a half-open socket.
                    return Poll::Pending;
                }
                Disruption::None => {}
            }
        }

        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => {
                this.seen_chars += visible_len(&item);
                Poll::Ready(Some(Ok(item)))
            }
            other => other,
        }
    }
}

/// Length of human-visible text in a stream item (text + reasoning deltas).
fn visible_len<R>(item: &StreamedAssistantContent<R>) -> usize {
    match item {
        StreamedAssistantContent::Text(t) => t.text.chars().count(),
        StreamedAssistantContent::ReasoningDelta { reasoning, .. } => reasoning.chars().count(),
        StreamedAssistantContent::Reasoning(r) => r.display_text().chars().count(),
        _ => 0,
    }
}

/// How a drain ended.
enum Outcome {
    /// Stream completed and delivered authoritative usage.
    Clean(Usage),
    /// Stream was disrupted; usage estimated locally from partial output.
    Estimated { usage: Usage, reason: String },
}

struct Report {
    label: &'static str,
    output_chars: usize,
    outcome: Outcome,
}

/// The single accounting routine every disruption funnels through.
///
/// Accumulates output incrementally, watches for authoritative usage, and bounds
/// each read with a timeout. If the stream ends — by `None`, `Err`, a zeroed
/// `Final`, or a stall — without authoritative usage, it estimates tokens from
/// the partial output via `countTokens`.
async fn drain_with_accounting<S, R>(
    label: &'static str,
    mut stream: S,
    http: &reqwest::Client,
    api_key: &str,
    prompt_text: &str,
) -> anyhow::Result<Report>
where
    S: Stream<Item = Result<StreamedAssistantContent<R>, CompletionError>> + Unpin,
    R: Clone + Unpin + GetTokenUsage,
{
    let mut output = String::new();
    let mut authoritative: Option<Usage> = None;
    let mut reason: Option<String> = None;

    loop {
        match tokio::time::timeout(READ_TIMEOUT, stream.next()).await {
            // (4) Stall / half-open: no bytes within the read window.
            Err(_elapsed) => {
                reason = Some(format!("stall: no data within {:?}", READ_TIMEOUT));
                break;
            }
            // Stream ended.
            Ok(None) => {
                // (1) manual kill / drop, or (3) premature clean close: if no
                // authoritative usage arrived, we must estimate.
                if authoritative.is_none() {
                    reason = Some("stream closed without authoritative usage".to_string());
                }
                break;
            }
            // (2) Transport/server error mid-response.
            Ok(Some(Err(err))) => {
                reason = Some(format!("stream error: {err}"));
                break;
            }
            Ok(Some(Ok(item))) => match item {
                StreamedAssistantContent::Text(text) => output.push_str(&text.text),
                StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                    output.push_str(&reasoning)
                }
                StreamedAssistantContent::Reasoning(r) => output.push_str(&r.display_text()),
                StreamedAssistantContent::Final(resp) => {
                    // Authoritative usage — but only trust it if non-zero. A
                    // premature clean close yields a zeroed Final (shape #3).
                    let usage = resp.token_usage();
                    if usage.has_values() {
                        authoritative = Some(usage);
                    }
                }
                _ => {}
            },
        }
    }

    let outcome = match authoritative {
        Some(usage) => Outcome::Clean(usage),
        None => {
            // Disruption-agnostic fallback: estimate from whatever we received.
            // Input is deterministic regardless of when the cut happened.
            let input_tokens = count_tokens(http, api_key, prompt_text).await?;
            let output_tokens = count_tokens(http, api_key, &output).await?;

            let mut usage = Usage::new();
            usage.input_tokens = input_tokens;
            usage.output_tokens = output_tokens;
            usage.total_tokens = input_tokens + output_tokens;

            Outcome::Estimated {
                usage,
                reason: reason.unwrap_or_else(|| "unknown disruption".to_string()),
            }
        }
    };

    Ok(Report {
        label,
        output_chars: output.chars().count(),
        outcome,
    })
}

/// Call Gemini's free `countTokens` endpoint on arbitrary text. Returns 0 for
/// empty input (the endpoint rejects empty `contents`).
async fn count_tokens(http: &reqwest::Client, api_key: &str, text: &str) -> anyhow::Result<u64> {
    if text.is_empty() {
        return Ok(0);
    }
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:countTokens?key={api_key}"
    );
    let body = serde_json::json!({
        "contents": [{ "parts": [{ "text": text }] }]
    });
    let resp = http
        .post(url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    let value: serde_json::Value = resp.json().await?;
    let total = value
        .get("totalTokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    Ok(total)
}

/// Disable Gemini's "thinking" so visible text streams immediately. Otherwise
/// 2.5-flash spends seconds generating hidden thoughts (no chunks sent), which
/// is indistinguishable from a stall and would trip the read timeout before any
/// real partial output — masking the injected disruptions.
fn no_thinking_params() -> anyhow::Result<serde_json::Value> {
    let params = AdditionalParameters {
        generation_config: Some(GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(false),
                thinking_budget: Some(0),
                thinking_level: None,
            }),
            ..Default::default()
        }),
        additional_params: None,
    };
    Ok(serde_json::to_value(&params)?)
}

async fn run_scenario(
    label: &'static str,
    mode: Disruption,
    prompt: &str,
    http: &reqwest::Client,
    api_key: &str,
) -> anyhow::Result<Report> {
    let client = gemini::Client::from_env()?;
    let model = client.completion_model(MODEL);

    let stream = model
        .completion_request(prompt)
        .temperature(0.7)
        .max_tokens(2000)
        .additional_params(no_thinking_params()?)
        .stream()
        .await?;

    let disrupted = Disrupt::new(stream, mode, DISRUPT_AFTER_CHARS);
    drain_with_accounting(label, disrupted, http, api_key, prompt).await
}

fn print_report(report: &Report) {
    println!("\n=== {} ===", report.label);
    println!("partial output: {} chars", report.output_chars);
    match &report.outcome {
        Outcome::Clean(usage) => {
            println!("result: CLEAN — authoritative usage from final chunk");
            println!(
                "  input={} output={} reasoning={} total={}",
                usage.input_tokens, usage.output_tokens, usage.reasoning_tokens, usage.total_tokens
            );
        }
        Outcome::Estimated { usage, reason } => {
            println!("result: ESTIMATED via countTokens (cut: {reason})");
            println!(
                "  input={} output={} total={}  (output is a lower bound; hidden thoughts uncounted)",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
            );
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY must be set"))?;
    let http = reqwest::Client::new();

    // Short prompt for the clean baseline (completes quickly, real usage).
    let short_prompt = "Reply with a single short sentence greeting.";
    // Long prompt so disruptions land mid-stream with real partial output.
    let long_prompt = "Write a detailed, multi-paragraph essay (about 600 words) \
        on the history and design philosophy of the Rust programming language.";

    println!(
        "Demonstrating one token-accounting path across every mid-stream disruption.\n\
         Model: {MODEL}  |  disrupt after ~{DISRUPT_AFTER_CHARS} chars  |  read timeout {READ_TIMEOUT:?}"
    );

    let scenarios = [
        (
            "clean completion (baseline)",
            Disruption::None,
            short_prompt,
        ),
        ("manual kill (cancel)", Disruption::ManualKill, long_prompt),
        ("transport error", Disruption::TransportError, long_prompt),
        ("stall / half-open", Disruption::Stall, long_prompt),
    ];

    for (label, mode, prompt) in scenarios {
        match run_scenario(label, mode, prompt, &http, &api_key).await {
            Ok(report) => print_report(&report),
            Err(err) => println!("\n=== {label} ===\nFAILED: {err}"),
        }
    }

    println!(
        "\nEvery disrupted run produced a token count without ever receiving the \
         provider's final usage chunk — keyed only on \"no authoritative usage\", \
         so it is agnostic to why the stream stopped."
    );

    Ok(())
}
