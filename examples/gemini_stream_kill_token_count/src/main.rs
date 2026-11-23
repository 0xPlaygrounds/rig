//! Estimates token usage with Gemini's `countTokens` endpoint when a streaming
//! generation stops before its final chunk.
//!
//! Gemini sends `usageMetadata` only on the last chunk of a stream. `Disrupt`
//! injects four disruptions into a live stream: a dropped stream, a transport
//! error, an early clean close, and a stall caught by a per-read timeout. Each goes
//! through `drain_with_accounting`, which counts the partial output whenever no
//! usage arrived. The count is an estimate, and a lower bound when thinking tokens
//! are hidden.
//!
//! Requires `GEMINI_API_KEY`. Run with `cargo run -p gemini_stream_kill_token_count`.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{Stream, StreamExt};
use rig::completion::Usage;
use rig::completion::{CompletionRequest, Reasoning};
use rig::error::ErrorReport;
use rig::error::ProviderError;
use rig::message::AssistantContent;
use rig::providers::gemini::Gemini;
use rig::providers::gemini::extension::GeminiOptions;
use rig::streaming::{Item, Relayed, StreamEvent, StreamEvents};

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

/// Wraps a live relayed completion stream and injects a disruption after
/// `after_chars` of output has been forwarded. This lets us exercise every
/// disruption shape against a genuine Gemini stream's partial output.
struct Disrupt {
    /// The live stream, until a manual kill drops it.
    inner: Option<StreamEvents>,
    mode: Disruption,
    after_chars: usize,
    seen_chars: usize,
    fired: bool,
}

impl Disrupt {
    fn new(inner: StreamEvents, mode: Disruption, after_chars: usize) -> Self {
        // For `None`, make the trigger unreachable so it never fires.
        let after_chars = match mode {
            Disruption::None => usize::MAX,
            _ => after_chars,
        };
        Self {
            inner: Some(inner),
            mode,
            after_chars,
            seen_chars: 0,
            fired: false,
        }
    }
}

impl Stream for Disrupt {
    type Item = Result<Relayed, ErrorReport>;

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
                    // Real cancellation: dropping the stream drops the HTTP
                    // body; surfaces to the consumer as `None`.
                    this.inner = None;
                    return Poll::Ready(None);
                }
                Disruption::TransportError => {
                    // The same in-band shape a real transport failure takes:
                    // the provider's error, mapped to the stream's report.
                    return Poll::Ready(Some(Err(ErrorReport::from(&ProviderError::Provider(
                        "injected mid-stream transport drop".to_string(),
                    )))));
                }
                Disruption::Stall => {
                    // Park forever without scheduling a wake: the consumer's
                    // timeout is what makes progress. Simulates a half-open socket.
                    return Poll::Pending;
                }
                Disruption::None => {}
            }
        }

        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(inner).poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => {
                this.seen_chars += visible_len(&item);
                Poll::Ready(Some(Ok(item)))
            }
            other => other,
        }
    }
}

/// Length of human-visible text in a stream item (text + reasoning deltas).
fn visible_len(item: &Relayed) -> usize {
    let Relayed::Item(Item::Event(item)) = item else {
        return 0;
    };
    match item {
        StreamEvent::Text { text, .. } => text.chars().count(),
        StreamEvent::Reasoning { text, .. } => text.chars().count(),
        StreamEvent::End {
            content: AssistantContent::Reasoning(r),
            ..
        } => r.text.chars().count(),
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
/// `Done`, or a stall — without authoritative usage, it estimates tokens from
/// the partial output via `countTokens`.
async fn drain_with_accounting<S>(
    label: &'static str,
    mut stream: S,
    http: &reqwest::Client,
    api_key: &str,
    prompt_text: &str,
) -> anyhow::Result<Report>
where
    S: Stream<Item = Result<Relayed, ErrorReport>> + Unpin,
{
    let mut output = String::new();
    let mut authoritative: Option<Usage> = None;
    let mut reason: Option<String> = None;

    loop {
        match tokio::time::timeout(READ_TIMEOUT, stream.next()).await {
            // (4) Stall / half-open: no bytes within the read window.
            Err(_elapsed) => {
                reason = Some(format!("stall: no data within {READ_TIMEOUT:?}"));
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
            Ok(Some(Ok(Relayed::Origin(_)))) => {}
            Ok(Some(Ok(Relayed::Item(Item::Event(item))))) => match item {
                StreamEvent::Text { text, .. } => output.push_str(&text),
                StreamEvent::Reasoning { text, .. } => {
                    output.push_str(&text);
                }
                StreamEvent::End {
                    content: AssistantContent::Reasoning(r),
                    ..
                } => {
                    output.push_str(&r.text.clone());
                }
                _ => {}
            },
            Ok(Some(Ok(Relayed::Done(resp)))) => {
                // Authoritative usage. A premature clean close (shape #3)
                // never emits a Done at all — the absence of a terminal
                // record is itself the truncation signal — so reaching this
                // arm means the stream completed. Still guard on reported
                // usage in case the provider sent none.
                let usage = resp.usage;
                if usage.is_reported() {
                    authoritative = Some(usage);
                }
            }
            // An unmodeled provider event carries no visible text.
            Ok(Some(Ok(Relayed::Item(Item::Unknown(_)) | Relayed::Usage(_)))) => {}
        }
    }

    let outcome = match authoritative {
        Some(usage) => Outcome::Clean(usage),
        None => {
            // Disruption-agnostic fallback: estimate from whatever we received.
            // Input is deterministic regardless of when the cut happened.
            let input_tokens = count_tokens(http, api_key, prompt_text).await?;
            let output_tokens = count_tokens(http, api_key, &output).await?;

            let usage = Usage::new()
                .input_tokens(input_tokens)
                .output_tokens(output_tokens)
                .total_tokens(input_tokens + output_tokens);

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
    let url =
        format!("https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:countTokens");
    let body = serde_json::json!({
        "contents": [{ "parts": [{ "text": text }] }]
    });
    let resp = http
        .post(url)
        .header("x-goog-api-key", api_key)
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
fn no_thinking(request: CompletionRequest) -> CompletionRequest {
    request
        .reasoning(Reasoning::Off)
        .provider_option(GeminiOptions::new().include_thoughts(false))
}

async fn run_scenario(
    label: &'static str,
    mode: Disruption,
    prompt: &str,
    http: &reqwest::Client,
    api_key: &str,
) -> anyhow::Result<Report> {
    let model = Gemini::from_env()?.completion(MODEL);

    let stream = model.stream(no_thinking(
        CompletionRequest::new(prompt)
            .temperature(0.7)
            .max_tokens(2000),
    ))?;

    let disrupted = Disrupt::new(stream.into_relay(), mode, DISRUPT_AFTER_CHARS);
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
                usage.input_tokens.unwrap_or(0),
                usage.output_tokens.unwrap_or(0),
                usage
                    .reasoning_tokens
                    .map_or_else(|| "unreported".to_string(), |n| n.to_string()),
                usage.total_tokens.unwrap_or(0)
            );
        }
        Outcome::Estimated { usage, reason } => {
            println!("result: ESTIMATED via countTokens (cut: {reason})");
            println!(
                "  input={} output={} total={}  (output is a lower bound; hidden thoughts uncounted)",
                usage.input_tokens.unwrap_or(0),
                usage.output_tokens.unwrap_or(0),
                usage.total_tokens.unwrap_or(0)
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
