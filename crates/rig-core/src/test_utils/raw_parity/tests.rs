//! Guarantee 5 over every recorded pair: one prompt answered unary and
//! streamed, on one API.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{comparable, interactions, raw_pair, recorded_reply, scalar};
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::providers::copilot::CopilotConfig;
use crate::providers::gemini::GeminiConfig;
use crate::providers::openai::wire::{
    COHERE, DEEPSEEK, DOUBLEWORD, Dialect, GROQ, LLAMACPP, MISTRAL, OLLAMA, OPENAI, OPENROUTER,
    OpenAIConfig, PERPLEXITY, VENICE,
};

/// The cassette corpus, beside this crate.
fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../rig-cassette/fixtures/cassettes")
}

/// One recorded interaction: its request path and body, and its reply.
struct Recorded {
    path: String,
    /// The request body, when it is JSON.
    request: Option<Value>,
    status: Option<u16>,
    /// Whether the reply is a stream.
    streamed: bool,
    reply: Bytes,
}

impl Recorded {
    /// One interaction's text, its reply body left unread.
    fn parse(interaction: &str) -> Option<Self> {
        let (sent, reply) = interaction.split_once("\nthen:\n")?;
        let path = scalar(sent, "path").unwrap_or_default();
        let request = scalar(sent, "body").and_then(|body| serde_json::from_str(&body).ok());
        let status = scalar(reply, "status").and_then(|status| status.parse().ok());
        let mut lines = reply.lines();
        let content_type = lines
            .by_ref()
            .position(|line| line.trim().eq_ignore_ascii_case("- name: content-type"))
            .and_then(|_| lines.next())
            .and_then(|line| line.trim().strip_prefix("value:"))
            .unwrap_or_default()
            .to_ascii_lowercase();
        let streamed = ["event-stream", "eventstream", "ndjson"]
            .iter()
            .any(|kind| content_type.contains(kind));
        Some(Self {
            path,
            request,
            status,
            streamed,
            reply: Bytes::new(),
        })
    }

    /// The interaction `relative` names under `provider`'s recordings:
    /// `file.yaml`, its first interaction, or `file.yaml#n`.
    fn named(provider: &str, relative: &str) -> Self {
        let (file, n) = interaction(relative);
        let file = corpus().join(provider).join(file);
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("{} is readable: {error}", file.display()));
        let recorded = n
            .checked_sub(1)
            .and_then(|index| interactions(&text).nth(index))
            .and_then(Self::parse)
            .zip(recorded_reply(&text, n));
        let Some((recorded, reply)) = recorded else {
            panic!("{provider}/{relative} records interaction {n} and its reply");
        };
        Self { reply, ..recorded }
    }

    /// The model the request names, when its body does.
    fn model(&self) -> Option<&str> {
        self.request.as_ref()?.get("model")?.as_str()
    }

    /// What makes two requests the same turn asked both ways: the endpoint
    /// without its streaming suffix, and the body without its streaming
    /// switches, with keys sorted.
    fn turn(&self) -> Option<(String, String)> {
        let mut request = self.request.clone()?;
        if let Some(body) = request.as_object_mut() {
            body.shift_remove("stream");
            body.shift_remove("stream_options");
        }
        let endpoint = ["-stream", ":streamGenerateContent", ":generateContent"]
            .iter()
            .fold(self.path.as_str(), |path, suffix| {
                path.strip_suffix(suffix).unwrap_or(path)
            });
        Some((endpoint.to_owned(), sorted(&request).to_string()))
    }
}

/// `file.yaml#n` as the file and the interaction number; `n` is 1 when
/// absent.
fn interaction(relative: &str) -> (&str, usize) {
    match relative.rsplit_once('#') {
        Some((file, n)) => (file, n.parse().unwrap_or(0)),
        None => (relative, 1),
    }
}

/// `value` with every object's keys sorted.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), sorted(value)))
                    .collect::<Map<String, Value>>(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// One prompt answered both ways on one API. `unary` and `streamed` name an
/// interaction under the provider's recordings (`file.yaml` or
/// `file.yaml#n`).
struct Pair {
    provider: &'static str,
    unary: &'static str,
    streamed: &'static str,
    /// JSON pointers to what the two answers of the prompt cannot share
    /// (generated text, signatures, counts), dropped before comparing.
    minted: &'static [&'static str],
}

const PAIRS: &[Pair] = &[
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_exposes_stop_sequence.yaml",
        streamed: "raw_stream_capture_matrix/raw_exposes_stop_sequence.yaml",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_exposes_thinking_block_and_signature.yaml",
        streamed: "raw_stream_capture_matrix/terminal_raw_round_trips_for_thinking_stream.yaml",
        minted: &["/content/0/signature"],
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_exposes_tool_use_block.yaml",
        streamed: "raw_stream_capture_matrix/terminal_raw_round_trips_for_tool_use_stream.yaml",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_round_trips_into_provider_type.yaml",
        streamed: "raw_stream_capture_matrix/terminal_raw_round_trips_into_provider_type.yaml",
        // The two recordings ask for different words.
        minted: &["/content/0/text", "/usage"],
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/normalized_fields_match_raw_renormalized.yaml",
        streamed: "raw_stream_capture_matrix/normalized_terminal_matches_raw_renormalized.yaml",
        // The two recordings ask for different words.
        minted: &["/content/0/text", "/usage"],
    },
    Pair {
        provider: "cohere",
        unary: "raw_capture_matrix/raw_exposes_envelope_fields.yaml",
        streamed: "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk.yaml",
        // The two recordings ask for different words.
        minted: &["/choices/0/message/content", "/usage"],
    },
    Pair {
        provider: "gemini",
        unary: "raw_capture_matrix/raw_exposes_forced_function_call.yaml",
        streamed: "raw_stream_capture_matrix/raw_terminal_keeps_stop_on_forced_function_call.yaml",
        minted: &["/responseId"],
    },
    Pair {
        provider: "gemini",
        unary: "generate_tool_args/nested_arguments_roundtrip_nonstreaming.yaml",
        streamed: "generate_tool_args/nested_arguments_streaming.yaml",
        minted: &[
            "/responseId",
            "/candidates/0/content/parts/0/thoughtSignature",
        ],
    },
    Pair {
        provider: "gemini",
        unary: "corpus_breadth/output_tool_unary.yaml",
        streamed: "corpus_breadth/output_tool_streamed.yaml",
        minted: &[
            "/responseId",
            "/candidates/0/content/parts/0/thoughtSignature",
        ],
    },
    Pair {
        provider: "gemini",
        unary: "corpus_matrix/causal_completion_concurrent.yaml",
        streamed: "corpus_matrix/causal_completion_streamed.yaml",
        // Gemini 3 states no `finishMessage` on a streamed call turn.
        minted: &[
            "/responseId",
            "/candidates/0/content/parts/0/thoughtSignature",
            "/candidates/0/finishMessage",
        ],
    },
    Pair {
        provider: "gemini",
        unary: "regression/structured_output_without_max_tokens.yaml",
        streamed: "regression/streaming_structured_output_without_max_tokens.yaml",
        // The two answers word the summary differently.
        minted: &[
            "/responseId",
            "/candidates/0/content/parts/0/text",
            "/candidates/0/content/parts/0/thoughtSignature",
            "/usageMetadata",
        ],
    },
    Pair {
        provider: "gemini",
        unary: "auto_caching/support_chat_100_current_turn.yaml",
        streamed: "auto_caching/support_chat_100_streamed.yaml",
        // Gemini 3 states no `finishMessage` on a streamed call turn.
        minted: &[
            "/responseId",
            "/candidates/0/content/parts/0/thoughtSignature",
            "/candidates/0/finishMessage",
            "/usageMetadata",
        ],
    },
    Pair {
        provider: "ollama",
        unary: "raw_capture_matrix/raw_exposes_envelope_fields.yaml",
        streamed: "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk.yaml",
        // The two recordings ask for different words.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/usage",
        ],
    },
    Pair {
        provider: "ollama",
        unary: "native/inline_think_whole.yaml",
        streamed: "native/inline_think_streamed.yaml",
        // The daemon's timings and its prompt cache differ between runs.
        minted: &[
            "/total_duration",
            "/load_duration",
            "/prompt_eval_duration",
            "/eval_duration",
            "/prompt_eval_cached_count",
        ],
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/chat_raw_round_trips_typed.yaml",
        streamed: "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/chat_tool_call_raw_round_trips_typed.yaml",
        streamed: "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed.yaml",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/responses_raw_exposes_service_tier_and_store.yaml",
        streamed: "raw_stream_capture_matrix/responses_stream_raw_exposes_status.yaml",
        // `billing` is in unary bodies only.
        minted: &["/billing"],
    },
    Pair {
        provider: "openrouter",
        unary: "raw_capture_matrix/raw_round_trips_openrouter_type.yaml",
        streamed: "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider.yaml",
        minted: &[],
    },
    Pair {
        provider: "perplexity",
        unary: "raw_capture_matrix/normalized_fields_match_raw_renormalized.yaml",
        streamed: "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object.yaml",
        minted: &[],
    },
    Pair {
        provider: "deepseek",
        unary: "followup_hunt_matrix/blocking_stop_sequence_reaches_the_wire_and_stops_generation.yaml",
        streamed: "followup_hunt_matrix/streaming_stop_sequence_reaches_the_wire_and_stops_generation.yaml",
        minted: &[],
    },
    Pair {
        provider: "deepseek",
        unary: "streaming_logprobs_matrix/blocking_disabled_length_top_absent.yaml",
        streamed: "streaming_logprobs_matrix/streaming_disabled_length_top_absent.yaml",
        // Each answer samples its own probabilities.
        minted: &["/choices/0/logprobs/content/0/logprob"],
    },
    Pair {
        provider: "deepseek",
        unary: "streaming_logprobs_matrix/blocking_low_length_top_absent.yaml",
        streamed: "streaming_logprobs_matrix/streaming_low_length_top_absent.yaml",
        // Each answer samples its own probabilities.
        minted: &["/choices/0/logprobs/reasoning_content/0/logprob"],
    },
    Pair {
        provider: "deepseek",
        unary: "streaming_logprobs_matrix/blocking_low_length_top_two.yaml",
        streamed: "streaming_logprobs_matrix/streaming_low_length_top_two.yaml",
        // Each answer samples its own probabilities.
        minted: &[
            "/choices/0/logprobs/reasoning_content/0/logprob",
            "/choices/0/logprobs/reasoning_content/0/top_logprobs/0/logprob",
            "/choices/0/logprobs/reasoning_content/0/top_logprobs/1/logprob",
        ],
    },
    Pair {
        provider: "deepseek",
        unary: "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap.yaml",
        streamed: "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap.yaml",
        minted: &[],
    },
    Pair {
        provider: "doubleword",
        unary: "finish_reason_matrix/blocking_natural_stop.yaml",
        streamed: "finish_reason_matrix/streaming_natural_stop.yaml",
        minted: &[],
    },
    Pair {
        provider: "doubleword",
        unary: "finish_reason_matrix/blocking_tool_calls.yaml",
        streamed: "finish_reason_matrix/streaming_tool_calls.yaml",
        // The unary answer reasoned before its call and the streamed one did
        // not.
        minted: &[
            "/choices/0/message/reasoning_content",
            "/choices/0/message/reasoning_details",
            "/choices/0/message/tool_calls/0/function/arguments",
            "/usage",
        ],
    },
    Pair {
        provider: "llamacpp",
        unary: "response_shape_matrix/two_candidates_blocking.yaml",
        streamed: "response_shape_matrix/two_candidates_streaming.yaml",
        // Each answer writes its own first candidate and times itself.
        minted: &["/choices/0/message/content", "/timings", "/usage"],
    },
    Pair {
        provider: "llamacpp",
        unary: "truncation_matrix/tool_call_cut_mid_arguments.yaml",
        streamed: "truncation_matrix/streaming_tool_call_cut_mid_arguments.yaml",
        // llama.cpp's unary body states an empty `content` beside its calls,
        // which its stream never sends; each answer times itself.
        minted: &["/choices/0/message/content", "/timings"],
    },
    Pair {
        provider: "openai",
        unary: "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_zero.yaml",
        streamed: "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_zero.yaml",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_absent.yaml",
        streamed: "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_absent.yaml",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_two.yaml",
        streamed: "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_two.yaml",
        minted: &[],
    },
    Pair {
        provider: "openrouter",
        unary: "reasoning_tool_order_matrix/blocking_single.yaml",
        streamed: "reasoning_tool_order_matrix/streaming_single.yaml",
        // Each answer reasons in its own words, and OpenRouter's unary body
        // keeps the stream `index` on its calls.
        minted: &[
            "/choices/0/message/reasoning",
            "/choices/0/message/reasoning_details/0/text",
            "/choices/0/message/reasoning_details/0/signature",
            "/choices/0/message/tool_calls/0/index",
            "/usage",
        ],
    },
    Pair {
        provider: "openrouter",
        unary: "refusal_matrix/blocking_refusal_with_tools_in_request.yaml",
        streamed: "refusal_matrix/streaming_refusal_emits_no_tool_calls.yaml",
        minted: &[],
    },
    Pair {
        provider: "openrouter",
        unary: "streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_absent.yaml",
        streamed: "streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_absent.yaml",
        minted: &[],
    },
    Pair {
        provider: "openrouter",
        unary: "terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_plain_two.yaml",
        streamed: "terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_plain_two.yaml",
        minted: &[],
    },
    Pair {
        provider: "openrouter",
        unary: "terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_tool.yaml",
        streamed: "terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_tool.yaml",
        // OpenRouter's unary body keeps the stream `index` on its calls.
        minted: &["/choices/0/message/tool_calls/0/index"],
    },
    Pair {
        provider: "openrouter",
        unary: "terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_one.yaml",
        streamed: "terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_one.yaml",
        minted: &[],
    },
    Pair {
        provider: "venice",
        unary: "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap.yaml",
        streamed: "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap.yaml",
        // Each answer is cut at its own words, and Venice states its
        // parameters in a unary body only.
        minted: &["/choices/0/message/content", "/venice_parameters"],
    },
    Pair {
        provider: "groq",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Each answer reasons and counts in its own words. A gpt-oss stream
        // names each delta's `channel`, which the unary message does not;
        // Groq states the seed in a unary body only and repeats the usage
        // under `x_groq` in a stream only.
        minted: &[
            "/choices/0/message/channel",
            "/choices/0/message/reasoning",
            "/system_fingerprint",
            "/usage",
            "/x_groq/seed",
            "/x_groq/usage",
        ],
    },
    Pair {
        provider: "groq",
        unary: "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming.yaml",
        streamed: "agent_tool_sessions/parallel_tool_calls_single_turn_streaming.yaml",
        // Each answer times itself. Groq states the seed in a unary body only
        // and repeats the usage under `x_groq` in a stream only.
        minted: &[
            "/usage/completion_time",
            "/usage/prompt_time",
            "/usage/queue_time",
            "/usage/total_time",
            "/x_groq/seed",
            "/x_groq/usage",
        ],
    },
    Pair {
        provider: "mistral",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Mistral's unary body states an empty `content` beside its calls and
        // keeps their stream `index`, which its stream does not; the cache
        // was cold for one answer only.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/tool_calls/0/index",
            "/choices/0/message/tool_calls/1/index",
            "/usage/prompt_tokens_details/cached_tokens",
        ],
    },
    Pair {
        provider: "openai",
        unary: "long_task_matrix/chat_repair.yaml",
        streamed: "long_task_matrix/chat_repair_streamed.yaml",
        minted: &[],
    },
    Pair {
        provider: "openrouter",
        unary: "reasoning_roundtrip/nonstreaming.yaml",
        streamed: "reasoning_roundtrip/streaming.yaml",
        // Each answer reasons and answers in its own words.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/choices/0/message/reasoning_details/0/summary",
            "/choices/0/message/reasoning_details/1/data",
            "/usage",
        ],
    },
    Pair {
        provider: "openrouter",
        unary: "agent_tool_sessions/sequential_complex_tool_calls_nonstreaming.yaml",
        streamed: "agent_tool_sessions/sequential_complex_tool_calls_streaming.yaml",
        // OpenRouter's unary body keeps the stream `index` on its calls.
        minted: &["/choices/0/message/tool_calls/0/index"],
    },
    Pair {
        provider: "venice",
        unary: "reasoning_matrix/tool_unary.yaml",
        streamed: "reasoning_matrix/tool_streamed.yaml",
        // Each answer reasons in its own words. Venice's unary body states an
        // empty `content` beside its calls and its parameters, which its
        // stream does not.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning_content",
            "/cost",
            "/usage",
            "/venice_parameters",
        ],
    },
    Pair {
        provider: "doubleword",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Each answer counts its own reasoning; Doubleword states the service
        // tier in a unary body only.
        minted: &["/service_tier", "/usage"],
    },
    Pair {
        provider: "anthropic",
        unary: "corpus_host/custom_at_completion_call.yaml",
        streamed: "corpus_host/custom_at_start_streamed.yaml",
        // Each answer words and counts its reply in its own way.
        minted: &["/content/0/text", "/usage"],
    },
    Pair {
        provider: "anthropic",
        unary: "corpus_memory/two_runs.yaml#2",
        streamed: "corpus_memory/two_runs_streamed.yaml#2",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "corpus_output/prompted_unary.yaml",
        streamed: "corpus_output/prompted_streamed.yaml",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "corpus_request_shape/output_schema_unary.yaml",
        streamed: "corpus_request_shape/output_schema_streamed.yaml",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "corpus_request_shape/thinking_unary.yaml",
        streamed: "corpus_request_shape/thinking_streamed.yaml",
        // Each answer signs its own thinking.
        minted: &["/content/0/signature"],
    },
    Pair {
        provider: "anthropic",
        unary: "corpus_request_shape/tool_choice_auto.yaml",
        streamed: "corpus_endings/tool_dispatch_cancelled_streamed.yaml",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "corpus_shaping/extra_context.yaml",
        streamed: "corpus_shaping/extra_context_streamed.yaml",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "models/sonnet_5_5/session.yaml#20",
        streamed: "models/sonnet_5_5/session.yaml#21",
        minted: &[],
    },
    Pair {
        provider: "anthropic",
        unary: "request_override/request_overridden_by_hook_blocking.yaml",
        streamed: "request_override/request_overridden_by_hook_streaming.yaml",
        minted: &[],
    },
    Pair {
        provider: "copilot",
        unary: "reasoning_roundtrip/nonstreaming.yaml",
        streamed: "reasoning_roundtrip/streaming.yaml",
        // Each answer reasons, words and counts in its own way. Copilot's stream
        // states `background`, the penalties, `store` and `top_logprobs`, which
        // its unary body omits.
        minted: &[
            "/copilot_usage",
            "/output/0/encrypted_content",
            "/output/1/content/0/text",
            "/safety_identifier",
            "/usage",
            "/background",
            "/frequency_penalty",
            "/presence_penalty",
            "/store",
            "/top_logprobs",
        ],
    },
    Pair {
        provider: "gemini",
        unary: "agent_run_recovery/repair_renames_tool_call_and_executes_it.yaml",
        streamed: "agent_run_streamed/streamed_repair_continues_the_same_stream.yaml",
        // Each answer signs and counts its own thinking.
        minted: &[
            "/candidates/0/content/parts/0/thoughtSignature",
            "/responseId",
            "/usageMetadata",
        ],
    },
    Pair {
        provider: "gemini",
        unary: "agent_run_recovery/repair_renames_tool_call_and_executes_it.yaml#2",
        streamed: "agent_run_streamed/streamed_repair_continues_the_same_stream.yaml#2",
        minted: &["/responseId"],
    },
    Pair {
        provider: "gemini",
        unary: "agent_run_recovery/repair_to_disallowed_name_fails_with_unknown_tool_call.yaml",
        streamed: "agent_run_streamed/builtin_streaming_cancellation_history_includes_assistant_turn.yaml",
        // Each answer signs its own thinking.
        minted: &[
            "/candidates/0/content/parts/0/thoughtSignature",
            "/responseId",
        ],
    },
    Pair {
        provider: "gemini",
        unary: "auto_caching/support_chat_100_compaction.yaml",
        streamed: "auto_caching/support_chat_100_streamed.yaml",
        // Each answer signs and counts its own thinking. Gemini 3 states no
        // `finishMessage` on a streamed call turn.
        minted: &[
            "/candidates/0/content/parts/0/thoughtSignature",
            "/candidates/0/finishMessage",
            "/responseId",
            "/usageMetadata",
        ],
    },
    Pair {
        provider: "groq",
        unary: "agent_tool_sessions/sequential_complex_tool_calls_nonstreaming.yaml",
        streamed: "agent_tool_sessions/sequential_complex_tool_calls_streaming.yaml",
        // Each answer times itself. Groq states the seed in a unary body only
        // and repeats the usage under `x_groq` in a stream only.
        minted: &["/usage", "/x_groq/seed", "/x_groq/usage"],
    },
    Pair {
        provider: "mistral",
        unary: "prompt_caching/blocking_probe.yaml",
        streamed: "prompt_caching/streaming_probe.yaml",
        minted: &[],
    },
    Pair {
        provider: "mistral",
        unary: "prompt_caching/blocking_probe.yaml#3",
        streamed: "prompt_caching/streaming_probe.yaml#3",
        // The cache was warm for one answer only.
        minted: &["/usage"],
    },
    Pair {
        provider: "ollama",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Each answer reasons and counts in its own words. Ollama's unary body
        // states an empty `content` beside its calls and keeps their stream
        // `index`, which its stream does not.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/choices/0/message/tool_calls/0/index",
            "/choices/0/message/tool_calls/1/index",
            "/usage",
        ],
    },
    Pair {
        provider: "ollama",
        unary: "reasoning_roundtrip/nonstreaming.yaml",
        streamed: "reasoning_roundtrip/streaming.yaml",
        // Each answer reasons, answers and counts in its own words.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/usage",
        ],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_5_4_nano/session.yaml#58",
        streamed: "models/gpt_5_4_nano/session.yaml#59",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_5_4_nano/session.yaml#84",
        streamed: "models/gpt_5_4_nano/session.yaml#86",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_6_luna/session.yaml#58",
        streamed: "models/gpt_6_luna/session.yaml#59",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_6_luna/session.yaml#84",
        streamed: "models/gpt_6_luna/session.yaml#86",
        minted: &[],
    },
    Pair {
        provider: "openai",
        unary: "refusal_matrix/cross_surface_refusal_parity.yaml#2",
        streamed: "refusal_matrix/chat_streaming_agent_surfaces_refusal.yaml",
        // Each answer words its refusal in its own way.
        minted: &["/choices/0/message/refusal", "/usage"],
    },
    Pair {
        provider: "openai",
        unary: "corpus_host/embed_prompt.yaml#2",
        streamed: "corpus_host/embed_prompt_streamed.yaml#2",
        // `billing` is in unary bodies only.
        minted: &["/billing"],
    },
    Pair {
        provider: "openai",
        unary: "corpus_output/prompted_unary.yaml",
        streamed: "corpus_breadth/prompted_streamed.yaml",
        // `billing` is in unary bodies only; each answer words its summary in
        // its own way.
        minted: &["/billing", "/output/0/content/0/text", "/usage"],
    },
    Pair {
        provider: "openai",
        unary: "corpus_output/tool_unary.yaml",
        streamed: "corpus_breadth/output_tool_streamed.yaml",
        // `billing` is in unary bodies only; each answer words its arguments in
        // its own way.
        minted: &[
            "/billing",
            "/output/0/arguments",
            "/output/0/call_id",
            "/usage",
        ],
    },
    Pair {
        provider: "openai",
        unary: "corpus_retrieval/dynamic_context_one.yaml#3",
        streamed: "corpus_retrieval/dynamic_context_one_streamed.yaml#3",
        // `billing` is in unary bodies only.
        minted: &["/billing"],
    },
    Pair {
        provider: "openai",
        unary: "corpus_retrieval/retrieved_tools_one.yaml#3",
        streamed: "corpus_retrieval/retrieved_tools_one_streamed.yaml#3",
        // `billing` is in unary bodies only.
        minted: &["/billing", "/output/0/call_id"],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_5_2_pro/session.yaml#16",
        streamed: "models/gpt_5_2_pro/session.yaml#17",
        // `billing` is in unary bodies only; each answer encrypts its own
        // reasoning.
        minted: &["/billing", "/output/0/encrypted_content"],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_5_2_pro/session.yaml#42",
        streamed: "models/gpt_5_2_pro/session.yaml#44",
        // `billing` is in unary bodies only; each answer encrypts its own
        // reasoning.
        minted: &[
            "/billing",
            "/output/0/encrypted_content",
            "/output/1/call_id",
            "/usage",
        ],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_5_4_nano/session.yaml#16",
        streamed: "models/gpt_5_4_nano/session.yaml#17",
        // `billing` is in unary bodies only.
        minted: &["/billing"],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_5_4_nano/session.yaml#42",
        streamed: "models/gpt_5_4_nano/session.yaml#44",
        // `billing` is in unary bodies only; each answer spaces its arguments in
        // its own way.
        minted: &[
            "/billing",
            "/output/0/arguments",
            "/output/0/call_id",
            "/usage",
        ],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_6_luna/session.yaml#16",
        streamed: "models/gpt_6_luna/session.yaml#17",
        // `billing` is in unary bodies only.
        minted: &["/billing"],
    },
    Pair {
        provider: "openai",
        unary: "models/gpt_6_luna/session.yaml#42",
        streamed: "models/gpt_6_luna/session.yaml#44",
        // `billing` is in unary bodies only.
        minted: &["/billing", "/output/0/call_id"],
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_agent_matrix/responses_blocking_hooks_see_raw.yaml",
        streamed: "raw_capture_agent_matrix/responses_streamed_hooks_see_raw.yaml",
        // `billing` is in unary bodies only.
        minted: &["/billing"],
    },
    Pair {
        provider: "openai",
        unary: "refusal_matrix/cross_surface_refusal_parity.yaml",
        streamed: "refusal_matrix/responses_agent_streaming_refusal_surfaces.yaml",
        // `billing` is in unary bodies only; each answer words its refusal in
        // its own way.
        minted: &["/billing", "/output/0/content/0/refusal", "/usage"],
    },
    Pair {
        provider: "openai",
        unary: "response_metadata_matrix/object_top_p_blocking_tool_call.yaml",
        streamed: "response_metadata_matrix/object_top_p_streaming_terminal_usage.yaml",
        // `billing` is in unary bodies only.
        minted: &["/billing"],
    },
    Pair {
        provider: "openai",
        unary: "web_search_citations/streamed_and_unary.yaml#2",
        streamed: "web_search_citations/streamed_and_unary.yaml",
        // `billing` is in unary bodies only; each answer searches, cites and
        // words its reply in its own way.
        minted: &[
            "/billing",
            "/output/0/encrypted_content",
            "/output/1/action",
            "/output/2/encrypted_content",
            "/output/3/content/0/annotations/0/end_index",
            "/output/3/content/0/annotations/0/start_index",
            "/output/3/content/0/annotations/0/title",
            "/output/3/content/0/annotations/0/url",
            "/output/3/content/0/text",
            "/usage",
        ],
    },
    Pair {
        provider: "openrouter",
        unary: "prompt_caching/blocking_probe.yaml",
        streamed: "prompt_caching/streaming_probe.yaml",
        // The two answers hit the cache differently.
        minted: &["/usage"],
    },
    Pair {
        provider: "openrouter",
        unary: "prompt_caching/blocking_probe.yaml#3",
        streamed: "prompt_caching/streaming_probe.yaml#3",
        minted: &[],
    },
    Pair {
        provider: "openrouter",
        unary: "reasoning_usage_matrix/transports_agree_on_reasoning_tokens.yaml",
        streamed: "reasoning_usage_matrix/transports_agree_on_reasoning_tokens.yaml#2",
        // Each answer reasons and answers in its own words.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/choices/0/message/reasoning_details/0/summary",
            "/choices/0/message/reasoning_details/1/data",
            "/usage",
        ],
    },
    Pair {
        provider: "openrouter",
        unary: "refusal_matrix/transports_agree_on_the_refusal_text.yaml",
        streamed: "refusal_matrix/transports_agree_on_the_refusal_text.yaml#2",
        // Each answer words its refusal in its own way.
        minted: &["/choices/0/message/refusal", "/usage"],
    },
    Pair {
        provider: "openrouter",
        unary: "upstream_switch_matrix/switch_unary.yaml",
        streamed: "upstream_switch_matrix/switch_streamed.yaml",
        // Each answer reasons and answers in its own words, and OpenRouter's
        // unary body keeps the stream `index` on its calls.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/choices/0/message/reasoning_details/0/signature",
            "/choices/0/message/reasoning_details/0/text",
            "/choices/0/message/tool_calls/0/index",
            "/usage",
        ],
    },
    Pair {
        provider: "openrouter",
        unary: "upstream_switch_matrix/responses_switch_unary.yaml",
        streamed: "upstream_switch_matrix/responses_switch_streamed.yaml",
        // Each answer reasons and answers in its own words.
        minted: &[
            "/output/0/content/0/text",
            "/output/0/signature",
            "/output/1/content/0/text",
            "/output/2/call_id",
            "/usage",
        ],
    },
    Pair {
        provider: "perplexity",
        unary: "prompt_caching/blocking_probe.yaml",
        streamed: "prompt_caching/streaming_probe.yaml",
        minted: &[],
    },
    Pair {
        provider: "perplexity",
        unary: "prompt_caching/blocking_probe.yaml#3",
        streamed: "prompt_caching/streaming_probe.yaml#3",
        minted: &[],
    },
    Pair {
        provider: "venice",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Venice's unary body states an empty `content` beside its calls and its
        // parameters, which its stream does not.
        minted: &["/choices/0/message/content", "/venice_parameters"],
    },
    Pair {
        provider: "xai",
        unary: "permission_control/permission_control_prompt_example.yaml",
        streamed: "permission_control/permission_control_streaming_example.yaml",
        // Each answer reasons and counts in its own words. xAI's stream states
        // `reasoning` as nulls and its sampling parameters as 32-bit floats.
        minted: &[
            "/output/0/encrypted_content",
            "/output/0/summary/0/text",
            "/output/1/call_id",
            "/output/2/call_id",
            "/reasoning",
            "/temperature",
            "/top_p",
            "/usage",
        ],
    },
    Pair {
        provider: "xai",
        unary: "prompt_caching/blocking_probe.yaml",
        streamed: "prompt_caching/streaming_probe.yaml",
        // Each answer reasons and counts in its own words. xAI's stream states
        // `reasoning` as nulls and `top_p` as a 32-bit float.
        minted: &[
            "/output/0/encrypted_content",
            "/reasoning",
            "/top_p",
            "/usage",
        ],
    },
    Pair {
        provider: "xai",
        unary: "web_search_citations/streamed_and_unary.yaml#2",
        streamed: "web_search_citations/streamed_and_unary.yaml",
        // Each answer searches, cites and words its reply in its own way. xAI's
        // stream states `reasoning` as nulls, its sampling parameters as 32-bit
        // floats and no search context size.
        minted: &[
            "/output/0/encrypted_content",
            "/output/1/action/query",
            "/output/1/action/sources",
            "/output/2/action/query",
            "/output/2/action/sources",
            "/output/3/encrypted_content",
            "/output/4/encrypted_content",
            "/output/5/encrypted_content",
            "/output/6/content/0/annotations/0/end_index",
            "/output/6/content/0/annotations/0/start_index",
            "/output/6/content/0/text",
            "/reasoning",
            "/temperature",
            "/tools/0/search_context_size",
            "/top_p",
            "/usage",
        ],
    },
];

/// Recorded pairs whose wire is not in this crate, each checked by the named
/// test of its own crate.
const CHECKED_ELSEWHERE: &[(&str, &str, &str, &str)] = &[(
    "bedrock",
    "tool_choice/specific_add_raw_nonstreaming.yaml",
    "tool_choice/specific_add_raw_streaming.yaml",
    "rig_bedrock::streaming::tests::the_recorded_pair_agrees",
)];

/// The OpenAI-compatible dialect `provider` recorded on.
fn dialect(provider: &str) -> &'static Dialect {
    match provider {
        "openai" => &OPENAI,
        "openrouter" => &OPENROUTER,
        "perplexity" => &PERPLEXITY,
        "deepseek" => &DEEPSEEK,
        "doubleword" => &DOUBLEWORD,
        "llamacpp" => &LLAMACPP,
        "groq" => &GROQ,
        "mistral" => &MISTRAL,
        "venice" => &VENICE,
        "cohere" => &COHERE,
        "ollama" => &OLLAMA,
        "xai" => &crate::providers::xai::DIALECT,
        provider => panic!("no wire for {provider}"),
    }
}

/// The `raw` pair `pair` decodes to on the wire and route that recorded it.
async fn decoded(pair: &Pair) -> Result<(Value, Value), crate::error::ProviderError> {
    let unary = Recorded::named(pair.provider, pair.unary);
    let streamed = Recorded::named(pair.provider, pair.streamed).reply;
    let model = unary.model().unwrap_or("model").to_owned();
    let responses = unary.path.ends_with("/responses");
    let native = unary.path == "/api/chat";
    let reply = unary.reply;
    match pair.provider {
        "anthropic" => {
            let wire = AnthropicConfig::new("test-key").completion(model);
            raw_pair(wire, reply, streamed).await
        }
        "gemini" => {
            let wire = GeminiConfig::new("test-key").completion("gemini-2.5-flash-lite");
            raw_pair(wire, reply, streamed).await
        }
        "copilot" => {
            let wire = CopilotConfig::new("tid=test-token").completion(model);
            raw_pair(wire, reply, streamed).await
        }
        "ollama" if native => {
            let wire = crate::providers::ollama::OllamaConfig::new().native_completion(model);
            raw_pair(wire, reply, streamed).await
        }
        provider => {
            let config = OpenAIConfig::new("test-key").with_dialect(dialect(provider));
            if responses {
                raw_pair(config.responses(model), reply, streamed).await
            } else {
                raw_pair(config.chat(model), reply, streamed).await
            }
        }
    }
}

/// Every pair agrees whole: a streamed reply's `raw` is the unary document
/// of the same turn.
#[tokio::test]
async fn every_recorded_pair_agrees() {
    let mut wrong = Vec::new();
    for pair in PAIRS {
        let (unary, streamed) = match decoded(pair).await {
            Ok(pair) => pair,
            Err(error) => {
                wrong.push(format!("{}/{}: {error}", pair.provider, pair.streamed));
                continue;
            }
        };
        assert!(
            !unary.is_null(),
            "{}/{}: the unary reply has a document",
            pair.provider,
            pair.unary
        );
        let (unary, streamed) = (
            comparable(&unary, pair.minted),
            comparable(&streamed, pair.minted),
        );
        if unary != streamed {
            wrong.push(format!(
                "{}/{}: the streamed raw is not the unary document\n  unary:    {unary}\n  streamed: {streamed}",
                pair.provider, pair.streamed,
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Interactions has no recorded pair: the recorded unary interaction
/// against the stream the same turn sends, hand-built in the event grammar
/// `gemini/corpus_delta/interactions_baseline.yaml` records, whose
/// `interaction.completed` is the resource without its steps.
#[tokio::test]
async fn interactions_rebuilds_the_recorded_unary_interaction() {
    let unary = Recorded::named(
        "gemini",
        "interactions_raw_capture_matrix/raw_roundtrips_interaction.yaml",
    )
    .reply;
    let mut interaction: Value =
        serde_json::from_slice(&unary).unwrap_or_else(|error| panic!("{error}"));
    if let Some(interaction) = interaction.as_object_mut() {
        interaction.shift_remove("steps");
    }
    let events = [
        json!({"event_type": "interaction.created", "interaction": {
            "id": interaction["id"], "model": interaction["model"],
            "object": "interaction", "status": "in_progress",
        }}),
        json!({"event_type": "step.start", "index": 0, "step": {"type": "thought"}}),
        json!({"event_type": "step.delta", "index": 0,
            "delta": {"type": "thought_signature", "signature": "streamed"}}),
        json!({"event_type": "step.stop", "index": 0}),
        json!({"event_type": "step.start", "index": 1, "step": {"type": "model_output"}}),
        json!({"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "capt"}}),
        json!({"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "ured"}}),
        json!({"event_type": "step.stop", "index": 1}),
        json!({"event_type": "interaction.completed", "interaction": interaction}),
    ];
    let streamed: String = events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["event_type"].as_str().unwrap_or_default()
            )
        })
        .chain(["event: done\ndata: [DONE]\n\n".to_owned()])
        .collect();
    let wire = crate::providers::gemini::interactions_api::Interactions::new(
        GeminiConfig::new("test-key"),
        "gemini-3-flash-preview",
    );
    let (unary, streamed) = raw_pair(wire, unary, streamed)
        .await
        .unwrap_or_else(|error| panic!("the interaction decodes: {error}"));
    let minted = ["/steps/0/signature"];
    assert_eq!(comparable(&unary, &minted), comparable(&streamed, &minted));
    assert_eq!(streamed["steps"][0]["signature"], "streamed");
}

/// Every turn the corpus records both unary and streamed (the same request
/// body, streaming switches aside, sent to the same endpoint) has a row here
/// or is checked by another crate's test, so such a pair cannot go
/// unchecked. Pairs whose two requests differ, such as two prompts asking
/// for different words, are rows by hand.
#[test]
fn every_recorded_pair_has_a_row() {
    type Turns = BTreeMap<(String, String, String), (BTreeSet<String>, BTreeSet<String>)>;
    let mut turns = Turns::new();
    let providers = std::fs::read_dir(corpus()).unwrap_or_else(|error| panic!("{error}"));
    for provider in providers.flatten() {
        let name = provider.file_name().to_string_lossy().into_owned();
        let mut files = vec![provider.path()];
        while let Some(path) = files.pop() {
            if path.is_dir() {
                let entries = std::fs::read_dir(&path).unwrap_or_else(|error| panic!("{error}"));
                files.extend(entries.flatten().map(|entry| entry.path()));
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "yaml") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let relative = path
                .strip_prefix(provider.path())
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            for (n, recorded) in interactions(&text).map(Recorded::parse).enumerate() {
                let Some((endpoint, request)) = recorded.as_ref().and_then(Recorded::turn) else {
                    continue;
                };
                let Some(recorded) = recorded.filter(|recorded| recorded.status == Some(200))
                else {
                    continue;
                };
                let (unary, streamed) = turns.entry((name.clone(), endpoint, request)).or_default();
                let named = format!("{relative}#{}", n + 1);
                if recorded.streamed {
                    streamed.insert(named);
                } else {
                    unary.insert(named);
                }
            }
        }
    }
    let numbered = |relative: &str| {
        let (file, n) = interaction(relative);
        format!("{file}#{n}")
    };
    let rows: Vec<(&str, String, String)> = PAIRS
        .iter()
        .map(|pair| (pair.provider, numbered(pair.unary), numbered(pair.streamed)))
        .chain(
            CHECKED_ELSEWHERE
                .iter()
                .map(|(provider, unary, streamed, _)| {
                    (*provider, numbered(unary), numbered(streamed))
                }),
        )
        .collect();
    let mut missing = Vec::new();
    for ((provider, _, _), (unary, streamed)) in &turns {
        let (Some(first_unary), Some(first_streamed)) = (unary.first(), streamed.first()) else {
            continue;
        };
        let covered = rows.iter().any(|(row, row_unary, row_streamed)| {
            row == provider && unary.contains(row_unary) && streamed.contains(row_streamed)
        });
        if !covered {
            missing.push(format!("{provider}: {first_unary} <-> {first_streamed}"));
        }
    }
    assert!(
        missing.is_empty(),
        "recorded pairs with no parity row:\n{}",
        missing.join("\n")
    );
}

#[test]
fn comparable_drops_empties_minted_keys_and_named_pointers() {
    let document = json!({
        "id": "a",
        "created": 1,
        "object": "chat.completion",
        "choices": [{"message": {"content": "pong", "annotations": [], "refusal": null}}],
        "usage": {"total_tokens": 3},
        "extra": {}
    });
    assert_eq!(
        comparable(&document, &["/usage"]),
        json!({"object": "chat.completion", "choices": [{"message": {"content": "pong"}}]})
    );
}
