//! Every bank reply through its provider's real decoder. A runtime scenario
//! runs once, on one wire, so the decode of each provider's reply shapes is
//! pinned here instead: each reply is decoded by the model of its provider
//! and encoder, and what the decoder made of it must agree with what the
//! bank read off the bytes: the tools it calls, how the turn ended, and a
//! status error for a rejection. Bedrock answers through the AWS SDK, not
//! an HTTP client the bank can stand in for, and has no decoder here.
//!
//! Every reply of a provider `sweeps!` names must decode. The coverage gate
//! reads that list and counts those replies as recordings of their reply
//! shapes, so a cassette holding only a decoded reply shape can go.

use rig::completion::{AssistantContent, FinishReason};
use rig::providers::anthropic::wire::AnthropicConfig;
use rig::providers::cohere::CohereConfig;
use rig::providers::copilot::CopilotConfig;
use rig::providers::gemini::GeminiConfig;
use rig::providers::ollama::OllamaConfig;
use rig::providers::openai::wire::{
    DEEPSEEK, DOUBLEWORD, Dialect, GROQ, LLAMACPP, MISTRAL, OPENROUTER, OpenAIConfig, PERPLEXITY,
    VENICE,
};
use rig_test_support::bank::{self, Ending, Entry};
use rig_test_support::cassette_models::{
    AnthropicModels, CohereModels, CopilotModels, GeminiModels, OllamaModels,
};

const KEY: &str = "bank";
const BASE_URL: &str = "http://bank.invalid";

/// What the decoder made of one reply.
#[derive(Debug)]
pub(crate) struct Decoded {
    pub(crate) finish: Option<FinishReason>,
    pub(crate) text: String,
    /// The distinct names of the tools the turn calls, sorted.
    pub(crate) calls: Vec<String>,
    /// The decoder's error, when it refused the reply.
    pub(crate) error: Option<String>,
}

async fn call<W, T>(model: rig::driver::Model<W, T>, streamed: bool) -> Decoded
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let response = if streamed {
        match model.stream("Reply.") {
            Ok(stream) => stream.finish().await,
            Err(error) => Err(error),
        }
    } else {
        model.call("Reply.").await
    };
    match response {
        Ok(response) => {
            let mut calls: Vec<String> = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(call) => {
                        Some(call.function.name.as_str().to_owned())
                    }
                    _ => None,
                })
                .collect();
            calls.sort();
            calls.dedup();
            Decoded {
                finish: response.finish_reason(),
                text: response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect(),
                calls,
                error: None,
            }
        }
        Err(error) => Decoded {
            finish: None,
            text: String::new(),
            calls: Vec::new(),
            error: Some(error.to_string()),
        },
    }
}

pub(crate) fn openai_family(provider: &str) -> Option<OpenAIConfig> {
    let dialect: Option<&Dialect> = match provider {
        "deepseek" => Some(&DEEPSEEK),
        "doubleword" => Some(&DOUBLEWORD),
        "groq" => Some(&GROQ),
        "llamacpp" => Some(&LLAMACPP),
        "mistral" => Some(&MISTRAL),
        "openrouter" => Some(&OPENROUTER),
        "perplexity" => Some(&PERPLEXITY),
        "venice" => Some(&VENICE),
        "xai" => Some(&rig::providers::xai::DIALECT),
        "chatgpt" => Some(&rig::providers::chatgpt::DIALECT),
        "openai" | "mistralrs" => None,
        _ => return None,
    };
    let config = match dialect {
        Some(dialect) => OpenAIConfig::with_key(dialect, KEY),
        None => OpenAIConfig::new(KEY),
    };
    Some(config.with_base_url(BASE_URL))
}

/// Bind `$model` to `$entry`'s provider and encoder model over `$http` and
/// evaluate `$body` with it: `Some` of the body, or `None` when the bank has
/// no model for that provider and encoder. A macro, because each arm's model
/// is of its own type.
macro_rules! with_model {
    ($entry:expr, $http:expr, |$model:ident| $body:expr) => {{
        let entry: &rig_test_support::bank::Entry = $entry;
        let http: rig::http_client::DynHttpClient = $http;
        let encoder = entry.encoder.as_str();
        match entry.provider.as_str() {
            "anthropic" if encoder.ends_with("/messages") => {
                let $model = $crate::decode::anthropic_model(http);
                Some($body)
            }
            "gemini" if encoder.ends_with("/interactions") => {
                let $model =
                    $crate::decode::gemini_model(http).interactions($crate::decode::GEMINI);
                Some($body)
            }
            "gemini" => {
                let $model = $crate::decode::gemini_model(http).completion($crate::decode::GEMINI);
                Some($body)
            }
            "cohere" => {
                let $model = $crate::decode::cohere_model(http);
                Some($body)
            }
            "ollama" => {
                let $model = $crate::decode::ollama_model(http);
                Some($body)
            }
            "copilot" => {
                let $model = $crate::decode::copilot_model(http, encoder);
                Some($body)
            }
            provider => match $crate::decode::openai_family(provider) {
                Some(config) if encoder.ends_with("/chat/completions") => {
                    let $model = rig_test_support::cassette_models::OpenAiModels::new(config, http)
                        .chat("model");
                    Some($body)
                }
                Some(config) if encoder.ends_with("/responses") => {
                    let $model = rig_test_support::cassette_models::OpenAiModels::new(config, http)
                        .responses("model");
                    Some($body)
                }
                _ => None,
            },
        }
    }};
}
pub(crate) use with_model;

pub(crate) const GEMINI: &str = rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

pub(crate) fn anthropic_model(
    http: rig::http_client::DynHttpClient,
) -> rig::Model<rig::providers::anthropic::wire::Messages> {
    AnthropicModels::new(AnthropicConfig::new(KEY).with_base_url(BASE_URL), http)
        .completion("claude-haiku-4-5-20251001")
}

pub(crate) fn gemini_model(http: rig::http_client::DynHttpClient) -> GeminiModels {
    GeminiModels::new(GeminiConfig::new(KEY).with_base_url(BASE_URL), http)
}

pub(crate) fn cohere_model(
    http: rig::http_client::DynHttpClient,
) -> rig::Model<rig::providers::openai::wire::Chat> {
    CohereModels::new(CohereConfig::new(KEY).with_base_url(BASE_URL), http)
        .completion("command-a-03-2025")
}

pub(crate) fn ollama_model(
    http: rig::http_client::DynHttpClient,
) -> rig::Model<rig::providers::openai::wire::Chat> {
    OllamaModels::new(OllamaConfig::new().with_base_url(BASE_URL), http).completion("qwen3:4b")
}

pub(crate) fn copilot_model(
    http: rig::http_client::DynHttpClient,
    encoder: &str,
) -> rig::Model<rig::providers::copilot::wire::CopilotWire> {
    let model = if encoder.ends_with("/responses") {
        "gpt-5.3-codex"
    } else {
        "gpt-4o"
    };
    CopilotModels::new(CopilotConfig::new(KEY).with_base_url(BASE_URL), http).completion(model)
}

/// `entry` through its provider's decoder; `None` when the bank has no
/// decoder for its provider and encoder.
pub(crate) async fn decode(entry: &Entry) -> Option<Decoded> {
    let streamed = entry.streamed();
    with_model!(entry, bank::client(std::slice::from_ref(entry)), |model| {
        call(model, streamed).await
    })
}

/// The finish reason a decoded turn of `ending` must report.
fn expected(ending: Ending, calls: &[String]) -> Option<FinishReason> {
    match ending {
        Ending::Stop if calls.is_empty() => Some(FinishReason::Stop),
        Ending::Stop | Ending::Tool => Some(FinishReason::ToolCalls),
        Ending::Length => Some(FinishReason::Length),
        Ending::Filtered => Some(FinishReason::ContentFilter),
        Ending::Other => None,
    }
}

/// One provider's bank, every reply decoded and compared with what the bank
/// read off its bytes. Returns how many replies were decoded.
async fn sweep(provider: &str) -> usize {
    let mut decoded_count = 0;
    let mut failures = Vec::new();
    for entry in bank::entries(provider).iter() {
        let Some(decoded) = decode(entry).await else {
            failures.push(format!(
                "{}: no decoder for `{}`",
                entry.source, entry.encoder
            ));
            continue;
        };
        decoded_count += 1;
        let status = entry.then.status;
        if !(200..300).contains(&status) {
            if decoded.error.is_none() {
                failures.push(format!(
                    "{}: status {status} decoded as a reply",
                    entry.source
                ));
            }
            continue;
        }
        if let Some(error) = &decoded.error {
            failures.push(format!("{}: {error}", entry.source));
            continue;
        }
        if decoded.calls != entry.calls {
            failures.push(format!(
                "{}: calls {:?}, the bank read {:?}",
                entry.source, decoded.calls, entry.calls
            ));
        }
        if let Some(finish) = expected(entry.ending(), &entry.calls)
            && decoded.finish.as_ref() != Some(&finish)
        {
            failures.push(format!(
                "{}: finish {:?}, the bank read {:?}",
                entry.source, decoded.finish, entry.ends
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{provider}: {} of {decoded_count} replies disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
    decoded_count
}

macro_rules! sweeps {
    ($($provider:ident),* $(,)?) => {
        $(
            #[tokio::test]
            async fn $provider() {
                assert!(sweep(stringify!($provider)).await > 0, "the bank decodes some reply");
            }
        )*
    };
}

sweeps!(
    anthropic, chatgpt, cohere, copilot, deepseek, doubleword, gemini, groq, llamacpp, mistral,
    mistralrs, ollama, openai, openrouter, perplexity, venice, xai,
);
