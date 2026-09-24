//! Every failure to build a provider request reports as a request failure,
//! whichever wire built it.

use serde_json::json;

use super::*;
use crate::completion::{CompletionRequest, CompletionRequestBuilder, ToolDefinition};
use crate::driver::{HasCompletion, HasEmbedding, HasModelListing, HasRerank, HasVerify};
use crate::message::ToolChoice;
use crate::providers::anthropic::Anthropic;
use crate::providers::cohere::Cohere;
use crate::providers::gemini::Gemini;
use crate::providers::ollama::Ollama;
use crate::providers::openai::OpenAI;
use crate::providers::openai::embedding::EncodingFormat;
use crate::providers::openai::wire::{GROQ, LLAMACPP, MISTRAL, OPENAI, PERPLEXITY, TOGETHER};
use crate::providers::voyageai::VoyageAi;
use crate::wire::{Mode, Wire};

/// A base URL `http` refuses, so building the request fails.
const BAD: &str = "http://bad host";

fn request() -> CompletionRequest {
    CompletionRequestBuilder::unbound("hi")
        .max_tokens(16)
        .build()
}

fn failure<T>(result: Result<T, EncodeError>) -> ProviderError {
    match result {
        Ok(_) => panic!("the request must not build"),
        Err(error) => error.into(),
    }
}

/// Asserts a request-building classification. The final exhaustive match
/// fails to compile when `ProviderError` gains a variant, so a new variant
/// has to be placed on one side of the rule.
fn assert_request_building(case: &str, error: &ProviderError) {
    assert_eq!(
        error.boundary(),
        AdapterErrorBoundary::Request,
        "{case}: {error}"
    );
    assert!(!error.is_retryable(), "{case}: {error}");
    assert_eq!(error.report().kind, ErrorKind::Request, "{case}: {error}");
    match error {
        ProviderError::Request(_) => {}
        ProviderError::Http(_)
        | ProviderError::Url(_)
        | ProviderError::Json(_)
        | ProviderError::Response(_)
        | ProviderError::Provider(_)
        | ProviderError::ProviderResponse(_)
        | ProviderError::InvalidAuthentication(_)
        | ProviderError::CacheExpired { .. }
        | ProviderError::MismatchedDimensions { .. } => {
            panic!("{case}: an encode failure must not classify as {error:?}")
        }
    }
}

#[test]
fn every_encode_error_constructor_classifies_as_request_building() {
    let json = serde_json::from_str::<serde_json::Value>("{").expect_err("malformed");
    let http = http::Request::builder()
        .uri("http://bad host")
        .body(())
        .expect_err("bad uri");
    let boxed: BoxError = Box::new(std::io::Error::other("io"));
    let message = crate::message::MessageError::ConversionError("nope".into());
    let cases: Vec<(&str, EncodeError)> = vec![
        ("request", EncodeError::request("no endpoint")),
        ("from json", json.into()),
        ("from http", http.into()),
        ("from boxed", boxed.into()),
        ("from message", message.into()),
    ];
    for (case, error) in cases {
        let error = ProviderError::from(error);
        assert_request_building(case, &error);
    }
}

/// The request builder's own failure, which each wire used to classify as it
/// liked (`response`, `http` or `request`), and the requests a wire refuses
/// before sending (`provider`, `response`).
#[test]
fn provider_encode_failures_classify_as_request_building() {
    let openai = OpenAI::with_key(&OPENAI, "k").with_base_url(BAD);
    let perplexity = OpenAI::with_key(&PERPLEXITY, "k");
    let anthropic = Anthropic::new("k").with_base_url(BAD);
    let gemini = Gemini::new("k").with_base_url(BAD);
    let specific_tool = CompletionRequestBuilder::unbound("hi")
        .tool(ToolDefinition {
            name: "f".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
        })
        .tool_choice(ToolChoice::Specific {
            function_names: vec!["f".into()],
        })
        .build();
    let unflattenable_schema = CompletionRequestBuilder::unbound("hi")
        .tool(ToolDefinition {
            name: "f".into(),
            description: "d".into(),
            parameters: json!({
                "type": "object",
                "$defs": 5,
                "properties": {"a": {"$ref": "#/$defs/x"}},
            }),
        })
        .build();
    let cases = [
        (
            "openai chat",
            failure(
                OpenAI::with_key(&GROQ, "k")
                    .with_base_url(BAD)
                    .completion("m")
                    .encode(request(), Mode::Unary),
            ),
            "RequestError: invalid uri character",
        ),
        (
            "openai responses",
            failure(openai.completion("gpt-5").encode(request(), Mode::Unary)),
            "RequestError: invalid uri character",
        ),
        (
            "openai embeddings",
            failure(
                openai
                    .embedding("text-embedding-3-small", None)
                    .encode(vec!["a".into()], Mode::Unary),
            ),
            "RequestError: invalid uri character",
        ),
        (
            "openai embeddings, base64",
            failure(
                OpenAI::with_key(&OPENAI, "k")
                    .embedding("text-embedding-3-small", None)
                    .with_encoding_format(EncodingFormat::Base64)
                    .encode(vec!["a".into()], Mode::Unary),
            ),
            "RequestError: Rig cannot decode openai embedding responses encoded as `base64`",
        ),
        (
            "openai-compatible embeddings, unsupported encoding format",
            failure(
                OpenAI::with_key(&TOGETHER, "k")
                    .embedding("m", None)
                    .with_encoding_format(EncodingFormat::Float)
                    .encode(vec!["a".into()], Mode::Unary),
            ),
            "RequestError: together embeddings do not support the `encoding_format` parameter",
        ),
        (
            "openai-compatible embeddings, unsupported user",
            failure(
                OpenAI::with_key(&MISTRAL, "k")
                    .embedding("mistral-embed", None)
                    .with_user("u")
                    .encode(vec!["a".into()], Mode::Unary),
            ),
            "RequestError: mistral embeddings do not support the `user` parameter",
        ),
        (
            "openai rerank without the endpoint",
            failure(perplexity.rerank("m").encode(
                crate::operation::RerankRequest {
                    query: "q".into(),
                    documents: vec!["d".into()],
                },
                Mode::Unary,
            )),
            "RequestError: perplexity offers no reranking endpoint",
        ),
        (
            "openai verify without the endpoint",
            failure(perplexity.verify().encode((), Mode::Unary)),
            "RequestError: perplexity offers no endpoint that checks a credential without \
             consuming tokens",
        ),
        (
            "llama.cpp specific tool choice",
            failure(
                OpenAI::with_key(&LLAMACPP, "k")
                    .completion("m")
                    .encode(specific_tool, Mode::Unary),
            ),
            "RequestError: llama.cpp cannot force a specific tool: `llama-server` accepts only \
             `auto`, `none` or `required` for tool_choice and silently treats anything else as \
             `auto`, so requesting `f` would return whichever tool the model picked. Use \
             `ToolChoice::Required` to force a call, or advertise only `f` in `tools`.",
        ),
        (
            "anthropic messages",
            failure(
                anthropic
                    .completion("claude-sonnet-4-5")
                    .encode(request(), Mode::Unary),
            ),
            "RequestError: invalid uri character",
        ),
        (
            "anthropic verify",
            failure(HasVerify::verify(&anthropic).encode((), Mode::Unary)),
            "RequestError: invalid uri character",
        ),
        (
            "anthropic models",
            failure(anthropic.model_listing().encode((), Mode::Unary)),
            "RequestError: invalid uri character",
        ),
        (
            "gemini generateContent",
            failure(
                gemini
                    .completion("gemini-2.5-flash")
                    .encode(request(), Mode::Unary),
            ),
            "RequestError: invalid uri character",
        ),
        (
            "gemini verify",
            failure(gemini.verify().encode((), Mode::Unary)),
            "RequestError: invalid uri character",
        ),
        (
            "gemini tool schema",
            failure(
                Gemini::new("k")
                    .completion("gemini-2.5-flash")
                    .encode(unflattenable_schema, Mode::Unary),
            ),
            "RequestError: Tool 'f' could not be converted to a schema: $defs must be an object",
        ),
        (
            "cohere embed",
            failure(
                Cohere::new("k")
                    .with_base_url(BAD)
                    .embedding("embed-english-v3.0", None)
                    .encode(vec!["a".into()], Mode::Unary),
            ),
            "RequestError: invalid uri character",
        ),
        (
            "voyage embed",
            failure(
                VoyageAi::new("k")
                    .with_base_url(BAD)
                    .embedding("voyage-3", None)
                    .encode(vec!["a".into()], Mode::Unary),
            ),
            "RequestError: invalid uri character",
        ),
        (
            "ollama chat",
            failure(
                Ollama::new()
                    .with_base_url(BAD)
                    .completion("llama3")
                    .encode(request(), Mode::Unary),
            ),
            "RequestError: invalid uri character",
        ),
    ];
    for (case, error, message) in cases {
        assert_request_building(case, &error);
        assert_eq!(error.to_string(), message, "{case}");
    }
}
