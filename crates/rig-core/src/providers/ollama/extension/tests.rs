use std::collections::BTreeSet;

use serde_json::json;

use super::*;
use crate::completion::{
    CompletionRequest, Effort, GenerationOptions, OnUnsupported, ProviderOptions, Reasoning,
    ToolDefinition,
};
use crate::message::ToolName;
use crate::operation::Completion;
use crate::providers::ollama::OllamaConfig;
use crate::test_utils::json_body;
use crate::wire::{Encoded, Mode, Operation, Wire};

const MODEL: &str = "qwen3:8b";

fn every_field() -> OllamaOptions {
    OllamaOptions::default()
        .keep_alive(KeepAlive::duration("5m"))
        .num_ctx(4096)
        .num_keep(4)
        .top_k(20)
        .min_p(0.05)
        .repeat_penalty(1.1)
        .repeat_last_n(64)
        .num_gpu(10)
        .num_thread(8)
        .logprobs(true)
        .top_logprobs(2)
}

fn with(options: &OllamaOptions) -> CompletionRequest {
    CompletionRequest::new("q").provider_options(
        ProviderOptions::new()
            .with::<OllamaExt>(options)
            .expect("the options serialize"),
    )
}

/// The body `wire` sends for `request`, prepared as the driver prepares it.
fn sent<W: Wire<Op = Completion, Payload = Encoded>>(
    wire: &W,
    request: CompletionRequest,
) -> Value {
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    json_body(
        &wire
            .encode(request, Mode::Unary)
            .expect("the request encodes")
            .request,
    )
}

/// The `/api/chat` body of `request`.
fn native(request: CompletionRequest) -> Value {
    sent(&OllamaConfig::new().native_completion(MODEL), request)
}

/// The `/v1/chat/completions` body of `request`.
fn v1(request: CompletionRequest) -> Value {
    sent(&OllamaConfig::new().completion(MODEL), request)
}

#[test]
fn keep_alive_goes_top_level_on_both_routes() {
    let duration = with(&OllamaOptions::default().keep_alive(KeepAlive::duration("5m")));
    assert_eq!(native(duration.clone())["keep_alive"], json!("5m"));
    assert_eq!(v1(duration)["keep_alive"], json!("5m"));
    let seconds = with(&OllamaOptions::default().keep_alive(KeepAlive::seconds(-1)));
    assert_eq!(native(seconds)["keep_alive"], json!(-1));
}

#[test]
fn num_ctx_goes_in_options() {
    let body = native(with(&OllamaOptions::default().num_ctx(4096)));
    assert_eq!(body["options"], json!({"num_ctx": 4096}));
}

#[test]
fn num_keep_goes_in_options() {
    let body = native(with(&OllamaOptions::default().num_keep(4)));
    assert_eq!(body["options"], json!({"num_keep": 4}));
}

#[test]
fn top_k_goes_in_options() {
    let body = native(with(&OllamaOptions::default().top_k(20)));
    assert_eq!(body["options"], json!({"top_k": 20}));
}

#[test]
fn min_p_goes_in_options() {
    let body = native(with(&OllamaOptions::default().min_p(0.05)));
    assert_eq!(body["options"], json!({"min_p": 0.05}));
}

#[test]
fn repeat_penalty_goes_in_options() {
    let body = native(with(&OllamaOptions::default().repeat_penalty(1.1)));
    assert_eq!(body["options"], json!({"repeat_penalty": 1.1}));
}

#[test]
fn repeat_last_n_goes_in_options() {
    let body = native(with(&OllamaOptions::default().repeat_last_n(-1)));
    assert_eq!(body["options"], json!({"repeat_last_n": -1}));
}

#[test]
fn num_gpu_goes_in_options() {
    let body = native(with(&OllamaOptions::default().num_gpu(10)));
    assert_eq!(body["options"], json!({"num_gpu": 10}));
}

#[test]
fn num_thread_goes_in_options() {
    let body = native(with(&OllamaOptions::default().num_thread(8)));
    assert_eq!(body["options"], json!({"num_thread": 8}));
}

#[test]
fn logprobs_goes_top_level() {
    let body = native(with(&OllamaOptions::default().logprobs(true)));
    assert_eq!(body["logprobs"], json!(true));
}

#[test]
fn top_logprobs_goes_top_level() {
    let body = native(with(&OllamaOptions::default().top_logprobs(2)));
    assert_eq!(body["top_logprobs"], json!(2));
}

/// The model parameters join the ones the request and its generation
/// options put in `options`.
#[test]
fn model_options_join_the_request_options() {
    let mut request = with(&OllamaOptions::default().num_ctx(4096).top_k(20))
        .options(GenerationOptions::default().seed(7).top_p(0.9));
    request.temperature = Some(0.0);
    request.max_tokens = Some(64);
    assert_eq!(
        native(request)["options"],
        json!({"temperature": 0.0, "num_predict": 64, "seed": 7, "top_p": 0.9,
               "num_ctx": 4096, "top_k": 20})
    );
}

/// The native section is skipped on `/v1`, with no error; the shared one is
/// sent.
#[test]
fn the_native_section_is_skipped_on_v1() {
    let body = v1(with(&every_field()));
    for key in ["options", "num_ctx", "logprobs", "top_logprobs"] {
        assert!(body.get(key).is_none(), "{key}: {body}");
    }
    assert_eq!(body["keep_alive"], json!("5m"));
}

#[test]
fn raw_beats_typed() {
    let mut request = with(&OllamaOptions::default().num_ctx(4096).logprobs(true));
    request.additional_params = Some(json!({"num_ctx": 2048, "logprobs": false}));
    let body = native(request);
    assert_eq!(body["options"], json!({"num_ctx": 2048}));
    assert_eq!(body["logprobs"], json!(false));
}

/// Every JSON pointer to a leaf of `value`, an empty object being a leaf.
fn leaves(value: &Value, at: &str, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(fields) if !fields.is_empty() => {
            for (key, field) in fields {
                leaves(field, &format!("{at}/{key}"), out);
            }
        }
        _ => {
            out.insert(at.to_owned());
        }
    }
}

/// No field writes a leaf that a mapped option or the request's own fields
/// write, or a leaf above or below one, on either route.
#[test]
fn no_field_writes_a_reserved_leaf() {
    let options = ProviderOptions::new()
        .with::<OllamaExt>(&every_field())
        .expect("the options serialize");
    let mut provider = BTreeSet::new();
    for section in options.get::<OllamaExt>().expect("an entry").values() {
        leaves(section, "", &mut provider);
    }
    for reasoning in [Reasoning::Off, Reasoning::Effort(Effort::High)] {
        let mut request = CompletionRequest::new("q").options(
            GenerationOptions::default()
                .reasoning(reasoning)
                .top_p(0.5)
                .seed(1)
                .stop(["END"])
                .on_unsupported(OnUnsupported::Ignore),
        );
        request.temperature = Some(0.5);
        request.max_tokens = Some(64);
        request.tools = vec![ToolDefinition::new(
            ToolName::new("lookup").expect("a tool name"),
            "a tool",
            json!({"type": "object"}),
        )];
        request.output_schema = Some(schemars::json_schema!({"type": "object"}));
        for body in [native(request.clone()), v1(request)] {
            let mut owned = BTreeSet::new();
            leaves(&body, "", &mut owned);
            for leaf in &provider {
                for reserved in &owned {
                    assert!(
                        !(leaf == reserved
                            || leaf.starts_with(&format!("{reserved}/"))
                            || reserved.starts_with(&format!("{leaf}/"))),
                        "{leaf} meets {reserved}"
                    );
                }
            }
        }
    }
}

/// Not a cassette test: no Ollama recording carries log probabilities, so
/// this final record is built here. The recorded fields are read in the
/// `ollama` cassette target.
#[test]
fn extras_read_the_logprobs_of_a_native_reply() {
    let raw = json!({
        "model": "qwen3:4b",
        "done": true,
        "done_reason": "stop",
        "eval_count": 1,
        "logprobs": [{
            "token": "Hi",
            "logprob": -0.25,
            "bytes": [72, 105],
            "top_logprobs": [{"token": "Hello", "logprob": -1.5, "bytes": [72]}]
        }]
    });
    let extras =
        OllamaExtras::from_reply(&Api::from_static("ollama.chat"), &raw).expect("the reply reads");
    assert_eq!(extras.done_reason.as_deref(), Some("stop"));
    assert_eq!(extras.eval_count, Some(1));
    assert_eq!(
        extras.logprobs,
        Some(vec![Logprob {
            token: Some("Hi".to_owned()),
            logprob: Some(-0.25),
            bytes: Some(vec![72, 105]),
            top_logprobs: Some(vec![Logprob {
                token: Some("Hello".to_owned()),
                logprob: Some(-1.5),
                bytes: Some(vec![72]),
                top_logprobs: None,
            }]),
        }])
    );
    assert!(
        OllamaExtras::from_reply(
            &Api::from_static("ollama.chat"),
            &json!({"eval_count": "x"})
        )
        .is_err()
    );
}
