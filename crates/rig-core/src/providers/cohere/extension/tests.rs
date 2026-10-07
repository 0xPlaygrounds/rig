use std::collections::{BTreeSet, HashMap};

use serde_json::json;

use super::*;
use crate::completion::{
    CompletionRequest, Document, GenerationOptions, OnUnsupported, ProviderOptions, Reasoning,
    SHARED, ToolDefinition,
};
use crate::message::{ToolChoice, ToolName};
use crate::operation::Completion;
use crate::providers::cohere::{ChatRoute, CohereChat, CohereConfig};
use crate::test_utils::json_body;
use crate::wire::{Encoded, Mode, Operation, Wire};

const MODEL: &str = "command-a-03-2025";

fn chat(route: ChatRoute) -> CohereChat {
    CohereConfig::new("key")
        .with_base_url("http://127.0.0.1:9")
        .completion(MODEL)
        .with_route(route)
}

fn every_field() -> CohereOptions {
    CohereOptions::default()
        .frequency_penalty(0.25)
        .presence_penalty(0.5)
        .citation_mode(CitationMode::Accurate)
        .safety_mode(SafetyMode::Strict)
        .priority(3)
        .top_k(40)
        .logprobs(true)
}

fn with(options: &CohereOptions) -> CompletionRequest {
    CompletionRequest::new("q").provider_options(
        ProviderOptions::new()
            .with::<CohereExt>(options)
            .expect("the options serialize"),
    )
}

/// The path and body `wire` sends for `request`, prepared as the driver
/// prepares it.
fn sent<W: Wire<Op = Completion, Payload = Encoded>>(
    wire: &W,
    request: CompletionRequest,
) -> (String, Value) {
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    (
        encoded.request.uri().path().to_owned(),
        json_body(&encoded.request),
    )
}

fn native(options: &CohereOptions) -> Value {
    let (path, body) = sent(&chat(ChatRoute::Native), with(options));
    assert_eq!(path, "/v2/chat");
    body
}

fn compatibility(options: &CohereOptions) -> Value {
    let (path, body) = sent(&chat(ChatRoute::Compatibility), with(options));
    assert_eq!(path, "/compatibility/v1/chat/completions");
    body
}

fn documents() -> Vec<Document> {
    vec![Document {
        id: "harbor-1".to_owned(),
        text: "Beacon amber-73 is at Dock Seven.".to_owned(),
        additional_props: HashMap::new(),
    }]
}

#[test]
fn frequency_penalty_goes_to_both_routes() {
    let options = CohereOptions::default().frequency_penalty(0.25);
    assert_eq!(native(&options)["frequency_penalty"], json!(0.25));
    assert_eq!(compatibility(&options)["frequency_penalty"], json!(0.25));
}

#[test]
fn presence_penalty_goes_to_both_routes() {
    let options = CohereOptions::default().presence_penalty(0.5);
    assert_eq!(native(&options)["presence_penalty"], json!(0.5));
    assert_eq!(compatibility(&options)["presence_penalty"], json!(0.5));
}

#[test]
fn citation_mode_goes_to_citation_options() {
    let options = CohereOptions::default().citation_mode(CitationMode::Fast);
    assert_eq!(
        native(&options)["citation_options"],
        json!({"mode": "FAST"})
    );
}

#[test]
fn safety_mode_is_sent() {
    let options = CohereOptions::default().safety_mode(SafetyMode::Strict);
    assert_eq!(native(&options)["safety_mode"], json!("STRICT"));
}

#[test]
fn priority_is_sent() {
    let options = CohereOptions::default().priority(3);
    assert_eq!(native(&options)["priority"], json!(3));
}

#[test]
fn top_k_is_sent_as_k() {
    let options = CohereOptions::default().top_k(40);
    let body = native(&options);
    assert_eq!(body["k"], json!(40));
    assert!(body.get("top_k").is_none(), "{body}");
}

#[test]
fn logprobs_is_sent() {
    let options = CohereOptions::default().logprobs(true);
    assert_eq!(native(&options)["logprobs"], json!(true));
}

/// The native section is skipped on a request the Compatibility API takes,
/// with no error, and read on one `Auto` sends natively for its documents.
#[test]
fn the_native_section_goes_only_to_the_native_route() {
    let options = every_field();
    let body = compatibility(&options);
    for key in [
        "citation_options",
        "safety_mode",
        "priority",
        "k",
        "logprobs",
    ] {
        assert!(body.get(key).is_none(), "{key}: {body}");
    }
    assert_eq!(body["presence_penalty"], json!(0.5));

    let (path, body) = sent(&chat(ChatRoute::Auto), with(&options));
    assert_eq!(path, "/compatibility/v1/chat/completions");
    assert!(body.get("k").is_none(), "{body}");

    let mut grounded = with(&options);
    grounded.documents = documents();
    let (path, body) = sent(&chat(ChatRoute::Auto), grounded);
    assert_eq!(path, "/v2/chat");
    assert_eq!(body["k"], json!(40));
}

#[test]
fn raw_beats_typed() {
    let mut request = with(&CohereOptions::default().top_k(7).frequency_penalty(0.1));
    request.additional_params = Some(json!({"k": 3, "frequency_penalty": 0.9}));
    let (_, body) = sent(&chat(ChatRoute::Native), request);
    assert_eq!(body["k"], json!(3));
    assert_eq!(body["frequency_penalty"], json!(0.9));
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
        .with::<CohereExt>(&every_field())
        .expect("the options serialize");
    let mut provider = BTreeSet::new();
    for section in options.get::<CohereExt>().expect("an entry").values() {
        leaves(section, "", &mut provider);
    }
    assert!(
        options
            .get::<CohereExt>()
            .is_some_and(|entry| entry.contains_key(SHARED))
    );
    for route in [ChatRoute::Native, ChatRoute::Compatibility] {
        for reasoning in [Reasoning::Off, Reasoning::Budget { tokens: 2048 }] {
            let mut request = CompletionRequest::new("q")
                .options(
                    GenerationOptions::default()
                        .reasoning(reasoning)
                        .top_p(0.5)
                        .seed(1)
                        .stop(["END"])
                        .parallel_tool_calls(true)
                        .on_unsupported(OnUnsupported::Ignore),
                )
                .preamble("be brief".to_owned());
            request.temperature = Some(0.5);
            request.max_tokens = Some(4096);
            request.documents = documents();
            request.tools = vec![ToolDefinition::new(
                ToolName::new("lookup").expect("a tool name"),
                "a tool",
                json!({"type": "object"}),
            )];
            request.tool_choice = Some(ToolChoice::Required);
            request.output_schema = Some(schemars::json_schema!({"type": "object"}));
            let (_, body) = sent(&chat(route), request);
            let mut owned = BTreeSet::new();
            leaves(&body, "", &mut owned);
            for leaf in &provider {
                for reserved in &owned {
                    assert!(
                        !(leaf == reserved
                            || leaf.starts_with(&format!("{reserved}/"))
                            || reserved.starts_with(&format!("{leaf}/"))),
                        "{route:?}: {leaf} meets {reserved}"
                    );
                }
            }
        }
    }
}

/// Not a cassette test: no unary Cohere recording carries a tool plan or
/// log probabilities, so this native reply is built here. The recorded
/// fields are read in the `cohere` cassette target.
#[test]
fn extras_read_the_tool_plan_and_logprobs_of_a_native_reply() {
    let raw = json!({
        "id": "r-1",
        "finish_reason": "TOOL_CALL",
        "message": {"role": "assistant", "tool_plan": "I will look it up.", "tool_calls": []},
        "usage": {
            "billed_units": {"input_tokens": 12, "output_tokens": 4, "search_units": 1},
            "tokens": {"input_tokens": 300, "output_tokens": 9},
            "cached_tokens": 128
        },
        "logprobs": [{"token_ids": [1, 2], "text": "Hi", "logprobs": [-0.1, -0.2]}]
    });
    let extras =
        CohereExtras::from_reply(&Api::from_static("cohere.chat"), &raw).expect("the reply reads");
    assert_eq!(extras.id.as_deref(), Some("r-1"));
    assert_eq!(extras.finish_reason.as_deref(), Some("TOOL_CALL"));
    assert_eq!(extras.tool_plan.as_deref(), Some("I will look it up."));
    assert_eq!(
        extras.billed_units,
        Some(BilledUnits {
            input_tokens: Some(12.0),
            output_tokens: Some(4.0),
            search_units: Some(1.0),
            classifications: None,
        })
    );
    assert_eq!(
        extras.tokens,
        Some(Tokens {
            input_tokens: Some(300.0),
            output_tokens: Some(9.0),
        })
    );
    assert_eq!(extras.cached_tokens, Some(128.0));
    assert_eq!(
        extras.logprobs,
        Some(vec![Logprob {
            token_ids: vec![1, 2],
            text: Some("Hi".to_owned()),
            logprobs: vec![-0.1, -0.2],
        }])
    );
    assert!(CohereExtras::from_reply(&Api::from_static("cohere.chat"), &json!({"id": 7})).is_err());
}
