//! Structured output: the four ways to constrain llama.cpp's answer, and which
//! of them actually constrain it.
//!
//! **Server**: the competent tier (`unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, b10964-b29c606e2) for the cells whose
//! claim is about the *model* holding a shape, and the default smoke tier for
//! the cells whose claim is about the *server* enforcing one. Grammar
//! adherence is a capability question, and the split says which side of it
//! each cell is on.
//!
//! | Cell | Mechanism | Constrained by | Result |
//! | --- | --- | --- | --- |
//! | [`json_object_response_format_is_enforced_as_an_object`] | `response_format: {type: json_object}` | GBNF for any JSON object | a one-word answer comes back wrapped in an object |
//! | [`a_gbnf_grammar_through_additional_params_is_enforced`] | `grammar` | GBNF verbatim | only the alternatives the grammar allows |
//! | [`a_schema_and_a_grammar_together_are_rejected`] | top-level `json_schema` + `grammar` | — | 500, `Cannot use both json_schema and grammar` |
//! | [`response_format_and_a_grammar_silently_let_the_schema_win`] | `response_format` + `grammar` | schema only | 200, the grammar is dropped with no diagnostic |
//!
//! # `json_object` constrains the answer to a JSON object
//!
//! OpenAI's `response_format: {"type": "json_object"}` guarantees syntactically
//! valid JSON. llama.cpp b10499 derived an empty schema from a bare
//! `json_object` and constrained nothing; b10964-b29c606e2 enforces a JSON
//! object. The cell asks for a single plain word, which only a grammar turns
//! into JSON, so it tells the two behaviors apart.
//!
//! Rig never sends a bare `json_object` on its own: `output_schema` maps to
//! `json_schema`, which llama.cpp also enforces.
//!
//! # The conflict guard has a hole, and rig's route is on the wrong side of it
//!
//! llama.cpp refuses a request that carries both a schema and a grammar —
//! `if (!json_schema.is_null() && !grammar.empty()) throw` — but that check
//! reads the *top-level* `json_schema` field and runs **before**
//! `response_format` is unpacked into the same variable. Rig's `output_schema`
//! travels as `response_format`, so pairing it with an explicit `grammar`
//! passes the guard and the schema silently wins. The two cells
//! [`a_schema_and_a_grammar_together_are_rejected`] and
//! [`response_format_and_a_grammar_silently_let_the_schema_win`] record both
//! sides; neither is a rig defect, and a caller who does not know about the
//! hole gets a constraint they did not ask for with no diagnostic.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};
use crate::support::assistant_text_response;

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

const NO_THINK: &str = "/no_think ";

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct CityFact {
    #[schemars(required)]
    city: String,
    #[schemars(required)]
    country: String,
    #[schemars(required)]
    population_millions: f64,
}

fn recorded_answer(scenario: &str) -> String {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// `json_object` alone constrains the answer to a JSON object.
#[tokio::test]
async fn json_object_response_format_is_enforced_as_an_object() {
    with_llamacpp_cassette(
        "structured_output_matrix/json_object_is_enforced",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(format!(
                        "{NO_THINK}Reply with the single word hello and nothing else."
                    ))
                    .max_tokens(256)
                    .additional_params(json!({
                        "response_format": { "type": "json_object" }
                    })),
                )
                .await
                .expect("a bare json_object response_format is accepted");

            let text = assistant_text_response(&response.choice).unwrap_or_default();
            assert!(!text.trim().is_empty(), "the turn produced an answer");
        },
    )
    .await;

    let request = recorded_json_request(
        "llamacpp",
        "structured_output_matrix/json_object_is_enforced",
    );
    assert_eq!(
        request["response_format"],
        json!({ "type": "json_object" }),
        "additional_params must merge the response_format through unchanged"
    );

    // Unconstrained, the model answers with the bare word; only the server's
    // grammar makes it a JSON object.
    let answer = recorded_answer("structured_output_matrix/json_object_is_enforced");
    let parsed: Value = serde_json::from_str(answer.trim()).unwrap_or_else(|error| {
        panic!("a bare json_object must yield a JSON object ({error}): {answer:?}")
    });
    assert!(
        parsed.is_object(),
        "the answer is a JSON object: {answer:?}"
    );
}

/// A GBNF grammar sent verbatim through `additional_params`.
///
/// This is llama.cpp's own constraint language and has no OpenAI equivalent,
/// so `additional_params` is the only route. The grammar admits exactly two
/// strings, which makes the assertion total rather than probabilistic — the
/// model cannot produce anything else even if it wants to.
#[tokio::test]
async fn a_gbnf_grammar_through_additional_params_is_enforced() {
    with_llamacpp_cassette(
        "structured_output_matrix/gbnf_grammar_is_enforced",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(format!(
                        "{NO_THINK}Answer with one word: is the sky blue?"
                    ))
                    .max_tokens(16)
                    .additional_params(json!({ "grammar": "root ::= \"yes\" | \"no\"" })),
                )
                .await
                .expect("a GBNF grammar should be accepted");

            let text = assistant_text_response(&response.choice).unwrap_or_default();
            assert!(
                matches!(text.trim(), "yes" | "no"),
                "the grammar admits exactly two strings: {text:?}"
            );
        },
    )
    .await;

    let request = recorded_json_request(
        "llamacpp",
        "structured_output_matrix/gbnf_grammar_is_enforced",
    );
    assert_eq!(
        request["grammar"],
        json!("root ::= \"yes\" | \"no\""),
        "the grammar must reach the wire verbatim"
    );
    let answer = recorded_answer("structured_output_matrix/gbnf_grammar_is_enforced");
    assert!(matches!(answer.trim(), "yes" | "no"), "{answer:?}");
}

/// A **top-level** `json_schema` beside a `grammar` is refused — with a 500.
///
/// llama.cpp guards the conflict at `server-common.cpp:1157`:
/// `if (!json_schema.is_null() && !grammar.empty()) throw`. Both constraints
/// compile to a GBNF grammar and it will not guess which wins. The status is
/// 500 rather than the 400 a caller error deserves — the same
/// misclassification the error matrix records for `--no-jinja` with tools.
#[tokio::test]
async fn a_schema_and_a_grammar_together_are_rejected() {
    with_llamacpp_cassette(
        "structured_output_matrix/schema_and_grammar_conflict",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let error = model
                .call(
                    CompletionRequest::new(format!("{NO_THINK}Give a fact about Paris."))
                        .max_tokens(128)
                        .additional_params(json!({
                            "json_schema": {
                                "type": "object",
                                "properties": { "city": { "type": "string" } },
                                "required": ["city"],
                            },
                            "grammar": "root ::= \"yes\" | \"no\"",
                        })),
                )
                .await
                .expect_err("a top-level schema and a grammar cannot both constrain one turn");

            let body = error
                .provider_response_body()
                .expect("the refusal body must be preserved");
            assert!(
                body.contains("Cannot use both json_schema and grammar"),
                "the message must name both constraints: {body}"
            );
        },
    )
    .await;

    let recorded = recorded_statuses_and_bodies(
        "llamacpp",
        "structured_output_matrix/schema_and_grammar_conflict",
    );
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(
        *status, 500,
        "llama.cpp reports this caller error as a server error: {body}"
    );
    let json: Value = serde_json::from_str(body).expect("error body should be JSON");
    assert_eq!(json["error"]["type"], json!("server_error"));
}

/// The **`response_format`** route silently drops the grammar instead.
///
/// The conflict guard above reads the *top-level* `json_schema` field, and
/// `response_format` is only unpacked into `json_schema` on the next lines
/// (`server-common.cpp:1163-1176`) — after the check has already passed. So
/// the route rig actually takes for `output_schema` can carry a `grammar`
/// alongside it and the schema wins with no diagnostic at all.
///
/// Measured on b10964-b29c606e2: a request pairing a `{city, country,
/// population_millions}` schema with `root ::= "yes" | "no"` answers with the
/// JSON object. Neither constraint is what the caller asked for jointly, and
/// nothing says so — which is why the pair of cells is worth more than either.
#[tokio::test]
async fn response_format_and_a_grammar_silently_let_the_schema_win() {
    with_llamacpp_competent_cassette(
        "structured_output_matrix/response_format_beats_grammar_silently",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(format!("{NO_THINK}Give a fact about Paris, France."))
                        .max_tokens(256)
                        .output_schema(schemars::schema_for!(CityFact))
                        .additional_params(json!({ "grammar": "root ::= \"yes\" | \"no\"" })),
                )
                .await
                .expect("the response_format route does not trip the conflict guard");

            let text = assistant_text_response(&response.choice).unwrap_or_default();
            assert!(
                !matches!(text.trim(), "yes" | "no"),
                "the explicit grammar was dropped, not applied: {text:?}"
            );
            serde_json::from_str::<CityFact>(text.trim())
                .unwrap_or_else(|error| panic!("the schema won silently: {error}: {text:?}"));
        },
    )
    .await;

    let request = recorded_json_request(
        "llamacpp",
        "structured_output_matrix/response_format_beats_grammar_silently",
    );
    assert_eq!(request["response_format"]["type"], json!("json_schema"));
    assert_eq!(request["grammar"], json!("root ::= \"yes\" | \"no\""));
    let recorded = recorded_statuses_and_bodies(
        "llamacpp",
        "structured_output_matrix/response_format_beats_grammar_silently",
    );
    assert_eq!(
        recorded[0].0, 200,
        "no conflict is reported on this route, which is the point"
    );
    let answer = recorded_answer("structured_output_matrix/response_format_beats_grammar_silently");
    serde_json::from_str::<CityFact>(answer.trim())
        .expect("the recorded answer follows the schema, not the grammar");
}
