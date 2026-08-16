//! What Z.AI does with the model handles this crate exports.
//!
//! `GLM_4_6_AIR` and `GLM_4_6_X` name models that appear in neither the `model`
//! enum of Z.AI's chat-completion reference nor its pricing table — the 4.6
//! generation shipped only as `glm-4.6`. They are deprecated in
//! `crates/rig-core/src/providers/zai.rs` on that documentary evidence, and a
//! unit test there pins the constant set against a transcription of Z.AI's
//! catalog. These cells are the wire half of the same claim: recording them
//! turns "Z.AI does not document this model" into "Z.AI rejects this model".
//!
//! The control cell is what makes the rejections mean anything — it proves the
//! failure is about the name and not about the request shape.

use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::CHEAP_GENERAL_MODEL;
use super::super::support::{recorded_request_body, with_zai_general_cassette};
use crate::support::assert_nonempty_response;

/// The prompt and cap every cell here sends. Sharing them is what makes the
/// control meaningful: the rejected cells and the control differ in the model
/// name and nothing else, so a 400 is attributable to the handle.
///
/// (Shared as consts rather than by reading a sibling cell's fixture — a cell
/// must read only its own cassette, since in record mode the others may not be
/// written yet.)
const PROBE_PROMPT: &str = "Say hi.";
const PROBE_MAX_TOKENS: u64 = 16;

/// Asserts that Z.AI rejected a model handle, and that the rejection carries
/// the provider's own error envelope rather than a bare status.
fn assert_model_rejected(error: rig::completion::CompletionError, scenario: &str) {
    let status = error
        .provider_response_status()
        .unwrap_or_else(|| panic!("{scenario}: the rejection should preserve its HTTP status"));
    assert!(
        status.is_client_error(),
        "{scenario}: an unknown model is a client error, got {status}"
    );

    let body = error
        .provider_response_json()
        .unwrap_or_else(|err| panic!("{scenario}: error body should be JSON: {err}"))
        .unwrap_or_else(|| panic!("{scenario}: the rejection should preserve its body"));
    assert!(
        body.get("error").is_some(),
        "{scenario}: Z.AI answers a rejection with an `error` envelope, got {body}"
    );
}

/// The request body every cell in this file sends, model name aside.
fn assert_probe_shape(request: &serde_json::Value) {
    assert_eq!(request["messages"][0]["content"], PROBE_PROMPT);
    assert_eq!(request["max_tokens"], PROBE_MAX_TOKENS);
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
// The dead constant is exactly what this cell is about.
#[allow(deprecated)]
async fn general_unknown_model_constant_glm_4_6_air() {
    with_zai_general_cassette(
        "general/unknown_model_constant_glm_4_6_air",
        |client| async move {
            let model = client.completion_model(rig::providers::zai::GLM_4_6_AIR);
            let request = model
                .completion_request(PROBE_PROMPT)
                .max_tokens(PROBE_MAX_TOKENS)
                .build();

            let error = model
                .completion(request)
                .await
                .expect_err("Z.AI should reject a model it does not serve");

            assert_model_rejected(error, "general/unknown_model_constant_glm_4_6_air");
        },
    )
    .await;

    let request = recorded_request_body("general/unknown_model_constant_glm_4_6_air");
    assert_eq!(
        request["model"], "glm-4.6-air",
        "the cell must have asked for the dead constant itself"
    );
    assert_probe_shape(&request);
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
// The dead constant is exactly what this cell is about.
#[allow(deprecated)]
async fn general_unknown_model_constant_glm_4_6_x() {
    with_zai_general_cassette(
        "general/unknown_model_constant_glm_4_6_x",
        |client| async move {
            let model = client.completion_model(rig::providers::zai::GLM_4_6_X);
            let request = model
                .completion_request(PROBE_PROMPT)
                .max_tokens(PROBE_MAX_TOKENS)
                .build();

            let error = model
                .completion(request)
                .await
                .expect_err("Z.AI should reject a model it does not serve");

            assert_model_rejected(error, "general/unknown_model_constant_glm_4_6_x");
        },
    )
    .await;

    let request = recorded_request_body("general/unknown_model_constant_glm_4_6_x");
    assert_eq!(
        request["model"], "glm-4.6-x",
        "the cell must have asked for the dead constant itself"
    );
    assert_probe_shape(&request);
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_known_model_control() {
    with_zai_general_cassette("general/known_model_control", |client| async move {
        let model = client.completion_model(CHEAP_GENERAL_MODEL);
        let request = model
            .completion_request(PROBE_PROMPT)
            .max_tokens(PROBE_MAX_TOKENS)
            .build();

        let response = model
            .completion(request)
            .await
            .expect("a documented Z.AI model should answer the same request");

        let text = crate::support::assistant_text_response(&response.choice)
            .expect("the control turn should carry assistant text");
        assert_nonempty_response(&text);
    })
    .await;

    // Same body, same endpoint, only the name differs — that is what makes the
    // two rejections above attributable to the model handle. Each cell checks
    // the shared shape against its own fixture rather than against a sibling's.
    assert_probe_shape(&recorded_request_body("general/known_model_control"));
}
