//! llama.cpp's surface beyond what rig calls, and the decision on each.
//!
//! Rig speaks four `llama-server` routes: chat completions, embeddings,
//! rerank, and `GET /props`, the only API-key-checked GET and so the
//! dialect's `verify_path`, which also reports the build tag and modalities
//! that produced a fixture
//! ([`props_states_which_model_and_modalities_produced_this_corpus`]). The
//! other cells pin excluded routes rig can reach but does not route to:
//! [`the_responses_api_is_reachable_but_rig_does_not_route_to_it`] and
//! [`transcription_is_a_501_unless_the_loaded_model_hears`].
//!
//! The rest are excluded. `/infill` has no rig capability to map into. The
//! Anthropic-wire `/v1/messages` is reachable through rig's Anthropic
//! provider with a base URL, and one provider speaks one wire. The legacy
//! `/completion` wire is prompt continuation, which rig does not model, so
//! its extra fields are absent rather than dropped. Operational routes such
//! as `/slots`, `/tokenize`, `/lora-adapters` and `/models/load` map to no
//! rig capability.

use serde_json::Value;

use crate::cassettes::{recorded_request_paths, recorded_statuses_and_bodies};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

/// `GET /props` is what a fixture uses to say which model and modalities
/// produced it.
///
/// Recorded once, on the vision server, because that is the configuration
/// where the answer is not obvious: `modalities` reports `vision: true` *and*
/// `video: true` there, which is what makes the video cell in
/// `multimodal_matrix.rs` a question worth asking rather than an assumption.
///
/// It doubles as the evidence for the dialect's `verify_path`: this is the
/// route `Model::verify()` issues, so a successful verification is a
/// successful `/props`.
#[tokio::test]
async fn props_states_which_model_and_modalities_produced_this_corpus() {
    with_llamacpp_vision_cassette("unmapped_surface/props", |client| async move {
        // `verify()` *is* the `/props` request; there is no separate typed
        // accessor, which is itself part of the exclude decision for the
        // operational routes.
        client
            .verify()
            .await
            .expect("an unkeyed server verifies successfully");
    })
    .await;

    assert_eq!(
        recorded_request_paths("llamacpp", "unmapped_surface/props"),
        vec!["/props".to_string()],
        "the operational namespace is not under /v1"
    );

    let recorded = recorded_statuses_and_bodies("llamacpp", "unmapped_surface/props");
    let (status, body) = &recorded[0];
    assert_eq!(*status, 200);
    let props: Value = serde_json::from_str(body).expect("props should be JSON");

    assert_eq!(
        props["build_info"],
        serde_json::json!("b10964-b29c606e2"),
        "every fixture in this corpus was recorded against this build; if this \
         assertion fails the corpus was re-recorded against another one and the \
         module docs need updating"
    );
    assert_eq!(
        props["modalities"]["vision"],
        serde_json::json!(true),
        "the vision configuration must actually load a projector: {}",
        props["modalities"]
    );
    assert!(
        props["chat_template_caps"]["supports_tools"].is_boolean(),
        "`chat_template_caps` is how a family's tool support is decided in \
         `model_family_matrix.rs`: {}",
        props["chat_template_caps"]
    );
}

/// `POST /v1/responses` exists and answers — and rig deliberately does not
/// route this provider to it.
///
/// llama.cpp implements the Responses API by converting the request into a
/// chat completion, so it is the *same* model, the same sampler and the same
/// template behind a second envelope. Declaring it would mean this provider
/// carried two completion paths that differ only in the wire, with no
/// capability on one that the other lacks — while doubling the matrix that has
/// to be recorded and maintained.
///
/// The exclusion is recorded rather than assumed: the cell reaches the route
/// through the plain `OpenAI` configuration's Responses wire, which is exactly
/// what a caller who wants it does today, and pins that it works. So "rig
/// cannot" is not the reason; "one provider, one wire" is.
#[tokio::test]
async fn the_responses_api_is_reachable_but_rig_does_not_route_to_it() {
    with_llamacpp_bare_openai_cassette("unmapped_surface/responses_api", |client| async move {
        // The wrapper hands out the plain OpenAI configuration; the Responses
        // surface is a *different* wire over the same socket and the same
        // base URL, which is the whole shape of the exclusion.
        let model = client.responses(CASSETTE_MODEL);
        let response = model
            .call(
                CompletionRequest::new("/no_think Reply with the single word: ok").max_tokens(256),
            )
            .await
            .expect("llama.cpp serves the Responses API end to end");
        assert!(!response.choice.is_empty());
    })
    .await;

    assert_eq!(
        recorded_request_paths("llamacpp", "unmapped_surface/responses_api"),
        vec!["/v1/responses".to_string()],
        "the cell must really have exercised the Responses route, not chat completions"
    );
    let recorded = recorded_statuses_and_bodies("llamacpp", "unmapped_surface/responses_api");
    let (status, body) = &recorded[0];
    assert_eq!(*status, 200, "{body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    assert_eq!(
        response["object"],
        serde_json::json!("response"),
        "llama.cpp answers with the Responses envelope, not a chat completion: {response}"
    );
    assert!(
        response["output"].is_array(),
        "the Responses envelope carries `output`: {response}"
    );
}

/// `POST /v1/audio/transcriptions` exists and 501s unless the loaded model
/// hears.
///
/// llama.cpp implements transcription by rewriting the upload into a
/// chat-template ASR prompt, so whether the route works is a property of the
/// *weights*, not of the server: without an audio projector it answers
/// `501 "The current model does not support audio input."`.
///
/// Rig's `TranscriptionModel` contract has no way to express "this endpoint
/// exists but depends on which model is loaded", and llama.cpp additionally
/// rejects every `response_format` except `json` while rig's shared multipart
/// driver sends none. Declaring the slot would ship a capability that 501s on
/// every text-only server, which is most of them. Excluded — and the 501 is
/// recorded here so the exclusion rests on a measurement.
#[tokio::test]
async fn transcription_is_a_501_unless_the_loaded_model_hears() {
    // Reached directly rather than through a capability, because the whole
    // point is that this provider declares none. The request is the minimal
    // multipart body llama.cpp's handler reads before it checks the model.
    with_llamacpp_raw_http_cassette(
        "unmapped_surface/transcription_not_supported",
        |base_url| async move {
            let body = "--rigboundary\r\n\
                Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
                Content-Type: audio/wav\r\n\r\nRIFF\r\n\
                --rigboundary--\r\n";
            let response = reqwest::Client::new()
                .post(format!(
                    "{}/v1/audio/transcriptions",
                    base_url.trim_end_matches('/')
                ))
                .header("content-type", "multipart/form-data; boundary=rigboundary")
                .body(body)
                .send()
                .await
                .expect("the route exists");
            let status = response.status();
            let text = response.text().await.expect("a body");

            assert_eq!(
                status.as_u16(),
                501,
                "a text-only model cannot transcribe: {text}"
            );
            let json: Value = serde_json::from_str(&text).expect("the 501 body should be JSON");
            assert_eq!(
                json["error"]["type"],
                serde_json::json!("not_supported_error")
            );
            assert!(
                json["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("audio")),
                "{json}"
            );
        },
    )
    .await;

    // And the exclusion rests on the recorded bytes, not on the assertion
    // above alone.
    let recorded =
        recorded_statuses_and_bodies("llamacpp", "unmapped_surface/transcription_not_supported");
    assert_eq!(recorded[0].0, 501);
}
