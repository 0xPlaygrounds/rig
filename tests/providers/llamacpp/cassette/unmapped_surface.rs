//! Part 3: llama.cpp's surface beyond what rig calls, and the decision on each.
//!
//! `llama-server` b10499 registers 40-odd routes. Rig speaks four of them, and
//! this file is where the rest are decided rather than left as an unexamined
//! gap. Every row is **implement** or **exclude**, with the reason attached.
//!
//! | Route | Decision | Where |
//! | --- | --- | --- |
//! | `POST /v1/chat/completions` | implement | the whole suite |
//! | `POST /v1/embeddings` | implement | `embeddings.rs`, `error_matrix.rs` |
//! | `GET /v1/models` | implement | [`the_model_listing_reads_the_openai_half_of_a_hybrid_body`] |
//! | `POST /v1/rerank` (+ `/rerank`, `/reranking`, `/v1/reranking`) | implement | `rerank_matrix.rs` |
//! | `GET /props` | implement, as `VERIFY_PATH` | [`props_states_which_model_and_modalities_produced_this_corpus`] |
//! | `POST /v1/responses` | **exclude** | [`the_responses_api_is_reachable_but_rig_does_not_route_to_it`] |
//! | `POST /v1/audio/transcriptions` | **exclude** | [`transcription_is_a_501_unless_the_loaded_model_hears`] |
//! | `POST /infill` | **exclude** | rig has no FIM capability; see below |
//! | `POST /v1/messages` (Anthropic wire) | **exclude** | see below |
//! | `POST /completion`, `/v1/completions` | **exclude** | the legacy text-completion wire; rig models chat |
//! | `GET /slots`, `/metrics`, `/lora-adapters`, `POST /tokenize`, `/detokenize`, `/apply-template`, `/models/load`, `/models/unload`, `/cors-proxy`, `/tools` | **exclude** | operational and administrative; no rig capability corresponds |
//!
//! # The exclusions, stated
//!
//! * **`POST /infill`** — fill-in-the-middle, taking `input_prefix`,
//!   `input_suffix` and `input_extra`. Rig has no capability for code infilling
//!   at all: there is no trait to implement it against and no normalized
//!   response to map it into. Adding one is a rig-wide design question, not a
//!   provider integration. Excluded, and it is the clearest case in the table.
//! * **`POST /v1/messages`** — llama.cpp also speaks the *Anthropic* Messages
//!   wire, converting it to chat completions internally. Rig has an Anthropic
//!   provider, so this is reachable today by pointing `anthropic::Client` at a
//!   llama.cpp base URL. It is excluded from *this* provider because a
//!   provider that spoke two wires would have to pick one for every capability,
//!   and the OpenAI wire is the one llama.cpp's own documentation leads with.
//! * **The legacy completion wire** (`/completion`, `/v1/completions`) — the
//!   only place llama.cpp reports `tokens_evaluated`, `truncated`, `stop_type`,
//!   `stopping_word` and `generation_settings`. Rig models chat completions;
//!   there is no prompt-continuation capability to route here, and the fields
//!   are therefore *absent* rather than dropped. Stated in
//!   `providers::llamacpp::completion`'s docs alongside the `timings` that are
//!   preserved.
//! * **Operational routes** — `/slots` exposes and mutates server state,
//!   `/tokenize` and `/detokenize` are tokenizer utilities, `/lora-adapters`
//!   swaps adapters, `/models/load` and `/models/unload` drive the multi-model
//!   router, and `/tools` and `/cors-proxy` are the web UI's own back end. None
//!   maps to a rig capability. `/props` is the exception and is implemented:
//!   it is the only API-key-checked GET, which makes it the only honest
//!   `VERIFY_PATH`, and it reports the build tag and modalities that say what
//!   produced a fixture.

use rig::client::{ModelListingClient, VerifyClient};
use serde_json::Value;

use crate::cassettes::{recorded_request_paths, recorded_statuses_and_bodies};

use super::super::cassette_support::*;

/// `GET /v1/models` returns a **hybrid** body, and rig reads the right half.
///
/// One response carries an Ollama-style `models: [...]` array *and* OpenAI's
/// `object: "list"` with `data: [...]`, describing the same models twice in
/// different shapes. Rig's shared OpenAI-style lister reads `data`, which is
/// the half that carries `id`, `created` and `owned_by`; the `models` half has
/// `name`/`model` and a `details` object rig has no home for.
///
/// The cell asserts the envelope shape from the recorded bytes rather than
/// just "at least one model came back", because a lister that read the wrong
/// half would still return one model — with an empty id.
#[tokio::test]
async fn the_model_listing_reads_the_openai_half_of_a_hybrid_body() {
    with_llamacpp_cassette("unmapped_surface/models_envelope", |client| async move {
        let models = client
            .list_models()
            .await
            .expect("listing llama.cpp models should succeed");

        assert_eq!(
            models.len(),
            1,
            "a single-model server lists exactly one: {models:#?}"
        );
        let model = &models.data[0];
        assert!(
            !model.id.is_empty(),
            "reading the wrong half of the hybrid body yields an empty id: {model:#?}"
        );
        assert_eq!(
            model.owned_by.as_deref(),
            Some("llamacpp"),
            "`owned_by` exists only on the OpenAI half: {model:#?}"
        );
        assert!(
            model.created_at.is_some(),
            "`created` exists only on the OpenAI half: {model:#?}"
        );
    })
    .await;

    assert_eq!(
        recorded_request_paths("llamacpp", "unmapped_surface/models_envelope"),
        vec!["/v1/models".to_string()]
    );

    let recorded = recorded_statuses_and_bodies("llamacpp", "unmapped_surface/models_envelope");
    let (status, body) = &recorded[0];
    assert_eq!(*status, 200);
    let response: Value = serde_json::from_str(body).expect("response should be JSON");

    // Both halves are present — that is what makes it a hybrid rather than
    // simply an OpenAI listing.
    assert_eq!(response["object"], serde_json::json!("list"));
    let data = response["data"].as_array().expect("the OpenAI half");
    let ollama = response["models"]
        .as_array()
        .expect("the Ollama-style half");
    assert_eq!(
        data.len(),
        ollama.len(),
        "the two halves describe the same models: {response}"
    );

    // And they are genuinely different shapes, so reading the wrong one is a
    // real failure mode rather than a hypothetical.
    assert!(
        data[0].get("id").is_some() && data[0].get("name").is_none(),
        "the OpenAI half keys on `id`: {}",
        data[0]
    );
    assert!(
        ollama[0].get("name").is_some() && ollama[0].get("id").is_none(),
        "the Ollama-style half keys on `name`: {}",
        ollama[0]
    );
}

/// `GET /props` is what a fixture uses to say which model and modalities
/// produced it.
///
/// Recorded once, on the vision server, because that is the configuration
/// where the answer is not obvious: `modalities` reports `vision: true` *and*
/// `video: true` there, which is what makes the video cell in
/// `multimodal_matrix.rs` a question worth asking rather than an assumption.
///
/// It doubles as the evidence for `VERIFY_PATH`: this is the route
/// `Client::verify()` issues, so a successful verification is a successful
/// `/props`.
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
        serde_json::json!("b10499-6d05498"),
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
/// through a bare `openai::Client`'s Responses surface, which is exactly what
/// a caller who wants it does today, and pins that it works. So "rig cannot"
/// is not the reason; "one provider, one wire" is.
#[tokio::test]
async fn the_responses_api_is_reachable_but_rig_does_not_route_to_it() {
    use rig::completion::CompletionModel as _;
    use rig::prelude::*;

    with_llamacpp_bare_openai_cassette("unmapped_surface/responses_api", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request("/no_think Reply with the single word: ok")
                    .max_tokens(256)
                    .build(),
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
