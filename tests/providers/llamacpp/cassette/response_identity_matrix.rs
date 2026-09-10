//! Response identity, and the typed route's parity with the normalized one.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10499-6d05498.
//!
//! Two dimensions every mature suite in this tree carries and llama.cpp did
//! not: what `provider_request_id` and `response_id` are worth here, and
//! whether `raw_completion` → `normalize` reproduces what `completion`
//! reports.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`the_transport_request_id_is_absent_because_the_server_sends_none`] | `provider_request_id` | `None`, and the recorded headers show why |
//! | [`the_response_id_reaches_the_caller_on_both_transports`] | `response_id` | llama.cpp's `chatcmpl-…`, blocking and streaming |
//! | [`the_typed_route_reproduces_the_normalized_one`] | `raw_completion` parity | provider, model, finish reason, usage and the absence of a transport id |
//!
//! # `provider_request_id` is `None`, and that is a measurement
//!
//! `Llamacpp` leaves `OpenAICompatibleProvider::REQUEST_ID_HEADER` at its
//! `None` default, so rig never looks for a transport id. Declaring a contract
//! it does not have would reclassify every non-success status from `HttpError`
//! to `ProviderResponse` provider-wide, so "no contract" needs to be right
//! rather than merely convenient.
//!
//! It is. Measured live against b10499-6d05498 on both transports, the full
//! response header set is `Server`, `Access-Control-Allow-Origin`,
//! `Content-Type`, and `Content-Length` (blocking) or `X-Accel-Buffering` +
//! `Transfer-Encoding` (streaming), plus `Keep-Alive` — no id among them.
//!
//! The *fixtures* cannot show that set: `RESPONSE_HEADER_ALLOWLIST` keeps only
//! `content-type` and the transport-id names, so every recording here carries
//! exactly one header. That is precisely what makes the recorded check
//! meaningful in the one direction it can go — the three id names **are** on
//! the allowlist, so if llama.cpp ever started sending one it would be
//! recorded, and the cell's absence check would fail. An empty result is the
//! server's silence rather than the recorder's filtering.
//!
//! # And `response_id` is not the same thing
//!
//! llama.cpp does mint a per-call `chatcmpl-…` id in the *body*, and rig
//! surfaces it as `response_id`. The two are easy to conflate and have
//! different lifetimes: one is a transport correlator a proxy can add, the
//! other is the provider's own handle for the turn.

use rig::completion::{CompletionModel, NormalizeCompletionResponse};
use rig::prelude::*;
use serde_json::Value;

use crate::cassettes::{CassetteMode, recorded_statuses_and_bodies};

use super::super::cassette_support::*;

const PROBE: &str = "/no_think Reply with exactly the word: cedar.";

/// Header names any provider in this tree uses for a transport request id.
///
/// All three are on the cassette response-header allowlist, so if llama.cpp
/// ever starts sending one it will be recorded and this list will find it.
const REQUEST_ID_HEADERS: &[&str] = &["request-id", "x-request-id", "mistral-correlation-id"];

#[tokio::test]
async fn the_transport_request_id_is_absent_because_the_server_sends_none() {
    use futures::StreamExt as _;
    use rig::streaming::StreamedAssistantContent;

    with_llamacpp_cassette(
        "response_identity_matrix/blocking_identity",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(model.completion_request(PROBE).max_tokens(256).build())
                .await
                .expect("completion should succeed");

            assert!(
                response.provider_request_id.is_none(),
                "llama.cpp sends no transport id and this provider declares no \
             contract for one: {:?}",
                response.provider_request_id
            );
        },
    )
    .await;

    with_llamacpp_cassette(
        "response_identity_matrix/streaming_identity",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let mut stream = model
                .stream(model.completion_request(PROBE).max_tokens(256).build())
                .await
                .expect("stream should start");

            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(record) =
                    item.expect("stream item should be ok")
                {
                    terminal = Some(record);
                }
            }
            let terminal = terminal.expect("the stream must terminate");
            assert!(
                terminal.provider_request_id.is_none(),
                "the streaming surface must agree: {:?}",
                terminal.provider_request_id
            );
        },
    )
    .await;

    // The measurement behind the contract: no recorded response carries an id
    // header, on either transport. The allowlist keeps these names, so an
    // empty result is the server's silence rather than the scrubber's work.
    for scenario in [
        "response_identity_matrix/blocking_identity",
        "response_identity_matrix/streaming_identity",
    ] {
        let headers = recorded_response_headers(scenario);
        assert!(!headers.is_empty(), "{scenario}: no interactions recorded");
        for interaction in &headers {
            for (name, _) in interaction {
                assert!(
                    !REQUEST_ID_HEADERS.contains(&name.as_str()),
                    "{scenario}: llama.cpp started sending `{name}` — \
                     `REQUEST_ID_HEADER` should now be declared, and this cell is \
                     how that gets noticed"
                );
            }
            assert!(
                interaction.iter().any(|(name, _)| name == "content-type"),
                "{scenario}: the recording must carry headers at all, or the check \
                 above is vacuous: {interaction:?}"
            );
        }
    }
}

/// The body-level `chatcmpl-…` id does reach the caller, on both transports.
///
/// Its own scenarios rather than a re-read of the cell above's: a cell that
/// depends on another cell's fixture passes or fails on test *ordering*, which
/// is exactly the shared-state trap this corpus already has a rule against.
#[tokio::test]
async fn the_response_id_reaches_the_caller_on_both_transports() {
    use futures::StreamExt as _;
    use rig::streaming::StreamedAssistantContent;

    with_llamacpp_cassette(
        "response_identity_matrix/blocking_response_id",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(model.completion_request(PROBE).max_tokens(256).build())
                .await
                .expect("completion should succeed");

            let id = response
                .response_id
                .expect("llama.cpp mints a response id and rig surfaces it");
            assert!(
                id.starts_with("chatcmpl-"),
                "the provider's own handle for the turn: {id:?}"
            );
        },
    )
    .await;

    with_llamacpp_cassette(
        "response_identity_matrix/streaming_response_id",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let mut stream = model
                .stream(model.completion_request(PROBE).max_tokens(256).build())
                .await
                .expect("stream should start");

            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(record) =
                    item.expect("stream item should be ok")
                {
                    terminal = Some(record);
                }
            }
            let id = terminal
                .expect("the stream must terminate")
                .response_id
                .expect("the streamed terminal must carry the same handle");
            assert!(id.starts_with("chatcmpl-"), "{id:?}");
        },
    )
    .await;

    // And it really is in the recorded bytes on both transports — a
    // placeholder the scrubber minted would still start with `chatcmpl-`, so
    // the check is that the *wire* carried one at all.
    for scenario in [
        "response_identity_matrix/blocking_response_id",
        "response_identity_matrix/streaming_response_id",
    ] {
        let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
        let (status, body) = &recorded[0];
        assert_eq!(*status, 200, "{scenario}");
        let id = body
            .lines()
            .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
            .filter(|line| !line.is_empty() && *line != "[DONE]")
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find_map(|json| json.get("id")?.as_str().map(str::to_string))
            .unwrap_or_else(|| panic!("{scenario}: llama.cpp mints a response id"));
        match CassetteMode::current() {
            CassetteMode::Replay => assert!(
                id.starts_with("chatcmpl-"),
                "{scenario}: the id keeps its wire shape through scrubbing: {id:?}"
            ),
            CassetteMode::Record => assert!(
                id.len() > "chatcmpl-".len(),
                "{scenario}: a live id must be non-trivial: {id:?}"
            ),
        }
    }
}

/// `raw_completion` → `normalize` reproduces `completion`, field for field.
///
/// llama.cpp sends no transport request id, so — unlike Groq or xAI, where
/// `raw_completion` necessarily drops one the normalized path reports — the
/// plain typed route is *already* complete here. That is the parity claim, and
/// it is worth recording rather than assuming: it is the reason this provider
/// needs no `raw_completion_with_request_id` dance.
///
/// The two turns are separate calls against a sampling server, so what is
/// compared is everything the wire makes equal — provider, model, finish
/// reason, prompt usage, and the absence of a transport id — plus the presence
/// of a response id on each side. The answer *text* is deliberately not
/// compared: two calls need not produce the same tokens, and requiring it
/// would make the cell a flake rather than a parity check.
#[tokio::test]
async fn the_typed_route_reproduces_the_normalized_one() {
    with_llamacpp_cassette(
        "response_identity_matrix/typed_route_parity",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);

            let normalized = model
                .completion(model.completion_request(PROBE).max_tokens(256).build())
                .await
                .expect("completion should succeed");
            let raw = model
                .raw_completion(model.completion_request(PROBE).max_tokens(256).build())
                .await
                .expect("raw completion should succeed");
            let from_raw: rig::completion::CompletionResponse = raw
                .normalize("llamacpp")
                .expect("the raw response should normalize");

            // The two turns are separate calls with separate ids, so identity is
            // compared for *shape* and everything the wire makes equal is compared
            // exactly.
            assert_eq!(from_raw.provider, normalized.provider);
            assert_eq!(from_raw.model, normalized.model);
            assert_eq!(from_raw.finish_reason(), normalized.finish_reason());
            assert_eq!(
                from_raw.provider_request_id, normalized.provider_request_id,
                "both are None, and neither route may invent one"
            );
            assert!(
                from_raw.response_id.is_some() && normalized.response_id.is_some(),
                "both routes must carry the provider's own id: {:?} vs {:?}",
                from_raw.response_id,
                normalized.response_id
            );
            assert_eq!(
                from_raw.usage.input_tokens, normalized.usage.input_tokens,
                "the same prompt bills the same either way"
            );
        },
    )
    .await;

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "response_identity_matrix/typed_route_parity");
    assert_eq!(
        recorded.len(),
        2,
        "the scenario records both turns so the comparison is against real bytes"
    );
}
