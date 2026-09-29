//! Citations of xAI's hosted `web_search` tool through the shared Responses
//! adapter, recorded live on the streamed and the unary route with the same
//! prompt. Each route's text part must end with the `output_text` extras its
//! recorded message item states, annotations included, once. Every
//! expectation is derived from the frozen recording.

use std::sync::{Arc, Mutex};

use rig::completion::{CompletionRequest, ProviderToolDefinition};
use rig::message::Text;
use rig_test_support::citations::{
    annotation_count, assert_texts_match_items, choice_texts, drain, item_snapshots, messages,
};
use serde_json::{Value, json};

use super::support::with_xai_cassette;

const SCENARIO: &str = "web_search_citations/streamed_and_unary";
const MODEL: &str = "grok-4.3";
const PROMPT: &str = "Use web search to find the latest stable Rust release and its release \
     date. Cite the source you used. Answer in one short sentence.";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT)
        .additional_params(json!({ "store": false }))
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .max_tokens(2048)
}

/// What each route delivered.
#[derive(Default)]
struct Observed {
    streamed: Vec<Text>,
    unary: Vec<Text>,
}

/// The hosted search answers with cited text on both routes, and each text
/// part carries its recorded message item's annotations once.
#[tokio::test]
async fn web_search_citations_reach_streamed_and_unary_text() {
    let observed = Arc::new(Mutex::new(Observed::default()));
    let sink = observed.clone();
    with_xai_cassette(
        "web_search_citations/streamed_and_unary",
        |client| async move {
            let model = client.completion(MODEL);
            let streamed = drain(model.stream(request()).expect("the stream starts")).await;
            let unary = model.call(request()).await.expect("the unary call answers");
            *sink.lock().expect("the observation lock") = Observed {
                streamed: streamed.texts,
                unary: choice_texts(&unary.choice),
            };
        },
    )
    .await;
    let observed = observed.lock().expect("the observation lock");

    let interactions = crate::cassettes::recorded_interaction_bodies("xai", SCENARIO);
    assert_eq!(interactions.len(), 2, "one streamed and one unary exchange");
    for (request, _) in &interactions {
        let request: Value = serde_json::from_str(request).expect("the request is JSON");
        assert_eq!(request["store"], false);
        assert_eq!(request["tools"], json!([{ "type": "web_search" }]));
    }

    let frames = crate::cassettes::recorded_sse_json_frames("xai", SCENARIO);
    let snapshots = item_snapshots(&frames);
    assert!(
        snapshots.iter().any(|item| annotation_count(item) > 0),
        "the recorded stream states citations in its message snapshot"
    );
    assert_texts_match_items("streamed", &observed.streamed, &snapshots);

    let body: Value = serde_json::from_str(&interactions[1].1).expect("the unary body is JSON");
    let unary_items = messages(body["output"].as_array().into_iter().flatten());
    assert!(
        unary_items.iter().any(|item| annotation_count(item) > 0),
        "the recorded unary reply states citations"
    );
    assert_texts_match_items("unary", &observed.unary, &unary_items);
}
