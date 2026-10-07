//! Citations of the Responses hosted `web_search` tool, recorded live on
//! the streamed and the unary route with the same prompt. Each route's text
//! part must end with the `output_text` extras its recorded message item
//! states, annotations included, once, and cite it once per annotation.
//! Every expectation is derived from the frozen recording; none names a
//! generated answer, source or count.

use std::sync::{Arc, Mutex};

use rig::completion::{CompletionRequest, ProviderToolDefinition};
use rig::message::Text;
use rig::providers::openai::{GPT_5_4_MINI, OpenAIConfig};
use rig_test_support::cassette_models::OpenAiModels;
use rig_test_support::citations::{
    annotation_count, assert_texts_match_items, choice_texts, drain, frames_of, item_snapshots,
    messages, recorded_extras,
};
use serde_json::{Value, json};

use rig::completion::Effort;

use super::super::support::{effort, sse_json_frames, stateless, with_openai_cassette};
use crate::stream_faults::{recorded_sse_frames, scripted, sse_bytes};

const SCENARIO: &str = "web_search_citations/streamed_and_unary";
const PROMPT: &str = "Use web search to find the latest stable Rust release and its release \
     date. Cite the source you used. Answer in one short sentence.";
/// The key the scripted cell sends; nothing is sent anywhere.
const SCRIPTED_KEY: &str = "sk-scripted-citation-key-5c1e";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT)
        .options(effort(Effort::Low))
        .provider_options(stateless())
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .max_tokens(2048)
}

/// What each route delivered.
#[derive(Default)]
struct Observed {
    streamed: Vec<Text>,
    streamed_unknown_types: Vec<String>,
    unary: Vec<Text>,
}

/// The hosted search answers with cited text on both routes. The streamed
/// text part carries the annotations its `output_item.done` snapshot states,
/// once, although `content_part.done` and `response.completed` restate them
/// and `annotation.added` announced them; the unary text part carries the
/// annotations of its body by the same rule.
#[tokio::test]
async fn web_search_citations_reach_streamed_and_unary_text() {
    let observed = Arc::new(Mutex::new(Observed::default()));
    let sink = observed.clone();
    with_openai_cassette(
        "web_search_citations/streamed_and_unary",
        |client| async move {
            let model = client.openai.completion(GPT_5_4_MINI);
            let streamed = drain(model.stream(request()).expect("the stream starts")).await;
            let unary = model.call(request()).await.expect("the unary call answers");
            *sink.lock().expect("the observation lock") = Observed {
                streamed: streamed.texts,
                streamed_unknown_types: streamed.unknown_types,
                unary: choice_texts(&unary.choice),
            };
        },
    )
    .await;
    let observed = observed.lock().expect("the observation lock");

    let interactions = crate::cassettes::recorded_interaction_bodies("openai", SCENARIO);
    assert_eq!(interactions.len(), 2, "one streamed and one unary exchange");
    let requests: Vec<Value> = interactions
        .iter()
        .map(|(request, _)| serde_json::from_str(request).expect("the request is JSON"))
        .collect();
    assert_eq!(requests[0]["stream"], true);
    assert!(
        requests[1]
            .get("stream")
            .is_none_or(|stream| stream == false)
    );
    for request in &requests {
        assert_eq!(request["store"], false);
        assert_eq!(request["tools"], json!([{ "type": "web_search" }]));
        assert_eq!(
            request["input"], requests[0]["input"],
            "one prompt on both routes"
        );
    }

    let frames = sse_json_frames(&interactions[0].1);
    let snapshots = item_snapshots(&frames);
    assert!(
        snapshots.iter().any(|item| annotation_count(item) > 0),
        "the recorded stream states citations in its message snapshot"
    );
    let completed = frames_of(&frames, "response.completed");
    assert_eq!(completed.len(), 1, "the stream records one terminal");
    let restated = messages(
        completed[0]["response"]["output"]
            .as_array()
            .into_iter()
            .flatten(),
    );
    assert_eq!(
        restated.iter().map(recorded_extras).collect::<Vec<_>>(),
        snapshots.iter().map(recorded_extras).collect::<Vec<_>>(),
        "the terminal restates the item snapshots"
    );
    assert_texts_match_items("streamed", &observed.streamed, &snapshots);
    let added = frames_of(&frames, "response.output_text.annotation.added").len();
    assert_eq!(
        observed
            .streamed_unknown_types
            .iter()
            .filter(|kind| *kind == "response.output_text.annotation.added")
            .count(),
        added,
        "each incremental annotation event still reaches the consumer raw"
    );

    let body: Value = serde_json::from_str(&interactions[1].1).expect("the unary body is JSON");
    let unary_items = messages(body["output"].as_array().into_iter().flatten());
    assert!(
        unary_items.iter().any(|item| annotation_count(item) > 0),
        "the recorded unary reply states citations"
    );
    assert_texts_match_items("unary", &observed.unary, &unary_items);
}

/// The recorded stream with every message `output_item.done` frame removed,
/// replayed through the real Responses adapter over a scripted transport: a
/// gateway that sends no item snapshot. That omission is the only
/// transformation; every kept frame is the recording's bytes. The terminal
/// then supplies the annotations, once.
#[tokio::test]
async fn recorded_citations_without_item_snapshots_come_from_the_terminal() {
    let data = |frame: &String| {
        frame
            .lines()
            .find_map(|line| line.strip_prefix("data:"))
            .and_then(|data| serde_json::from_str::<Value>(data.trim()).ok())
    };
    let (dropped, kept): (Vec<String>, Vec<String>) = recorded_sse_frames("openai", SCENARIO, 0)
        .into_iter()
        .partition(|frame| {
            data(frame).is_some_and(|data| {
                data["type"] == "response.output_item.done" && data["item"]["type"] == "message"
            })
        });
    assert!(
        !dropped.is_empty(),
        "the recording has message snapshots to drop"
    );
    let kept_data: Vec<Value> = kept.iter().filter_map(data).collect();
    let completed = frames_of(&kept_data, "response.completed");
    assert_eq!(completed.len(), 1, "the recording keeps its terminal");
    let items = messages(
        completed[0]["response"]["output"]
            .as_array()
            .into_iter()
            .flatten(),
    );
    assert!(
        items.iter().any(|item| annotation_count(item) > 0),
        "the recorded terminal restates the citations"
    );

    let model = OpenAiModels::new(
        OpenAIConfig::new(SCRIPTED_KEY),
        scripted(vec![sse_bytes(&kept)]),
    )
    .completion(GPT_5_4_MINI);
    let streamed = drain(model.stream(request()).expect("the stream starts")).await;
    assert_texts_match_items("terminal only", &streamed.texts, &items);
}
