//! The Deep Research agent through the Interactions API, recorded: a
//! background stream dropped after its first cursor and resumed after it.

use futures::StreamExt;
use rig::completion::CompletionRequest;
use rig::providers::gemini::extension::{
    DeepResearchConfig, GeminiOptions, InteractionsOptions, ThinkingSummaries, Visualization,
};
use rig::providers::gemini::interactions_api::DEEP_RESEARCH_PREVIEW_04_2026;
use rig::providers::gemini::interactions_api::streaming::StreamCursor;

use super::super::support::{recorded_request_bodies, with_gemini_interactions_cassette};

/// With collaborative planning, the agent answers with a plan. The stream
/// is dropped at its first cursor, and the resumed stream, after that
/// cursor, carries the plan to its end.
#[tokio::test]
async fn a_planning_stream_resumes_after_its_cursor() {
    const SCENARIO: &str = "deep_research/planning_stream_resumed";
    with_gemini_interactions_cassette(
        "deep_research/planning_stream_resumed",
        |client| async move {
            let agent = InteractionsOptions::new()
                .agent(DEEP_RESEARCH_PREVIEW_04_2026)
                .background(true)
                .agent_config(
                    DeepResearchConfig::new()
                        .thinking_summaries(ThinkingSummaries::Auto)
                        .visualization(Visualization::Off)
                        .collaborative_planning(true)
                        .into(),
                );
            let request = CompletionRequest::new(
                "Plan a short research report on the origins of the Rust borrow checker.",
            )
            .provider_option(GeminiOptions::new().interactions(agent));
            let mut stream = client
                .interactions(DEEP_RESEARCH_PREVIEW_04_2026)
                .stream(request)
                .expect("the interaction opens");
            let mut cursor = None;
            while let Some(item) = stream.next().await {
                cursor = StreamCursor::of(&item.expect("the stream reads"));
                if cursor.is_some() {
                    break;
                }
            }
            drop(stream);
            let cursor = cursor.expect("the stream yields a cursor");
            let interaction_id = cursor
                .interaction_id
                .clone()
                .expect("the cursor names its interaction");

            let mut resumed = client
                .interaction_resumed(interaction_id.as_str(), Some(&cursor.event_id))
                .stream(CompletionRequest::new("resume"))
                .expect("the stream resumes");
            while let Some(item) = resumed.next().await {
                item.expect("the resumed stream reads");
            }
            let plan = resumed.finish().await.expect("the plan completes");
            assert_eq!(plan.response_id(), Some(interaction_id.as_str()));
            let text: String = plan
                .choice
                .iter()
                .filter_map(|content| match content {
                    rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(!text.trim().is_empty(), "the plan is text");
        },
    )
    .await;
    let body = recorded_request_bodies(SCENARIO);
    let body = body.first().expect("the create request");
    assert_eq!(
        body["agent_config"],
        serde_json::json!({
            "type": "deep-research",
            "thinking_summaries": "auto",
            "visualization": "off",
            "collaborative_planning": true
        })
    );
}
