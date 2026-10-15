//! The Deep Research agent through the Interactions API, recorded: a
//! background stream dropped at its first resume point and resumed after
//! it.

use futures::StreamExt;
use rig::completion::CompletionRequest;
use rig::providers::gemini::extension::{
    DeepResearchConfig, GeminiOptions, InteractionsOptions, ThinkingSummaries, Visualization,
};
use rig::providers::gemini::interactions_api::DEEP_RESEARCH_PREVIEW_04_2026;
use rig::providers::gemini::interactions_api::streaming::StreamCursor;

use super::super::support::{recorded_request_bodies, with_gemini_interactions_cassette};

/// With collaborative planning, the agent answers with a plan. The stream
/// names its interaction as it is created, is dropped at its first resume
/// point, and the stream resumed after that point carries the plan to its
/// end.
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
            let (mut created, mut cursor) = (None, None);
            while let Some(item) = stream.next().await {
                let Some(next) = StreamCursor::of(&item.expect("the stream reads")) else {
                    continue;
                };
                match next.event_id {
                    None => created = created.or(Some(next)),
                    Some(_) => {
                        cursor = Some(next);
                        break;
                    }
                }
            }
            drop(stream);
            let created = created.expect("the creation yields a cursor");
            let cursor = cursor.expect("the stream yields a resume point");
            let interaction_id = created
                .interaction_id
                .expect("the creation names its interaction");
            assert_eq!(
                cursor.interaction_id.as_deref(),
                Some(interaction_id.as_str())
            );

            let mut resumed = client
                .interaction_resumed(interaction_id.as_str(), cursor.event_id.as_deref())
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
