//! Provider-adapter execution through native ECS, using the original agent
//! fixtures. The original tests remain independent baseline executions.

use rig::providers::gemini;
use rig_ecs::agent::{AdditionalParams, Temperature};

use super::super::support::with_gemini_cassette;
use crate::{ecs_agent::EcsAgent, support::assert_nonempty_response};

#[tokio::test]
async fn example_streaming_prompt() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_gemini_cassette("streaming/example_streaming_prompt", |client| async move {
                let mut ecs = EcsAgent::new(
                    client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW),
                    "Be precise and concise.",
                    1,
                );
                ecs.app.world_mut().entity_mut(ecs.agent).insert((
                    Temperature(Some(0.5)),
                    AdditionalParams(Some(serde_json::json!({
                        "generationConfig": {
                            "thinkingConfig": {
                                "thinkingLevel": "medium",
                                "includeThoughts": true
                            }
                        }
                    }))),
                ));
                assert_nonempty_response(
                    &ecs.prompt(
                        "When and where and what type is the next solar eclipse?",
                        true,
                    )
                    .await,
                );
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "gemini_parity_example_streaming_prompt",
                log,
            )
        },
    )
    .await
}
