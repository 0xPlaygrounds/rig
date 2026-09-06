//! Native ECS provider completions preserving the original cassette assertions.
use crate::bedrock::support::with_bedrock_cassette;
use crate::ecs_agent::EcsAgent;
use crate::support::{
    Adder, CONTEXT_DOCS, CONTEXT_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT,
    Subtract, assert_contains_any_case_insensitive, assert_mentions_expected_number,
};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::bedrock;
use rig::prelude::*;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    with_bedrock_cassette("agent/completion_smoke", |client| async move {
        let mut ecs = EcsAgent::new(
            client.completion_model(bedrock::completion::AMAZON_NOVA_LITE),
            BASIC_PREAMBLE,
            1,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(DefaultMaxTurns(None));
        let response = ecs.prompt(BASIC_PROMPT, false).await;
        assert_nonempty_response(&response);
    })
    .await;
}
#[tokio::test]
async fn completion_with_context_smoke() {
    with_bedrock_cassette("agent/completion_with_context_smoke", |client| async move {
        let mut ecs = EcsAgent::new(
            client.completion_model(bedrock::completion::AMAZON_NOVA_LITE),
            "Answer the user using only the supplied context.",
            1,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(DefaultMaxTurns(None));
        let document = ecs
            .app
            .world_mut()
            .spawn((
                rig_ecs::agent::DocumentId("static_doc_0".into()),
                rig_ecs::agent::DocumentText((CONTEXT_DOCS[0]).into()),
            ))
            .id();
        ecs.app.world_mut().spawn((
            rig_ecs::agent::Context(document),
            rig_ecs::agent::Order(0),
            bevy_ecs::prelude::ChildOf(ecs.agent),
        ));
        let document = ecs
            .app
            .world_mut()
            .spawn((
                rig_ecs::agent::DocumentId("static_doc_1".into()),
                rig_ecs::agent::DocumentText((CONTEXT_DOCS[1]).into()),
            ))
            .id();
        ecs.app.world_mut().spawn((
            rig_ecs::agent::Context(document),
            rig_ecs::agent::Order(1),
            bevy_ecs::prelude::ChildOf(ecs.agent),
        ));
        let document = ecs
            .app
            .world_mut()
            .spawn((
                rig_ecs::agent::DocumentId("static_doc_2".into()),
                rig_ecs::agent::DocumentText((CONTEXT_DOCS[2]).into()),
            ))
            .id();
        ecs.app.world_mut().spawn((
            rig_ecs::agent::Context(document),
            rig_ecs::agent::Order(2),
            bevy_ecs::prelude::ChildOf(ecs.agent),
        ));
        let response = ecs.prompt(CONTEXT_PROMPT, false).await;
        assert_contains_any_case_insensitive(&response, &["ancient tool", "farm"]);
    })
    .await;
}
#[tokio::test]
async fn tool_roundtrip_smoke() {
    with_bedrock_cassette("agent/tool_roundtrip_smoke", |client| async move {
        let mut ecs = EcsAgent::new(
            client.completion_model(bedrock::completion::AMAZON_NOVA_LITE),
            STREAMING_TOOLS_PREAMBLE,
            2,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(rig_ecs::agent::MaxTokens(Some(1024)));
        ecs.tool(Adder);
        ecs.tool(Subtract);
        let response = ecs.prompt(STREAMING_TOOLS_PROMPT, false).await;
        assert_mentions_expected_number(&response, -3);
    })
    .await;
}
#[tokio::test]
async fn prompt_caching_completion_smoke() {
    with_bedrock_cassette(
        "agent/prompt_caching_completion_smoke",
        |client| async move {
            let model = client
                .completion_model(bedrock::completion::AMAZON_NOVA_LITE)
                .with_prompt_caching();
            let mut ecs = EcsAgent::new(model, BASIC_PREAMBLE, 1);
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(DefaultMaxTurns(None));
            let response = ecs.prompt(BASIC_PROMPT, false).await;
            assert_nonempty_response(&response);
        },
    )
    .await;
}
