//! Matrix P's live cells (`CLAUDE_SONNET_4_6`, temperature 0): the
//! equivalence row — hand-written layers standing in for the corpus's
//! hooks, over the hooks cells' own cassettes, so the records can be
//! asserted byte-equal to the hook goldens' — a host's denial and a
//! host's patch beneath the agent's. The layers are in `test-support/rig-test-support/src/goldens.rs`; the enumeration and
//! the replays live in `crates/rig-cassette/tests/corpus_layers.rs`.

use rig::agent::AgentBuilder;
use rig::effect::{EffectFamily, EffectKind, HandlerKey};
use rig::handlers::Handlers;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::serve::ErasedHandler;
use rig_cassette::agent::AgentReplayExt;
use rig_test_support::cassette_models::AnthropicModels;

use super::super::support::with_anthropic_corpus_hooks_cassette;
use crate::goldens::{
    DENY_REASON, DenyAddLayer, PATCHED_AGAIN_ARGS, PatchAddArgs, PatchAddArgsLayer,
    PatchAgainLayer, ReplaceAddResultLayer, add_tool_under, families,
};
use crate::support::TOOLS_PREAMBLE;

pub(super) const ADD_PROMPT: &str =
    "Use the add tool to add 17 and 25, then reply with just the number.";

pub(super) fn tool_record_args(log: &rig::cassette::effect_log::EffectLog) -> Vec<String> {
    log.records
        .iter()
        .filter_map(|record| match &record.kind {
            EffectKind::ToolCall { args, .. } => Some(args.clone()),
            _ => None,
        })
        .collect()
}

pub(super) fn tool_record_outputs(log: &rig::cassette::effect_log::EffectLog) -> Vec<String> {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(rig::effect::Answered::ToolResult { result, .. }) => Some(result.output().render()),
            _ => None,
        })
        .collect()
}

/// The program of the hooks cells with `layers` around `add` instead of
/// a hook stack, on the agent's own bus.
async fn own_bus(
    client: AnthropicModels,
    layers: impl FnOnce(ErasedHandler) -> ErasedHandler,
    hooks: impl FnOnce(AgentBuilder) -> AgentBuilder,
) -> rig::cassette::effect_log::EffectLog {
    let server = add_tool_under(layers);
    let builder = rig::AgentBuilder::new(client.completion(CLAUDE_SONNET_4_6))
        .name("golden")
        .preamble(TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool_server_handle(server);
    let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
    let agent = hooks(builder).record_to(recorder.clone()).build();
    let response = agent
        .prompt(ADD_PROMPT)
        .max_turns(3)
        .await
        .expect("the agent answers");
    assert!(!response.output().is_empty());
    agent.stamp(recorder.take())
}

#[tokio::test]
async fn deny_tool_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool", |client| async move {
        let log = own_bus(client, |adder| adder.layered(DenyAddLayer), |b| b).await;
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert_eq!(log.header.hooks, ["DenyAddLayer"]);
        crate::goldens::golden_effects("anthropic_layers_deny_tool", &log);
    })
    .await;
}

#[tokio::test]
async fn patch_tool_args_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/patch_tool_args", |client| async move {
        let log = own_bus(client, |adder| adder.layered(PatchAddArgsLayer), |b| b).await;
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        assert_eq!(log.header.hooks, ["PatchAddArgsLayer"]);
        crate::goldens::golden_effects("anthropic_layers_patch_tool_args", &log);
    })
    .await;
}

#[tokio::test]
async fn replace_tool_result_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/replace_tool_result", |client| async move {
        let log = own_bus(client, |adder| adder.layered(ReplaceAddResultLayer), |b| b).await;
        assert_eq!(
            tool_record_outputs(&log),
            ["42"],
            "the record holds the tool's answer"
        );
        assert_eq!(log.header.hooks, ["ReplaceAddResultLayer"]);
        crate::goldens::golden_effects("anthropic_layers_replace_tool_result", &log);
    })
    .await;
}

#[tokio::test]
async fn two_layers_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/two_hooks", |client| async move {
        // Outermost first in the header: the patch sees the dispatch
        // first, the replacement sees the answer first.
        let log = own_bus(
            client,
            |adder| {
                adder
                    .layered(ReplaceAddResultLayer)
                    .layered(PatchAddArgsLayer)
            },
            |b| b,
        )
        .await;
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(
            log.header.hooks,
            ["PatchAddArgsLayer", "ReplaceAddResultLayer"]
        );
        crate::goldens::golden_effects("anthropic_layers_two_layers", &log);
    })
    .await;
}

#[tokio::test]
async fn host_deny_over_host_bus_effect_log_is_the_golden_fixture() {
    // The host's own policy on the agent's tool key, over the host's bus.
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool", |client| async move {
        let handlers = Handlers::default();
        let model_key = HandlerKey::from("golden/model:default");
        handlers
            .register_erased(
                model_key.clone(),
                ErasedHandler::new(rig::serve::adapters::ModelAdapter::new(
                    "default",
                    client.completion(CLAUDE_SONNET_4_6),
                )),
            )
            .expect("a fresh key");
        let recorder = rig::cassette::effect_log::EffectLogRecorder::new();
        handlers.record_to(recorder.clone());
        let server = add_tool_under(|adder| adder.layered(DenyAddLayer));
        let agent = AgentBuilder::over_handlers(handlers.clone(), "golden", model_key)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool_server_handle(server)
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(!response.output().is_empty());
        let log = agent.stamp(recorder.take());
        drop(agent);
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert_eq!(log.header.hooks, ["DenyAddLayer"]);
        let _ = DENY_REASON;
        crate::goldens::golden_effects("anthropic_layers_host_deny_over_host_bus", &log);
    })
    .await;
}

#[tokio::test]
async fn patch_beneath_hook_patch_effect_log_is_the_golden_fixture() {
    // The agent's hook patches first (40 + 2); the host's layer beneath it
    // patches again (30 + 12): the record holds what was served.
    with_anthropic_corpus_hooks_cassette("corpus_hooks/patch_tool_args", |client| async move {
        let log = own_bus(
            client,
            |adder| adder.layered(PatchAgainLayer),
            |b| b.add_hook(PatchAddArgs),
        )
        .await;
        assert_eq!(tool_record_args(&log), [PATCHED_AGAIN_ARGS]);
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(log.header.hooks, ["PatchAddArgs", "PatchAgainLayer"]);
        crate::goldens::golden_effects("anthropic_layers_patch_beneath_hook_patch", &log);
    })
    .await;
}
