//! Adversarial response-identity coverage (rig#2265 / PR #2313 follow-up):
//! feature collisions, failure/recovery paths, and replay semantics, chosen
//! because a plausible implementation error would make each cell fail.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext, ModelTurnAction,
    ModelTurnFinished,
};
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_cassette;
use crate::support::{Adder, IdentityProbe, assert_transport_request_id};

/// Family B: an invalid tool call (provider-advertised alias rig cannot
/// execute) repaired by a hook. The recovered turn's identity-bearing events
/// stay suppressed — intentional — while its `CompletionCall` still records
/// the attempt's identity.
#[tokio::test]
async fn repaired_invalid_call_keeps_call_identity() {
    #[derive(Clone, Default)]
    struct RepairToAdd {
        probe: IdentityProbe,
        repaired: Arc<AtomicBool>,
    }

    impl AgentHook for RepairToAdd {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            context: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            assert_eq!(context.tool_name, "sum_values");
            self.repaired.store(true, Ordering::SeqCst);
            Some(InvalidToolCallAction::repair("add"))
        }

        async fn on_model_turn_finished(
            &self,
            ctx: &HookContext,
            event: ModelTurnFinished<'_>,
        ) -> ModelTurnAction {
            self.probe.on_model_turn_finished(ctx, event).await
        }
    }

    with_anthropic_cassette(
        "response_identity_edge/repaired_invalid_call_keeps_call_identity",
        |client| async move {
            let hook = RepairToAdd::default();
            let agent = rig::AgentBuilder::new(client.completion(CLAUDE_SONNET_4_6))
                .preamble(
                    "Call the sum_values tool exactly once for the sum. As soon as any \
                     tool result arrives — whatever tool name it shows — state the final \
                     answer in plain text and make no further tool calls.",
                )
                .max_tokens(1024)
                .tool(Adder)
                .add_hook(hook.clone())
                .build();

            let response = agent
                .prompt("What is 2 + 3? Use the sum_values tool.")
                .merge_additional_params(
                    serde_json::json!({
                        "tools": [{
                            "name": "sum_values",
                            "description": "Add x and y.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"}
                                },
                                "required": ["x", "y"]
                            }
                        }]
                    })
                    .as_object()
                    .expect("params are an object")
                    .clone(),
                )
                .max_turns(4)
                .await
                .expect("repaired run should succeed");

            assert!(
                hook.repaired.load(Ordering::SeqCst),
                "the invalid-call hook must have fired"
            );
            // Every recorded completion call carries identity, recovered or not.
            assert!(!response.completion_calls.is_empty());
            for call in &response.completion_calls {
                assert_transport_request_id(
                    call.provider_request_id.as_deref(),
                    "recovered-run completion call",
                );
            }
            // The recovered turn fires no ModelTurnFinished — intentional
            // suppression — so hook observations count fewer events than
            // completion calls when a repair occurred.
            let turns = hook.probe.turn_identities();
            assert!(
                turns.len() < response.completion_calls.len(),
                "recovered turn suppressed: {} events vs {} calls",
                turns.len(),
                response.completion_calls.len()
            );
        },
    )
    .await;
}
