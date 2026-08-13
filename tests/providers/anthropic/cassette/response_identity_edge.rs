//! Adversarial response-identity coverage (rig#2265 / PR #2313 follow-up):
//! feature collisions, failure/recovery paths, and replay semantics, chosen
//! because a plausible implementation error would make each cell fail.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext, ModelTurnAction,
    ModelTurnFinished,
};
use rig::completion::{CompletionModel, Document, Message, Prompt};
use rig::prelude::*;
use rig::providers::anthropic::completion::{CLAUDE_SONNET_4_6, CacheTtl};
use rig::streaming::StreamedAssistantContent;
use rig::tool::{Tool, ToolContext, ToolExecutionError};

use super::super::support::with_anthropic_cassette;
use crate::support::{Adder, IdentityProbe, TOOLS_PREAMBLE, assert_transport_request_id};

fn cache_padding(label: &str) -> String {
    format!(
        "You are a deterministic identity-edge test assistant for {label}. {}",
        "This identity-edge fixture paragraph is stable provider test padding about request \
         routing, tool schemas, system instructions, and deterministic replay behavior. "
            .repeat(120)
    )
}

/// Family A: the #2312 cache-TTL feature and #2313 identity capture share
/// `send_completion` and landed a day apart — assert both on the same
/// recorded turns. The warm turn reads the cache yet reports its *own*
/// transport id, distinct from the cold turn's.
#[tokio::test]
async fn caching_and_identity_share_the_wire_blocking() {
    with_anthropic_cassette(
        "response_identity_edge/caching_and_identity_share_the_wire_blocking",
        |client| async move {
            let model = client
                .completion_model(CLAUDE_SONNET_4_6)
                .with_prompt_caching()
                .with_static_prefix_cache_ttl(CacheTtl::OneHour);
            let send = |model: rig::providers::anthropic::completion::CompletionModel| async move {
                model
                    .raw_completion(
                        model
                            .completion_request("Reply with exactly: edge probe")
                            .preamble(cache_padding("caching-blocking"))
                            .temperature(0.0)
                            .max_tokens(16)
                            .build(),
                    )
                    .await
                    .expect("cached completion should succeed")
            };

            let first = send(model.clone()).await;
            let second = send(model).await;

            // Identity present on both turns, and per-attempt distinct.
            assert_transport_request_id(first.provider_request_id.as_deref(), "cold turn");
            assert_transport_request_id(second.provider_request_id.as_deref(), "warm turn");
            assert_ne!(
                first.provider_request_id, second.provider_request_id,
                "the warm turn reports its own request id, not the cold turn's"
            );

            // The cache story holds on the same interactions: the cold turn
            // wrote (or re-read) the 1h prefix, the warm turn read it.
            let split = first
                .usage
                .cache_creation
                .as_ref()
                .expect("per-TTL split reported");
            assert_eq!(
                split.ephemeral_1h_input_tokens + split.ephemeral_5m_input_tokens,
                first.usage.cache_creation_input_tokens.unwrap_or_default()
            );
            assert!(
                second.usage.cache_read_input_tokens.unwrap_or_default() > 0,
                "warm turn reads the cache, got {:?}",
                second.usage
            );
        },
    )
    .await;
}

/// Streaming half of the cache × identity collision: the SSE-captured ids
/// are per-connection distinct while the cache warms across them.
#[tokio::test]
async fn caching_and_identity_share_the_wire_streaming() {
    with_anthropic_cassette(
        "response_identity_edge/caching_and_identity_share_the_wire_streaming",
        |client| async move {
            let model = client
                .completion_model(CLAUDE_SONNET_4_6)
                .with_prompt_caching()
                .with_static_prefix_cache_ttl(CacheTtl::OneHour);
            let send = |model: rig::providers::anthropic::completion::CompletionModel| async move {
                let mut stream = model
                    .completion_request("Reply with exactly: stream edge probe")
                    .preamble(cache_padding("caching-streaming"))
                    .temperature(0.0)
                    .max_tokens(16)
                    .stream()
                    .await
                    .expect("stream should open");
                let mut terminal = None;
                while let Some(item) = stream.next().await {
                    if let StreamedAssistantContent::Final(final_record) =
                        item.expect("stream item")
                    {
                        terminal = Some(final_record);
                    }
                }
                terminal.expect("terminal record")
            };

            let first = send(model.clone()).await;
            let second = send(model).await;
            assert_transport_request_id(first.provider_request_id.as_deref(), "cold stream");
            assert_transport_request_id(second.provider_request_id.as_deref(), "warm stream");
            assert_ne!(first.provider_request_id, second.provider_request_id);
            assert!(
                second.usage.cached_input_tokens > 0,
                "warm stream reads the cache, got {:?}",
                second.usage
            );
        },
    )
    .await;
}

/// Family A: strict tool schemas rebuild the request; identity still rides it.
#[tokio::test]
async fn strict_tools_and_identity() {
    with_anthropic_cassette(
        "response_identity_edge/strict_tools_and_identity",
        |client| async move {
            let model = client
                .completion_model(CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let response = model
                .completion_request("Use the add tool: what is 2 + 3?")
                .preamble(TOOLS_PREAMBLE.to_string())
                .max_tokens(1024)
                .tool(rig::tool::tool_definition(&Adder))
                .tool_choice(rig::message::ToolChoice::Required)
                .send()
                .await
                .expect("strict tool completion should succeed");
            assert_transport_request_id(
                response.provider_request_id.as_deref(),
                "strict-tools response",
            );
            assert!(response.message_id.is_some());
        },
    )
    .await;
}

/// Family A: extended thinking's altered message shape still routes through
/// the capturing driver.
#[tokio::test]
async fn extended_thinking_and_identity() {
    with_anthropic_cassette(
        "response_identity_edge/extended_thinking_and_identity",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let response = model
                .completion_request("Think briefly, then reply with exactly: thought probe")
                .max_tokens(2048)
                .additional_params(serde_json::json!({
                    "thinking": {"type": "enabled", "budget_tokens": 1024}
                }))
                .send()
                .await
                .expect("thinking completion should succeed");
            assert_transport_request_id(
                response.provider_request_id.as_deref(),
                "thinking response",
            );
        },
    )
    .await;
}

/// Family A: document context blocks still route through the capturing driver.
#[tokio::test]
async fn documents_and_identity() {
    with_anthropic_cassette(
        "response_identity_edge/documents_and_identity",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let response = model
                .completion_request("Per the document, reply with exactly the code word.")
                .document(Document {
                    id: "doc-1".to_string(),
                    text: "The code word is: heliotrope.".to_string(),
                    additional_props: Default::default(),
                })
                .max_tokens(32)
                .send()
                .await
                .expect("document completion should succeed");
            assert_transport_request_id(
                response.provider_request_id.as_deref(),
                "document response",
            );
        },
    )
    .await;
}

/// A tool that fails its first invocation and succeeds afterwards.
#[derive(Clone, Default)]
struct FlakyAdder {
    failed_once: Arc<AtomicBool>,
}

impl Tool for FlakyAdder {
    const NAME: &'static str = "add";
    type Error = ToolExecutionError;
    type Args = serde_json::Value;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y. May fail transiently; retry on failure.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        if !self.failed_once.swap(true, Ordering::SeqCst) {
            return Err(ToolExecutionError::timeout("transient failure; retry"));
        }
        Ok(args["x"].as_i64().unwrap_or(0) + args["y"].as_i64().unwrap_or(0))
    }
}

/// Family B: a tool error fed back to the model produces extra completion
/// calls — every one observed with its *own* transport id, none leaked
/// across the tool-error boundary.
#[tokio::test]
async fn tool_error_retry_reports_distinct_ids_blocking() {
    with_anthropic_cassette(
        "response_identity_edge/tool_error_retry_reports_distinct_ids_blocking",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("Use the add tool. If it fails transiently, call it again once.")
                .max_tokens(1024)
                .tool(FlakyAdder::default())
                .add_hook(probe.clone())
                .build();

            let response = agent
                .prompt("What is 2 + 3? Use the tool.")
                .max_turns(5)
                .extended_details()
                .await
                .expect("run should recover from the transient tool failure");

            let turns = probe.turn_identities();
            assert!(
                turns.len() >= 3,
                "error → retry → answer is at least three model calls, got {}",
                turns.len()
            );
            let mut ids: Vec<_> = turns
                .iter()
                .map(|turn| turn.provider_request_id.clone())
                .collect();
            for (index, id) in ids.iter().enumerate() {
                assert_transport_request_id(id.as_deref(), &format!("turn {index}"));
            }
            let calls: Vec<_> = response
                .completion_calls
                .iter()
                .map(|call| call.provider_request_id.clone())
                .collect();
            assert_eq!(ids, calls, "hooks and completion_calls agree in order");
            ids.dedup();
            assert_eq!(
                ids.len(),
                turns.len(),
                "every attempt across the tool-error boundary has its own id"
            );
        },
    )
    .await;
}

/// Family B: same recovery on the streamed surface.
#[tokio::test]
async fn tool_error_retry_reports_distinct_ids_streamed() {
    with_anthropic_cassette(
        "response_identity_edge/tool_error_retry_reports_distinct_ids_streamed",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("Use the add tool. If it fails transiently, call it again once.")
                .max_tokens(1024)
                .tool(FlakyAdder::default())
                .add_hook(probe.clone())
                .build();

            let mut stream = agent
                .runner(Message::user("What is 2 + 3? Use the tool."))
                .max_turns(5)
                .stream()
                .await;
            while let Some(item) = stream.next().await {
                item.expect("stream item should succeed");
            }

            let turns = probe.turn_identities();
            assert!(turns.len() >= 3, "got {} turns", turns.len());
            let mut ids: Vec<_> = turns
                .iter()
                .map(|turn| turn.provider_request_id.clone())
                .collect();
            for id in &ids {
                assert_transport_request_id(id.as_deref(), "streamed retry turn");
            }
            ids.dedup();
            assert_eq!(
                ids.len(),
                turns.len(),
                "distinct SSE connection per attempt"
            );
        },
    )
    .await;
}

/// Family B: a *live* hook-driven retry on the streamed surface. The retried
/// attempt opens a new SSE connection; the second `ModelTurnFinished` must
/// carry that second connection's id — the real-world test of the shared
/// request-id slot the unit tests only mock.
#[tokio::test]
async fn streamed_hook_retry_uses_second_connections_id() {
    #[derive(Clone, Default)]
    struct RetryOnce {
        probe: IdentityProbe,
        retried: Arc<AtomicBool>,
    }

    impl AgentHook for RetryOnce {
        async fn on_model_turn_finished(
            &self,
            ctx: &HookContext,
            event: ModelTurnFinished<'_>,
        ) -> ModelTurnAction {
            let action = if !self.retried.swap(true, Ordering::SeqCst) {
                ModelTurnAction::retry_with_feedback(
                    "Please answer again with exactly: retried probe",
                )
            } else {
                ModelTurnAction::continue_run()
            };
            // Record through the shared probe so ordering matches.
            let _ = self.probe.on_model_turn_finished(ctx, event).await;
            action
        }
    }

    with_anthropic_cassette(
        "response_identity_edge/streamed_hook_retry_uses_second_connections_id",
        |client| async move {
            let hook = RetryOnce::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("You are a terse assistant.")
                .max_tokens(64)
                .add_hook(hook.clone())
                .build();

            let mut stream = agent
                .runner(Message::user("Reply with exactly: first probe"))
                .max_turns(3)
                .stream()
                .await;
            while let Some(item) = stream.next().await {
                item.expect("stream item should succeed");
            }

            let turns = hook.probe.turn_identities();
            assert_eq!(turns.len(), 2, "one rejected attempt plus its retry");
            assert_transport_request_id(turns[0].provider_request_id.as_deref(), "attempt 1");
            assert_transport_request_id(turns[1].provider_request_id.as_deref(), "attempt 2");
            assert_ne!(
                turns[0].provider_request_id, turns[1].provider_request_id,
                "the retried attempt reports the second connection's id"
            );
        },
    )
    .await;
}

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
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
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
                .extended_details()
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

/// Family B: `max_turns` exhaustion mid-tool-run — every *completed* call was
/// observed with identity before the abort.
#[tokio::test]
async fn max_turns_exhaustion_still_observed_completed_calls() {
    with_anthropic_cassette(
        "response_identity_edge/max_turns_exhaustion_still_observed_completed_calls",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
                .tool(Adder)
                .add_hook(probe.clone())
                .build();

            let error = agent
                .prompt("What is 2 + 3? Use the tool.")
                .max_turns(1)
                .extended_details()
                .await
                .expect_err("a tool run under max_turns(1) must exhaust");
            let message = error.to_string();
            assert!(
                message.to_lowercase().contains("turn"),
                "expected a max-turns error, got: {message}"
            );

            let turns = probe.turn_identities();
            assert_eq!(turns.len(), 1, "the one completed call was observed");
            assert_transport_request_id(
                turns[0].provider_request_id.as_deref(),
                "exhausted run's completed call",
            );
        },
    )
    .await;
}

/// Family B: several tool calls in one turn are one completion call with one
/// identity — never duplicated per tool execution.
#[tokio::test]
async fn parallel_tool_calls_one_identity_per_turn() {
    with_anthropic_cassette(
        "response_identity_edge/parallel_tool_calls_one_identity_per_turn",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble(
                    "Use the add tool for arithmetic. When asked for several sums, emit all \
                     the tool calls in one single response.",
                )
                .max_tokens(1024)
                .tool(Adder)
                .add_hook(probe.clone())
                .build();

            let response = agent
                .prompt("Compute 2 + 3 and 10 + 20. Emit both add calls together, then state both results.")
                .max_turns(4)
                .extended_details()
                .await
                .expect("parallel tool run should succeed");

            let turns = probe.turn_identities();
            assert_eq!(
                turns.len(),
                response.completion_calls.len(),
                "one identity-bearing event per completion call, regardless of \
                 how many tools ran inside a turn"
            );
            for (turn, call) in turns.iter().zip(&response.completion_calls) {
                assert_eq!(turn.provider_request_id, call.provider_request_id);
            }
        },
    )
    .await;
}

/// Family D: run B is fed run A's messages as history; run B's identity is
/// its own — persisted `message_id`s in history never resurface as run B's
/// identity, and the request bodies carry no identity fields at all (the
/// harness's request-boundary match would fail otherwise).
#[tokio::test]
async fn history_replay_does_not_leak_prior_run_identity() {
    with_anthropic_cassette(
        "response_identity_edge/history_replay_does_not_leak_prior_run_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("You are a terse assistant.")
                .max_tokens(64)
                .add_hook(probe.clone())
                .build();

            let first = agent
                .prompt("Reply with exactly: run A probe")
                .extended_details()
                .await
                .expect("run A should succeed");
            let history = first.messages.clone().expect("run A history");
            let run_a_identity = probe.turn_identities()[0].clone();

            let second = agent
                .runner(Message::user("Reply with exactly: run B probe"))
                .history(history)
                .run()
                .await
                .expect("run B should succeed");
            let _ = second;

            let identities = probe.turn_identities();
            assert_eq!(identities.len(), 2);
            let run_b_identity = identities[1].clone();
            assert_ne!(
                run_a_identity.provider_request_id, run_b_identity.provider_request_id,
                "run B reports its own transport id"
            );
            assert_ne!(
                run_a_identity.message_id, run_b_identity.message_id,
                "run B's message id is run B's, not the replayed history's"
            );
        },
    )
    .await;
}

/// Family D: live pin of the stream→`CompletionResponse` conversion — the
/// terminal's transport id survives into the converted response.
#[tokio::test]
async fn stream_conversion_carries_live_identity() {
    with_anthropic_cassette(
        "response_identity_edge/stream_conversion_carries_live_identity",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let mut stream = model
                .completion_request("Reply with exactly: conversion probe")
                .max_tokens(32)
                .stream()
                .await
                .expect("stream should open");
            let mut terminal_id = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(final_record) = item.expect("stream item") {
                    terminal_id = final_record.provider_request_id.clone();
                }
            }
            assert_transport_request_id(terminal_id.as_deref(), "live terminal");

            let response: rig::completion::CompletionResponse = stream.into();
            assert_eq!(
                response.provider_request_id, terminal_id,
                "conversion carries the live terminal's id"
            );
        },
    )
    .await;
}

/// Family B: a provider 4xx — identity capture must not disturb the error
/// path, and the recorded fixture documents whether the error *response*
/// carried the `request-id` header (evidence for the error-path follow-up;
/// capture on errors is documented out of scope today).
#[tokio::test]
async fn provider_error_response_surfaces_cleanly() {
    with_anthropic_cassette(
        "response_identity_edge/provider_error_response_surfaces_cleanly",
        |client| async move {
            let model = client.completion_model("claude-nonexistent-model-for-identity-edge");
            let error = model
                .completion_request("Reply with exactly: never sent successfully")
                .max_tokens(16)
                .send()
                .await
                .expect_err("a nonexistent model must fail");
            let message = error.to_string();
            assert!(
                message.contains("not_found")
                    || message.contains("404")
                    || message.contains("model"),
                "expected a provider not-found error, got: {message}"
            );
        },
    )
    .await;
}
