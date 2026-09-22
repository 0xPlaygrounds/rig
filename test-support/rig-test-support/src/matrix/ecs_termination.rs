//! Shared ecs termination bodies; each wire supplies explicit scenario rows.

/// Emit registered test rows with the shared execution body.
#[macro_export]
macro_rules! ecs_termination_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_truncated_turn_reports_length_and_cap_7, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), CONCISE_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(TINY_CAP))));
                    ecs_termination::install(&mut ecs, probe, None);
                    ecs.prompt_with_max_turns(TRUNCATING_PROMPT, false, None)
                        .await;
                })
                .await;
                assert_eq!(
                    observed.first_reason(),
                    Some(FinishReason::Length),
                    "the wire `length` must reach the hook as FinishReason::Length"
                );
                assert_eq!(
                    observed.first_max_tokens(),
                    Some(TINY_CAP),
                    "the hook must report the cap this attempt actually ran under"
                );
                assert!(
                    observed
                        .first_reason()
                        .is_some_and(|reason| reason.truncated_output()),
                    "a truncated turn must satisfy the portable retry predicate"
                );
                assert_recorded_wire_reason(SCENARIO, "length");
                assert_recorded_request_cap(SCENARIO, TINY_CAP);
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_truncated_turn_reports_length_and_cap_8, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), CONCISE_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(TINY_CAP))));
                    ecs_termination::install(&mut ecs, probe, None);
                    ecs.prompt_with_max_turns(TRUNCATING_PROMPT, true, None)
                        .await;
                })
                .await;
                assert_eq!(
                    observed.first_reason(),
                    Some(FinishReason::Length),
                    "the streaming surface must report the same reason as the blocking one"
                );
                assert_eq!(observed.first_max_tokens(), Some(TINY_CAP));
                assert_recorded_wire_reason(SCENARIO, "length");
                assert_recorded_request_cap(SCENARIO, TINY_CAP);
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_completed_turn_reports_stop_and_cap_9, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), CONCISE_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                    ecs_termination::install(&mut ecs, probe, None);
                    ecs.prompt_with_max_turns(SHORT_PROMPT, false, None).await;
                })
                .await;
                assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
                assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
                assert!(
                    !observed
                        .first_reason()
                        .is_some_and(|reason| reason.truncated_output()),
                    "a completed turn must not satisfy the retry predicate"
                );
                assert_recorded_wire_reason(SCENARIO, "stop");
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_completed_turn_reports_stop_and_cap_10, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), CONCISE_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                    ecs_termination::install(&mut ecs, probe, None);
                    ecs.prompt_with_max_turns(SHORT_PROMPT, true, None).await;
                })
                .await;
                assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
                assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
                assert_recorded_wire_reason(SCENARIO, "stop");
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_tool_turn_reports_tool_calls_11, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), TOOL_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                    ecs.tool(Adder);
                    ecs_termination::install(&mut ecs, probe, None);
                    ecs.prompt_with_max_turns(TOOL_PROMPT, false, Some(3)).await;
                })
                .await;
                assert_eq!(
                    observed.first_reason(),
                    Some(FinishReason::ToolCalls),
                    "the turn that issued the tool call must read as ToolCalls"
                );
                assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
                assert!(
                    !observed
                        .first_reason()
                        .is_some_and(|reason| reason.truncated_output()),
                    "a tool turn must not satisfy the retry predicate"
                );
                assert_recorded_wire_reason(SCENARIO, "tool_calls");
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_tool_turn_reports_tool_calls_12, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), TOOL_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                    ecs.tool(Adder);
                    ecs_termination::install(&mut ecs, probe, None);
                    ecs.prompt_with_max_turns(TOOL_PROMPT, true, Some(3)).await;
                })
                .await;
                assert_eq!(
                    observed.first_reason(),
                    Some(FinishReason::ToolCalls),
                    "streaming must resolve the tool turn exactly as blocking does"
                );
                assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
                assert_recorded_wire_reason(SCENARIO, "tool_calls");
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_escalating_retry_reports_each_attempts_own_cap_13, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
                let observed = probe.clone();
                let escalations = escalate.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), CONCISE_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(64))));
                    ecs_termination::install(&mut ecs, probe, Some(escalate));
                    ecs.prompt_with_max_turns(RETRY_PROMPT, false, Some(2))
                        .await;
                })
                .await;
                assert_eq!(
                    observed.observations(),
                    vec![
                        (Some(FinishReason::Length), Some(TINY_CAP)),
                        (Some(FinishReason::Stop), Some(ROOMY_CAP)),
                    ],
                    "each attempt must report its own post-patch cap, never the agent's baseline of 64"
                );
                assert_eq!(escalations.escalations(), vec![ROOMY_CAP]);
                assert_eq!(escalations.retries(), 1);
                assert_eq!(recorded_request_caps(SCENARIO), vec![TINY_CAP, ROOMY_CAP]);
                assert_eq!(
                    recorded_wire_reasons(SCENARIO),
                    vec!["length".to_owned(), "stop".to_owned()]
                );
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_escalating_retry_reports_each_attempts_own_cap_14, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
$crate::goldens::world_golden_test(async {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
                let observed = probe.clone();
                let escalations = escalate.clone();
                $wrapper($scenario, |client| async move {
                    let mut ecs = EcsAgent::new(client.completion(MODEL), CONCISE_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert((Temperature(Some(0.0)), MaxTokens(Some(64))));
                    ecs_termination::install(&mut ecs, probe, Some(escalate));
                    ecs.prompt_with_max_turns(RETRY_PROMPT, true, Some(2)).await;
                })
                .await;
                assert_eq!(
                    observed.observations(),
                    vec![
                        (Some(FinishReason::Length), Some(TINY_CAP)),
                        (Some(FinishReason::Stop), Some(ROOMY_CAP)),
                    ],
                    "the streaming surface must escalate and report identically to blocking"
                );
                assert_eq!(escalations.escalations(), vec![ROOMY_CAP]);
                assert_eq!(recorded_request_caps(SCENARIO), vec![TINY_CAP, ROOMY_CAP]);
            }
}, |log| $crate::goldens::world_golden_effects($golden, log)).await;
        }
    };
}

pub use ecs_termination_case;
