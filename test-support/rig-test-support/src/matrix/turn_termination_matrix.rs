//! Shared turn termination matrix bodies; each wire supplies explicit scenario rows.

/// Emit registered test rows with the shared execution body.
#[macro_export]
macro_rules! turn_termination_matrix_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_truncated_turn_reports_length_and_cap_15) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    {
                        client
                            .agent(MODEL)
                            .preamble(CONCISE_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(TINY_CAP)
                            .add_hook(probe)
                            .build()
                            .prompt(TRUNCATING_PROMPT)
                            .run()
                            .await
                            .expect("a partially truncated turn still carries an answer");
                    }
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
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_truncated_turn_reports_length_and_cap_16) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    {
                        let agent = client
                            .agent(MODEL)
                            .preamble(CONCISE_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(TINY_CAP)
                            .build();
                        let mut stream = agent.prompt(TRUNCATING_PROMPT).add_hook(probe).stream();
                        let _ = collect_stream_final_response(&mut stream).await;
                    }
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
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_completed_turn_reports_stop_and_cap_17) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    {
                        client
                            .agent(MODEL)
                            .preamble(CONCISE_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(ROOMY_CAP)
                            .add_hook(probe)
                            .build()
                            .prompt(SHORT_PROMPT)
                            .run()
                            .await
                            .expect("a short answer under a roomy cap");
                    }
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
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_completed_turn_reports_stop_and_cap_18) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    {
                        let agent = client
                            .agent(MODEL)
                            .preamble(CONCISE_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(ROOMY_CAP)
                            .build();
                        let mut stream = agent.prompt(SHORT_PROMPT).add_hook(probe).stream();
                        let _ = collect_stream_final_response(&mut stream).await;
                    }
                })
                .await;
                assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
                assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
                assert_recorded_wire_reason(SCENARIO, "stop");
            }
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_tool_turn_reports_tool_calls_19) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    {
                        client
                            .agent(MODEL)
                            .preamble(TOOL_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(ROOMY_CAP)
                            .tool(Adder)
                            .add_hook(probe)
                            .build()
                            .prompt(TOOL_PROMPT)
                            .max_turns(3)
                            .run()
                            .await
                            .expect("the tool turn should complete the run");
                    }
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
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_tool_turn_reports_tool_calls_20) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();
                $wrapper($scenario, |client| async move {
                    {
                        let agent = client
                            .agent(MODEL)
                            .preamble(TOOL_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(ROOMY_CAP)
                            .tool(Adder)
                            .build();
                        let mut stream = agent
                            .prompt(TOOL_PROMPT)
                            .add_hook(probe)
                            .max_turns(3)
                            .stream();
                        let _ = collect_stream_final_response(&mut stream).await;
                    }
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
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, blocking_escalating_retry_reports_each_attempts_own_cap_21) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
                let observed = probe.clone();
                let escalations = escalate.clone();
                $wrapper($scenario, |client| async move {
                    {
                        client
                            .agent(MODEL)
                            .preamble(CONCISE_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(64)
                            .add_hook(probe)
                            .add_hook(escalate)
                            .build()
                            .prompt(RETRY_PROMPT)
                            .max_turns(2)
                            .run()
                            .await
                            .expect("the retried attempt should answer");
                    }
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
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, streaming_escalating_retry_reports_each_attempts_own_cap_22) => {
        $(#[$attribute])*
        async fn $name() {
            {
                const SCENARIO: &str = $scenario;
                let probe = TurnTerminationProbe::default();
                let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
                let observed = probe.clone();
                let escalations = escalate.clone();
                $wrapper($scenario, |client| async move {
                    {
                        let agent = client
                            .agent(MODEL)
                            .preamble(CONCISE_PREAMBLE)
                            .temperature(0.0)
                            .max_tokens(64)
                            .build();
                        let mut stream = agent
                            .prompt(RETRY_PROMPT)
                            .add_hook(probe)
                            .add_hook(escalate)
                            .max_turns(2)
                            .stream();
                        let _ = collect_stream_final_response(&mut stream).await;
                    }
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
        }
    };
}

pub use turn_termination_matrix_case;
