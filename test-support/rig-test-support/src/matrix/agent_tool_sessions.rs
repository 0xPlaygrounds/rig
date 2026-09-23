//! Shared agent tool sessions bodies; each wire supplies explicit scenario rows.
//!
//! The invoking module defines `SESSION_MODEL` and `SESSION_MAX_TOKENS`, an
//! `Option<u64>` output cap for wires whose rate limiter rejects the default.

/// The session agent for `client`: `SESSION_MODEL`, capped at
/// `SESSION_MAX_TOKENS` when the invoking wire sets one.
#[doc(hidden)]
#[macro_export]
macro_rules! session_agent {
    ($client:expr) => {{
        let agent = $client.agent(SESSION_MODEL);
        match SESSION_MAX_TOKENS {
            Some(max_tokens) => agent.max_tokens(max_tokens),
            None => agent,
        }
    }};
}

/// Emit registered test rows with the shared execution body.
#[macro_export]
macro_rules! agent_tool_sessions_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, sequential_complex_tool_calls_nonstreaming_0) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let log = Arc::new(Mutex::new(Vec::new()));
                let (ping, manifest, labels, echo) = complex_tools(&log);
                let agent = $crate::matrix::session_agent!(client)
                    .preamble(COMPLEX_SESSION_PREAMBLE)
                    .tool(ping)
                    .tool(manifest)
                    .tool(labels)
                    .tool(echo)
                    .additional_params(json ! ({ "parallel_tool_calls" : false }))
                    .default_max_turns(10)
                    .build();
                let mut history = Vec::<Message>::new();
                let response = agent.chat(COMPLEX_SESSION_PROMPT, &mut history).await?;
                assert_contains_all_case_insensitive(
                    &response.output,
                    &["EMPTY-OK", "MANIFEST-OK", "LABELS-OK", "ESCAPE-OK"],
                );
                assert_complex_invocations(&log);
                assert_history_records_sequential_tool_roundtrips(
                    &history,
                    &[
                        PingEmpty::NAME,
                        InspectManifest::NAME,
                        JoinLabels::NAME,
                        EscapeEcho::NAME,
                    ],
                );
                Ok(())
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, sequential_complex_tool_calls_nonstreaming_1) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let log = Arc::new(Mutex::new(Vec::new()));
                let (ping, manifest, labels, optional, echo) = complex_tools(&log);
                let agent = $crate::matrix::session_agent!(client)
                    .preamble(COMPLEX_SESSION_PREAMBLE)
                    .tool(ping)
                    .tool(manifest)
                    .tool(labels)
                    .tool(optional)
                    .tool(echo)
                    .additional_params(json ! ({ "parallel_tool_calls" : false }))
                    .default_max_turns(10)
                    .build();
                let mut history = Vec::<Message>::new();
                let response = agent.chat(COMPLEX_SESSION_PROMPT, &mut history).await?;
                assert_contains_all_case_insensitive(
                    &response.output,
                    &[
                        "EMPTY-OK",
                        "MANIFEST-OK",
                        "LABELS-OK",
                        "OPTIONAL-OK",
                        "ESCAPE-OK",
                    ],
                );
                assert_complex_invocations(&log);
                assert_history_records_sequential_tool_roundtrips(
                    &history,
                    &[
                        PingEmpty::NAME,
                        InspectManifest::NAME,
                        JoinLabels::NAME,
                        OptionalNullableProbe::NAME,
                        EscapeEcho::NAME,
                    ],
                );
                Ok(())
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, sequential_complex_tool_calls_streaming_2) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let log = Arc::new(Mutex::new(Vec::new()));
                let (ping, manifest, labels, optional, echo) = complex_tools(&log);
                let agent = $crate::matrix::session_agent!(client)
                    .preamble(COMPLEX_SESSION_PREAMBLE)
                    .tool(ping)
                    .tool(manifest)
                    .tool(labels)
                    .tool(optional)
                    .tool(echo)
                    .additional_params(json ! ({ "parallel_tool_calls" : false }))
                    .build();
                let mut stream = agent
                    .prompt(COMPLEX_SESSION_PROMPT)
                    .history(Vec::<Message>::new())
                    .max_turns(10)
                    .stream();
                let observation = collect_stream_observation(&mut stream).await;
                anyhow::ensure!(
                    observation.errors.is_empty(),
                    "stream should not emit errors: {:?}",
                    observation.errors
                );
                anyhow::ensure!(
                    observation.tool_calls
                        == vec![
                            PingEmpty::NAME.to_string(),
                            InspectManifest::NAME.to_string(),
                            JoinLabels::NAME.to_string(),
                            OptionalNullableProbe::NAME.to_string(),
                            EscapeEcho::NAME.to_string(),
                        ],
                    "stream should expose ordered tool calls, saw {:?}",
                    observation.tool_calls
                );
                anyhow::ensure!(
                    observation.tool_results == 5,
                    "expected 5 streamed tool results"
                );
                let response = observation
                    .final_response_text
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("stream should produce final response text"))?;
                assert_contains_all_case_insensitive(
                    response,
                    &[
                        "EMPTY-OK",
                        "MANIFEST-OK",
                        "LABELS-OK",
                        "OPTIONAL-OK",
                        "ESCAPE-OK",
                    ],
                );
                assert_complex_invocations(&log);
                Ok(())
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, parallel_tool_calls_single_turn_nonstreaming_3) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let agent = $crate::matrix::session_agent!(client)
                    .preamble(TWO_TOOL_STREAM_PREAMBLE)
                    .tool(AlphaSignal)
                    .tool(BetaSignal)
                    .additional_params(json ! ({ "parallel_tool_calls" : true }))
                    .default_max_turns(5)
                    .build();
                let mut history = Vec::<Message>::new();
                let response = agent.chat(TWO_TOOL_STREAM_PROMPT, &mut history).await?;
                assert_contains_all_case_insensitive(
                    &response.output,
                    &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
                );
                let calls = history_tool_calls(&history);
                let call_names = calls
                    .iter()
                    .map(|call| call.name.as_str())
                    .collect::<Vec<_>>();
                anyhow::ensure!(
                    calls.len() == 2
                        && call_names.contains(&AlphaSignal::NAME)
                        && call_names.contains(&BetaSignal::NAME),
                    "expected both zero-argument tools, saw {call_names:?}"
                );
                anyhow::ensure!(
                    calls[0].message_index == calls[1].message_index,
                    "parallel tool calls should be recorded on one assistant message"
                );
                anyhow::ensure!(
                    history_tool_results(&history).len() == 2,
                    "expected two tool results"
                );
                Ok(())
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, parallel_tool_calls_single_turn_streaming_4) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper($scenario, |client| async move {
                let agent = $crate::matrix::session_agent!(client)
                    .preamble(TWO_TOOL_STREAM_PREAMBLE)
                    .tool(AlphaSignal)
                    .tool(BetaSignal)
                    .additional_params(json ! ({ "parallel_tool_calls" : true }))
                    .build();
                let mut stream = agent.prompt(TWO_TOOL_STREAM_PROMPT).max_turns(5).stream();
                let observation = collect_stream_observation(&mut stream).await;
                assert_two_tool_roundtrip_contract(
                    &observation,
                    &[AlphaSignal::NAME, BetaSignal::NAME],
                    &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
                );
                Ok(())
            })
            .await
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, long_history_replay_with_tool_result_continuation_5) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            $wrapper ($scenario , | client | async move { let model = client . completion (SESSION_MODEL) ; let tool_call_id = "call_REDACTED_1" ; let request = model . completion_request ("Answer in one short sentence: what is my favorite color, which label came from the tool, and which release lane did I choose? Do not call any tools." ,) . preamble ("You are concise and should rely on the provided chat history." . to_string ()) . message (Message :: user ("My favorite color is teal. Please remember it.")) . message (Message :: assistant ("Noted: your favorite color is teal.")) . message (Message :: user ("For this release, use the canary lane.")) . message (Message :: assistant ("Understood: the release lane is canary.")) . message (Message :: user ("Look up the harbor label with the tool.")) . message (Message :: Assistant { id : None , content : vec ! [AssistantContent :: tool_call (tool_call_id , AlphaSignal :: NAME , json ! ({ }) ,)] , }) . message (Message :: tool_result (tool_call_id , AlphaSignal :: NAME , ALPHA_SIGNAL_OUTPUT ,)) . message (Message :: assistant ("The harbor label is crimson-harbor.")) . tool (rig :: tool :: tool_definition (& AlphaSignal)) . tool_choice (ToolChoice :: None) . max_tokens (SESSION_MAX_TOKENS) . build () ; let (raw , response) = raw_and_normalized_completion (& model , request) . await ? ; let text = assistant_text_response (& response . choice) . ok_or_else (| | anyhow :: anyhow ! ("response should include assistant text")) ? ; assert_contains_all_case_insensitive (& text , & ["teal" , ALPHA_SIGNAL_OUTPUT , "canary"]) ; assert_response_metadata (& response , & raw) ; Ok (()) } ,) . await
        }
    };
}

pub use agent_tool_sessions_case;
pub use session_agent;
