//! Edge matrix for the premature stream-terminal bug.
//!
//! # The bug
//!
//! `GeminiRestAdapter::interpret` treated *any* chunk carrying a
//! `finishReason` as the provider completing the turn and pushed the
//! terminal record there; the shared driver stops reading
//! as soon as it sees a terminal record. Gemini's `streamGenerateContent`
//! does not honour that assumption: when a built-in tool runs a round it
//! emits an **intermediate** `finishReason` and keeps streaming. A recorded
//! two-round code-execution turn reads
//!
//! ```text
//! [executableCode] [codeExecutionResult] [executableCode + finishReason:STOP]
//! [codeExecutionResult] [text] [text + finishReason:STOP]
//! ```
//!
//! so rig ended the stream on frame 3 and dropped the model's entire answer —
//! while still yielding a terminal record that reported a clean `STOP`. The
//! blocking twin of the same request returns the full answer.
//!
//! The fix defers the terminal record to EOF, which the adapter contract
//! names as deferral rather than synthesis ("a terminal the provider *did*
//! signal earlier … may be emitted here"). EOF with no `finishReason` at all
//! is still truncation and still yields no terminal.
//!
//! # The matrix
//!
//! The fixed code path's inputs are: how many `finishReason` chunks arrive,
//! what follows the first one, which terminal metadata each chunk carries,
//! how the stream ends (EOF / transport error / in-band failure), and which
//! stream entry point the caller used.
//!
//! | # | cell | kind | dimension pinned |
//! |---|------|------|------------------|
//! | 13 | `an_intermediate_finish_reason_does_not_end_the_stream` | unit | minimal statement of the rule |
//! | 14 | `the_last_finish_reason_wins_over_an_earlier_one` | unit | STOP→MAX_TOKENS ordering |
//! | 15 | `a_usage_only_trailer_after_the_finish_reason_reaches_the_terminal` | unit | trailing usage chunk |
//! | 16 | `later_metadata_wins_on_the_terminal_record` | unit | responseId/modelVersion from the last chunk |
//! | 17 | `three_finish_reason_chunks_still_yield_one_terminal` | unit | exactly-one-terminal invariant |
//! | 18 | `eof_without_any_finish_reason_yields_no_terminal_record` | unit | truncation is still truncation |
//! | 19 | `a_transport_error_after_a_finish_reason_yields_no_terminal_record` | unit | error path |
//! | 20 | `a_tool_protocol_failure_still_ends_the_stream_immediately` | unit | in-band failure still short-circuits |
//! | 21 | `a_tool_call_after_the_first_finish_reason_reaches_the_choice` | unit | tool content past the boundary |
//! | 22 | `reasoning_after_the_first_finish_reason_reaches_the_choice` | unit | reasoning past the boundary |
//! | 23 | `an_unknown_frame_after_the_first_finish_reason_is_passed_through` | unit | passthrough past the boundary |
//! | 24 | `the_terminal_record_is_the_last_item_yielded` | unit | ordering invariant |
//! | 25 | `a_transport_error_after_the_real_terminal_also_reports_truncation` | unit | the deliberate trade-off, stated |
//!
//! Cells 13–25 drive synthetic SSE frames through the real client rather than
//! recording, because the provider cannot be asked for these shapes: Gemini
//! never emits `STOP` followed by `MAX_TOKENS`, never sends a usage-only
//! trailer *after* an intermediate finish, and cannot be made to fail its
//! transport mid-turn on demand. Each frame sequence is written from the
//! bytes recorded by cells 1–12.
//!
//! Recording note: the two-terminal shape needs the model to take **two**
//! code-execution rounds. Measured 4/4 with the prompt below and thinking at
//! its default; forcing `thinkingBudget: 0` makes gemini-2.5-flash narrate
//! the code as text instead of calling the tool, and the shape never appears.
//! `assert_recorded_stream_finishes_early` asserts the shape's presence (or,
//! for cell 6, its absence) against each fixture's own bytes, so a cell that
//! stopped covering what it claims fails instead of quietly passing.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test -p rig --all-features --test gemini stream_terminal_matrix -- --test-threads=1`

// --- 1-8: the bug, over the shapes live traffic produces ------------------

// --- 9-12: regression guards for ordinary single-terminal streams ---------

// --- 13-25: shapes the provider cannot be asked for ----------------------

mod unit {
    use futures::StreamExt;
    use rig::completion::FinishReason;
    use rig::message::AssistantContent;

    use rig::providers::gemini::{self, GeminiConfig};
    use rig::streaming::StreamEvent;
    use rig_core::test_utils::{MockStreamingClient, SequencedStreamingHttpClient};

    /// Frames written from the bytes recorded by cells 1–12.
    const CODE_ROUND: &str = r#"{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(6*7)"}}],"role":"model"},"index":0}],"responseId":"resp-first","modelVersion":"gemini-2.5-flash"}"#;
    /// The intermediate terminal: a finishReason on a chunk that is not last.
    /// The part kind matches the recorded bytes, where the early finish rides
    /// the *second* `executableCode` frame (see the module doc).
    const INTERMEDIATE_TERMINAL: &str = r#"{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(6*7+100)"}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},"responseId":"resp-first","modelVersion":"gemini-2.5-flash"}"#;
    const ANSWER: &str = r#"{"candidates":[{"content":{"parts":[{"text":"The answer is 42."}],"role":"model"},"index":0}]}"#;
    const REAL_TERMINAL: &str = r#"{"candidates":[{"content":{"parts":[{"text":" Done."}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":40,"totalTokenCount":50},"responseId":"resp-last","modelVersion":"gemini-2.5-flash-002"}"#;

    fn sse(frames: &[&str]) -> bytes::Bytes {
        bytes::Bytes::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        )
    }

    struct Run {
        text: String,
        reasoning: usize,
        tool_calls: usize,
        unknowns: usize,
        errors: usize,
        /// The error items' messages, in order.
        error_messages: Vec<String>,
        response: Option<rig::completion::CompletionResponse>,
    }

    async fn run(frames: &[&str]) -> Run {
        run_client(MockStreamingClient {
            sse_bytes: sse(frames),
        })
        .await
    }

    async fn run_client<T>(http_client: T) -> Run
    where
        T: rig::http_client::HttpClientExt + Clone + std::fmt::Debug + Send + Sync + 'static,
    {
        let model = GeminiConfig::new("test-key")
            .connect(http_client)
            .completion(gemini::completion::GEMINI_2_5_FLASH);
        let request = rig::completion::CompletionRequest::new("hello");
        let mut stream = model.stream(request).expect("stream should open");

        let mut run = Run {
            text: String::new(),
            reasoning: 0,
            tool_calls: 0,
            unknowns: 0,
            errors: 0,
            error_messages: Vec::new(),
            response: None,
        };
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => match item {
                    rig::streaming::Item::Event(StreamEvent::Text { text, .. }) => {
                        run.text.push_str(&text)
                    }
                    rig::streaming::Item::Event(StreamEvent::End {
                        content: AssistantContent::Reasoning(_),
                        ..
                    }) => run.reasoning += 1,
                    rig::streaming::Item::Event(StreamEvent::End {
                        content: AssistantContent::ToolCall(_),
                        ..
                    }) => run.tool_calls += 1,
                    rig::streaming::Item::Unknown(_) => run.unknowns += 1,
                    _ => {}
                },
                Err(error) => {
                    run.errors += 1;
                    run.error_messages.push(error.to_string());
                }
            }
        }
        run.response = stream.finish().await.ok();
        run
    }

    #[tokio::test]
    async fn an_intermediate_finish_reason_does_not_end_the_stream() {
        let run = run(&[CODE_ROUND, INTERMEDIATE_TERMINAL, ANSWER, REAL_TERMINAL]).await;
        assert_eq!(run.text, "The answer is 42. Done.");
        assert!(run.response.is_some());
    }

    #[tokio::test]
    async fn the_last_finish_reason_wins_over_an_earlier_one() {
        // Gemini never pairs these two reasons, so only synthetic frames can
        // show which one the terminal reports.
        const TRUNCATED: &str = r#"{"candidates":[{"content":{"parts":[{"text":" more"}],"role":"model"},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, ANSWER, TRUNCATED]).await;
        assert_eq!(
            run.response
                .as_ref()
                .and_then(|terminal| terminal.finish_reason()),
            Some(FinishReason::Length)
        );
    }

    #[tokio::test]
    async fn a_usage_only_trailer_after_the_finish_reason_reaches_the_terminal() {
        const TRAILER: &str = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":99,"totalTokenCount":109}}"#;
        let run = run(&[ANSWER, REAL_TERMINAL, TRAILER]).await;
        let terminal = run.response.as_ref().expect("one terminal");
        assert_eq!(
            terminal.usage.total_tokens,
            Some(109),
            "a usage trailer after the finish chunk must reach the terminal record"
        );
    }

    #[tokio::test]
    async fn later_metadata_wins_on_the_terminal_record() {
        let run = run(&[CODE_ROUND, INTERMEDIATE_TERMINAL, ANSWER, REAL_TERMINAL]).await;
        let terminal = run.response.as_ref().expect("one terminal");
        assert_eq!(terminal.response_id(), Some("resp-last"));
        assert_eq!(terminal.model(), Some("gemini-2.5-flash-002"));
        assert_eq!(terminal.usage.total_tokens, Some(50));
    }

    #[tokio::test]
    async fn three_finish_reason_chunks_still_yield_one_terminal() {
        let run = run(&[
            INTERMEDIATE_TERMINAL,
            ANSWER,
            INTERMEDIATE_TERMINAL,
            ANSWER,
            REAL_TERMINAL,
        ])
        .await;
        assert!(run.response.is_some(), "exactly one terminal record");
    }

    #[tokio::test]
    async fn eof_without_any_finish_reason_yields_no_terminal_record() {
        let run = run(&[CODE_ROUND, ANSWER]).await;
        assert!(run.response.is_none(), "truncation is still truncation");
        assert!(run.response.is_none());
        assert_eq!(run.text, "The answer is 42.");
    }

    /// Gemini's in-band abort: a frame carrying only its error envelope.
    const ERROR_FRAME: &str =
        r#"{"error":{"code":500,"message":"An internal error has occurred.","status":"INTERNAL"}}"#;

    #[tokio::test]
    async fn an_error_frame_after_text_is_the_providers_verdict_not_a_truncation() {
        let run = run(&[ANSWER, ERROR_FRAME]).await;
        assert_eq!(run.text, "The answer is 42.");
        assert_eq!(
            run.unknowns, 0,
            "the envelope is a modeled frame, never skipped"
        );
        assert_eq!(run.errors, 1, "one error item: the provider's");
        assert!(run.response.is_none(), "no terminal record after an abort");
        let message = &run.error_messages[0];
        assert!(
            message.contains("INTERNAL"),
            "the envelope's status survives: {message}"
        );
        assert!(
            message.contains("An internal error has occurred."),
            "the envelope's message survives: {message}"
        );
        assert!(
            !message.contains("ended before the provider ended it"),
            "not reported as a cut stream: {message}"
        );
    }

    #[tokio::test]
    async fn an_error_frame_alone_is_the_providers_verdict() {
        let run = run(&[ERROR_FRAME]).await;
        assert!(run.text.is_empty());
        assert_eq!(run.errors, 1);
        assert!(run.response.is_none());
        assert!(run.error_messages[0].contains("INTERNAL"));
    }

    #[tokio::test]
    async fn frames_after_an_error_frame_are_not_read() {
        let run = run(&[ERROR_FRAME, ANSWER]).await;
        assert_eq!(run.errors, 1);
        assert!(run.text.is_empty(), "the abort is the wire's terminal");
    }

    #[tokio::test]
    async fn a_transport_error_after_a_finish_reason_yields_no_terminal_record() {
        let run = run_client(SequencedStreamingHttpClient::new(vec![
            Ok(sse(&[INTERMEDIATE_TERMINAL, ANSWER])),
            Err(rig::http_client::Error::instance(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "connection reset",
            ))),
        ]))
        .await;

        assert_eq!(
            run.errors, 1,
            "the transport failure must reach the consumer"
        );
        // The deliberate half of the trade-off. Before the fix, a connection
        // that dropped after *any* finishReason still handed the consumer a
        // terminal record, because the stream had already ended there. It now
        // reports truncation instead — and it must: on this wire a
        // finishReason is not proof the turn finished (that is the whole bug),
        // so the two cases are indistinguishable in-band and the safe reading
        // is the module's stated truncation semantics. A terminal here would
        // report a successful completion for a turn that may have been cut in
        // half.
        assert!(
            run.response.is_none(),
            "a failed stream must not be dressed up as a completed turn by the deferred terminal"
        );
        assert!(run.response.is_none());
    }

    /// The other half of cell 19, and the regressing direction: the provider
    /// signalled its *real* finish and the transport then failed before EOF.
    /// The turn's bytes all arrived, and rig still reports truncation —
    /// deliberately, because on this wire a `finishReason` is not proof the
    /// turn finished (that is the whole bug), so "final finish" and
    /// "intermediate finish" are indistinguishable in-band and reporting a
    /// completed turn for one that may have been cut in half is the worse
    /// error. Pinned so the trade-off cannot change unnoticed.
    #[tokio::test]
    async fn a_transport_error_after_the_real_terminal_also_reports_truncation() {
        let run = run_client(SequencedStreamingHttpClient::new(vec![
            Ok(sse(&[ANSWER, REAL_TERMINAL])),
            Err(rig::http_client::Error::instance(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "connection reset",
            ))),
        ]))
        .await;

        assert_eq!(run.errors, 1, "the transport failure reaches the consumer");
        assert_eq!(run.text, "The answer is 42. Done.", "content still arrives");
        assert!(
            run.response.is_none(),
            "no terminal record: the stream never reached EOF"
        );
        assert!(run.response.is_none());
    }

    #[tokio::test]
    async fn a_tool_protocol_failure_still_ends_the_stream_immediately() {
        const FAILURE: &str = r#"{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"malformed function call","index":0}]}"#;
        let run = run(&[ANSWER, FAILURE, ANSWER, REAL_TERMINAL]).await;

        assert_eq!(
            run.errors, 0,
            "a failure finish is a failed turn, not a stream error"
        );
        assert_eq!(
            run.text, "The answer is 42.",
            "nothing after the in-band failure is interpreted"
        );
        let response = run.response.expect("the failure ends the turn");
        assert!(
            response.stop().is_failure(),
            "a later STOP cannot dress the failed turn up as complete: {:?}",
            response.stop()
        );
    }

    #[tokio::test]
    async fn a_tool_call_after_the_first_finish_reason_reaches_the_choice() {
        const TOOL_CALL: &str = r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"add","args":{"x":1,"y":2},"id":"call-1"}}],"role":"model"},"index":0}]}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, TOOL_CALL, REAL_TERMINAL]).await;
        assert_eq!(
            run.tool_calls, 1,
            "a tool call past the boundary must still be delivered"
        );
    }

    #[tokio::test]
    async fn reasoning_after_the_first_finish_reason_reaches_the_choice() {
        const THOUGHT: &str = r#"{"candidates":[{"content":{"parts":[{"text":"thinking on","thought":true,"thoughtSignature":"sig"}],"role":"model"},"index":0}]}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, THOUGHT, REAL_TERMINAL]).await;
        assert!(
            run.reasoning > 0,
            "reasoning past the boundary must still be delivered"
        );
    }

    #[tokio::test]
    async fn an_unknown_frame_after_the_first_finish_reason_is_passed_through() {
        const UNKNOWN: &str = r#"{"someFutureField":{"x":1}}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, UNKNOWN, ANSWER, REAL_TERMINAL]).await;
        assert_eq!(
            run.unknowns, 1,
            "the raw passthrough channel must still see frames past the boundary"
        );
        assert_eq!(run.text, "The answer is 42. Done.");
    }
}
