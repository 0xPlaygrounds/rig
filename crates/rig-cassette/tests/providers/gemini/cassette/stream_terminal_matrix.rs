//! Gemini stream terminals: an intermediate `finishReason` does not end the
//! stream.
//!
//! Gemini's `streamGenerateContent` emits an intermediate `finishReason`
//! when a built-in tool runs a round, then keeps streaming. A two-round
//! code-execution turn reads:
//!
//! ```text
//! [executableCode] [codeExecutionResult] [executableCode + finishReason:STOP]
//! [codeExecutionResult] [text] [text + finishReason:STOP]
//! ```
//!
//! So the terminal record waits for EOF (`GenerateContentDecoder::eof`), where
//! the last finish reason and metadata win, and exactly one terminal is
//! yielded last. EOF with no `finishReason`, or a transport error after one,
//! is truncation and yields no terminal. These cells drive synthetic SSE
//! frames, modelled on recorded Gemini streams, through the real client,
//! since the provider cannot be asked for these shapes on demand.

mod unit {
    use futures::StreamExt;
    use rig::completion::FinishReason;
    use rig::message::AssistantContent;

    use rig::providers::gemini::{self, GeminiConfig};
    use rig::streaming::StreamEvent;
    use rig_core::test_utils::{MockStreamingClient, SequencedStreamingHttpClient};

    /// Frames written from the bytes of recorded Gemini streams.
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
            Err(rig::http_client::Error::transient(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "connection reset",
            ))),
        ]))
        .await;

        assert_eq!(
            run.errors, 1,
            "the transport failure must reach the consumer"
        );
        // On this wire a finishReason is not proof the turn finished, so a
        // terminal here could report success for a turn cut in half.
        assert!(
            run.response.is_none(),
            "a failed stream must not be dressed up as a completed turn by the deferred terminal"
        );
        assert!(run.response.is_none());
    }

    /// The provider signalled its real finish and the transport then failed
    /// before EOF. Rig still reports truncation: a final and an intermediate
    /// finish are indistinguishable in band, and reporting a completed turn
    /// for one cut in half is the worse error.
    #[tokio::test]
    async fn a_transport_error_after_the_real_terminal_also_reports_truncation() {
        let run = run_client(SequencedStreamingHttpClient::new(vec![
            Ok(sse(&[ANSWER, REAL_TERMINAL])),
            Err(rig::http_client::Error::transient(std::io::Error::new(
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
