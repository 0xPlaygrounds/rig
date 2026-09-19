use super::*;

fn citation(n: u64) -> serde_json::Value {
    json!({"type":"url_citation", "url":format!("https://example.com/{n}"),
        "title":format!("Source {n}"), "start_index":0, "end_index":5})
}

fn message(id: &str, annotations: serde_json::Value) -> serde_json::Value {
    json!({"type":"message", "id":id, "role":"assistant", "status":"completed",
        "content":[{"type":"output_text", "text":"hello", "annotations":annotations}]})
}

fn delta(id: Option<&str>) -> serde_json::Value {
    json!({"type":"response.output_text.delta", "item_id":id,
        "output_index":0, "content_index":0, "sequence_number":1, "delta":"hello"})
}

fn annotation(id: &str, n: u64) -> serde_json::Value {
    json!({"type":"response.output_text.annotation.added", "item_id":id,
        "output_index":0, "content_index":0, "sequence_number":2,
        "annotation_index":0, "annotation":citation(n)})
}

fn fold(
    mut frames: Vec<serde_json::Value>,
    output: Vec<serde_json::Value>,
) -> crate::completion::CompletionResponse {
    let mut response = sample_response(ResponseStatus::Completed);
    response.output = output
        .into_iter()
        .map(|v| serde_json::from_value(v).expect("valid item"))
        .collect();
    frames.push(json!({"type":"response.completed", "sequence_number":10, "response":response}));
    let body = frames
        .iter()
        .map(|v| format!("data: {v}\n"))
        .collect::<String>();
    let events = stream_events_from_sse_body("openai", &body, None).expect("buffered decode");
    folded_stream_events("openai", events, &response).expect("fold")
}

fn parts(response: &crate::completion::CompletionResponse) -> Vec<(String, serde_json::Value)> {
    response
        .choice
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(text) => Some((
                text.text.clone(),
                text.additional_params
                    .as_ref()
                    .and_then(|p| p.get("openai_responses"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            )),
            _ => None,
        })
        .collect()
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn idless_delta() {
    let response = fold(
        vec![delta(None)],
        vec![message("msg_1", json!([citation(1)]))],
    );
    assert_eq!(
        parts(&response),
        vec![("hello".into(), json!({"annotations":[citation(1)]}))]
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn missing_annotation_index() {
    let mut added = annotation("msg_1", 1);
    added
        .as_object_mut()
        .expect("object")
        .remove("annotation_index");
    let response = fold(
        vec![delta(Some("msg_1")), added],
        vec![message("msg_1", json!([citation(1)]))],
    );
    assert_eq!(
        parts(&response),
        vec![("hello".into(), json!({"annotations":[citation(1)]}))]
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn repaired_output_indexes_collide() {
    let mut frames = vec![
        delta(Some("msg_1")),
        annotation("msg_1", 1),
        delta(Some("msg_2")),
        annotation("msg_2", 2),
        json!({"type":"response.content_part.done", "item_id":"msg_1",
            "content_index":0, "sequence_number":5,
            "part":{"type":"output_text", "text":"hello", "annotations":[citation(1)]}}),
        json!({"type":"response.content_part.done", "item_id":"msg_2",
            "content_index":0, "sequence_number":6,
            "part":{"type":"output_text", "text":"hello", "annotations":[citation(2)]}}),
    ];
    for frame in &mut frames {
        frame
            .as_object_mut()
            .expect("object")
            .remove("output_index");
    }
    let response = fold(frames, vec![]);
    assert_eq!(
        parts(&response),
        vec![
            ("hello".into(), json!({"annotations":[citation(1)]})),
            ("hello".into(), json!({"annotations":[citation(2)]})),
        ]
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn snapshot_position_shift() {
    let response = fold(
        vec![delta(Some("msg_1")), annotation("msg_1", 1)],
        vec![
            json!({"type":"web_search_call", "id":"ws_1", "status":"completed"}),
            message("msg_1", json!([citation(1)])),
        ],
    );
    assert_eq!(
        parts(&response),
        vec![("hello".into(), json!({"annotations":[citation(1)]}))]
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn unary_non_array_annotations() {
    let mut response = sample_response(ResponseStatus::Completed);
    response.output =
        vec![serde_json::from_value(message("msg_1", json!({"future":"value"}))).expect("item")];
    let mut accumulator = RawChoiceAccumulator::new("openai", None);
    let mut output = AdapterOutput::default();
    accumulator.replay_whole_response(response.clone(), &mut output);
    let events = output
        .into_items()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("events");
    let result = folded_stream_events("openai", events, &response).expect("fold");
    assert_eq!(
        parts(&result),
        vec![("hello".into(), json!({"annotations":{"future":"value"}}))]
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn done_then_changed_terminal_id() {
    let response = fold(
        vec![json!({"type":"response.output_item.done", "output_index":0,
        "sequence_number":1, "item":message("msg_old", json!([citation(1)]))})],
        vec![message("msg_new", json!([citation(1)]))],
    );
    assert_eq!(
        parts(&response),
        vec![("hello".into(), json!({"annotations":[citation(1)]}))]
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn phase_and_logprobs_snapshot() {
    let mut item = message("msg_1", json!([citation(1)]));
    item["phase"] = json!("final_answer");
    item["content"][0]["logprobs"] = json!([{"token":"hello", "logprob":-0.1}]);
    let response = fold(vec![delta(Some("msg_1"))], vec![item]);
    assert_eq!(parts(&response)[0].1["phase"], "final_answer");
    assert_eq!(
        parts(&response)[0].1["logprobs"],
        json!([{"token":"hello", "logprob":-0.1}])
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn snapshot_fills_annotation_gap_in_order() {
    let mut added = annotation("msg_1", 2);
    added["annotation_index"] = json!(1);
    let response = fold(
        vec![delta(Some("msg_1")), added],
        vec![message("msg_1", json!([citation(1), citation(2)]))],
    );
    assert_eq!(
        parts(&response),
        vec![(
            "hello".into(),
            json!({"annotations":[citation(1), citation(2)]})
        )]
    );
}

/// Synthetic regression for a replay/identity edge not covered by recorded citations.
#[test]
fn annotation_offsets_are_relative_to_their_content_part() {
    let mut item = message("msg_1", json!([citation(1)]));
    item["content"]
        .as_array_mut()
        .expect("content")
        .push(json!({"type":"output_text", "text":"world", "annotations":[citation(2)]}));
    let mut second_delta = delta(Some("msg_1"));
    second_delta["content_index"] = json!(1);
    second_delta["delta"] = json!("world");
    let mut second_annotation = annotation("msg_1", 2);
    second_annotation["content_index"] = json!(1);
    let response = fold(
        vec![
            delta(Some("msg_1")),
            annotation("msg_1", 1),
            second_delta,
            second_annotation,
        ],
        vec![item],
    );
    // Each annotation spans 0..5 in its own content part. Keeping the parts
    // separate preserves those offsets without rewriting the wire payload.
    assert_eq!(
        parts(&response),
        vec![
            ("hello".into(), json!({"annotations":[citation(1)]})),
            ("world".into(), json!({"annotations":[citation(2)]})),
        ]
    );
}

/// Exercises the strict streaming transport with synthetic annotation frames.
#[tokio::test]
async fn valid_annotation_through_live_strict_decoder() {
    for provider in [
        OpenAI::new("test-key"),
        OpenAI::with_key(&crate::providers::xai::DIALECT, "test-key"),
    ] {
        let mut raw = sample_response(ResponseStatus::Completed);
        let mut item = message("msg_1", json!([citation(1)]));
        item["content"][0]["text"] = json!("hello!");
        raw.output = vec![serde_json::from_value(item).expect("item")];
        let events = [
            delta(Some("msg_1")),
            annotation("msg_1", 1),
            json!({"type":"response.output_text.delta", "item_id":"msg_1",
                "output_index":0, "content_index":0, "sequence_number":3, "delta":"!"}),
            json!({"type":"response.content_part.done", "item_id":"msg_1",
                "output_index":0, "content_index":0, "sequence_number":4,
                "part":{"type":"output_text", "text":"hello!", "annotations":[citation(1)]}}),
            json!({"type":"response.completed", "sequence_number":5, "response":raw}),
        ];
        let model = Bound::new(
            provider.responses("test-model"),
            MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&events),
            },
        );
        let mut stream = model
            .stream(model.completion_request("hello").build())
            .await
            .expect("stream");
        let mut metadata_count = 0;
        let mut saw_last_delta = false;
        while let Some(event) = stream.next().await {
            let event = event.expect("valid strict event");
            if matches!(
                &event,
                StreamEvent::BlockDelta {
                    delta: Delta::TextMeta { .. },
                    ..
                }
            ) {
                assert!(
                    !saw_last_delta,
                    "citation must precede the final text delta"
                );
                metadata_count += 1;
            }
            if matches!(&event, StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if text == "!")
            {
                saw_last_delta = true;
            }
        }
        assert!(saw_last_delta);
        assert_eq!(metadata_count, 1, "snapshots must not repeat the citation");
    }
}

/// Replay-only repair must not mask malformed live frames.
#[test]
fn strict_decoder_rejects_missing_annotation_index() {
    let mut added = annotation("msg_1", 1);
    added
        .as_object_mut()
        .expect("object")
        .remove("annotation_index");
    assert!(matches!(
        classify_responses_frame(&added.to_string()),
        WireEvent::Corrupt(_)
    ));
}

/// Synthetic replay frames have no annotation positions or final snapshot.
#[test]
fn missing_annotation_indexes_preserve_all_incremental_citations() {
    let mut frames = vec![
        delta(Some("msg_1")),
        annotation("msg_1", 1),
        annotation("msg_1", 2),
    ];
    for frame in &mut frames[1..] {
        frame
            .as_object_mut()
            .expect("object")
            .remove("annotation_index");
    }
    let response = fold(frames, vec![]);
    assert_eq!(
        parts(&response),
        vec![(
            "hello".into(),
            json!({"annotations":[citation(1), citation(2)]})
        )]
    );
}

/// A replay can omit both correlators on text, then identify its annotation.
#[test]
fn annotation_names_an_anonymous_replay_part() {
    let mut fragment = delta(None);
    fragment
        .as_object_mut()
        .expect("object")
        .remove("output_index");
    let response = fold(
        vec![fragment, annotation("msg_1", 1)],
        vec![message("msg_1", json!([citation(1)]))],
    );
    assert_eq!(
        parts(&response),
        vec![("hello".into(), json!({"annotations":[citation(1)]}))]
    );
}

fn drive(frames: Vec<serde_json::Value>) -> Vec<Result<StreamEvent, CompletionError>> {
    let mut driver: WireDriver<Completion, _> = WireDriver::new(
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()).with_envelope_repair(),
    );
    let mut events = Vec::new();
    for frame in frames {
        driver.push(WireFrame::Text(frame.to_string()));
        events.extend(driver.drain());
    }
    driver.finish();
    events.extend(driver.drain());
    events
}

fn metadata(events: &[Result<StreamEvent, CompletionError>]) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|event| match event {
            Ok(StreamEvent::BlockDelta {
                delta: Delta::TextMeta { additional_params },
                ..
            }) => additional_params.get("openai_responses").cloned(),
            _ => None,
        })
        .collect()
}

#[test]
fn conflicting_snapshots_fail_without_final_or_later_text() {
    for annotations in [json!([]), json!([citation(2)])] {
        let events = drive(vec![
            delta(Some("msg_1")),
            annotation("msg_1", 1),
            json!({"type":"response.output_item.done", "output_index":0,
                "sequence_number":3, "item":message("msg_1", annotations)}),
            delta(Some("msg_1")),
            json!({"type":"response.completed", "sequence_number":5,
                "response":sample_response(ResponseStatus::Completed)}),
        ]);
        assert!(events.last().is_some_and(Result::is_err));
        assert_eq!(events.iter().filter(|event| event.is_err()).count(), 1);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(StreamEvent::Final(_))))
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    Ok(StreamEvent::BlockDelta {
                        delta: Delta::Text { .. },
                        ..
                    })
                ))
                .count(),
            1
        );
    }
}

#[test]
fn conflicting_incremental_annotation_fails() {
    let events = drive(vec![annotation("msg_1", 1), annotation("msg_1", 2)]);
    assert!(events.last().is_some_and(Result::is_err));
    assert_eq!(
        metadata(&events),
        vec![json!({"annotations":[citation(1)]})]
    );
}

/// Synthetic fragments pin block identity independently of token boundaries.
/// Recorded Responses effect-log tests also check the exact event sequence.
#[test]
fn text_fragments_and_late_metadata_start_their_block_only_once() {
    let events = drive(vec![
        delta(Some("msg_1")),
        delta(Some("msg_1")),
        annotation("msg_1", 1),
        json!({"type":"response.completed", "sequence_number":4,
            "response":sample_response(ResponseStatus::Completed)}),
    ]);
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                Ok(StreamEvent::BlockStart {
                    kind: BlockKind::Text { .. },
                    ..
                })
            ))
            .count(),
        1
    );
    assert_eq!(
        metadata(&events),
        vec![json!({"annotations":[citation(1)]})]
    );
}

#[test]
fn out_of_order_annotations_wait_for_gap_and_ignore_repeats() {
    let mut second = annotation("msg_1", 2);
    second["annotation_index"] = json!(1);
    let events = drive(vec![
        second.clone(),
        annotation("msg_1", 1),
        second,
        annotation("msg_1", 1),
        json!({"type":"response.completed", "sequence_number":5,
            "response":sample_response(ResponseStatus::Completed)}),
    ]);
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        metadata(&events),
        vec![json!({"annotations":[citation(1), citation(2)]})]
    );
}

#[test]
fn pending_annotations_flush_before_eof_and_provider_errors() {
    let mut indexed = annotation("msg_1", 1);
    indexed["annotation_index"] = json!(3);
    let mut unindexed = annotation("msg_1", 2);
    unindexed
        .as_object_mut()
        .expect("object")
        .remove("annotation_index");
    for terminal in [
        None,
        Some(json!({"type":"error", "code":"server_error", "message":"failed"})),
        Some(json!({"type":"response.failed", "sequence_number":3,
            "response":sample_response(ResponseStatus::Failed)})),
    ] {
        let expects_error = terminal.is_some();
        let mut frames = vec![delta(Some("msg_1")), unindexed.clone(), indexed.clone()];
        frames.extend(terminal);
        let events = drive(frames);
        assert_eq!(
            metadata(&events),
            vec![json!({"annotations":[citation(1), citation(2)]})]
        );
        if expects_error {
            assert!(events.last().is_some_and(Result::is_err));
        } else {
            // EOF is not a provider terminal; the outer stream owns its
            // truncation diagnostic. The decoder must not manufacture Final.
            assert!(events.iter().all(Result::is_ok));
        }
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(StreamEvent::Final(_))))
        );
    }
}

#[test]
fn content_part_snapshot_recovers_metadata_without_terminal_output() {
    let response = fold(
        vec![
            delta(Some("msg_1")),
            json!({"type":"response.content_part.done", "item_id":"msg_1",
            "output_index":0, "content_index":0, "sequence_number":2,
            "part":{"type":"output_text", "text":"hello", "annotations":[citation(1)],
                "logprobs":[{"token":"hello", "logprob":-0.1}]}}),
        ],
        vec![],
    );
    assert_eq!(
        parts(&response),
        vec![(
            "hello".into(),
            json!({
        "annotations":[citation(1)], "logprobs":[{"token":"hello", "logprob":-0.1}]})
        )]
    );
}

#[test]
fn repeated_snapshots_do_not_duplicate_logprobs() {
    let mut item = message("msg_1", json!([citation(1)]));
    item["content"][0]["logprobs"] = json!([{"token":"hello", "logprob":-0.1}]);
    let response = fold(
        vec![
            delta(Some("msg_1")),
            json!({"type":"response.output_item.done", "output_index":0,
            "sequence_number":2, "item":item}),
        ],
        vec![item],
    );
    assert_eq!(
        parts(&response)[0].1["logprobs"],
        json!([{"token":"hello", "logprob":-0.1}])
    );
}

#[test]
fn pending_annotations_flush_before_transport_failure() {
    let mut driver: WireDriver<Completion, _> = WireDriver::new(ResponsesDecoder::new(
        "openai",
        ResponsesStreamOptions::strict(),
    ));
    let mut added = annotation("msg_1", 1);
    added["annotation_index"] = json!(2);
    driver.push(WireFrame::Text(added.to_string()));
    assert!(metadata(&driver.drain().collect::<Vec<_>>()).is_empty());
    driver.fail(CompletionError::ResponseError("transport failed".into()));
    driver.finish();
    let events = driver.drain().collect::<Vec<_>>();
    assert_eq!(
        metadata(&events),
        vec![json!({"annotations":[citation(1)]})]
    );
    assert!(events.last().is_some_and(Result::is_err));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Final(_))))
    );
}

#[test]
fn ambiguous_anonymous_replay_part_fails_instead_of_misattributing_metadata() {
    let mut frames = vec![
        delta(Some("msg_1")),
        delta(Some("msg_2")),
        annotation("", 1),
    ];
    for frame in &mut frames {
        frame
            .as_object_mut()
            .expect("object")
            .remove("output_index");
    }
    let events = drive(frames);
    assert!(events.last().is_some_and(Result::is_err));
    assert!(metadata(&events).is_empty());
}
