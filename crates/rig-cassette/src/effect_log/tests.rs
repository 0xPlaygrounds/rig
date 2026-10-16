use rig_core::{
    completion::{AssistantContent, CompletionRequest, CompletionResponse, Message, Usage},
    effect::{Answered, EffectId, EffectKind, EffectRecord, HandlerKey},
    error::{ErrorKind, ErrorReport},
};

use super::*;

fn request() -> CompletionRequest {
    CompletionRequest::from(vec![Message::user("hi")])
}

#[test]
fn replay_keeps_recorded_model_semantics_even_when_it_has_records() {
    use rig_core::{
        completion::{ModelRef, ProviderCapabilities},
        effect::{FamilyDescriptor, HandlerDescriptor},
        serve::Serve,
    };
    let mut log = two_records();
    for record in &mut log.records {
        record.kind = EffectKind::Completion {
            request: request().into(),
            stream: false,
        };
        record.outcome = Ok(Answered::Completion(
            CompletionResponse::new(
                vec![AssistantContent::text("ok")],
                Usage::default(),
                rig_core::message::Origin::new("test.api", "composing-model", ""),
                serde_json::json!({}),
            )
            .into(),
        ));
    }
    log.header.signature = log
        .records
        .iter()
        .map(|record| (record.key.clone(), record.kind.family()))
        .collect();
    let key = log.records[0].key.clone();
    let family = FamilyDescriptor::Completion {
        model: ModelRef::new("composing-model"),
        capabilities: ProviderCapabilities::new().with_native_output_tool_composition(true),
    };
    log.header.handlers.push(HandlerDescriptor {
        key: key.clone(),
        family: family.clone(),
        layers: vec!["ApplicationLayer".into()],
    });
    let log: EffectLog = serde_json::from_str(&serde_json::to_string(&log).unwrap()).unwrap();
    let replayer = EffectLogReplayer::for_key(&log, &key).unwrap();
    assert_eq!(replayer.descriptor().family, family);
    assert!(
        replayer.descriptor().layers.is_empty(),
        "middleware must be reapplied by the program"
    );
}

#[test]
fn stream_error_metadata_is_validated_and_released_with_its_records() {
    use rig_core::serve::{DispatchContext, Recorder};
    let recorder = EffectLogRecorder::keeping_stream_events();
    for discarded in [false, true] {
        let id = EffectId::from_raw(u64::from(discarded));
        recorder.begin(
            id,
            "stream".into(),
            EffectKind::Completion {
                request: request().into(),
                stream: true,
            },
            DispatchContext::default(),
        );
        let error = ErrorReport::new(ErrorKind::Response, "error item");
        recorder.stream_error(id, &error);
        if discarded {
            recorder.discard(id);
            recorder.stream_error(id, &error);
            assert!(recorder.header().stream_errors.is_empty());
        } else {
            recorder.resolve(id, Err(error));
            let log = recorder.take();
            assert_eq!(log.header.stream_errors[&id][0].item, 0);
            EffectLogReplayer::check_header(&log).unwrap();
            assert!(log.tail(1).header.stream_errors.is_empty());
            assert!(recorder.header().stream_errors.is_empty());
            let mut invalid = log.clone();
            invalid.header.stream_errors.get_mut(&id).unwrap()[0].item = 99;
            assert!(
                EffectLogReplayer::check_header(&invalid)
                    .unwrap_err()
                    .message
                    .contains("positions")
            );
            let mut invalid = log;
            invalid.records.clear();
            assert!(
                EffectLogReplayer::check_header(&invalid)
                    .unwrap_err()
                    .message
                    .contains("kept stream")
            );
        }
    }
}

#[test]
fn effect_record_and_log_round_trip() {
    let log: EffectLog = EffectLog::from_records(vec![
        EffectRecord {
            purpose: None,
            turn: None,
            started_ms: None,
            finished_ms: None,
            reviewed: None,
            disposition: Default::default(),
            decided_by: None,
            stream_origin: None,
            published: None,
            parent: None,
            scope: None,
            id: EffectId::from_raw(1),
            key: HandlerKey::from("model"),
            kind: EffectKind::Completion {
                request: request().into(),
                stream: false,
            },
            outcome: Ok(Answered::Completion(
                CompletionResponse::new(
                    vec![AssistantContent::text("hi")],
                    Usage::default(),
                    rig_core::message::Origin::new("test.api", "mock", ""),
                    serde_json::json!({}),
                )
                .into(),
            )),
            events: None,
        },
        EffectRecord {
            purpose: None,
            turn: None,
            started_ms: None,
            finished_ms: None,
            reviewed: None,
            disposition: Default::default(),
            decided_by: None,
            stream_origin: None,
            published: None,
            parent: None,
            scope: None,
            id: EffectId::from_raw(2),
            key: HandlerKey::from("tool:add"),
            kind: EffectKind::ToolCall {
                name: "add".into(),
                args: "{}".into(),
            },
            outcome: Err(ErrorReport::new(ErrorKind::Timeout, "slow")),
            events: None,
        },
    ]);
    let json = serde_json::to_string(&log).expect("serializes");
    let back: EffectLog = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(
        serde_json::to_string(&back).expect("serializes"),
        json,
        "log round trip"
    );
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].id, EffectId::from_raw(1));
    assert_eq!(back[1].key.as_str(), "tool:add");
    assert!(matches!(&back[1].outcome, Err(report) if report.kind == ErrorKind::Timeout));
}

#[test]
fn headers_have_no_format_and_refuse_an_unknown_key() {
    let log = two_records();
    let mut json = serde_json::to_value(&log).expect("serializes");
    assert!(json["header"].get("format").is_none());
    // A header key this rig does not know — a retired `format` number
    // included — is refused on load rather than ignored.
    json["header"]["format"] = serde_json::json!(999);
    let error =
        serde_json::from_value::<EffectLog>(json).expect_err("an unknown header key is refused");
    assert!(error.to_string().contains("format"), "{error}");
}

#[test]
fn missing_published_output_is_unknown_not_explicitly_absent() {
    let mut json = serde_json::to_value(two_records()).unwrap();
    assert!(json["records"][0].get("tool_output").is_some());
    assert!(json["records"][0]["tool_output"].is_null());
    serde_json::from_value::<EffectLog>(json.clone()).expect("explicit absence is valid");
    json["records"][0]
        .as_object_mut()
        .unwrap()
        .shift_remove("tool_output");
    let error = serde_json::from_value::<EffectLog>(json).expect_err("old recorder omitted output");
    assert!(error.to_string().contains("tool_output"));
}

/// The recorder finds a slot from the back: the newest slot with an id is
/// the one an event or an outcome lands on. Probed with two slots begun
/// under one id (a shape the recorder never sees from a bus, which mints
/// ids once): the later slot resolves, the earlier stays in flight.
#[test]
fn the_recorder_finds_the_newest_slot_first() {
    use rig_core::serve::Recorder;
    let recorder = EffectLogRecorder::new();
    let kind = || EffectKind::Custom {
        kind: std::sync::Arc::from("test:probe"),
        payload: serde_json::Value::Null,
    };
    let id = EffectId::from_raw(7);
    recorder.begin(
        id,
        HandlerKey::from("k"),
        kind(),
        rig_core::serve::DispatchContext::default(),
    );
    recorder.begin(
        id,
        HandlerKey::from("k"),
        kind(),
        rig_core::serve::DispatchContext::default(),
    );
    recorder.resolve(
        id,
        Ok(Answered::Custom {
            payload: serde_json::json!("newest"),
        }),
    );
    assert_eq!(recorder.resolved(), 1);
    let log = recorder.take();
    assert_eq!(log.records.len(), 1, "one slot resolved");
    assert_eq!(recorder.resolved(), 0);
    assert_eq!(
        recorder.in_flight(),
        1,
        "the earlier slot is still in flight"
    );
}

/// A key the handler table names is described from it whether or not any
/// record dispatched to it and whether or not the required row names it:
/// a host's handler nothing asked, or one a layer denied every dispatch
/// to, replays as what the table says it was.
#[test]
fn a_key_the_handler_table_names_is_described_without_records_or_a_row_entry() {
    let mut log = EffectLog::from_records(vec![]);
    log.header
        .handlers
        .push(rig_core::effect::HandlerDescriptor {
            key: HandlerKey::from("host/note"),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: "corpus:note".to_owned(),
            },
            layers: vec!["DenyAllLayer".to_owned()],
        });
    let replayer = super::EffectLogReplayer::for_key(&log, &HandlerKey::from("host/note"))
        .expect("described from the handler table");
    let descriptor = rig_core::serve::Serve::descriptor(&replayer);
    assert_eq!(descriptor.key.as_str(), "host/note");
    assert!(matches!(
        descriptor.family,
        rig_core::effect::FamilyDescriptor::Custom { ref kind } if kind == "corpus:note"
    ));
    assert!(
        descriptor.layers.is_empty(),
        "the replayer is the handler beneath; whoever replays re-applies the layers"
    );
    let refused = match super::EffectLogReplayer::for_key(&log, &HandlerKey::from("host/other")) {
        Ok(_) => panic!("nothing describes an unknown key"),
        Err(report) => report,
    };
    assert_eq!(refused.kind, rig_core::error::ErrorKind::HandlerUnavailable);
    assert!(
        refused.message.contains("`host/other`"),
        "{}",
        refused.message
    );
}

fn custom_kind() -> EffectKind {
    EffectKind::Custom {
        kind: std::sync::Arc::from("test"),
        payload: serde_json::json!({"ask": 1}),
    }
}

fn two_records() -> EffectLog {
    EffectLog::from_records(vec![
        EffectRecord {
            purpose: None,
            turn: None,
            started_ms: None,
            finished_ms: None,
            reviewed: None,
            disposition: Default::default(),
            decided_by: None,
            stream_origin: None,
            published: None,
            parent: None,
            scope: None,
            id: EffectId::from_raw(1),
            key: HandlerKey::from("model"),
            kind: custom_kind(),
            outcome: Ok(rig_core::effect::Answered::Custom {
                payload: serde_json::json!({"n": 1}),
            }),
            events: None,
        },
        EffectRecord {
            purpose: None,
            turn: None,
            started_ms: None,
            finished_ms: None,
            reviewed: None,
            disposition: Default::default(),
            decided_by: None,
            stream_origin: None,
            published: None,
            parent: None,
            scope: None,
            id: EffectId::from_raw(2),
            key: HandlerKey::from("model"),
            kind: custom_kind(),
            outcome: Ok(rig_core::effect::Answered::Custom {
                payload: serde_json::json!({"n": 2}),
            }),
            events: None,
        },
    ])
}

#[tokio::test]
async fn payload_mode_refuses_by_the_pointer() {
    use rig_core::serve::Serve;
    let log = two_records();
    let key = HandlerKey::from("model");
    let mut other = custom_kind();
    if let EffectKind::Custom { payload, .. } = &mut other {
        *payload = serde_json::json!({"changed": true});
    }
    let answer = |replayer: super::EffectLogReplayer, kind: EffectKind| async move {
        replayer
            .serve(
                kind,
                rig_core::serve::Dispatch::new(EffectId::from_raw(9), false),
            )
            .await
            .into_outcome()
            .await
    };
    let replayer = super::EffectLogReplayer::for_key(&log, &key).expect("records");
    assert_eq!(replayer.check(), super::RequestCheck::Payload);
    answer(replayer, custom_kind())
        .await
        .expect("the record answers");
    let by_payload = super::EffectLogReplayer::for_key(&log, &key).expect("records");
    let report = answer(by_payload, other).await.expect_err("diverged");
    assert_eq!(report.kind, rig_core::error::ErrorKind::Divergence);
    assert!(report.message.contains("payload"), "{}", report.message);
}

#[tokio::test]
async fn positional_replay_preserves_recorded_order_when_ids_were_reserved_earlier() {
    use rig_core::serve::{Dispatch, DispatchContext, Recorder, Serve};
    let recorder = EffectLogRecorder::new();
    let key = HandlerKey::from("ordered");
    let kind = |label: &str| EffectKind::Custom {
        kind: "ordered".into(),
        payload: serde_json::json!(label),
    };
    // Dispatch futures reserve A=1, B=2, but the executor serves B first.
    for (id, label) in [(2, "B"), (1, "A")] {
        let id = EffectId::from_raw(id);
        recorder.begin(id, key.clone(), kind(label), DispatchContext::default());
        recorder.resolve(
            id,
            Ok(Answered::Custom {
                payload: serde_json::json!(label),
            }),
        );
    }
    let log = recorder.log();
    assert_eq!(
        log.records
            .iter()
            .map(|record| record.id)
            .collect::<Vec<_>>(),
        vec![EffectId::from_raw(2), EffectId::from_raw(1)]
    );
    let positional = EffectLogReplayer::for_key(&log, &key).unwrap();
    for (id, label) in [(10, "B"), (11, "A")] {
        let answer = positional
            .serve(kind(label), Dispatch::new(EffectId::from_raw(id), false))
            .await
            .into_outcome()
            .await;
        assert!(
            matches!(answer.unwrap(), Answered::Custom { payload } if payload == serde_json::json!(label))
        );
    }
}

/// Local namespace tags survive nested log payloads and replay request matching.
/// This is a format/handler contract test; it does not simulate provider traffic.
#[tokio::test]
async fn typed_tool_namespaces_survive_log_roundtrip_and_replay() {
    use rig_core::{
        message::{CallId, ToolCall, ToolFunction, ToolResultContent, UserContent},
        serve::{Dispatch, Serve},
        streaming::StreamRecord,
    };
    let generated = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(
            rig_core::message::ToolName::new("add").expect("tool name"),
            serde_json::json!({"x": 1}),
        ),
    );
    let explicit = ToolCall::from_wire(
        "tool-0",
        ToolFunction::new(
            rig_core::message::ToolName::new("add").expect("tool name"),
            serde_json::json!({"x": 2}),
        ),
    );
    assert_ne!(generated.id, explicit.id);
    let calls = [generated, explicit];
    let choice: Vec<_> = calls
        .iter()
        .cloned()
        .map(AssistantContent::ToolCall)
        .collect();
    // Each call as a stream carries it: its start, its arguments, its end.
    let events = StreamRecord::parse(serde_json::Value::Array(
        calls
            .iter()
            .enumerate()
            .flat_map(|(part, call)| {
                [
                    serde_json::json!({
                        "event": "start",
                        "part": part,
                        "kind": "tool_call",
                        "name": call.function.name,
                    }),
                    serde_json::json!({
                        "event": "arguments",
                        "part": part,
                        "json": call.function.arguments_value().to_string(),
                    }),
                    serde_json::json!({
                        "event": "end",
                        "part": part,
                        "content": AssistantContent::ToolCall(call.clone()),
                    }),
                ]
            })
            .map(|event| serde_json::json!({"item": "event", "value": event}))
            .collect(),
    ))
    .expect("a transcript");
    let mut next = request();
    next.chat_history
        .push(Message::Assistant(rig_core::message::AssistantMessage::new(choice.clone())).into());
    next.chat_history.push(
        Message::User {
            content: calls
                .iter()
                .rev()
                .map(|call| {
                    UserContent::tool_result(
                        call.id.clone(),
                        rig_core::message::ToolName::new("add").expect("tool name"),
                        vec![ToolResultContent::text("ok")],
                    )
                })
                .collect::<Vec<_>>(),
        }
        .into(),
    );
    let mut records = two_records().records;
    records[0].kind = EffectKind::Completion {
        request: request().into(),
        stream: false,
    };
    records[0].outcome = Ok(Answered::Completion(
        CompletionResponse::new(
            choice,
            Usage::default(),
            rig_core::message::Origin::new("test.api", "test", ""),
            serde_json::json!({}),
        )
        .into(),
    ));
    records[0].events = Some(events);
    records[1].kind = EffectKind::Completion {
        request: next.into(),
        stream: false,
    };
    records[1].outcome = Ok(Answered::Completion(
        CompletionResponse::new(
            vec![AssistantContent::text("done")],
            Usage::default(),
            rig_core::message::Origin::new("test.api", "test", ""),
            serde_json::json!({}),
        )
        .into(),
    ));
    let log = EffectLog::from_records(records);
    let json = serde_json::to_value(&log).unwrap();
    let restored: EffectLog = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), json);
    let replay = EffectLogReplayer::for_key(&restored, &HandlerKey::from("model")).unwrap();
    for record in &log.records {
        let answer = replay
            .serve(record.kind.clone(), Dispatch::new(record.id, false))
            .await
            .into_outcome()
            .await;
        assert_eq!(
            serde_json::to_value(answer).unwrap(),
            serde_json::to_value(&record.outcome).unwrap()
        );
    }
    // The request check must reject identity collapse, even though the
    // generated and explicit handles have the same candidate wire spelling.
    // The replayer answers in order: the first record, then the changed one.
    let replay = EffectLogReplayer::for_key(&restored, &HandlerKey::from("model")).unwrap();
    let _ = replay
        .serve(
            log.records[0].kind.clone(),
            Dispatch::new(log.records[0].id, false),
        )
        .await
        .into_outcome()
        .await;
    let mut changed = log.records[1].kind.clone();
    let EffectKind::Completion { request, .. } = &mut changed else {
        unreachable!()
    };
    let Some(Message::User { content }) = std::sync::Arc::make_mut(request)
        .chat_history
        .last_mut()
        .map(std::sync::Arc::make_mut)
    else {
        unreachable!()
    };
    let Some(UserContent::ToolResult(result)) = content.last_mut() else {
        unreachable!()
    };
    assert!(result.call.is_local());
    result.call = calls[1].id.clone();
    let error = replay
        .serve(changed, Dispatch::new(log.records[1].id, false))
        .await
        .into_outcome()
        .await
        .expect_err("collapsed namespace must diverge");
    assert!(error.message.contains("diverg"), "{error:?}");
    // Replace every nested identity, including events and next-turn history,
    // with the old ambiguous spelling: the persisted payload must be refused.
    fn erase_tags(value: &mut serde_json::Value) -> usize {
        let is_call_id = value.as_object().is_some_and(|object| {
            object.len() == 1
                && (object
                    .get("local")
                    .is_some_and(serde_json::Value::is_string)
                    || object
                        .get("provider")
                        .is_some_and(serde_json::Value::is_string))
        });
        if is_call_id {
            *value = serde_json::json!("tool-0");
            return 1;
        }
        match value {
            serde_json::Value::Array(values) => values.iter_mut().map(erase_tags).sum(),
            serde_json::Value::Object(values) => values.values_mut().map(erase_tags).sum(),
            _ => 0,
        }
    }
    let mut legacy = json;
    assert_eq!(erase_tags(&mut legacy), 8);
    assert!(serde_json::from_value::<EffectLog>(legacy).is_err());
}

/// A timed recorder stamps when each dispatch began and resolved, and
/// records what it was for; an untimed one, as goldens use, keeps no times.
#[test]
fn a_timed_recorder_stamps_its_records_and_an_untimed_one_does_not() {
    use rig_core::serve::{DispatchContext, Recorder};
    let origin = DispatchContext {
        purpose: Some("summary".into()),
        ..DispatchContext::default()
    };
    for (recorder, timed) in [
        (EffectLogRecorder::new().timed(), true),
        (EffectLogRecorder::new(), false),
    ] {
        let id = EffectId::from_raw(1);
        let kind = EffectKind::ToolCall {
            name: "t".into(),
            args: "{}".into(),
        };
        recorder.begin(id, "t".into(), kind, origin.clone());
        recorder.resolve(id, Err(ErrorReport::new(ErrorKind::Internal, "no")));
        let log = recorder.take();
        let record = log.records.first().expect("one record");
        assert_eq!(record.purpose.as_deref(), Some("summary"));
        let times = (record.started_ms.is_some(), record.finished_ms.is_some());
        assert_eq!(times, (timed, timed));
    }
}

/// Discarding a dispatch the recorder never saw changes nothing, and the
/// recorder shows its counts.
#[test]
fn an_unknown_discard_is_a_no_op() {
    use rig_core::serve::Recorder;
    let recorder = EffectLogRecorder::new();
    recorder.discard(EffectId::from_raw(9));
    let header = recorder.header();
    assert!(header.signature.is_empty());
    assert_eq!(
        format!("{recorder:?}"),
        "EffectLogRecorder { records: 0, in_flight: 0 }"
    );
}
