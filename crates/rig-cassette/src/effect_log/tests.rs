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

/// Serve `kind` from `replayer` as dispatch `id`, unary.
async fn answer(
    replayer: &super::EffectLogReplayer,
    id: u64,
    kind: EffectKind,
) -> Result<Answered, ErrorReport> {
    use rig_core::serve::Serve;
    replayer
        .serve(
            kind,
            rig_core::serve::Dispatch::new(EffectId::from_raw(id), false),
        )
        .await
        .into_outcome()
        .await
}

fn changed_kind() -> EffectKind {
    let mut other = custom_kind();
    if let EffectKind::Custom { payload, .. } = &mut other {
        *payload = serde_json::json!({"changed": true});
    }
    other
}

#[tokio::test]
async fn hash_mode_refuses_by_the_hash_pair_and_payload_mode_by_the_pointer() {
    let log = two_records();
    let key = HandlerKey::from("model");
    let other = changed_kind();
    // Both modes accept the recorded request.
    for check in [super::RequestCheck::Payload, super::RequestCheck::Hash] {
        let replayer = super::EffectLogReplayer::for_key(&log, &key)
            .expect("records")
            .checking(check);
        assert_eq!(replayer.check(), check);
        answer(&replayer, 9, custom_kind())
            .await
            .expect("the record answers");
    }
    // Hash mode names the pair; payload mode the pointer.
    let hashed = super::EffectLogReplayer::for_key(&log, &key)
        .expect("records")
        .checking(super::RequestCheck::Hash);
    let report = answer(&hashed, 9, other.clone())
        .await
        .expect_err("diverged");
    assert_eq!(report.kind, ErrorKind::Divergence);
    let recorded = super::stable_hash(&custom_kind()).expect("hashes");
    let arrived = super::stable_hash(&other).expect("hashes");
    assert!(
        report.message.ends_with(&format!(
            "hash {recorded:#018x} was recorded, {arrived:#018x} arrived"
        )),
        "{}",
        report.message
    );
    let by_payload = super::EffectLogReplayer::for_key(&log, &key).expect("records");
    let report = answer(&by_payload, 9, other).await.expect_err("diverged");
    assert!(report.message.contains("payload"), "{}", report.message);
    assert!(!report.message.contains("hash "), "{}", report.message);
}

#[tokio::test]
async fn refusals_are_signalled_and_recorded_failures_are_not() {
    let mut log = two_records();
    log.records[0].outcome = Err(ErrorReport::new(ErrorKind::Divergence, "recorded failure"));
    let key = HandlerKey::from("model");
    let refusals = super::ReplayRefusals::default();
    let replayer = super::EffectLogReplayer::for_key(&log, &key)
        .expect("records")
        .reporting_refusals(refusals.clone());
    // A failure the log recorded is replayed as it was, not signalled.
    let replayed = answer(&replayer, 1, custom_kind())
        .await
        .expect_err("recorded");
    assert_eq!(replayed.message, "recorded failure");
    assert!(refusals.take().is_none());
    // A request that diverges is refused, and the refusal signalled once.
    let refused = answer(&replayer, 2, changed_kind())
        .await
        .expect_err("refused");
    assert_eq!(refusals.take(), Some(refused));
    assert!(refusals.take().is_none());
    // So is a dispatch after the log ran out.
    let refused = answer(&replayer, 3, custom_kind())
        .await
        .expect_err("ran out");
    assert_eq!(refusals.take(), Some(refused));
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
    // By id, each dispatch takes the record of its id, whatever order
    // they arrive in.
    let by_id = EffectLogReplayer::for_key_by_id(&log, &key).unwrap();
    for (id, label) in [(1, "A"), (2, "B")] {
        let answer = by_id
            .serve(kind(label), Dispatch::new(EffectId::from_raw(id), false))
            .await
            .into_outcome()
            .await;
        assert!(
            matches!(answer.unwrap(), Answered::Custom { payload } if payload == serde_json::json!(label))
        );
    }
    // An id the log never recorded is refused by name.
    let by_id = EffectLogReplayer::for_key_by_id(&log, &key).unwrap();
    let refused = by_id
        .serve(kind("A"), Dispatch::new(EffectId::from_raw(10), false))
        .await
        .into_outcome()
        .await
        .expect_err("no record of that id");
    assert_eq!(refused.kind, ErrorKind::Divergence);
    assert!(refused.message.contains("effect:10"), "{}", refused.message);
    assert_eq!(by_id.remaining(), 2, "a refused dispatch takes no record");

    // Two dispatchers each numbering from 1 into one log: answering by id
    // could serve one program the other's record, so it is refused.
    let shared = EffectLogRecorder::new();
    for label in ["first table", "second table"] {
        let id = EffectId::from_raw(1);
        shared.begin(id, key.clone(), kind(label), DispatchContext::default());
        shared.resolve(
            id,
            Ok(Answered::Custom {
                payload: serde_json::json!(label),
            }),
        );
    }
    let refused = EffectLogReplayer::for_key_by_id(&shared.log(), &key)
        .err()
        .expect("one key's id recorded twice");
    assert_eq!(refused.kind, ErrorKind::Divergence);
    assert!(refused.message.contains("effect:1"), "{}", refused.message);
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

#[test]
fn a_checkpoint_names_the_position_the_next_id_and_the_state() {
    let log = two_records();
    let (checkpoint, tail) = log.checkpoint(1, serde_json::json!({"turn": 1}));
    assert_eq!(checkpoint.format, super::CHECKPOINT_FORMAT);
    assert_eq!(checkpoint.at, 1);
    assert_eq!(checkpoint.last, Some(EffectId::from_raw(1)));
    assert_eq!(checkpoint.next, Some(EffectId::from_raw(2)));
    assert_eq!(checkpoint.state, serde_json::json!({"turn": 1}));
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].id, EffectId::from_raw(2));
    assert_eq!(tail.header, log.header, "the tail keeps the header");
    // Serde round trip, then the continuation.
    let json = serde_json::to_string(&checkpoint).expect("serializes");
    let restored: super::Checkpoint<serde_json::Value> =
        serde_json::from_str(&json).expect("restores");
    assert_eq!(restored, checkpoint);
    let continuation = EffectLog::from_checkpoint(&restored, tail).expect("the tail follows");
    assert_eq!(continuation.len(), 1);
    // At the end: an empty tail, `next` absent on the wire.
    let (end, tail) = log.checkpoint(2, serde_json::Value::Null);
    assert_eq!(end.next, None);
    assert!(tail.is_empty());
    let json: serde_json::Value = serde_json::to_value(&end).expect("serializes");
    assert!(json.get("next").is_none());
    EffectLog::from_checkpoint(&end, tail).expect("an empty tail follows the end");
}

#[test]
fn a_continuation_that_does_not_follow_its_checkpoint_is_refused_by_name() {
    let log = two_records();
    let (checkpoint, tail) = log.checkpoint(1, serde_json::Value::Null);
    // The full log in the tail's place: its first id is not the next one.
    let refused = EffectLog::from_checkpoint(&checkpoint, log.clone())
        .expect_err("a full log is not the tail");
    assert_eq!(refused.kind, ErrorKind::Divergence);
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint at 1 expects record effect:2 next, the tail begins at effect:1"
    );
    let refused = EffectLog::from_checkpoint(&checkpoint, EffectLog::from_records(vec![]))
        .expect_err("nothing follows");
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint at 1 expects record effect:2 next, the tail is empty"
    );
    let (end, _) = log.checkpoint(2, serde_json::Value::Null);
    let refused = EffectLog::from_checkpoint(&end, tail.clone()).expect_err("the log ended");
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint at 2 ends the log, the tail begins at effect:2"
    );
    let mut old = checkpoint.clone();
    old.format = 3;
    let refused = EffectLog::from_checkpoint(&old, tail).expect_err("format 3");
    assert_eq!(refused.kind, ErrorKind::Divergence);
    assert_eq!(
        refused.message,
        format!(
            "resume refused: the checkpoint is format 3, this rig reads format {}",
            super::CHECKPOINT_FORMAT
        )
    );
}

/// Each program sharing a log names its required keys under its scope: a
/// replay serves every one of them, refuses a key no descriptor covers,
/// and a log of one program carries no `programs` field at all.
#[test]
fn every_programs_requirements_are_replayed_and_checked() {
    use rig_core::effect::{EffectFamily, EffectRow, FamilyDescriptor, HandlerDescriptor};
    let bare = serde_json::to_value(LogHeader::default()).expect("serializes");
    assert!(
        bare.get("programs").is_none(),
        "no programs, no field: {bare}"
    );
    let recorder = EffectLogRecorder::new();
    for (scope, keys) in [("one", ["shared", "first"]), ("two", ["shared", "second"])] {
        let mut required = EffectRow::new();
        for key in keys {
            required.insert_if_absent(HandlerKey::from(key), EffectFamily::Retrieve);
        }
        recorder.set_program_identity(
            scope,
            ProgramIdentity {
                required,
                policy: 7,
            },
        );
    }
    let mut log = recorder.log();
    log.header
        .describe(["shared", "first", "second"].map(|key| HandlerDescriptor {
            key: HandlerKey::from(key),
            family: FamilyDescriptor::Retrieve {},
            layers: vec![],
        }));
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&log).expect("serializes")).expect("restores");
    assert_eq!(
        log.header.programs.get("two").map(|program| program.policy),
        Some(7)
    );
    let keys: Vec<String> = EffectLogReplayer::for_log(&log)
        .expect("described")
        .iter()
        .map(|replayer| replayer.key().to_string())
        .collect();
    assert_eq!(keys, ["first", "shared", "second"]);

    let mut missing = log.clone();
    missing
        .header
        .handlers
        .retain(|handler| handler.key.as_str() != "second");
    let refused = EffectLogReplayer::check_header(&missing).expect_err("undescribed");
    assert_eq!(
        refused.message,
        "replay refused: scope `two` requires `second` but its recorded descriptor is missing"
    );
    let mut conflicting = log;
    let mut required = EffectRow::new();
    required.insert_if_absent(HandlerKey::from("shared"), EffectFamily::Completion);
    conflicting.header.programs.insert(
        "three".into(),
        ProgramIdentity {
            required,
            policy: 0,
        },
    );
    let refused = EffectLogReplayer::check_header(&conflicting).expect_err("conflicting");
    assert!(refused.message.contains("`shared`"), "{}", refused.message);
}
