use rig_core::{
    completion::{AssistantContent, CompletionRequest, CompletionResponse, Message, Usage},
    effect::{EffectId, EffectKind, EffectRecord, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
};

use super::*;

fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("hi")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[test]
fn effect_record_and_log_round_trip() {
    let log: EffectLog = EffectLog::from_records(vec![
        EffectRecord {
            parent: None,
            scope: None,
            id: EffectId::from_raw(1),
            key: HandlerKey::from("model"),
            kind: EffectKind::Completion {
                request: request(),
                stream: false,
            },
            outcome: Ok(Outcome::Completion(CompletionResponse::new(
                vec![AssistantContent::text("hi")],
                Usage::new(),
                "mock",
            ))),
            events: None,
        },
        EffectRecord {
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
        rig_core::serve::Origin::default(),
    );
    recorder.begin(
        id,
        HandlerKey::from("k"),
        kind(),
        rig_core::serve::Origin::default(),
    );
    recorder.resolve(id, Ok(Outcome::Custom(serde_json::json!("newest"))));
    let log = recorder.take();
    assert_eq!(log.records.len(), 1, "one slot resolved");
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
            parent: None,
            scope: None,
            id: EffectId::from_raw(1),
            key: HandlerKey::from("model"),
            kind: custom_kind(),
            outcome: Ok(rig_core::effect::Outcome::Custom(
                serde_json::json!({"n": 1}),
            )),
            events: None,
        },
        EffectRecord {
            parent: None,
            scope: None,
            id: EffectId::from_raw(2),
            key: HandlerKey::from("model"),
            kind: custom_kind(),
            outcome: Ok(rig_core::effect::Outcome::Custom(
                serde_json::json!({"n": 2}),
            )),
            events: None,
        },
    ])
}

#[test]
fn a_checkpoint_names_the_position_the_next_id_and_the_state() {
    let log = two_records();
    let (checkpoint, tail) = log.checkpoint(1, serde_json::json!({"turn": 1}));
    assert_eq!(checkpoint.format, super::EFFECT_LOG_FORMAT);
    assert_eq!(checkpoint.at, 1);
    assert_eq!(checkpoint.next, Some(EffectId::from_raw(2)));
    assert_eq!(checkpoint.state, serde_json::json!({"turn": 1}));
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].id, EffectId::from_raw(2));
    assert_eq!(tail.header, log.header, "the tail keeps the header");
    // Serde round trip, then the continuation.
    let json = serde_json::to_string(&checkpoint).expect("serializes");
    let restored: super::Checkpoint = serde_json::from_str(&json).expect("restores");
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
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint at 1 expects record effect:2 next, the tail begins at effect:1"
    );
    // An empty tail where a record was expected.
    let refused = EffectLog::from_checkpoint(&checkpoint, EffectLog::from_records(vec![]))
        .expect_err("nothing follows");
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint at 1 expects record effect:2 next, the tail is empty"
    );
    // A tail after a checkpoint that ended the log.
    let (end, _) = log.checkpoint(2, serde_json::Value::Null);
    let refused = EffectLog::from_checkpoint(&end, tail.clone()).expect_err("the log ended");
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint at 2 ends the log, the tail begins at effect:2"
    );
    // Another format, on either side.
    let mut old = checkpoint.clone();
    old.format = 3;
    let refused = EffectLog::from_checkpoint(&old, tail.clone()).expect_err("format 3");
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint is format 3, this rig reads format 5"
    );
    let mut old_tail = tail;
    old_tail.header.format = 3;
    let refused = EffectLog::from_checkpoint(&checkpoint, old_tail).expect_err("format 3");
    assert_eq!(
        refused.message,
        "resume refused: the tail is format 3, this rig reads format 5"
    );
}

#[tokio::test]
async fn hash_mode_refuses_by_the_hash_pair_and_payload_mode_by_the_pointer() {
    use rig_core::serve::Serve;
    let log = two_records();
    let key = HandlerKey::from("model");
    let mut other = custom_kind();
    if let EffectKind::Custom { payload, .. } = &mut other {
        *payload = serde_json::json!({"changed": true});
    }
    let answer = |replayer: super::EffectLogReplayer, kind: EffectKind| async move {
        let (reply, receiver) = futures::channel::oneshot::channel();
        replayer
            .serve(
                kind,
                rig_core::serve::OutcomeSink::unary(EffectId::from_raw(9), reply),
            )
            .await;
        receiver.await.expect("answered")
    };
    // Both modes accept the recorded request.
    for check in [super::RequestCheck::Payload, super::RequestCheck::Hash] {
        let replayer = super::EffectLogReplayer::for_key(&log, &key)
            .expect("records")
            .checking(check);
        assert_eq!(replayer.check(), check);
        answer(replayer, custom_kind())
            .await
            .expect("the record answers");
    }
    // Hash mode names the pair; payload mode the pointer.
    let hashed = super::EffectLogReplayer::for_key(&log, &key)
        .expect("records")
        .checking(super::RequestCheck::Hash);
    let report = answer(hashed, other.clone()).await.expect_err("diverged");
    assert_eq!(report.kind, rig_core::error::ErrorKind::Divergence);
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
    let report = answer(by_payload, other).await.expect_err("diverged");
    assert!(report.message.contains("payload"), "{}", report.message);
    assert!(!report.message.contains("hash "), "{}", report.message);
}
