use rig_core::{
    completion::{AssistantContent, CompletionRequest, CompletionResponse, Message, Usage},
    effect::{EffectId, EffectKind, EffectRecord, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    tool::ToolContext,
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
                context: ToolContext::new(),
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
