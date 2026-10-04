use super::*;
use serde_json::json;

fn observed(recording: &str, recorded: Option<&str>, sent: &str) -> Observed {
    Observed {
        provider: "openai".into(),
        encoder: "POST /v1/chat/completions".into(),
        recording: recording.into(),
        fixture: "f1".into(),
        recorded: recorded.map(str::to_owned),
        sent: sent.into(),
    }
}

#[test]
fn a_sent_skeleton_maps_to_a_recording_that_still_sends_it() {
    let corpus = [
        observed("a.yaml#0", Some("s1"), "s2"),
        observed("b.yaml#0", Some("s1"), "s1"),
        observed("c.yaml#0", Some("s2"), "s2"),
    ];
    let (index, missing) = build(&corpus, &Index::default());
    assert!(missing.is_empty(), "{missing:?}");
    let recordings: Vec<(&str, &str)> = index
        .skeletons
        .iter()
        .map(|entry| (entry.skeleton.as_str(), entry.recording.as_str()))
        .collect();
    assert_eq!(recordings, [("s1", "b.yaml#0"), ("s2", "c.yaml#0")]);
}

#[test]
fn a_skeleton_no_recording_holds_is_missing_with_where_it_is_sent() {
    let corpus = [
        observed("a.yaml#0", Some("s1"), "s3"),
        observed("b.yaml#1", Some("s1"), "s3"),
    ];
    let (index, missing) = build(&corpus, &Index::default());
    assert!(index.skeletons.is_empty());
    assert_eq!(
        missing,
        [Missing {
            provider: "openai".into(),
            encoder: "POST /v1/chat/completions".into(),
            skeleton: "s3".into(),
            sent_in: "a.yaml#0".into(),
        }]
    );
}

#[test]
fn a_bodyless_recording_keeps_its_pin_until_its_fixture_changes() {
    let pinned = Index {
        skeletons: Vec::new(),
        unrecorded: vec![Pin {
            recording: "audio.yaml#0".into(),
            fixture: "f1".into(),
            skeleton: "old".into(),
        }],
    };
    // A first index pins what the snapshot shows.
    let (first, missing) = build(&[observed("audio.yaml#0", None, "old")], &Index::default());
    assert!(missing.is_empty());
    assert_eq!(
        first,
        Index {
            skeletons: vec![Entry {
                provider: "openai".into(),
                encoder: "POST /v1/chat/completions".into(),
                skeleton: "old".into(),
                recording: "audio.yaml#0".into(),
            }],
            ..pinned.clone()
        }
    );
    // An encoder change shows in the snapshot, not in the pin.
    let (_, missing) = build(&[observed("audio.yaml#0", None, "new")], &pinned);
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0].skeleton, "new");
    // A re-recorded fixture is pinned afresh.
    let mut recorded = observed("audio.yaml#0", None, "new");
    recorded.fixture = "f2".into();
    let (index, missing) = build(&[recorded], &pinned);
    assert!(missing.is_empty());
    assert_eq!(index.unrecorded[0].skeleton, "new");
}

#[test]
fn the_index_round_trips() {
    let index = Index {
        skeletons: vec![Entry {
            provider: "gemini".into(),
            encoder: "POST /v1beta/models/{model}:generateContent".into(),
            skeleton: "0123456789abcdef".into(),
            recording: "gemini/a \"b\".yaml#2".into(),
        }],
        unrecorded: vec![Pin {
            recording: "openai/t.yaml#0".into(),
            fixture: "fedcba9876543210".into(),
            skeleton: "00".into(),
        }],
    };
    assert_eq!(parse(&render(&index)), Ok(index));
    assert!(parse("skeletons = [\n  { provider = \"x\" },\n]\n").is_err());
}

#[test]
fn snapshot_changes_apply_to_the_recorded_view() {
    let change = |text: Value| -> Change { serde_json::from_value(text).expect("a change") };
    let mut view = json!({ "messages": [{ "content": "hi" }], "model": "m", "seed": 1 });
    for item in [
        json!({ "path": "/messages/0/content", "recorded": "hi", "sent": [{ "type": "text", "text": "hi" }] }),
        json!({ "path": "/seed", "recorded": 1 }),
        json!({ "path": "/stream", "sent": true }),
        json!({ "path": "/messages", "splice": 1, "recorded": [], "sent": [{ "content": "more" }] }),
    ] {
        apply(&mut view, &change(item)).expect("the change applies");
    }
    assert_eq!(
        view,
        json!({
            "messages": [{ "content": [{ "type": "text", "text": "hi" }] }, { "content": "more" }],
            "model": "m",
            "stream": true,
        })
    );
    let mut whole = Value::Null;
    apply(
        &mut whole,
        &change(json!({ "path": "", "recorded": null, "sent": { "multipart": [] } })),
    )
    .expect("the root is replaced");
    assert_eq!(whole, json!({ "multipart": [] }));
    assert!(
        apply(
            &mut whole,
            &change(json!({ "path": "/missing/key", "sent": 1 }))
        )
        .is_err()
    );
}

#[test]
fn a_view_has_the_skeleton_its_recording_would() {
    let part = |name: &str| json!({ "headers": { "content-disposition": format!("form-data; name=\"{name}\"") }, "body": "x" });
    assert_eq!(
        view_skeleton(&json!({ "multipart": [part("model"), part("file")] })),
        "multipart[file,model]"
    );
    assert_eq!(view_skeleton(&Value::Null), "empty");
    assert_eq!(
        view_skeleton(&json!({ "bytes": 3, "fnv1a64": "00" })),
        "binary"
    );
    assert_eq!(view_skeleton(&json!({ "a": [1, "b"] })), "{a:[num|str]}");
}
