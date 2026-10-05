use super::*;
use serde_json::json;

fn fact(object: &str) -> Fact {
    Fact {
        path: "$".into(),
        object: object.into(),
    }
}

fn set(objects: &[&str]) -> Facts {
    objects.iter().map(|object| fact(object)).collect()
}

fn observed(recording: &str, size: usize, recorded: Option<&[&str]>, sent: &[&str]) -> Observed {
    Observed {
        provider: "openai".into(),
        encoder: "POST /v1/chat/completions".into(),
        recording: recording.into(),
        fixture: "f1".into(),
        size,
        recorded: recorded.map(set),
        sent: set(sent),
    }
}

fn missing(object: &str, rerecord: &str) -> Missing {
    Missing {
        provider: "openai".into(),
        encoder: "POST /v1/chat/completions".into(),
        fact: fact(object),
        rerecord: rerecord.into(),
    }
}

#[test]
fn a_sent_fact_maps_to_a_recording_that_still_sends_it() {
    let corpus = [
        observed("a.yaml#0", 1, Some(&["f1", "f3"]), &["f2", "f3"]),
        observed("b.yaml#0", 1, Some(&["f1"]), &["f1"]),
        observed("c.yaml#0", 1, Some(&["f2"]), &["f2"]),
    ];
    let (index, missing) = build(&corpus, &Index::default());
    assert!(missing.is_empty(), "{missing:?}");
    let recordings: Vec<(&str, &str)> = index
        .facts
        .iter()
        .map(|entry| (entry.fact.object.as_str(), entry.recording.as_str()))
        .collect();
    assert_eq!(
        recordings,
        [("f1", "b.yaml#0"), ("f2", "c.yaml#0"), ("f3", "a.yaml#0")]
    );
}

#[test]
fn a_fact_no_recording_holds_names_the_smallest_cassette_that_sends_it() {
    let corpus = [
        observed("a.yaml#0", 30, Some(&["f1"]), &["f1", "f3"]),
        observed("b.yaml#1", 20, Some(&["f1"]), &["f3"]),
        observed("c.yaml#0", 20, Some(&["f1"]), &["f3"]),
    ];
    let (index, found) = build(&corpus, &Index::default());
    assert_eq!(index.facts.len(), 1);
    assert_eq!(found, [missing("f3", "b.yaml#1")]);
}

#[test]
fn a_bodyless_recording_keeps_its_pins_until_its_fixture_changes() {
    let pin = |object: &str| Pin {
        recording: "audio.yaml#0".into(),
        fixture: "f1".into(),
        fact: fact(object),
    };
    let pinned = Index {
        facts: Vec::new(),
        unrecorded: vec![pin("old"), pin("shared")],
    };
    // A first index pins what the snapshot shows.
    let (first, missing) = build(
        &[observed("audio.yaml#0", 1, None, &["old", "shared"])],
        &Index::default(),
    );
    assert!(missing.is_empty());
    assert_eq!(first.unrecorded, pinned.unrecorded);
    assert_eq!(first.facts.len(), 2);
    // An encoder change shows in the snapshot, not in the pins.
    let (_, found) = build(
        &[observed("audio.yaml#0", 1, None, &["new", "shared"])],
        &pinned,
    );
    assert_eq!(found, [self::missing("new", "audio.yaml#0")]);
    // A re-recorded fixture is pinned afresh.
    let mut recorded = observed("audio.yaml#0", 1, None, &["new"]);
    recorded.fixture = "f2".into();
    let (index, missing) = build(&[recorded], &pinned);
    assert!(missing.is_empty());
    assert_eq!(index.unrecorded.len(), 1);
    assert_eq!(index.unrecorded[0].fact, fact("new"));
}

#[test]
fn the_index_round_trips() {
    let index = Index {
        facts: vec![Entry {
            provider: "gemini".into(),
            encoder: "POST /v1beta/models/{model}:generateContent".into(),
            fact: Fact {
                path: "$.contents[]".into(),
                object: "{parts:[{}],role:\"user\"}".into(),
            },
            recording: "gemini/a \"b\".yaml#2".into(),
        }],
        unrecorded: vec![Pin {
            recording: "openai/t.yaml#0".into(),
            fixture: "fedcba9876543210".into(),
            fact: fact("multipart[file,model]"),
        }],
    };
    assert_eq!(parse(&render(&index)), Ok(index));
    assert!(parse("facts = [\n  { provider = \"x\" },\n]\n").is_err());
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
fn a_view_has_the_facts_its_recording_would() {
    let part = |name: &str| json!({ "headers": { "content-disposition": format!("form-data; name=\"{name}\"") }, "body": "x" });
    assert_eq!(
        view_facts(&json!({ "multipart": [part("model"), part("file")] })),
        set(&["multipart[file,model]"])
    );
    assert_eq!(view_facts(&Value::Null), set(&["empty"]));
    assert_eq!(
        view_facts(&json!({ "bytes": 3, "fnv1a64": "00" })),
        set(&["binary"])
    );
    let body = json!({ "a": [1, "b"], "messages": [{ "role": "user" }] });
    assert_eq!(
        view_facts(&body),
        shapes::request_facts(&Exchange {
            path: String::new(),
            method: String::new(),
            status: 0,
            header: Vec::new(),
            body: Some(body.to_string()),
            body_encoding: None,
        })
    );
}
