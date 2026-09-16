use super::{NdjsonFramer, SseEvent, SseFramer};
use std::time::Duration;

/// Feed `body` in one chunk.
fn events(body: &str) -> Vec<SseEvent> {
    let mut framer = SseFramer::new();
    framer.push(body.as_bytes()).collect()
}

/// Feed `body` one byte at a time — chunk boundaries are the transport's,
/// never the grammar's.
fn events_split_everywhere(body: &str) -> Vec<SseEvent> {
    let mut framer = SseFramer::new();
    let mut out = Vec::new();
    for byte in body.as_bytes() {
        out.extend(framer.push(std::slice::from_ref(byte)));
    }
    out
}

fn event(data: &str) -> SseEvent {
    SseEvent {
        event: "message".to_owned(),
        data: data.to_owned(),
        id: None,
        retry: None,
    }
}

#[test]
fn dispatches_on_blank_line_with_every_line_ending() {
    for line_ending in ["\n", "\r\n", "\r"] {
        let body = format!("data: a{line_ending}{line_ending}data: b{line_ending}{line_ending}");
        assert_eq!(events(&body), vec![event("a"), event("b")]);
    }
}

#[test]
fn skips_a_leading_bom() {
    assert_eq!(events("\u{feff}data: a\n\n"), vec![event("a")]);
}

#[test]
fn ignores_comments() {
    assert_eq!(events(": keepalive\ndata: a\n\n"), vec![event("a")]);
}

#[test]
fn joins_multiple_data_lines() {
    assert_eq!(events("data: a\ndata: b\n\n"), vec![event("a\nb")]);
}

#[test]
fn the_last_id_persists_across_events() {
    let body = "id: 1\ndata: a\n\ndata: b\n\n";
    assert_eq!(
        events(body),
        vec![
            SseEvent {
                event: "message".to_owned(),
                data: "a".to_owned(),
                id: Some("1".to_owned()),
                retry: None,
            },
            SseEvent {
                event: "message".to_owned(),
                data: "b".to_owned(),
                id: Some("1".to_owned()),
                retry: None,
            },
        ]
    );
}

#[test]
fn a_retry_field_belongs_to_its_own_event_only() {
    let body = "retry: 1000\ndata: a\n\ndata: b\n\n";
    assert_eq!(
        events(body),
        vec![
            SseEvent {
                event: "message".to_owned(),
                data: "a".to_owned(),
                id: None,
                retry: Some(Duration::from_millis(1000)),
            },
            event("b"),
        ]
    );
}

#[test]
fn a_blank_line_without_data_dispatches_nothing() {
    assert!(events("\n\nevent: message\n\n").is_empty());
}

#[test]
fn named_events_are_reported_by_name() {
    let body = "event: message_start\ndata: {}\n\n";
    let mut expected = event("{}");
    expected.event = "message_start".to_owned();
    assert_eq!(events(body), vec![expected]);
}

#[test]
fn a_truncated_event_is_never_dispatched() {
    let mut framer = SseFramer::new();
    assert!(
        framer
            .push(b"data: complete\n\ndata: cut")
            .eq(vec![event("complete")])
    );
    assert_eq!(framer.pending(), "data: cut".len());
}

#[test]
fn chunk_boundaries_never_change_the_event_sequence() {
    let body = "id: 1\nevent: greeting\ndata: hello\ndata: world\n\n: comment\ndata: next\n\n";
    assert_eq!(events(body), events_split_everywhere(body));
}

/// Feed `body` in one chunk.
fn lines(body: &str) -> Vec<String> {
    let mut framer = NdjsonFramer::new();
    framer
        .push(body.as_bytes())
        .map(|line| String::from_utf8_lossy(&line).into_owned())
        .collect()
}

#[test]
fn ndjson_yields_one_frame_per_line() {
    assert_eq!(
        lines("{\"a\":1}\n{\"b\":2}\n"),
        vec!["{\"a\":1}", "{\"b\":2}"]
    );
}

#[test]
fn ndjson_trims_a_trailing_carriage_return_and_skips_blank_lines() {
    assert_eq!(
        lines("{\"a\":1}\r\n\n{\"b\":2}\n"),
        vec!["{\"a\":1}", "{\"b\":2}"]
    );
}

#[test]
fn ndjson_finish_yields_the_unterminated_last_line() {
    let mut framer = NdjsonFramer::new();
    assert!(framer.push(b"{\"a\":1}").next().is_none());
    assert_eq!(framer.finish().as_deref(), Some(&b"{\"a\":1}"[..]));
    assert_eq!(framer.finish(), None);
}
