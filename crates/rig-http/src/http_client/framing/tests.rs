use super::{Framing, NdjsonFramer, SseFramer, WireFrame};

/// Every event a framer dispatches over one whole body.
fn sse_all(body: &[u8]) -> Vec<(String, String, String)> {
    let mut framer = SseFramer::new();
    framer
        .push(body)
        .map(|event| (event.event.into_owned(), event.data, event.id))
        .collect()
}

#[test]
fn every_line_ending_terminates_a_line() {
    let expected = vec![("message".into(), "one".into(), String::new())];
    assert_eq!(sse_all(b"data: one\n\n"), expected);
    assert_eq!(sse_all(b"data: one\r\n\r\n"), expected);
    // Bare CR ends a line too, so both events dispatch: a CR does not wait
    // for an LF that never comes.
    assert_eq!(
        sse_all(b"data: one\r\rdata: two\r\r"),
        vec![
            ("message".into(), "one".into(), String::new()),
            ("message".into(), "two".into(), String::new()),
        ]
    );
}

#[test]
fn a_leading_bom_is_not_data() {
    assert_eq!(
        sse_all("\u{feff}data: one\n\n".as_bytes()),
        vec![("message".into(), "one".into(), String::new())]
    );
}

#[test]
fn a_bom_split_across_chunks_is_still_ignored() {
    let mut framer = SseFramer::new();
    assert_eq!(framer.push(&[0xef]).len(), 0);
    assert_eq!(framer.push(&[0xbb]).len(), 0);
    let events: Vec<_> = framer.push(&[0xbf]).collect();
    assert!(events.is_empty());
    let events: Vec<_> = framer.push(b"data: one\n\n").collect();
    assert_eq!(events.first().map(|event| event.data.as_str()), Some("one"));
}

#[test]
fn a_leading_colon_is_a_comment() {
    assert_eq!(
        sse_all(b": keep-alive\ndata: one\n\n"),
        vec![("message".into(), "one".into(), String::new())]
    );
}

#[test]
fn an_id_persists_across_events() {
    assert_eq!(
        sse_all(b"id: 7\ndata: one\n\ndata: two\n\n"),
        vec![
            ("message".into(), "one".into(), "7".into()),
            ("message".into(), "two".into(), "7".into()),
        ]
    );
}

#[test]
fn a_retry_does_not_persist_across_events() {
    let mut framer = SseFramer::new();
    let retries: Vec<_> = framer
        .push(b"retry: 500\ndata: one\n\ndata: two\n\n")
        .map(|event| event.retry)
        .collect();
    assert_eq!(retries, vec![Some(500), None]);
}

#[test]
fn a_blank_line_without_data_dispatches_nothing_and_resets_the_type() {
    assert_eq!(
        sse_all(b"event: ping\n\ndata: one\n\n"),
        vec![("message".into(), "one".into(), String::new())]
    );
}

#[test]
fn a_truncated_trailing_event_is_pending_bytes_never_a_frame() {
    let mut framer = SseFramer::new();
    let events: Vec<_> = framer
        .push(b"data: one\n\nevent: message_stop\ndata: {}")
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(framer.pending(), b"event: message_stop\ndata: {}".len());
}

/// The two cassette-derived shapes: a body whose last event has no trailing
/// blank line delivers one event fewer than it spells, on any chunking.
#[test]
fn a_body_whose_last_event_lacks_its_blank_line_delivers_one_event_fewer() {
    let body = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}"
    );
    assert_eq!(sse_all(body.as_bytes()).len(), 1);
    let terminated = format!("{body}\n\n");
    assert_eq!(sse_all(terminated.as_bytes()).len(), 2);
}

#[test]
fn splitting_a_body_at_every_offset_yields_the_same_events() {
    let body = concat!(
        "\u{feff}: hello\r\n",
        "event: a\r\ndata: {\"n\":1}\r\n\r\n",
        "id: x\ndata: line one\ndata: line two\n\n",
        "\rdata: after a bare cr\r\r",
        "retry: 12\nevent: b\ndata: last\n\n",
    )
    .as_bytes();
    let whole = sse_all(body);
    assert_eq!(whole.len(), 4);
    for split in 0..=body.len() {
        let (head, tail) = body.split_at(split);
        let mut framer = SseFramer::new();
        let mut events: Vec<_> = framer
            .push(head)
            .map(|event| (event.event.into_owned(), event.data, event.id))
            .collect();
        events.extend(
            framer
                .push(tail)
                .map(|event| (event.event.into_owned(), event.data, event.id)),
        );
        assert_eq!(events, whole, "split at {split}");
    }
    let mut framer = SseFramer::new();
    let mut bytewise = Vec::new();
    for byte in body {
        bytewise.extend(
            framer
                .push(&[*byte])
                .map(|event| (event.event.into_owned(), event.data, event.id)),
        );
    }
    assert_eq!(bytewise, whole, "one byte at a time");
}

#[test]
fn ndjson_yields_terminated_lines_and_flushes_the_last_one() {
    let mut framer = NdjsonFramer::new();
    let lines: Vec<_> = framer.push(b"{\"a\":1}\n\n{\"b\":2}\n{\"c\"").collect();
    assert_eq!(lines, vec![b"{\"a\":1}".to_vec(), b"{\"b\":2}".to_vec()]);
    assert_eq!(framer.pending(), 4);
    assert_eq!(framer.finish(), Some(b"{\"c\"".to_vec()));
    assert_eq!(framer.finish(), None);
}

#[test]
fn ndjson_lines_split_across_chunks_rejoin() {
    let mut framer = NdjsonFramer::new();
    assert_eq!(framer.push(b"{\"a\":").len(), 0);
    let lines: Vec<_> = framer.push(b"1}\r\n").collect();
    assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
    let mut lines = Vec::new();
    for byte in b"{\"b\":2}\r\n\n{\"c\":3}\n{\"d\"" {
        lines.extend(framer.push(&[*byte]));
    }
    assert_eq!(lines, vec![b"{\"b\":2}".to_vec(), b"{\"c\":3}".to_vec()]);
    assert_eq!(framer.finish(), Some(b"{\"d\"".to_vec()));
}

#[test]
fn a_whitespace_only_event_is_a_heartbeat_and_never_a_frame() {
    let Some(mut framer) = Framing::Sse.framer() else {
        panic!("SSE is framed incrementally");
    };
    let frames: Vec<_> = framer.push(b"data:  \n\ndata: {}\n\ndata: {").collect();
    assert_eq!(frames, [WireFrame::Text("{}".into())]);
    assert_eq!(framer.pending(), b"data: {".len());
    assert_eq!(framer.finish(), None);
}

/// A reply whose last chunk ends after the CR of the final CRLF still
/// dispatches the body's last event: that CR is a line end of its own, and no
/// LF follows it before the stream ends.
#[test]
fn a_final_bare_cr_dispatches_the_last_event() {
    assert_eq!(
        sse_all(b"data: [DONE]\r\r"),
        vec![("message".into(), "[DONE]".into(), String::new())]
    );
    let mut framer = SseFramer::new();
    let events: Vec<_> = framer
        .push(b"data: one\r\n\r\n")
        .map(|event| event.data)
        .collect();
    assert_eq!(events, ["one"]);
    assert_eq!(framer.pending(), 0);
}

/// A CRLF split between chunks is one line end, not a line end followed by a
/// blank line, so it cannot reset the event type between an event's fields.
#[test]
fn a_crlf_split_between_chunks_stays_one_line_end() {
    let mut framer = SseFramer::new();
    let mut events = Vec::new();
    for chunk in [&b"event: ping\r"[..], b"\ndata: one\r\n\r", b"\n"] {
        events.extend(
            framer
                .push(chunk)
                .map(|event| (event.event.into_owned(), event.data)),
        );
    }
    assert_eq!(events, vec![("ping".to_string(), "one".to_string())]);
}

/// A body that ends after a field line's CR has no blank line, so its event
/// stays pending: the CR ends the field line, not the event.
#[test]
fn a_body_ending_after_a_field_lines_cr_is_still_incomplete() {
    let mut framer = SseFramer::new();
    assert_eq!(framer.push(b"data: one\r").len(), 0);
    assert_eq!(framer.pending(), b"data: one\r".len());
}

#[test]
fn split_delivers_the_event_a_bare_cr_body_ends_with() {
    assert_eq!(
        Framing::Sse.split(b"data: one\r\r"),
        vec![WireFrame::Text("one".into())]
    );
}

/// A `retry` value the grammar accepts but `u64` cannot hold still names a
/// reconnection time: it saturates instead of clearing the pending value the
/// way a malformed field is ignored.
#[test]
fn a_retry_beyond_u64_saturates() {
    let mut framer = SseFramer::new();
    let retries: Vec<_> = framer
        .push(b"retry: 500\ndata: one\n\nretry: 99999999999999999999\ndata: two\n\n")
        .map(|event| event.retry)
        .collect();
    assert_eq!(retries, vec![Some(500), Some(u64::MAX)]);
}
