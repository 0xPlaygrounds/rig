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
    let message = |data: &str| ("message".to_string(), data.to_string(), String::new());
    assert_eq!(sse_all(b"data: one\n\n"), [message("one")]);
    assert_eq!(sse_all(b"data: one\r\n\r\n"), [message("one")]);
    // A CR ends its line at once, so a body ending on one dispatches.
    assert_eq!(
        sse_all(b"data: one\r\rdata: two\r\r"),
        [message("one"), message("two")]
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
        "retry: 12\nevent: b\r\ndata: last\r\n\r\n",
        "data: open\r\n",
    )
    .as_bytes();
    let mut whole = SseFramer::new();
    let events = |framer: &mut SseFramer, chunk: &[u8]| -> Vec<_> {
        framer
            .push(chunk)
            .map(|event| (event.event.into_owned(), event.data, event.id))
            .collect()
    };
    let expected = events(&mut whole, body);
    assert_eq!(expected.len(), 4);
    assert_eq!(whole.pending(), b"data: open\r\n".len());
    for split in 0..=body.len() {
        let (head, tail) = body.split_at(split);
        let mut framer = SseFramer::new();
        let mut split_events = events(&mut framer, head);
        split_events.extend(events(&mut framer, b""));
        split_events.extend(events(&mut framer, tail));
        assert_eq!(split_events, expected, "split at {split}");
        assert_eq!(framer.pending(), whole.pending(), "split at {split}");
    }
    let mut framer = SseFramer::new();
    let mut bytewise = Vec::new();
    for byte in body {
        bytewise.extend(events(&mut framer, &[*byte]));
    }
    assert_eq!(bytewise, expected, "one byte at a time");
    assert_eq!(framer.pending(), whole.pending(), "one byte at a time");
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
fn ndjson_finish_drops_a_trailing_cr() {
    let mut framer = NdjsonFramer::new();
    assert_eq!(framer.push(b"{}\n{\"a\":1}\r").count(), 1);
    assert_eq!(framer.finish(), Some(b"{\"a\":1}".to_vec()));
    let mut framer = NdjsonFramer::new();
    assert_eq!(framer.push(b"{}\n\r").count(), 1);
    assert_eq!(framer.finish(), None);
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

#[test]
fn a_retry_beyond_u64_saturates() {
    let mut framer = SseFramer::new();
    let retries: Vec<_> = framer
        .push(b"retry: 500\nretry: 99999999999999999999\ndata: one\n\n")
        .map(|event| event.retry)
        .collect();
    assert_eq!(retries, [Some(u64::MAX)]);
}
