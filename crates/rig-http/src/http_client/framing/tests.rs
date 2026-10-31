use super::{Framing, NdjsonFramer, SseFramer, TooLarge, WireFrame};

/// Every event a framer dispatches over one whole body.
fn sse_all(body: &[u8]) -> Vec<(String, String, String)> {
    let mut framer = SseFramer::new();
    framer
        .push(body)
        .flatten()
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
    let events: Vec<_> = framer.push(b"data: one\n\n").flatten().collect();
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
        .flatten()
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
            .flatten()
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
    assert_eq!(
        lines,
        vec![Ok(b"{\"a\":1}".to_vec()), Ok(b"{\"b\":2}".to_vec())]
    );
    assert_eq!(framer.pending(), 4);
    assert_eq!(framer.finish(), Some(b"{\"c\"".to_vec()));
    assert_eq!(framer.finish(), None);
}

#[test]
fn ndjson_lines_split_across_chunks_rejoin() {
    let mut framer = NdjsonFramer::new();
    assert_eq!(framer.push(b"{\"a\":").len(), 0);
    let lines: Vec<_> = framer.push(b"1}\r\n").flatten().collect();
    assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
    let mut lines = Vec::new();
    for byte in b"{\"b\":2}\r\n\n{\"c\":3}\n{\"d\"" {
        lines.extend(framer.push(&[*byte]).flatten());
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
    let frames: Vec<_> = framer
        .push(b"data:  \n\ndata: {}\n\ndata: {")
        .flatten()
        .collect();
    assert_eq!(frames, [WireFrame::Text("{}".into())]);
    assert_eq!(framer.pending(), b"data: {".len());
    assert_eq!(framer.finish(), None);
}

#[test]
fn a_retry_beyond_u64_saturates() {
    let mut framer = SseFramer::new();
    let retries: Vec<_> = framer
        .push(b"retry: 500\nretry: 99999999999999999999\ndata: one\n\n")
        .flatten()
        .map(|event| event.retry)
        .collect();
    assert_eq!(retries, [Some(u64::MAX)]);
}

#[test]
fn recording_ranges_and_tail_use_the_normal_grammar_at_every_chunk_boundary() {
    let body = "\u{feff}id: 7\rdata: one\r\rdata: two\r\n\r\nevent: last\ndata: tail";
    for split in 0..=body.len() {
        let (head, tail) = body.as_bytes().split_at(split);
        let mut framer = SseFramer::new();
        let mut events = Vec::new();
        let mut lines = Vec::new();
        for chunk in [head, tail] {
            let batch = framer.push_recording(chunk);
            events.extend(batch.events);
            lines.extend(batch.lines);
        }
        let batch = framer.finish_recording();
        events.extend(batch.events);
        lines.extend(batch.lines);
        let payloads: Vec<_> = events
            .iter()
            .map(|frame| (frame.event.data.as_str(), frame.event.id.as_str()))
            .collect();
        assert_eq!(
            payloads,
            [("one", "7"), ("two", "7"), ("tail", "7")],
            "split {split}"
        );
        assert_eq!(events.last().map(|frame| frame.end), Some(body.len()));
        assert_eq!(
            events.last().map(|frame| frame.event.event.as_ref()),
            Some("last")
        );
        assert_eq!(lines.first().map(|line| line.content.clone()), Some(3..8));
        let data: Vec<_> = lines
            .iter()
            .filter(|line| line.field == super::SseField::Data)
            .filter_map(|line| line.value.clone().and_then(|range| body.get(range)))
            .collect();
        assert_eq!(data, ["one", "two", "tail"]);
        for line in lines {
            assert!(body.get(line.span.clone()).is_some());
            assert!(line.content.start >= line.span.start);
            assert!(line.content.end <= line.span.end);
        }
    }
    assert_eq!(sse_all(body.as_bytes()).len(), 2);
}

/// An SSE event that holds exactly the limit passes, on any chunking; one
/// byte more is refused once, with the events before it, and nothing after.
#[test]
fn an_sse_event_past_the_limit_is_refused_without_growing_past_it() {
    let line = "data: 0123456789";
    let limit = line.len();
    let fits = format!(": ping\n\n{line}\n\n");
    for size in [1, 3, fits.len()] {
        let mut framer = SseFramer::with_limit(limit);
        let mut events = Vec::new();
        for chunk in fits.as_bytes().chunks(size) {
            events.extend(framer.push(chunk));
        }
        let data: Vec<_> = events
            .into_iter()
            .map(|event| event.map(|event| event.data))
            .collect();
        assert_eq!(data, [Ok("0123456789".to_owned())], "chunks of {size}");
    }
    let over = format!("data: one\n\n{line}X\n\ndata: two\n\n");
    let mut framer = SseFramer::with_limit(limit);
    let events: Vec<_> = framer
        .push(over.as_bytes())
        .map(|event| event.map(|event| event.data))
        .collect();
    let refused = TooLarge {
        limit,
        asked: limit + 1,
    };
    assert_eq!(events, [Ok("one".to_owned()), Err(refused)]);
    assert!(framer.pending() <= limit + 1);
    assert_eq!(framer.push(b"data: three\n\n").count(), 0);
}

/// Heartbeats between events hold nothing, however many come.
#[test]
fn sse_heartbeats_never_fill_the_limit() {
    let mut framer = SseFramer::with_limit(16);
    for _ in 0..100 {
        assert_eq!(framer.push(b": keep-alive\n").count(), 0);
    }
    let events: Vec<_> = framer.push(b"data: {}\n\n").collect();
    assert_eq!(events.len(), 1);
    assert!(events.iter().all(Result::is_ok));
}

/// An NDJSON line of exactly the limit passes; a longer one is refused in
/// its place, without holding more than one byte past the limit, and the
/// lines after it still come.
#[test]
fn an_ndjson_line_past_the_limit_is_refused_and_reading_goes_on() {
    let mut framer = NdjsonFramer::with_limit(4);
    let lines: Vec<_> = framer.push(b"abcd\nabcde").collect();
    assert_eq!(
        lines,
        [Ok(b"abcd".to_vec()), Err(TooLarge { limit: 4, asked: 5 })]
    );
    assert_eq!(framer.pending(), 0);
    for byte in b"fghij" {
        assert_eq!(framer.push(&[*byte]).count(), 0);
        assert_eq!(framer.pending(), 0);
    }
    let lines: Vec<_> = framer.push(b"k\nnext\n").collect();
    assert_eq!(lines, [Ok(b"next".to_vec())]);
}

/// A line of exactly the limit passes with a CRLF end, its CR in a chunk of
/// its own; a CR inside a line counts.
#[test]
fn an_ndjson_crlf_end_does_not_count_against_the_limit() {
    let mut framer = NdjsonFramer::with_limit(4);
    assert_eq!(framer.push(b"null\r").count(), 0);
    let lines: Vec<_> = framer.push(b"\n").collect();
    assert_eq!(lines, [Ok(b"null".to_vec())]);
    let lines: Vec<_> = framer.push(b"nu\rll\n").collect();
    assert_eq!(lines, [Err(TooLarge { limit: 4, asked: 5 })]);
}

/// The last id is copied into every event it precedes, so it counts once
/// for each event a push holds: a large id cannot multiply in memory.
#[test]
fn an_sse_id_counts_for_each_event_it_is_copied_into() {
    let id = "x".repeat(10);
    let mut body = format!("id: {id}\n");
    for _ in 0..4 {
        body.push_str("data: a\n\n");
    }
    let mut framer = SseFramer::with_limit(32);
    let events: Vec<_> = framer.push(body.as_bytes()).collect();
    assert_eq!(events.iter().filter(|event| event.is_ok()).count(), 2);
    assert!(events.last().is_some_and(Result::is_err));
    let mut framer = SseFramer::with_limit(64);
    assert_eq!(framer.push(body.as_bytes()).flatten().count(), 4);
}
