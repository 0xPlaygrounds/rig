use super::*;

#[test]
fn delimiters_bom_and_partial_frames_are_independent_of_byte_chunking() {
    for (bytes, pending) in [
        (&b"data: hi\n\n"[..], 0),
        (&b"data: hi\r\n\r\n"[..], 0),
        (&b"data: hi\r\r"[..], 0),
        (&b"data: hi\r\n\n"[..], 0),
        (&b"\xef\xbb\xbfdata: hi\n\n"[..], 0),
        (&b"\xef\xbb\xbf"[..], 0),
        (&b"\xef\xbb"[..], 2),
        (&b"data: hi\n\ndata: {"[..], 7),
        (&b"data: hi\r\n"[..], 10),
        (&b": heartbeat\n\ndata: hi\n\n"[..], 0),
        (&b"data: one\ndata: two\n"[..], 20),
    ] {
        for size in 1..=bytes.len() {
            let mut tail = SseTail::default();
            for chunk in bytes.chunks(size) {
                tail.feed(chunk);
            }
            assert_eq!(tail.pending(), pending, "{bytes:?}, chunks {size}");
        }
    }
}
