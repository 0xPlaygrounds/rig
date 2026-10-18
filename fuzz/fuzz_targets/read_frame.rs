//! Any bytes, read as a stream of frames: reading never panics, never
//! returns a line over MAX_FRAME, always moves on after a bad line, and
//! every frame it accepts writes back to bytes that read as the same frame.
#![no_main]

use libfuzzer_sys::fuzz_target;
use rig_protocol::{Frame, FrameError, MAX_FRAME, read_frame, write_frame};

fuzz_target!(|data: &[u8]| {
    let mut reader = data;
    let mut line = Vec::new();
    // Each call consumes at least one byte or ends, so this terminates.
    for _ in 0..=data.len() {
        let before = reader.len();
        match read_frame(&mut reader, &mut line) {
            Ok(None) => break,
            Ok(Some(frame)) => {
                assert!(line.len() <= MAX_FRAME);
                let mut wire = Vec::new();
                write_frame(&mut wire, &frame).expect("a read frame writes");
                let mut again = Vec::new();
                let reread = read_frame(&mut wire.as_slice(), &mut again)
                    .expect("a written frame reads")
                    .expect("a written frame is not EOF");
                assert_eq!(
                    serde_json::to_value(&frame).ok(),
                    serde_json::to_value(&reread).ok(),
                    "a frame round-trips"
                );
                let _ = matches!(frame, Frame::Request(_));
            }
            Err(FrameError::Io(error)) => panic!("reading a slice failed: {error}"),
            Err(error) => {
                // Every refusal but I/O has an answer for the peer.
                assert!(error.rpc().is_some());
            }
        }
        assert!(reader.len() < before || reader.is_empty(), "no progress");
    }
});
