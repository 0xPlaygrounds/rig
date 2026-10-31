use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use rig_ecs::tasks::Wake;

use super::{DRAIN, ROOM, Stream, TooLarge, Writer, lines, read_lines};

/// A reader that stopped reading: every write waits until `go`.
struct Stalled(Arc<AtomicBool>);

impl Write for Stalled {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        while !self.0.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// What a slow reader read, a write at a time.
#[derive(Clone, Default)]
struct Slow(Arc<Mutex<Vec<u8>>>);

impl Write for Slow {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        std::thread::sleep(Duration::from_millis(30));
        if let Ok(mut read) = self.0.lock() {
            read.extend_from_slice(bytes);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A pipe whose reader is gone.
struct Broken;

impl Write for Broken {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A stalled reader never holds the frame: handing bytes returns at once,
/// the front is told to hold back, and once the run is interrupted the
/// end waits at most `DRAIN`.
#[test]
fn a_stalled_reader_holds_no_write_and_an_interrupted_end_is_bounded() {
    let (go, interrupted) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let signal = interrupted.clone();
    let writer = Writer::spawn(
        Stalled(go.clone()),
        Slow::default(),
        Wake::default(),
        move || signal.load(Ordering::Acquire),
    );
    let started = Instant::now();
    for _ in 0..3 {
        writer.write(Stream::Out, vec![b'x'; ROOM]);
    }
    assert!(started.elapsed() < Duration::from_secs(1));
    assert!(!writer.has_room());
    interrupted.store(true, Ordering::Release);
    let ending = Instant::now();
    writer.finish();
    assert!(ending.elapsed() < DRAIN + Duration::from_secs(1));
    go.store(true, Ordering::Release);
}

/// Without an interruption the end waits for a slow reader, so nothing is
/// cut short, and the loop is woken once room came back.
#[test]
fn a_normal_end_waits_for_a_slow_reader_and_room_wakes_the_loop() {
    let woken = Arc::new(AtomicUsize::new(0));
    let counted = woken.clone();
    let wake = Wake::new(move || {
        counted.fetch_add(1, Ordering::AcqRel);
    });
    let read = Slow::default();
    let writer = Writer::spawn(read.clone(), Slow::default(), wake, || false);
    writer.write(Stream::Out, vec![b'a'; ROOM + 1]);
    writer.write(Stream::Out, b"end".to_vec());
    assert!(!writer.has_room());
    writer.finish();
    let read = read.0.lock().map(|read| read.len()).unwrap_or_default();
    assert_eq!(read, ROOM + 4);
    assert!(writer.has_room());
    assert!(woken.load(Ordering::Acquire) >= 1);
    writer.write(Stream::Out, b"late".to_vec());
    assert!(writer.has_room());
}

/// A pipe that broke drops only its own bytes: the other one is written.
#[test]
fn a_broken_pipe_drops_only_its_own_bytes() {
    let out = Slow::default();
    let writer = Writer::spawn(out.clone(), Broken, Wake::default(), || false);
    writer.write(Stream::Err, b"lost\n".to_vec());
    writer.write(Stream::Out, b"answer\n".to_vec());
    writer.finish();
    let read = out.0.lock().map(|read| read.clone()).unwrap_or_default();
    assert_eq!(read, b"answer\n");
}

/// A line past the limit is refused in its place, and the lines after it
/// still come, as does the unterminated last one.
#[test]
fn the_line_readers_refuse_a_long_line_and_read_on() {
    let input: &[u8] = b"abcd\nabcdefgh\nnext\nlast";
    let expected = vec![
        Ok(b"abcd".to_vec()),
        Err(TooLarge { limit: 4, asked: 5 }),
        Ok(b"next".to_vec()),
        Ok(b"last".to_vec()),
    ];
    let read = read_lines(input, 4);
    let mut got = Vec::new();
    while let Ok(line) = read.recv_blocking() {
        got.push(line);
    }
    assert_eq!(got, expected);
    let got: Vec<_> = futures::executor::block_on(lines(input, 4).collect());
    assert_eq!(got, expected);
}
