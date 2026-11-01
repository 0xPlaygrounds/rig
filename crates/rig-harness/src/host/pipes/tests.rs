use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use rig_ecs::tasks::Wake;

use super::{BACKLOG_LIMIT, DRAIN, Limited, ROOM, Stream, TooLarge, Writer, lines, read_lines};

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

/// A reader that takes its first write, then stops reading until `0`.
struct OnceThenStalled(bool, Arc<AtomicBool>);

impl Write for OnceThenStalled {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        while std::mem::replace(&mut self.0, true) && !self.1.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A reader that reads into `1` once `0` lets it.
struct Gated(Arc<AtomicBool>, Slow);

impl Write for Gated {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        while !self.0.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(5));
        }
        self.1.write(bytes)
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

/// Without an interruption the end waits for a reader that is behind, so
/// nothing is cut short; room wakes the loop and every task waiting for it.
#[test]
fn a_normal_end_waits_for_the_reader_and_room_wakes_every_waiter() {
    let woken = Arc::new(AtomicUsize::new(0));
    let counted = woken.clone();
    let wake = Wake::new(move || {
        counted.fetch_add(1, Ordering::AcqRel);
    });
    let (go, read) = (Arc::new(AtomicBool::new(false)), Slow::default());
    let reader = Gated(go.clone(), read.clone());
    let writer = Writer::spawn(reader, Slow::default(), wake, || false);
    writer.write(Stream::Out, vec![b'a'; ROOM + 1]);
    writer.write(Stream::Out, b"end".to_vec());
    assert!(!writer.has_room());
    // Both wait before room comes; one token would wake only one.
    let mut waiters = [Box::pin(writer.room()), Box::pin(writer.room())];
    for waiter in &mut waiters {
        let poll = futures::executor::block_on(async { futures::poll!(waiter.as_mut()) });
        assert!(poll.is_pending());
    }
    go.store(true, Ordering::Release);
    writer.finish();
    for waiter in waiters {
        futures::executor::block_on(waiter);
    }
    let read = read.0.lock().map(|read| read.len()).unwrap_or_default();
    assert_eq!(read, ROOM + 4);
    assert!(writer.has_room());
    assert!(woken.load(Ordering::Acquire) >= 1);
}

/// More than `BACKLOG_LIMIT` waiting for a reader that never reads counts
/// as the reader gone: the end waits at most `DRAIN`.
#[test]
fn a_backlog_past_its_limit_ends_within_the_drain() {
    let go = Arc::new(AtomicBool::new(false));
    let writer = Writer::spawn(
        Stalled(go.clone()),
        Slow::default(),
        Wake::default(),
        || false,
    );
    writer.write(Stream::Out, vec![b'x'; BACKLOG_LIMIT + 1]);
    assert!(writer.is_gone());
    futures::executor::block_on(writer.gone());
    let ending = Instant::now();
    writer.finish();
    assert!(ending.elapsed() < DRAIN + Duration::from_secs(1));
    go.store(true, Ordering::Release);
}

/// A backlog that passed its limit has no reader, even once it drained a
/// little: the end still waits at most `DRAIN`.
#[test]
fn a_backlog_that_passed_its_limit_stays_gone() {
    let go = Arc::new(AtomicBool::new(false));
    let reader = OnceThenStalled(false, go.clone());
    let writer = Writer::spawn(reader, Slow::default(), Wake::default(), || false);
    writer.write(Stream::Out, vec![b'x'; BACKLOG_LIMIT + 1]);
    writer.write(Stream::Out, b"stalled".to_vec());
    let since = Instant::now();
    let past = || writer.0.queue().backlog > BACKLOG_LIMIT;
    while past() && since.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!past() && writer.is_gone());
    let ending = Instant::now();
    writer.finish();
    assert!(ending.elapsed() < DRAIN + Duration::from_secs(1));
    go.store(true, Ordering::Release);
}

/// Nothing is written after the end, in place or by a thread alike.
#[test]
fn nothing_is_written_after_the_end() {
    let written = Slow::default();
    let inline = Writer::inline(written.clone(), std::io::sink());
    inline.finish();
    inline.write(Stream::Out, b"late".to_vec());
    let spawned = Writer::spawn(written.clone(), std::io::sink(), Wake::default(), || false);
    spawned.finish();
    spawned.write(Stream::Out, b"late".to_vec());
    std::thread::sleep(Duration::from_millis(100));
    let written = written
        .0
        .lock()
        .map(|written| written.clone())
        .unwrap_or_default();
    assert!(written.is_empty());
}

/// A pipe that broke drops only its own bytes: the other one is written;
/// a broken stdout says so.
#[test]
fn a_broken_pipe_drops_only_its_own_bytes() {
    let out = Slow::default();
    let writer = Writer::spawn(out.clone(), Broken, Wake::default(), || false);
    writer.write(Stream::Err, b"lost\n".to_vec());
    writer.write(Stream::Out, b"answer\n".to_vec());
    writer.finish();
    let broken = Writer::inline(Broken, std::io::sink());
    broken.write(Stream::Out, b"lost\n".to_vec());
    futures::executor::block_on(broken.gone());
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

/// A pipe's bytes pass as they are until a line is past the limit, which
/// ends them there with an error and is noted; a CRLF end does not count,
/// and the lines after the refused one never come.
#[test]
fn limited_bytes_stop_at_a_line_past_the_limit() {
    use tokio::io::{AsyncRead, ReadBuf};

    let input: &[u8] = b"abcd\r\nefgh\nijklm\nnext\n";
    let refused = Arc::new(Mutex::new(None));
    let mut limited = Limited::new(input, 4, refused.clone());
    let mut read = Vec::new();
    let failed = loop {
        let mut chunk = [0_u8; 3];
        let mut buf = ReadBuf::new(&mut chunk);
        let poll =
            std::future::poll_fn(|cx| std::pin::Pin::new(&mut limited).poll_read(cx, &mut buf));
        if futures::executor::block_on(poll).is_err() {
            break true;
        }
        if buf.filled().is_empty() {
            break false;
        }
        read.extend_from_slice(buf.filled());
    };
    assert!(
        failed,
        "the cut line ends in an error, never as a whole line"
    );
    assert_eq!(read, b"abcd\r\nefgh\nijkl");
    let refused = *refused
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(refused, Some(TooLarge { limit: 4, asked: 5 }));
}
