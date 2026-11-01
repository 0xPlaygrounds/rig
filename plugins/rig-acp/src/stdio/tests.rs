use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use rig_harness::host::pipes::{ROOM, Stream, Writer};
use rig_harness::rig_ecs::tasks::Wake;

use super::incoming;

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

/// What was written, shared with the test.
#[derive(Clone, Default)]
struct Written(Arc<Mutex<Vec<u8>>>);

impl Write for Written {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut written) = self.0.lock() {
            written.extend_from_slice(bytes);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// While the peer does not read what waits for it, its lines are still
/// read, so a cancel or the end of its input comes through.
#[test]
fn lines_are_read_while_the_writer_is_behind() {
    let go = Arc::new(AtomicBool::new(false));
    let writer = Writer::spawn(
        Stalled(go.clone()),
        std::io::sink(),
        Wake::default(),
        || false,
    );
    writer.write(Stream::Out, vec![b'x'; ROOM + 1]);
    let input: &[u8] = b"{\"a\":1}\n{\"b\":2}\n";
    let lines =
        incoming(input, writer.clone(), 1024, "test").filter_map(|line| async { line.ok() });
    let lines: Vec<String> = futures::executor::block_on(lines.collect());
    assert_eq!(lines, ["{\"a\":1}", "{\"b\":2}"]);
    go.store(true, Ordering::Release);
}

/// A line that is not UTF-8 is said and ends the input, as one past the
/// limit does.
#[test]
fn a_line_not_utf8_ends_the_input_saying_so() {
    let said = Written::default();
    let writer = Writer::inline(std::io::sink(), said.clone());
    let input: &[u8] = b"{\"a\":1}\n{\"b\":\"\xff\"}\n{\"c\":3}\n";
    let lines = incoming(input, writer, 1024, "test").filter_map(|line| async { line.ok() });
    let lines: Vec<String> = futures::executor::block_on(lines.collect());
    assert_eq!(lines, ["{\"a\":1}"]);
    let said = said.0.lock().map(|said| said.clone()).unwrap_or_default();
    let said = String::from_utf8_lossy(&said);
    assert_eq!(
        said,
        "test: a line that is not UTF-8 came on stdin; reading ends\n"
    );
}
