//! A front's pipes. A [`Writer`] owns the front's stdout and stderr on a
//! thread of its own, so the frame never waits on a reader: the frame
//! hands it bytes, and holds back its work while more than [`ROOM`] waits.
//! [`read_lines`] and [`lines`] read a pipe as lines of at most a limit,
//! refusing a longer one ([`TooLarge`]) before holding more than it;
//! [`Limited`] passes a pipe's bytes on to a reader of its own until a line
//! passes the limit.
//!
//! ```no_run
//! use rig_harness::host::pipes::{Stream, Writer};
//! use rig_harness::rig_ecs::tasks::Wake;
//!
//! let writer = Writer::spawn(std::io::stdout(), std::io::stderr(), Wake::default(), || false);
//! writer.write(Stream::Out, b"{\"type\":\"ready\"}\n".to_vec());
//! writer.finish();
//! ```

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use rig_ecs::tasks::Wake;
use rig_http::http_client::framing::NdjsonFramer;
pub use rig_http::http_client::framing::TooLarge;
use tokio::io::{AsyncRead, ReadBuf};

/// The backlog under which a front's work goes on: past it, the front
/// holds back what makes more output until the reader caught up.
pub const ROOM: usize = 1024 * 1024;

/// The backlog past which the reader counts as gone: only a driver that
/// keeps sending requests and never reads their replies gets there, as the
/// front's work waits at [`ROOM`].
pub const BACKLOG_LIMIT: usize = 64 * 1024 * 1024;

/// How long the output may still take once the run was interrupted.
pub const DRAIN: Duration = Duration::from_millis(500);

/// How often a wait for the writer looks whether the run was interrupted.
const LOOK: Duration = Duration::from_millis(20);

/// Which of the front's pipes bytes go to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

/// Where a [`Writer`] writes, and whether a pipe failed, after which its
/// bytes are dropped while the other goes on.
struct Targets {
    out: Box<dyn Write + Send>,
    err: Box<dyn Write + Send>,
    out_failed: bool,
    err_failed: bool,
}

impl Targets {
    fn new(out: impl Write + Send + 'static, err: impl Write + Send + 'static) -> Self {
        Self {
            out: Box::new(out),
            err: Box::new(err),
            out_failed: false,
            err_failed: false,
        }
    }

    fn write(&mut self, stream: Stream, bytes: &[u8]) {
        let (target, failed) = match stream {
            Stream::Out => (&mut self.out, &mut self.out_failed),
            Stream::Err => (&mut self.err, &mut self.err_failed),
        };
        if !*failed && (target.write_all(bytes).is_err() || target.flush().is_err()) {
            *failed = true;
        }
    }
}

/// What waits for the writer thread.
#[derive(Default)]
struct Queue {
    items: VecDeque<(Stream, Vec<u8>)>,
    /// The bytes handed and not written yet, the one being written too.
    backlog: usize,
    /// Whether the backlog went past [`ROOM`] since the loop was last
    /// woken for room.
    full: bool,
    closed: bool,
    /// Whether the thread wrote everything and ended.
    done: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    changed: Condvar,
    /// Written in place, without a thread: a target that never blocks.
    inline: Option<Mutex<Targets>>,
    wake: Wake,
    interrupted: Box<dyn Fn() -> bool + Send + Sync>,
    /// Counts the times the backlog came back under [`ROOM`], for every
    /// task that waits for room.
    roomy: tokio::sync::watch::Sender<u64>,
    /// Whether nobody reads stdout in effect: writing it failed, or the
    /// backlog passed [`BACKLOG_LIMIT`].
    gone: tokio::sync::watch::Sender<bool>,
}

impl Shared {
    fn queue(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Notes that nobody reads stdout in effect, once `gone`.
    fn broke(&self, gone: bool) {
        if gone {
            self.gone
                .send_if_modified(|was| !std::mem::replace(was, true));
        }
    }
}

/// The one writer of a front's stdout and stderr. The frame hands it bytes
/// and never waits; a thread writes them in order. While its backlog is
/// past [`ROOM`] the front holds back its work ([`Writer::has_room`]), and
/// the thread wakes the loop once it is under again. Cloning shares it.
#[derive(Clone)]
pub struct Writer(Arc<Shared>);

impl Writer {
    /// A writer of `out` and `err` on a thread of its own; `wake` wakes the
    /// loop when room comes, and `interrupted` says the run was interrupted,
    /// which bounds [`Writer::finish`] by [`DRAIN`].
    pub fn spawn(
        out: impl Write + Send + 'static,
        err: impl Write + Send + 'static,
        wake: Wake,
        interrupted: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        let writer = Self::new(None, wake, Box::new(interrupted));
        let shared = writer.0.clone();
        let mut targets = Targets::new(out, err);
        let spawned = std::thread::Builder::new()
            .name("rig-output".to_owned())
            .spawn(move || write_all(&shared, &mut targets));
        if spawned.is_err() {
            // No thread: nothing will write, so nothing waits to be.
            writer.0.queue().done = true;
        }
        writer
    }

    /// A writer that writes `out` and `err` in place, for targets that never
    /// block, such as a test's buffer.
    pub fn inline(out: impl Write + Send + 'static, err: impl Write + Send + 'static) -> Self {
        let targets = Mutex::new(Targets::new(out, err));
        Self::new(Some(targets), Wake::default(), Box::new(|| false))
    }

    fn new(
        inline: Option<Mutex<Targets>>,
        wake: Wake,
        interrupted: Box<dyn Fn() -> bool + Send + Sync>,
    ) -> Self {
        Self(Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            changed: Condvar::new(),
            inline,
            wake,
            interrupted,
            roomy: tokio::sync::watch::Sender::new(0),
            gone: tokio::sync::watch::Sender::new(false),
        }))
    }

    /// Hands `bytes` to `stream`; never waits. Nothing is written after
    /// [`Writer::finish`], or when no thread could write.
    pub fn write(&self, stream: Stream, bytes: Vec<u8>) {
        let mut queue = self.0.queue();
        if queue.closed || queue.done {
            return;
        }
        if let Some(targets) = &self.0.inline {
            drop(queue);
            let mut targets = targets.lock().unwrap_or_else(PoisonError::into_inner);
            targets.write(stream, &bytes);
            self.0.broke(targets.out_failed);
            return;
        }
        queue.backlog = queue.backlog.saturating_add(bytes.len());
        queue.full |= queue.backlog > ROOM;
        self.0.broke(queue.backlog > BACKLOG_LIMIT);
        queue.items.push_back((stream, bytes));
        self.0.changed.notify_all();
    }

    /// Whether at most [`ROOM`] waits to be written: the front's work goes
    /// on only then.
    pub fn has_room(&self) -> bool {
        self.0.queue().backlog <= ROOM
    }

    /// Done once nobody reads stdout in effect: writing it failed, as when
    /// its reader closed it, or more than [`BACKLOG_LIMIT`] waits. A front
    /// ends then, as when its input ends.
    pub async fn gone(&self) {
        self.0.gone.subscribe().wait_for(|gone| *gone).await.ok();
    }

    /// Whether nobody reads stdout in effect ([`Writer::gone`]), now or
    /// ever since it was so.
    pub fn is_gone(&self) -> bool {
        *self.0.gone.borrow()
    }

    /// Done once at most [`ROOM`] waits; every task that waits is woken.
    pub async fn room(&self) {
        let mut roomy = self.0.roomy.subscribe();
        while !self.has_room() {
            if roomy.changed().await.is_err() {
                return;
            }
        }
    }

    /// Writes nothing more after what it was handed, and waits until that
    /// is written: for as long as the reader takes, so a normal end never
    /// cuts the output short, but once the run is interrupted, or more than
    /// [`BACKLOG_LIMIT`] waits, for at most [`DRAIN`] more.
    pub fn finish(&self) {
        let mut queue = self.0.queue();
        queue.closed = true;
        self.0.changed.notify_all();
        if self.0.inline.is_some() {
            return;
        }
        let mut deadline: Option<Instant> = None;
        while !queue.done {
            // Gone once is gone: a backlog that passed its limit and then
            // drained a little still has no reader.
            let gone = *self.0.gone.borrow();
            if deadline.is_none() && (gone || (self.0.interrupted)()) {
                deadline = Some(Instant::now() + DRAIN);
            }
            let wait = match deadline {
                Some(deadline) if Instant::now() >= deadline => return,
                Some(deadline) => deadline.saturating_duration_since(Instant::now()).min(LOOK),
                None => LOOK,
            };
            queue = match self.0.changed.wait_timeout(queue, wait) {
                Ok((queue, _)) => queue,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
    }
}

/// The writer thread: writes each item in order until the queue is closed
/// and empty, waking the loop when the backlog comes back under [`ROOM`].
fn write_all(shared: &Shared, targets: &mut Targets) {
    loop {
        let mut queue = shared.queue();
        while queue.items.is_empty() && !queue.closed {
            queue = shared
                .changed
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
        let Some((stream, bytes)) = queue.items.pop_front() else {
            queue.done = true;
            shared.changed.notify_all();
            return;
        };
        drop(queue);
        targets.write(stream, &bytes);
        shared.broke(targets.out_failed);
        let mut queue = shared.queue();
        queue.backlog = queue.backlog.saturating_sub(bytes.len());
        if queue.full && queue.backlog <= ROOM {
            queue.full = false;
            shared.roomy.send_modify(|times| *times += 1);
            shared.wake.wake();
        }
        shared.changed.notify_all();
    }
}

/// How many lines a reader keeps ready before it waits for them to be
/// taken.
const READY_LINES: usize = 64;

/// The size of each read from a pipe.
const CHUNK: usize = 64 * 1024;

/// Reads `read` on a thread of its own as lines of at most `limit` bytes
/// ([`NdjsonFramer::with_limit`]): each line, [`TooLarge`] in the place of
/// a longer one, whose rest is skipped, and the unterminated last line at
/// the end of the input (not after a read error). A few lines wait to be
/// taken at most; the thread waits for room before it reads on. The channel
/// closes at the end of the input, a read error, or once nobody takes the
/// lines.
pub fn read_lines(
    mut read: impl Read + Send + 'static,
    limit: usize,
) -> async_channel::Receiver<Result<Vec<u8>, TooLarge>> {
    let (sender, lines) = async_channel::bounded(READY_LINES);
    // No thread: the channel closes at once, as at the end of the input.
    let _ = std::thread::Builder::new()
        .name("rig-input".to_owned())
        .spawn(move || {
            let mut framer = NdjsonFramer::with_limit(limit);
            let mut chunk = vec![0; CHUNK];
            loop {
                let got = match read.read(&mut chunk) {
                    Ok(0) => break,
                    // A line a read error cut is no line.
                    Err(_) => return,
                    Ok(got) => got,
                };
                for line in framer.push(chunk.get(..got).unwrap_or_default()) {
                    if sender.send_blocking(line).is_err() {
                        return;
                    }
                }
            }
            if let Some(line) = framer.finish() {
                sender.send_blocking(Ok(line)).ok();
            }
        });
    lines
}

/// [`read_lines`] over an async pipe, such as a child's stdout, read as
/// the stream is polled.
pub fn lines<R: AsyncRead + Unpin>(read: R, limit: usize) -> Lines<R> {
    Lines {
        read,
        framer: NdjsonFramer::with_limit(limit),
        ready: VecDeque::new(),
        chunk: vec![0; CHUNK],
        ended: false,
    }
}

/// The stream [`lines`] returns.
pub struct Lines<R> {
    read: R,
    framer: NdjsonFramer,
    ready: VecDeque<Result<Vec<u8>, TooLarge>>,
    chunk: Vec<u8>,
    ended: bool,
}

impl<R: AsyncRead + Unpin> futures::Stream for Lines<R> {
    type Item = Result<Vec<u8>, TooLarge>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(line) = this.ready.pop_front() {
                return Poll::Ready(Some(line));
            }
            if this.ended {
                return Poll::Ready(None);
            }
            let mut buffer = ReadBuf::new(&mut this.chunk);
            match Pin::new(&mut this.read).poll_read(cx, &mut buffer) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(())) if !buffer.filled().is_empty() => {
                    let filled = buffer.filled().len();
                    let chunk = this.chunk.get(..filled).unwrap_or_default();
                    this.ready.extend(this.framer.push(chunk));
                }
                Poll::Ready(Ok(())) => {
                    this.ended = true;
                    this.ready.extend(this.framer.finish().map(Ok));
                }
                // A read error ends the input too; the line it cut is no line.
                Poll::Ready(Err(_)) => this.ended = true,
            }
        }
    }
}

/// An async pipe's bytes as they come, for a reader that frames them
/// itself, until a line is longer than its limit: the bytes stop there, and
/// the next read fails, so the reader never takes the cut line for a whole
/// one; its `refused` says why. A line counts its bytes before its line
/// end, LF or CRLF.
pub struct Limited<R> {
    read: R,
    limit: usize,
    /// The bytes of the line read so far.
    line: usize,
    /// Whether the last byte was a CR, which counts only if no LF follows.
    cr: bool,
    refused: Arc<Mutex<Option<TooLarge>>>,
}

impl<R> Limited<R> {
    /// `read`, refusing a line of more than `limit` bytes, which it notes in
    /// `refused`, shared so it can be read once the reader was handed on.
    pub fn new(read: R, limit: usize, refused: Arc<Mutex<Option<TooLarge>>>) -> Self {
        Self {
            read,
            limit,
            line: 0,
            cr: false,
            refused,
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for Limited<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let refused = *this.refused.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(refused) = refused {
            return Poll::Ready(Err(std::io::Error::other(refused)));
        }
        let before = buf.filled().len();
        std::task::ready!(Pin::new(&mut this.read).poll_read(cx, buf))?;
        let read = buf.filled().get(before..).unwrap_or_default();
        for (at, byte) in read.iter().enumerate() {
            match byte {
                b'\n' => (this.line, this.cr) = (0, false),
                b'\r' => this.line += usize::from(std::mem::replace(&mut this.cr, true)),
                _ => this.line += 1 + usize::from(std::mem::take(&mut this.cr)),
            }
            if this.line > this.limit {
                let refused = TooLarge {
                    limit: this.limit,
                    asked: this.line,
                };
                *this.refused.lock().unwrap_or_else(PoisonError::into_inner) = Some(refused);
                buf.set_filled(before + at);
                // A read that brings nothing would read as the end of a
                // whole last line: the refusal is an error at once.
                if at == 0 {
                    return Poll::Ready(Err(std::io::Error::other(refused)));
                }
                break;
            }
        }
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests;
