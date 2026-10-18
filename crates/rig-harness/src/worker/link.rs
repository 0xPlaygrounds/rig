//! The worker's connection to `rigd`: requests answered on a reader thread,
//! and what `rigd` sends unasked (commands, the events of the sessions the
//! worker subscribed to) queued for the app, which the reader wakes.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use rig_ecs::tasks::Wake;
use rig_protocol::methods::{EVENT, WORKER_COMMAND, WorkerCommand};
use rig_protocol::{
    EventRecord, Frame, Method, Notification, Outcome, Request, RequestId, RpcError, codes,
    read_frame, write_frame,
};

/// How long a request waits for its answer before it fails.
const ANSWER_WITHIN: Duration = Duration::from_secs(60);

/// What `rigd` sent unasked, in the order it came.
pub(crate) enum Inbound {
    Command(WorkerCommand),
    Event(EventRecord),
    /// The connection closed: `rigd` is gone.
    Closed,
}

type Answer = Sender<Result<serde_json::Value, RpcError>>;

/// The requests not answered yet: by id, who waits for the answer, if
/// anyone does, and the method's name.
type Pending = Arc<Mutex<HashMap<u64, (Option<Answer>, &'static str)>>>;

/// One connection, shared by the app and the threads tools call from.
pub(crate) struct Link {
    writer: Mutex<Box<dyn Write + Send>>,
    next_id: AtomicU64,
    pending: Pending,
}

impl Link {
    /// Starts the reader thread on `reader`; what it reads unasked goes to
    /// the returned receiver, each followed by a `wake`.
    pub(crate) fn start(
        reader: impl BufRead + Send + 'static,
        writer: impl Write + Send + 'static,
        wake: Wake,
    ) -> (Arc<Link>, Receiver<Inbound>) {
        let (inbound, received) = unbounded();
        let pending: Pending = Arc::default();
        let link = Arc::new(Link {
            writer: Mutex::new(Box::new(writer)),
            next_id: AtomicU64::new(1),
            pending: pending.clone(),
        });
        std::thread::Builder::new()
            .name("rig worker channel".into())
            .spawn(move || read(reader, &pending, &inbound, &wake))
            .ok();
        (link, received)
    }

    /// Calls `M` and waits for its answer.
    pub(crate) fn call<M: Method>(&self, params: &M::Params) -> Result<M::Result, RpcError> {
        let (answer, answered) = bounded(1);
        self.request(M::NAME, params, Some(answer))?;
        let value = answered
            .recv_timeout(ANSWER_WITHIN)
            .map_err(|_| RpcError::new(codes::INTERNAL, format!("{} got no answer", M::NAME)))??;
        serde_json::from_value(value)
            .map_err(|e| RpcError::new(codes::INTERNAL, format!("{}: {e}", M::NAME)))
    }

    /// Calls `M` without waiting: a failure is logged when it comes.
    pub(crate) fn send<M: Method>(&self, params: &M::Params) {
        if let Err(e) = self.request(M::NAME, params, None) {
            bevy_log::warn!("could not send {}: {e}", M::NAME);
        }
    }

    fn request(
        &self,
        method: &'static str,
        params: &impl serde::Serialize,
        answer: Option<Answer>,
    ) -> Result<(), RpcError> {
        let params = serde_json::to_value(params)
            .map_err(|e| RpcError::new(codes::INVALID_PARAMS, e.to_string()))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        lock(&self.pending).insert(id, (answer, method));
        let frame = Frame::Request(Request::new(RequestId::Num(id), method, params));
        let written = write_frame(&mut *lock(&self.writer), &frame);
        if let Err(e) = written {
            lock(&self.pending).remove(&id);
            return Err(RpcError::new(codes::INTERNAL, format!("rigd is gone: {e}")));
        }
        Ok(())
    }
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Reads frames until the connection closes.
fn read(mut reader: impl BufRead, pending: &Pending, inbound: &Sender<Inbound>, wake: &Wake) {
    let mut line = Vec::new();
    loop {
        let frame = match read_frame(&mut reader, &mut line) {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(e) => {
                bevy_log::warn!("rigd sent a frame the worker cannot read: {e}");
                continue;
            }
        };
        match frame {
            Frame::Response(response) => {
                let RequestId::Num(id) = response.id else {
                    continue;
                };
                let Some((answer, method)) = lock(pending).remove(&id) else {
                    continue;
                };
                let outcome = match response.outcome {
                    Outcome::Result(value) => Ok(value),
                    Outcome::Error(e) => Err(e),
                };
                match (answer, outcome) {
                    (Some(answer), outcome) => drop(answer.send(outcome)),
                    (None, Err(e)) => bevy_log::warn!("rigd refused {method}: {e}"),
                    (None, Ok(_)) => {}
                }
            }
            Frame::Notification(note) => {
                if let Some(item) = unasked(note) {
                    inbound.send(item).ok();
                    wake.wake();
                }
            }
            // rigd asks a worker nothing: its calls come as commands.
            Frame::Request(_) => {}
        }
    }
    lock(pending).clear();
    inbound.send(Inbound::Closed).ok();
    wake.wake();
}

fn unasked(note: Notification) -> Option<Inbound> {
    match note.method.as_str() {
        WORKER_COMMAND => serde_json::from_value(note.params)
            .map(Inbound::Command)
            .map_err(|e| bevy_log::warn!("rigd sent a command the worker cannot read: {e}"))
            .ok(),
        EVENT => serde_json::from_value(note.params).ok().map(Inbound::Event),
        _ => None,
    }
}
