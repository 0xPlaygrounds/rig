#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]
//! The Rust client of the rig agent server (`rigd`). [`Client::connect`]
//! reaches the local server, starting it when none answers, or a remote one
//! through its `[remote.hosts]` transport; [`Client::call`] calls a typed
//! [`Method`]; [`Client::subscribe`] follows a session's events. It is sync,
//! on std threads, with no Bevy and no async runtime.
//!
//! When the connection drops, the client reconnects with backoff (starting
//! the server again when it is gone), says hello again, resends every call
//! still waiting for its answer that is safe to repeat, and resubscribes
//! every event stream from the last `seq` it received. A call is safe to
//! repeat when it carries an idempotency key (every mutating `rig/` method
//! does; the server answers a repeat with the original result) or changes
//! nothing; any other call, such as a plugin method without a `key`, fails
//! with [`ClientError::Unavailable`] saying it may have run. Each [`Events`] stream yields every event once, in
//! `seq` order.
//!
//! ```no_run
//! use rig_client::{Client, Target};
//! use rig_protocol::methods::{List, Subscribe};
//!
//! let client = Client::connect(&Target::Local { rig_home: "/home/me/.rig".into() }, "example")?;
//! for session in client.call::<List>(&List::default())? {
//!     println!("{} {:?}", session.session_id.0, session.state);
//! }
//! # Ok::<(), rig_client::ClientError>(())
//! ```

mod link;

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use rig_protocol::methods::{
    self, ClientInfo, EventNotice, Hello, HelloResult, Subscribe, Subscribed, SubscriptionId,
    Unsubscribe, VersionRange,
};
use rig_protocol::{
    EventRecord, Exit, Frame, Method, Notification, Outcome, Request, RequestId, RpcError,
    SUPPORTED, Seq, SessionId, Side, codes, read_frame, write_frame,
};

/// Where the server is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// The local `rigd` of this rig home, started when none answers.
    Local { rig_home: PathBuf },
    /// The `rigd` of another host, reached by running `transport` with
    /// `rig connect --stdio` as its last argument. The remote side's bridge
    /// authenticates as that host's user: the client's hello token is
    /// ignored there. [`Target::remote`] reads it from `[remote.hosts]`.
    Remote {
        host: String,
        transport: Vec<OsString>,
    },
}

impl Target {
    /// The host `[remote.hosts].<host>` of `rig_home/settings.toml` names.
    pub fn remote(rig_home: &std::path::Path, host: &str) -> Result<Target, ClientError> {
        Ok(Target::Remote {
            host: host.to_owned(),
            transport: link::transport(rig_home, host)?,
        })
    }
}

/// How a local client starts `rigd` when none answers. The default
/// ([`Autostart::detect`]) is the rig agent binary itself when this process
/// is one (its `RIG_ROLE` is set), else `$RIG_BIN serve`, else `rig serve`
/// from `PATH`: a library user's own program is never started as `rigd`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Autostart {
    /// This executable with `RIG_ROLE=server`: for the rig agent binary.
    CurrentExe,
    /// Another program with `RIG_ROLE=server`, such as `rig serve` or a
    /// test's server.
    Program {
        program: PathBuf,
        args: Vec<OsString>,
    },
    /// Never: an absent server is [`ClientError::Unavailable`].
    Off,
}

impl Default for Autostart {
    fn default() -> Autostart {
        Autostart::detect(
            std::env::var("RIG_ROLE").ok().as_deref(),
            std::env::var_os("RIG_BIN"),
        )
    }
}

impl Autostart {
    /// The autostart of a process whose `RIG_ROLE` is `role` and whose
    /// `RIG_BIN` (a rig launcher) is `rig_bin`.
    pub fn detect(role: Option<&str>, rig_bin: Option<OsString>) -> Autostart {
        match role {
            Some(role) if !role.is_empty() => Autostart::CurrentExe,
            _ => Autostart::Program {
                program: rig_bin
                    .filter(|bin| !bin.is_empty())
                    .map_or_else(|| PathBuf::from("rig"), PathBuf::from),
                args: vec![OsString::from("serve")],
            },
        }
    }
}

/// A callback the reader thread runs.
pub type Hook<T> = Arc<dyn Fn(T) + Send + Sync>;

/// How [`Client::connect_with`] connects and behaves.
#[derive(Clone)]
pub struct Options {
    /// The hello token. `None` reads `run/token` of a local target (a
    /// worker passes its own); a remote target sends none.
    pub token: Option<String>,
    pub autostart: Autostart,
    /// Whether a dropped connection is reconnected. A worker turns it off:
    /// it exits when `rigd` goes.
    pub reconnect: bool,
    /// How long a call or event stream waits for a lost server before it
    /// fails with [`ClientError::Unavailable`].
    pub give_up: Duration,
    /// Every notification but events, such as a worker's `worker/command`.
    pub on_notification: Option<Hook<Notification>>,
    /// Run after each frame the reader handles, and when the connection
    /// ends: a Bevy app wakes its runner here.
    pub wake: Option<Hook<()>>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            token: None,
            autostart: Autostart::default(),
            reconnect: true,
            give_up: Duration::from_secs(30),
            on_notification: None,
            wake: None,
        }
    }
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("token", &self.token.as_ref().map(|_| "…"))
            .field("autostart", &self.autostart)
            .field("reconnect", &self.reconnect)
            .field("give_up", &self.give_up)
            .finish_non_exhaustive()
    }
}

/// A connection to `rigd` that has said hello. Cheap to share across
/// threads (`&Client` calls from many at once); dropping it closes the
/// connection.
pub struct Client {
    shared: Arc<Shared>,
}

struct Shared {
    target: Target,
    name: String,
    options: Options,
    /// The current wire's writing half. Taken before `state` when both are
    /// held, and never held while waiting for the reader.
    writer: Mutex<Option<Box<dyn Write + Send>>>,
    state: Mutex<State>,
    changed: Condvar,
    next_id: AtomicU64,
}

struct State {
    link: Link,
    /// Bumped on every new connection.
    generation: u64,
    hello: Option<HelloResult>,
    pending: HashMap<u64, Pending>,
    subs: HashMap<u64, Sub>,
    /// Set by `Client::drop`.
    closed: bool,
}

enum Link {
    Up {
        closer: link::Closer,
    },
    /// Lost (or never made) at `since`; `error` is the last reason.
    Down {
        since: Instant,
        error: Option<String>,
    },
    /// No more connections: reconnect is off, or a refusal no retry fixes.
    Failed(Failure),
}

#[derive(Clone)]
enum Failure {
    Version { update: Side, supported: Vec<u32> },
    Rpc(RpcError),
    Gone(String),
}

impl Failure {
    fn error(&self) -> ClientError {
        match self {
            Failure::Version { update, supported } => ClientError::Version {
                update: *update,
                supported: supported.clone(),
            },
            Failure::Rpc(error) => ClientError::Rpc(error.clone()),
            Failure::Gone(why) => ClientError::Unavailable {
                next: format!(
                    "the rig server closed the connection ({why}); run `rig server status`"
                ),
            },
        }
    }
}

struct Pending {
    request: Request,
    /// The subscription this call makes, routed when it answers.
    subscribes: Option<u64>,
    /// When the caller gives up waiting for a lost server.
    started: Instant,
    /// Whether a caller waits for `done`; a resubscribe has none.
    waiter: bool,
    done: Option<Result<serde_json::Value, ClientError>>,
}

struct Sub {
    session: SessionId,
    /// The last `seq` received; a resubscribe starts after it.
    last: Seq,
    follow: bool,
    /// The server's id on the current connection, once it answered.
    server: Option<SubscriptionId>,
    tx: Sender<Result<EventRecord, ClientError>>,
}

/// The longest pause between reconnect attempts.
const BACKOFF_MAX: Duration = Duration::from_secs(2);
const BACKOFF_MIN: Duration = Duration::from_millis(20);

impl Client {
    /// Connects to `target` as client `name`, says hello and checks versions.
    pub fn connect(target: &Target, name: &str) -> Result<Client, ClientError> {
        Client::connect_with(target, name, Options::default())
    }

    /// [`connect`](Client::connect) with `options`.
    pub fn connect_with(
        target: &Target,
        name: &str,
        options: Options,
    ) -> Result<Client, ClientError> {
        let shared = Arc::new(Shared {
            target: target.clone(),
            name: name.to_owned(),
            options,
            state: Mutex::new(State {
                link: Link::Down {
                    since: Instant::now(),
                    error: None,
                },
                generation: 0,
                hello: None,
                pending: HashMap::new(),
                subs: HashMap::new(),
                closed: false,
            }),
            changed: Condvar::new(),
            writer: Mutex::new(None),
            next_id: AtomicU64::new(1),
        });
        let reader = shared.establish()?;
        let thread = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("rig-client".into())
            .spawn(move || thread.run(reader))?;
        Ok(Client { shared })
    }

    /// What the server said to this connection's hello (the latest one,
    /// after a reconnect).
    pub fn hello(&self) -> Option<HelloResult> {
        self.shared.lock().hello.clone()
    }

    /// This connection's id from `rig/hello`. A reconnect says hello again
    /// and gets a new one, so read it when needed rather than keeping it.
    pub fn id(&self) -> Option<rig_protocol::ClientId> {
        self.shared
            .lock()
            .hello
            .as_ref()
            .map(|hello| hello.client_id.clone())
    }

    /// Calls `M` and waits for its answer, across reconnects.
    pub fn call<M: Method>(&self, params: &M::Params) -> Result<M::Result, ClientError> {
        let value = self.call_raw(M::NAME, serde_json::to_value(params).map_err(io)?)?;
        serde_json::from_value(value).map_err(|error| {
            ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("the answer to {} does not parse: {error}", M::NAME),
            ))
        })
    }

    /// Calls `method` by name, for plugin methods (`x/<plugin>/<verb>`)
    /// this build has no type for.
    pub fn call_raw(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.shared.call(method, params, None)
    }

    /// `session`'s events after `after`, then live ones.
    pub fn subscribe(&self, session: &SessionId, after: Seq) -> Result<Events, ClientError> {
        self.subscribe_with(session, after, true)
    }

    /// [`subscribe`](Client::subscribe), ending after the journal's
    /// current end when `follow` is false.
    pub fn subscribe_with(
        &self,
        session: &SessionId,
        after: Seq,
        follow: bool,
    ) -> Result<Events, ClientError> {
        let local = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = crossbeam_channel::unbounded();
        self.shared.lock().subs.insert(
            local,
            Sub {
                session: session.clone(),
                last: after,
                follow,
                server: None,
                tx,
            },
        );
        let params = Subscribe {
            session_id: session.clone(),
            after_seq: after,
            follow,
        };
        let subscribed = self
            .shared
            .call(
                Subscribe::NAME,
                serde_json::to_value(&params).map_err(io)?,
                Some(local),
            )
            .and_then(|value| serde_json::from_value::<Subscribed>(value).map_err(io));
        match subscribed {
            Ok(subscribed) => {
                // Without follow the stream ends at the journal's end as of now.
                let end = (!follow).then(|| Seq(subscribed.next_seq.0.saturating_sub(1)));
                Ok(Events {
                    shared: Arc::downgrade(&self.shared),
                    local,
                    rx,
                    last: after,
                    end,
                    ended: end.is_some_and(|end| end <= after),
                })
            }
            Err(error) => {
                self.shared.lock().subs.remove(&local);
                Err(error)
            }
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.closed = true;
        if let Link::Up { closer, .. } = &state.link {
            closer.close();
        }
        self.shared.changed.notify_all();
    }
}

fn io(error: serde_json::Error) -> ClientError {
    ClientError::Io(error.into())
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn wake(&self) {
        if let Some(wake) = &self.options.wake {
            wake(());
        }
    }

    fn token(&self) -> Result<String, ClientError> {
        if let Some(token) = &self.options.token {
            return Ok(token.clone());
        }
        match &self.target {
            Target::Local { rig_home } => {
                rig_protocol::read_token(rig_home).map_err(|error| ClientError::Unavailable {
                    next: format!(
                        "could not read {} ({error}); run `rig server status`",
                        rig_home.join("run").join("token").display()
                    ),
                })
            }
            Target::Remote { .. } => Ok(String::new()),
        }
    }

    /// Opens a wire, says hello on it, and makes it the link, resending
    /// what waits. Returns the reading half.
    fn establish(&self) -> Result<Box<dyn std::io::BufRead + Send>, ClientError> {
        let autostart = match self.options.reconnect || self.lock().generation == 0 {
            true => &self.options.autostart,
            false => &Autostart::Off,
        };
        let link::Wire {
            mut reader,
            mut writer,
            closer,
        } = link::open(&self.target, autostart)?;
        let hello = Hello {
            token: self.token()?,
            client: ClientInfo {
                name: self.name.clone(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
            },
            rig: VersionRange {
                min: *SUPPORTED.start(),
                max: *SUPPORTED.end(),
            },
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = Request::new(
            RequestId::Num(id),
            Hello::NAME,
            serde_json::to_value(&hello).map_err(io)?,
        );
        let said = write_frame(&mut writer, &Frame::Request(request));
        let answer = said.map_err(ClientError::Io).and_then(|()| {
            let mut line = Vec::new();
            loop {
                match read_frame(&mut reader, &mut line) {
                    Ok(Some(Frame::Response(response))) if response.id == RequestId::Num(id) => {
                        return match response.outcome {
                            Outcome::Result(value) => {
                                serde_json::from_value::<HelloResult>(value).map_err(io)
                            }
                            Outcome::Error(error) => Err(refusal_error(error)),
                        };
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        return Err(ClientError::Io(std::io::ErrorKind::UnexpectedEof.into()));
                    }
                    Err(error) => {
                        // A server of another version answers in a frame we
                        // can't read: refuse it as its reader.
                        return Err(match error.refusal(Side::Client) {
                            Some(refusal) if refusal.code == codes::VERSION => {
                                refusal_error(refusal)
                            }
                            _ => ClientError::Io(std::io::Error::other(error.to_string())),
                        });
                    }
                }
            }
        });
        // The version the server settled on must be one this client speaks.
        let answer = answer.and_then(|hello| match SUPPORTED.contains(&hello.rig) {
            true => Ok(hello),
            false => Err(refusal_error(RpcError::version_read_by(
                Some(hello.rig),
                Side::Client,
            ))),
        });
        let hello = match answer {
            Ok(hello) => hello,
            Err(error) => {
                closer.close();
                return Err(error);
            }
        };
        let mut slot = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut state = self.lock();
        if state.closed {
            closer.close();
            return Err(ClientError::Unavailable {
                next: "the client was dropped".into(),
            });
        }
        state.generation += 1;
        state.hello = Some(hello);
        // Resubscribe each stream after what it has; its own subscribe call,
        // still unanswered, is resent with the rest.
        let mut resubscribes = Vec::new();
        for (local, sub) in &mut state.subs {
            if sub.server.take().is_none() {
                continue;
            }
            let params = Subscribe {
                session_id: sub.session.clone(),
                after_seq: sub.last,
                follow: sub.follow,
            };
            if let Ok(params) = serde_json::to_value(&params) {
                resubscribes.push((*local, params));
            }
        }
        for (local, params) in resubscribes {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            state.pending.insert(
                id,
                Pending {
                    request: Request::new(RequestId::Num(id), Subscribe::NAME, params),
                    subscribes: Some(local),
                    started: Instant::now(),
                    waiter: false,
                    done: None,
                },
            );
        }
        let mut resend: Vec<(u64, Request)> = state
            .pending
            .iter()
            .filter(|(_, pending)| pending.done.is_none())
            .map(|(id, pending)| (*id, pending.request.clone()))
            .collect();
        resend.sort_unstable_by_key(|(id, _)| *id);
        state.link = Link::Up { closer };
        drop(state);
        for (_, request) in resend {
            if write_frame(&mut writer, &Frame::Request(request)).is_err() {
                break;
            }
        }
        *slot = Some(writer);
        drop(slot);
        self.changed.notify_all();
        Ok(reader)
    }

    /// The reader thread: routes frames, and reconnects when the wire ends.
    fn run(self: Arc<Self>, mut reader: Box<dyn std::io::BufRead + Send>) {
        loop {
            let why = match self.read(&mut reader) {
                Ok(why) => why,
                Err(failure) => {
                    *self
                        .writer
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
                    if let Link::Up { closer, .. } = &self.lock().link {
                        closer.close();
                    }
                    self.fail(failure);
                    return;
                }
            };
            *self
                .writer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            let mut state = self.lock();
            if let Link::Up { closer, .. } = &state.link {
                closer.close();
            }
            if state.closed {
                return;
            }
            if !self.options.reconnect {
                drop(state);
                self.fail(Failure::Gone(why));
                return;
            }
            // A call that may have run and can't be told from a new one
            // (no key) isn't resent: its caller learns the outcome is unknown.
            for pending in state.pending.values_mut() {
                if pending.waiter && pending.done.is_none() && !resendable(&pending.request) {
                    pending.done = Some(Err(ClientError::Unavailable {
                        next: format!(
                            "the connection to the rig server dropped before {} answered, so it may have run; check its effect before calling it again",
                            pending.request.method
                        ),
                    }));
                }
            }
            state.link = Link::Down {
                since: Instant::now(),
                error: Some(why),
            };
            drop(state);
            self.changed.notify_all();
            self.wake();
            match self.reconnect() {
                Some(next) => reader = next,
                None => return,
            }
        }
    }

    /// Reads frames until the wire ends; returns why it ended.
    /// Reads frames until the wire ends: `Ok` with why it ended, or `Err`
    /// when the server writes a version this client doesn't speak, which no
    /// reconnect fixes.
    fn read(&self, reader: &mut Box<dyn std::io::BufRead + Send>) -> Result<String, Failure> {
        let mut line = Vec::new();
        loop {
            match read_frame(reader, &mut line) {
                Ok(Some(frame)) => self.route(frame),
                Ok(None) => return Ok("end of stream".into()),
                Err(rig_protocol::FrameError::Io(error)) => return Ok(error.to_string()),
                Err(error @ rig_protocol::FrameError::Version { .. }) => {
                    let refusal = error.refusal(Side::Client);
                    let update = refusal.as_ref().and_then(|r| r.data.as_ref()?.update);
                    return Err(Failure::Version {
                        update: update.unwrap_or(Side::Client),
                        supported: SUPPORTED.collect(),
                    });
                }
                // Another line we can't read is the server's bug; skip it.
                Err(_) => {}
            }
            self.wake();
        }
    }

    fn route(&self, frame: Frame) {
        match frame {
            Frame::Response(response) => {
                let RequestId::Num(id) = response.id else {
                    return;
                };
                let mut state = self.lock();
                let Some(pending) = state.pending.get_mut(&id) else {
                    return;
                };
                let outcome = match response.outcome {
                    Outcome::Result(value) => Ok(value),
                    Outcome::Error(error) => Err(error),
                };
                let waiter = pending.waiter;
                let local = pending.subscribes;
                if let Some(local) = local {
                    match &outcome {
                        Ok(value) => {
                            let server = serde_json::from_value::<Subscribed>(value.clone())
                                .ok()
                                .map(|s| s.subscription);
                            if let Some(sub) = state.subs.get_mut(&local) {
                                sub.server = server;
                            }
                        }
                        Err(error) => {
                            if let Some(sub) = state.subs.remove(&local) {
                                let _ = sub.tx.send(Err(ClientError::Rpc(error.clone())));
                            }
                        }
                    }
                }
                if !waiter {
                    state.pending.remove(&id);
                } else if let Some(pending) = state.pending.get_mut(&id) {
                    pending.done = Some(outcome.map_err(ClientError::Rpc));
                }
                drop(state);
                self.changed.notify_all();
            }
            Frame::Notification(notification) if notification.method == methods::EVENT => {
                let Ok(notice) = serde_json::from_value::<EventNotice>(notification.params) else {
                    return;
                };
                let mut state = self.lock();
                let sub = state
                    .subs
                    .values_mut()
                    .find(|sub| sub.server == Some(notice.subscription));
                if let Some(sub) = sub
                    && notice.record.seq > sub.last
                {
                    sub.last = notice.record.seq;
                    let _ = sub.tx.send(Ok(notice.record));
                }
            }
            // A lagged subscription is followed by the connection closing;
            // the resubscribe after it catches up.
            Frame::Notification(notification) if notification.method == methods::LAGGED => {}
            Frame::Notification(notification) => {
                if let Some(hook) = &self.options.on_notification {
                    hook(notification);
                }
            }
            Frame::Request(_) => {}
        }
    }

    /// Reconnects with backoff while anyone waits; `None` once dropped.
    fn reconnect(&self) -> Option<Box<dyn std::io::BufRead + Send>> {
        let mut pause = BACKOFF_MIN;
        loop {
            {
                let mut state = self.lock();
                // Park until there is demand: a call or a stream.
                while !state.closed && state.pending.is_empty() && state.subs.is_empty() {
                    state = self
                        .changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                if state.closed {
                    return None;
                }
                self.expire(&mut state);
            }
            match self.establish() {
                Ok(reader) => return Some(reader),
                Err(ClientError::Version { update, supported }) => {
                    return self.fail(Failure::Version { update, supported });
                }
                Err(ClientError::Rpc(error)) => return self.fail(Failure::Rpc(error)),
                Err(error) => {
                    let mut state = self.lock();
                    if let Link::Down { error: last, .. } = &mut state.link {
                        *last = Some(error.to_string());
                    }
                    let state = self
                        .changed
                        .wait_timeout(state, pause)
                        .map(|(state, _)| state)
                        .unwrap_or_else(|e| e.into_inner().0);
                    if state.closed {
                        return None;
                    }
                    pause = (pause * 2).min(BACKOFF_MAX);
                }
            }
        }
    }

    /// Ends the link for good: a refusal no retry fixes.
    fn fail(&self, failure: Failure) -> Option<Box<dyn std::io::BufRead + Send>> {
        let mut state = self.lock();
        fail_all(&mut state, &failure);
        state.link = Link::Failed(failure);
        drop(state);
        self.changed.notify_all();
        self.wake();
        None
    }

    /// Fails the calls and streams that waited longer than `give_up` for
    /// a lost server.
    fn expire(&self, state: &mut State) {
        let Link::Down { since, error } = &state.link else {
            return;
        };
        let give_up = self.options.give_up;
        let now = Instant::now();
        if now.duration_since(*since) < give_up {
            return;
        }
        let next = format!(
            "the rig server is gone ({}); run `rig server status`",
            error.as_deref().unwrap_or("no connection")
        );
        for pending in state.pending.values_mut() {
            if pending.done.is_none() && now.duration_since(pending.started) >= give_up {
                pending.done = Some(Err(ClientError::Unavailable { next: next.clone() }));
            }
        }
        state.pending.retain(|_, p| p.waiter);
        for (_, sub) in state.subs.drain() {
            let _ = sub
                .tx
                .send(Err(ClientError::Unavailable { next: next.clone() }));
        }
        self.changed.notify_all();
    }

    /// Writes `request` on the current wire, if there is one.
    fn send(&self, request: &Request) {
        let mut slot = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(writer) = slot.as_mut() {
            // A failed write is a lost connection, which the reader sees.
            let _ = write_frame(writer, &Frame::Request(request.clone()));
        }
    }

    fn call(
        &self,
        method: &str,
        params: serde_json::Value,
        subscribes: Option<u64>,
    ) -> Result<serde_json::Value, ClientError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = Request::new(RequestId::Num(id), method, params);
        let mut state = self.lock();
        if let Link::Failed(failure) = &state.link {
            return Err(failure.error());
        }
        state.pending.insert(
            id,
            Pending {
                request: request.clone(),
                subscribes,
                started: Instant::now(),
                waiter: true,
                done: None,
            },
        );
        drop(state);
        self.changed.notify_all();
        // Without a wire, or when the write fails, the reconnect resends it.
        self.send(&request);
        let mut state = self.lock();
        loop {
            if let Some(done) = state.pending.get_mut(&id).and_then(|p| p.done.take()) {
                state.pending.remove(&id);
                return done;
            }
            if let Link::Failed(failure) = &state.link {
                let error = failure.error();
                state.pending.remove(&id);
                return Err(error);
            }
            let wait = match &state.link {
                Link::Down { .. } => {
                    self.expire(&mut state);
                    Duration::from_millis(100)
                }
                _ => Duration::from_secs(3600),
            };
            state = self
                .changed
                .wait_timeout(state, wait)
                .map(|(state, _)| state)
                .unwrap_or_else(|e| e.into_inner().0);
        }
    }
}

/// Methods that change nothing, or nothing twice, so a call of one lost
/// with its connection is resent without a key.
const RESENDABLE: &[&str] = &[
    Hello::NAME,
    methods::Wait::NAME,
    methods::List::NAME,
    methods::Read::NAME,
    Subscribe::NAME,
    Unsubscribe::NAME,
    methods::ReadEvents::NAME,
    methods::Attach::NAME,
    methods::Detach::NAME,
];

/// Whether `request` may be sent again after its connection dropped: it
/// carries an idempotency key, or its method is [`RESENDABLE`].
fn resendable(request: &Request) -> bool {
    request.params.get("key").is_some() || RESENDABLE.contains(&request.method.as_str())
}

fn fail_all(state: &mut State, failure: &Failure) {
    for pending in state.pending.values_mut() {
        if pending.done.is_none() {
            pending.done = Some(Err(failure.error()));
        }
    }
    for (_, sub) in state.subs.drain() {
        let _ = sub.tx.send(Err(failure.error()));
    }
}

/// A refusal as the error a caller acts on: a version refusal names the
/// side to update.
fn refusal_error(error: RpcError) -> ClientError {
    match (error.code, &error.data) {
        (codes::VERSION, Some(data)) => ClientError::Version {
            update: data.update.unwrap_or(Side::Client),
            supported: data.supported.clone(),
        },
        _ => ClientError::Rpc(error),
    }
}

/// A session's events in `seq` order, each once. The iterator blocks for
/// the next one; it ends after an error, or, when not following, once the
/// server ends the replay.
pub struct Events {
    shared: std::sync::Weak<Shared>,
    local: u64,
    rx: Receiver<Result<EventRecord, ClientError>>,
    last: Seq,
    /// The last seq of a stream that doesn't follow.
    end: Option<Seq>,
    ended: bool,
}

impl Events {
    /// The `seq` of the last event this stream yielded.
    pub fn last_seq(&self) -> Seq {
        self.last
    }

    /// The next event if one is here, without blocking (for a frame loop).
    pub fn try_next(&mut self) -> Option<Result<EventRecord, ClientError>> {
        if self.ended {
            return None;
        }
        match self.rx.try_recv() {
            Ok(item) => self.take(item),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.ended = true;
                None
            }
        }
    }

    /// The next event, waiting at most `timeout`; `None` when none came.
    pub fn next_timeout(&mut self, timeout: Duration) -> Option<Result<EventRecord, ClientError>> {
        if self.ended {
            return None;
        }
        match self.rx.recv_timeout(timeout) {
            Ok(item) => self.take(item),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => {
                self.ended = true;
                None
            }
        }
    }

    fn take(
        &mut self,
        item: Result<EventRecord, ClientError>,
    ) -> Option<Result<EventRecord, ClientError>> {
        match item {
            Ok(record) if record.seq <= self.last => self.try_next(),
            Ok(record) => {
                self.last = record.seq;
                self.ended = self.end.is_some_and(|end| record.seq >= end);
                Some(Ok(record))
            }
            Err(error) => {
                self.ended = true;
                Some(Err(error))
            }
        }
    }
}

impl Iterator for Events {
    type Item = Result<EventRecord, ClientError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.ended {
                return None;
            }
            match self.rx.recv() {
                Ok(Ok(record)) if record.seq <= self.last => continue,
                Ok(item) => return self.take(item),
                Err(_) => {
                    self.ended = true;
                    return None;
                }
            }
        }
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let sub = shared.lock().subs.remove(&self.local);
        if let Some(subscription) = sub.and_then(|sub| sub.server) {
            let id = shared.next_id.fetch_add(1, Ordering::Relaxed);
            if let Ok(params) = serde_json::to_value(Unsubscribe { subscription }) {
                // Nobody waits for the answer; `route` ignores unknown ids.
                shared.send(&Request::new(RequestId::Num(id), Unsubscribe::NAME, params));
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// No server could be reached or started; `next` says what to run.
    #[error("the rig server is unavailable: {next}")]
    Unavailable { next: String },
    /// The two sides share no wire version; `update` must be updated.
    #[error("no common rig wire version: update the {update:?} (it speaks {supported:?})")]
    Version { update: Side, supported: Vec<u32> },
    #[error(transparent)]
    Rpc(RpcError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl ClientError {
    /// The exit code a command ends with on this error.
    pub fn exit(&self) -> Exit {
        match self {
            ClientError::Unavailable { .. } | ClientError::Version { .. } | ClientError::Io(_) => {
                Exit::Unavailable
            }
            ClientError::Rpc(e) => match e.code {
                codes::UNAUTHORIZED
                | codes::ALREADY_RESOLVED
                | codes::GRANT_WIDER
                | codes::SANDBOX_UNAVAILABLE => Exit::NotAllowed,
                codes::INVALID_PARAMS | codes::METHOD_NOT_FOUND | codes::NOT_FOUND => Exit::Usage,
                _ => Exit::Failed,
            },
        }
    }

    /// The command that fixes it, when one is known.
    pub fn next(&self) -> Option<&str> {
        match self {
            ClientError::Unavailable { next } => Some(next),
            ClientError::Rpc(e) => e.data.as_ref().and_then(|d| d.next.as_deref()),
            ClientError::Version { update, .. } => Some(match update {
                Side::Client => {
                    "install a newer rig on this side, or the newest clients/python/rig_client.py"
                }
                Side::Server => {
                    "install a newer rig where the server runs, then `rig server stop` so the new one starts"
                }
                Side::Worker => "rebuild the agent with `rig build`",
            }),
            ClientError::Io(_) => None,
        }
    }
}

#[cfg(test)]
mod tests;

const _: () = {
    const fn shareable<T: core::marker::Send + Sync>() {}
    const fn sendable<T: core::marker::Send>() {}
    shareable::<Client>();
    sendable::<Events>();
};
