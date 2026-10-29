//! `rig acp`: an ACP v1 agent on stdin and stdout that keeps each session
//! in a rig process of its own (`rig acp-session`, [`crate::session`]),
//! started in the session's directory, and forwards that session's
//! requests to it and its updates and questions back. rig runs one session
//! per process, with the process's directory as the project.
//!
//! The router holds each session's lock while its process runs, gives the
//! process a run directory of its own, and kills the process groups it left
//! there once it ended, however it ended. Every process runs the build the
//! router started with: a hard link to it, so a rebuild that replaces it
//! leaves the router a binary to start.

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use agent_client_protocol::schema::v1 as acp;
use agent_client_protocol::{Agent, Client, ConnectionTo, Error, Lines, Responder};
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, StreamExt};
use rig_basics::session_picker::SessionEntry;
use rig_harness::harness_protocol::{Home, RunDir, SessionId, StartReason, env, lock_file};
use rig_harness::host::session::StoreOpener;
use rig_harness::prelude::*;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{oneshot, watch};

use crate::{Reply, session};

/// How long a session process may take to end once asked, before it is
/// killed.
const GRACE: Duration = Duration::from_secs(5);

/// How the router starts a session's process: the program, and the
/// arguments before `acp-session`'s. The tests point it at their binary.
#[derive(Resource, Clone, Debug)]
pub(crate) struct SessionCommand {
    /// The program.
    pub(crate) program: PathBuf,
    /// Its arguments.
    pub(crate) args: Vec<OsString>,
}

/// The router's state, shared by its handlers.
struct Router {
    home: Home,
    stores: StoreOpener,
    command: SessionCommand,
    /// Where each session's run directory goes.
    runs: PathBuf,
    /// Numbers the run directories.
    next: AtomicU64,
    /// The client's `initialize`, which each session process gets too.
    client: Mutex<Option<acp::InitializeRequest>>,
    sessions: Mutex<HashMap<String, Arc<Child>>>,
}

/// A session's process, as the router runs it.
struct Child {
    cx: ConnectionTo<Agent>,
    /// The number of its run directory, which tells this process from a
    /// later one of the same session.
    run: u64,
    /// Ends the connection, which closes the process's stdin.
    stop: watch::Sender<bool>,
    /// Done once the process ended and was cleaned up, for every waiter.
    gone: Shared<BoxFuture<'static, ()>>,
}

/// Locks `mutex`, whatever a panicking holder left.
fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Child {
    /// Ends the process and waits until it was cleaned up.
    async fn end(&self) {
        self.stop.send(true).ok();
        self.gone.clone().await;
    }
}

/// `path` with its links resolved, else as it is.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned())
}

/// An error the client gets for a request it made wrong.
fn invalid(why: impl ToString) -> Error {
    crate::error(Error::invalid_params(), why)
}

/// An error the client gets for one of the router's own failures.
fn internal(why: impl ToString) -> Error {
    crate::error(Error::internal_error(), why)
}

impl Router {
    fn child(&self, id: &acp::SessionId) -> Result<Arc<Child>, Error> {
        let sessions = locked(&self.sessions);
        let child = sessions.get(id.0.as_ref()).cloned();
        child.ok_or_else(|| Error::session_not_found(id.clone()))
    }

    /// Starts `id`'s process in `cwd`, with `lock` held for it, and connects
    /// to it; the map has it before any request goes to it.
    async fn start(
        self: &Arc<Self>,
        id: &SessionId,
        cwd: &Path,
        reason: StartReason,
        lock: File,
        client: ConnectionTo<Client>,
    ) -> Result<Arc<Child>, Error> {
        let number = self.next.fetch_add(1, Ordering::Relaxed);
        let run = RunDir::at(self.runs.join(number.to_string()));
        std::fs::create_dir_all(run.path()).map_err(internal)?;
        let run_lock = lock_file(&run.lock(), true).map_err(internal)?;
        run_lock.lock().map_err(internal)?;
        let mut command = tokio::process::Command::new(&self.command.program);
        command
            .args(&self.command.args)
            .arg(session::FRONT)
            .current_dir(cwd)
            .env(env::HOME, self.home.root())
            .env(env::SESSION, id.as_str())
            .env(env::RUN, run.path())
            .env(env::START, reason.as_str())
            .env_remove(env::NOTICE)
            .env_remove(env::BUILD_FAILURE)
            .env_remove(env::LAUNCHER)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut process = command.spawn().map_err(internal)?;
        let (Some(input), Some(output)) = (process.stdin.take(), process.stdout.take()) else {
            return Err(internal("the session process has no pipes"));
        };
        let (ready, connected) = oneshot::channel();
        let (stop, stopping) = watch::channel(false);
        let (router, key) = (self.clone(), id.as_str().to_owned());
        let gone = tokio::spawn(async move {
            let connection = forward(client, lines(input, output), ready, stopping.clone());
            tokio::pin!(connection);
            // Asked to end, the connection has its grace to close, as a
            // stopped process never drains what was written to it.
            let ended = tokio::select! {
                ended = &mut connection => ended,
                () = asked(stopping) => {
                    let closed = tokio::time::timeout(GRACE, &mut connection).await;
                    closed.unwrap_or_else(|_| Err(internal("it did not close in time")))
                }
            };
            if let Err(error) = ended {
                eprintln!("rig acp: session {key}: {error}");
            }
            if tokio::time::timeout(GRACE, process.wait()).await.is_err() {
                process.kill().await.ok();
            }
            run.kill_left_groups().ok();
            std::fs::remove_dir_all(run.path()).ok();
            drop((run_lock, lock));
            let mut sessions = locked(&router.sessions);
            if sessions.get(&key).is_some_and(|child| child.run == number) {
                sessions.remove(&key);
            }
        });
        let gone = gone.map(drop).boxed().shared();
        let cx = connected
            .await
            .map_err(|_| internal("the session process did not start"))?;
        let child = Arc::new(Child {
            cx,
            run: number,
            stop,
            gone,
        });
        let mut sessions = locked(&self.sessions);
        sessions.insert(id.as_str().to_owned(), child.clone());
        // A process that ended before it was listed is no session.
        if child.gone.clone().now_or_never().is_some() {
            sessions.remove(id.as_str());
        }
        drop(sessions);
        Ok(child)
    }

    /// Opens `id` in `cwd` for the client: starts its process, which gets
    /// the client's `initialize` and then `request`, whose answer goes to
    /// the client in order with the process's updates: a loaded session's
    /// history before it, and what the session says once open after it.
    async fn open<Request>(
        self: &Arc<Self>,
        id: SessionId,
        cwd: &Path,
        reason: StartReason,
        lock: File,
        client: ConnectionTo<Client>,
        request: Request,
        reply: Arc<Reply<Request::Response>>,
    ) -> Result<(), Error>
    where
        Request: agent_client_protocol::JsonRpcRequest,
    {
        let initialize = locked(&self.client).clone();
        let initialize =
            initialize.ok_or_else(|| crate::error(Error::invalid_request(), "initialize first"))?;
        let child = self.start(&id, cwd, reason, lock, client).await?;
        if let Err(error) = child.cx.send_request(initialize).block_task().await {
            child.end().await;
            return Err(error);
        }
        let (answered, failed) = oneshot::channel();
        let forwarding = reply.clone();
        let forwarded =
            child
                .cx
                .prepare_request(request)
                .on_receiving_result(async move |result| {
                    answered.send(result.is_err()).ok();
                    forwarding.send(result);
                    Ok(())
                });
        // A process that ended first never answered.
        if forwarded.is_err() || failed.await.unwrap_or(true) {
            reply.send(Err(internal(
                "the session's process ended before it opened",
            )));
            child.end().await;
        }
        Ok(())
    }

    /// The client's working directory, which must be an existing absolute
    /// directory.
    fn cwd(cwd: &Path) -> Result<PathBuf, Error> {
        if !cwd.is_absolute() || !cwd.is_dir() {
            return Err(invalid(format!(
                "{} is not an existing absolute directory",
                cwd.display()
            )));
        }
        Ok(canonical(cwd))
    }

    /// A new session in `cwd`, locked, with its directory written.
    fn new_session(&self, cwd: &Path) -> Result<(SessionId, File), Error> {
        let (id, dir) = self.home.new_session().map_err(internal)?;
        let lock = dir.hold().map_err(internal)?;
        let lock = lock.ok_or_else(|| internal(format!("session {id} is already running")))?;
        dir.set_working_directory(cwd).map_err(internal)?;
        Ok((id, lock))
    }

    /// The saved session `id`, locked for this router, if it runs in `cwd`.
    fn saved(&self, id: &acp::SessionId, cwd: &Path) -> Result<(SessionId, File), Error> {
        let id = SessionId::from_str(&id.0).map_err(invalid)?;
        if locked(&self.sessions).contains_key(id.as_str()) {
            return Err(invalid(format!("session {id} is already open")));
        }
        let dir = self.home.session(&id);
        if !dir.is_saved() {
            return Err(Error::session_not_found(id.as_str().to_owned()));
        }
        let lock = dir.hold().map_err(internal)?;
        let lock = lock.ok_or_else(|| invalid(format!("session {id} is open in another rig")))?;
        match dir.working_directory() {
            Some(directory) => {
                let directory = canonical(&directory);
                if directory != cwd {
                    return Err(invalid(format!(
                        "session {id} runs in {}, not {}",
                        directory.display(),
                        cwd.display()
                    )));
                }
            }
            None => {
                dir.set_working_directory(cwd).map_err(internal)?;
            }
        }
        Ok((id, lock))
    }

    /// The saved sessions, newest first, of `cwd` when given.
    fn list(&self, cwd: Option<&Path>) -> Vec<acp::SessionInfo> {
        let cwd = cwd.map(canonical);
        let mut entries: Vec<SessionEntry> = self
            .home
            .saved_sessions()
            .iter()
            .filter_map(|(_, dir)| SessionEntry::read(dir, &self.stores))
            .filter(|entry| {
                let directory = entry.directory.as_ref();
                let directory = directory.map(|it| canonical(it));
                cwd.is_none() || directory == cwd
            })
            .collect();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.updated));
        entries
            .into_iter()
            .filter_map(|entry| {
                let directory = entry.directory.clone()?;
                let title = entry
                    .name
                    .clone()
                    .unwrap_or_else(|| entry.meta.title.clone());
                let updated = chrono::DateTime::<chrono::Utc>::from(entry.updated);
                let updated = updated.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
                let info = acp::SessionInfo::new(entry.id.as_str().to_owned(), directory)
                    .updated_at(updated);
                Some(match title.is_empty() {
                    true => info,
                    false => info.title(title),
                })
            })
            .collect()
    }

    /// Ends every session's process, as the router ends.
    async fn end(&self) {
        let children: Vec<_> = locked(&self.sessions).values().cloned().collect();
        futures::future::join_all(children.iter().map(|child| child.end())).await;
    }
}

/// Runs the client side of a session's connection over `transport` until
/// `stopped` or the process closes it: its updates and questions go to the
/// client as they come, in the order they come.
async fn forward(
    client: ConnectionTo<Client>,
    transport: impl agent_client_protocol::ConnectTo<Client> + 'static,
    ready: oneshot::Sender<ConnectionTo<Agent>>,
    stopping: watch::Receiver<bool>,
) -> Result<(), Error> {
    let (updates, permissions, forms) = (client.clone(), client.clone(), client);
    Client
        .builder()
        .name("rig-acp-router")
        .on_receive_notification(
            async move |update: acp::SessionNotification, _| updates.send_notification(update),
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: acp::RequestPermissionRequest, responder, _| {
                permissions
                    .prepare_request(request)
                    .forward_response_to(responder)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: acp::CreateElicitationRequest, responder, _| {
                forms
                    .prepare_request(request)
                    .forward_response_to(responder)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(transport, async move |cx| {
            ready.send(cx.clone()).ok();
            tokio::select! {
                () = asked(stopping) => {}
                () = cx.incoming_closed() => {}
            }
            Ok(())
        })
        .await
}

/// Done once `stopping` says to end; never when nothing will say so.
async fn asked(mut stopping: watch::Receiver<bool>) {
    if stopping.wait_for(|stop| *stop).await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// A session process's stdin and stdout as ACP lines. A line it writes that
/// is not a JSON object, such as a plugin's stray print, goes to stderr
/// instead, so it never breaks the client's connection.
pub(crate) fn lines(
    input: tokio::process::ChildStdin,
    output: tokio::process::ChildStdout,
) -> Lines<
    impl futures::Sink<String, Error = std::io::Error> + Send + 'static,
    impl futures::Stream<Item = std::io::Result<String>> + Send + 'static,
> {
    let sink = futures::sink::unfold(input, async |mut input, line: String| {
        input.write_all(line.as_bytes()).await?;
        input.write_all(b"\n").await?;
        input.flush().await?;
        Ok::<_, std::io::Error>(input)
    });
    let read = BufReader::new(output).lines();
    let stream = futures::stream::unfold(read, async |mut read| {
        loop {
            match read.next_line().await {
                Ok(Some(line)) if line.trim_start().starts_with('{') => {
                    return Some((Ok(line), read));
                }
                Ok(Some(line)) if line.trim().is_empty() => {}
                Ok(Some(line)) => {
                    eprintln!("rig acp: a session wrote a line that is not ACP: {line}")
                }
                Ok(None) => return None,
                Err(error) => return Some((Err(error), read)),
            }
        }
    });
    Lines::new(Box::pin(sink), stream.boxed())
}

/// Serves the client on stdio until it closes the connection, then ends
/// every session's process.
async fn serve(router: Arc<Router>) -> Result<(), Error> {
    let (init, new, load, resume, list) = (
        router.clone(),
        router.clone(),
        router.clone(),
        router.clone(),
        router.clone(),
    );
    let (prompt, cancel, config, close) = (
        router.clone(),
        router.clone(),
        router.clone(),
        router.clone(),
    );
    let served = Agent
        .builder()
        .name("rig-acp")
        .on_receive_request(
            async move |request: acp::InitializeRequest, responder, _| {
                *locked(&init.client) = Some(request);
                responder.respond(crate::initialized())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: acp::NewSessionRequest, responder, cx| {
                let at = (request.cwd.clone(), request.additional_directories.clone());
                let (reason, found) = (StartReason::New, |router: &Router, cwd: &Path| {
                    router.new_session(cwd)
                });
                opening(new.clone(), cx, request, responder, at, reason, found)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: acp::LoadSessionRequest, responder, cx| {
                let at = (request.cwd.clone(), request.additional_directories.clone());
                let id = request.session_id.clone();
                let found = move |router: &Router, cwd: &Path| router.saved(&id, cwd);
                opening(
                    load.clone(),
                    cx,
                    request,
                    responder,
                    at,
                    StartReason::Resumed,
                    found,
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: acp::ResumeSessionRequest, responder, cx| {
                let at = (request.cwd.clone(), request.additional_directories.clone());
                let id = request.session_id.clone();
                let found = move |router: &Router, cwd: &Path| router.saved(&id, cwd);
                opening(
                    resume.clone(),
                    cx,
                    request,
                    responder,
                    at,
                    StartReason::Resumed,
                    found,
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: acp::ListSessionsRequest, responder, _| {
                let sessions = list.list(request.cwd.as_deref());
                responder.respond(acp::ListSessionsResponse::new(sessions))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: acp::PromptRequest, responder, _| {
                let child = prompt.child(&request.session_id)?;
                child
                    .cx
                    .prepare_request(request)
                    .forward_response_to(responder)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |request: acp::CancelNotification, _| match cancel.child(&request.session_id)
            {
                Ok(child) => child.cx.send_notification(request),
                Err(_) => Ok(()),
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: acp::SetSessionConfigOptionRequest, responder, _| {
                let child = config.child(&request.session_id)?;
                child
                    .cx
                    .prepare_request(request)
                    .forward_response_to(responder)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: acp::CloseSessionRequest, responder, cx| {
                let router = close.clone();
                cx.spawn(async move {
                    let closed = async {
                        let Ok(child) = router.child(&request.session_id) else {
                            return Ok(acp::CloseSessionResponse::new());
                        };
                        // The session cancels its prompt and answers; one
                        // that does not in time is ended anyway. The close
                        // is done once the process ended and its lock is
                        // free, so the session can be loaded again, whatever
                        // another close of it did meanwhile.
                        let asked = child.cx.send_request(request).block_task();
                        tokio::time::timeout(GRACE, asked).await.ok();
                        child.end().await;
                        Ok(acp::CloseSessionResponse::new())
                    };
                    responder.respond_with_result(closed.await)
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_to(agent_client_protocol::Stdio::new())
        .await;
    router.end().await;
    served
}

/// Answers a request that opens a session in a task of its own: in the
/// client's directory `at` (the session's, and any others, which rig does
/// not take), the session `found` finds there and locks.
fn opening<Request>(
    router: Arc<Router>,
    cx: ConnectionTo<Client>,
    request: Request,
    responder: Responder<Request::Response>,
    (cwd, others): (PathBuf, Vec<PathBuf>),
    reason: StartReason,
    found: impl FnOnce(&Router, &Path) -> Result<(SessionId, File), Error> + Send + 'static,
) -> Result<(), Error>
where
    Request: agent_client_protocol::JsonRpcRequest + Send + 'static,
{
    cx.clone().spawn(async move {
        let reply = Arc::new(Reply::new(responder));
        let opened = async {
            refuse_directories(&others)?;
            let cwd = Router::cwd(&cwd)?;
            let (id, lock) = found(&router, &cwd)?;
            router
                .open(id, &cwd, reason, lock, cx, request, reply.clone())
                .await
        };
        if let Err(error) = opened.await {
            reply.send(Err(error));
        }
        Ok(())
    })
}

/// Refuses directories besides the session's, which rig does not take.
fn refuse_directories(directories: &[PathBuf]) -> Result<(), Error> {
    match directories.is_empty() {
        true => Ok(()),
        false => Err(invalid(
            "rig works in one directory: additionalDirectories is not supported",
        )),
    }
}

/// The router's run: its task, and how it is stopped.
#[derive(Resource)]
struct Served {
    stop: Option<oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<()>>,
    done: async_channel::Receiver<bool>,
}

/// Makes this app the router: serves the client from the first frame and
/// exits when the client closes the connection or the app is asked to.
pub(crate) fn install(app: &mut App) {
    let world = app.world();
    let (Some(home), Some(stores), Some(runtime), Some(wake)) = (
        world.get_resource::<RigHome>(),
        world.get_resource::<StoreOpener>(),
        world.get_resource::<TokioRuntime>(),
        world.get_resource::<Wake>(),
    ) else {
        app.refuse_run("rig acp needs rig's home, session store and tokio runtime".to_owned());
        return;
    };
    let home = home.0.clone();
    let (stores, runtime, wake) = (stores.clone(), runtime.0.clone(), wake.clone());
    let base = match runs(&home) {
        Ok(base) => base,
        Err(why) => {
            app.refuse_run(format!("rig acp could not make its run directory: {why}"));
            return;
        }
    };
    let command = match world.get_resource::<SessionCommand>() {
        Some(command) => command.clone(),
        None => match own_command(&base) {
            Ok(command) => command,
            Err(why) => {
                app.refuse_run(format!("rig acp could not find its own program: {why}"));
                return;
            }
        },
    };
    let router = Arc::new(Router {
        home,
        stores,
        command,
        runs: base.path.clone(),
        next: AtomicU64::new(1),
        client: Mutex::new(None),
        sessions: Mutex::new(HashMap::new()),
    });
    let (stop, stopped) = oneshot::channel();
    let (done, finished) = async_channel::bounded(1);
    let task = runtime.spawn(async move {
        let _held = base;
        let served = tokio::select! {
            served = serve(router.clone()) => served,
            _ = stopped => {
                router.end().await;
                Ok(())
            }
        };
        if let Err(error) = &served {
            eprintln!("rig acp: {error}");
        }
        done.send(served.is_ok()).await.ok();
        wake.wake();
    });
    app.insert_resource(Served {
        stop: Some(stop),
        task: Some(task),
        done: finished,
    })
    .add_systems(First, served_ended)
    .add_systems(
        Last,
        shutdown
            .in_set(OnAppExitSystems)
            .run_if(on_message::<AppExit>),
    );
}

/// The router's own run directory, which it holds while it runs; the
/// launcher's, when it started the router.
struct Runs {
    path: PathBuf,
    _lock: Option<File>,
}

/// Where the router keeps the session processes' run directories and its
/// binary's link: in the launcher's run directory when there is one, else in
/// one of its own under `RIG_HOME/runs`, which a launcher sweeps once it is
/// free.
fn runs(home: &Home) -> std::io::Result<Runs> {
    if let Some(run) = RunDir::from_env() {
        let path = run.path().join("acp");
        std::fs::create_dir_all(&path)?;
        return Ok(Runs { path, _lock: None });
    }
    let run = home.run(std::process::id());
    std::fs::create_dir_all(run.path())?;
    let lock = lock_file(&run.lock(), true)?;
    lock.try_lock().map_err(std::io::Error::other)?;
    let path = run.path().join("acp");
    std::fs::create_dir_all(&path)?;
    Ok(Runs {
        path,
        _lock: Some(lock),
    })
}

/// This program, linked (else copied) into `runs` so a build that replaces
/// it leaves one to start. A session takes its model and tools from the
/// client's options and the session, not from `rig acp`'s.
fn own_command(runs: &Runs) -> std::io::Result<SessionCommand> {
    let exe = std::env::current_exe()?;
    let program = runs.path.join("rig-agent");
    std::fs::remove_file(&program).ok();
    // A copy where a link cannot be made, such as on another file system.
    if std::fs::hard_link(&exe, &program).is_err() {
        std::fs::copy(&exe, &program)?;
    }
    Ok(SessionCommand {
        program,
        args: Vec::new(),
    })
}

/// Exits once the router stopped serving.
fn served_ended(served: Res<Served>, mut commands: Commands) {
    if let Ok(ok) = served.done.try_recv() {
        let reason = match ok {
            true => ExitReason::Quit,
            false => ExitReason::Failed(1),
        };
        commands.trigger(Exit(reason));
    }
}

/// Stops serving as the app exits, after every session's process ended.
fn shutdown(mut served: ResMut<Served>, runtime: Res<TokioRuntime>) {
    if let Some(stop) = served.stop.take() {
        stop.send(()).ok();
    }
    if let Some(task) = served.task.take() {
        runtime.0.block_on(task).ok();
    }
}
