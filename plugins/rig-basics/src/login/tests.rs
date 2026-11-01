use std::collections::VecDeque;
use std::future::{Future, ready};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use rig_core::catalog::{Catalog, Connector, ModelStatus};
use rig_core::http_client::{
    self, Bytes, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request, Response,
    StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use rig_ecs::prelude::*;
use rig_harness::commands::RunCommand;
use rig_harness::harness_protocol::SessionId;
use rig_harness::host::launcher::Invoked;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, run_until, scratch_home,
};

use super::*;
use crate::models::FirstPick;

/// The ChatGPT backend as a test scripts it: each `GET /models` answers
/// the next of its lists, with its status, once it is let go; every other
/// streamed request is refused as `model_not_found`. A credential refresh
/// fails, or is never answered when it stalls. It counts lists and
/// requests.
#[derive(Clone, Default)]
struct Backend {
    lists: Arc<Mutex<VecDeque<(u16, &'static str)>>>,
    listed: Arc<AtomicUsize>,
    sent: Arc<AtomicUsize>,
    /// Lists wait until this closes.
    held: Arc<Mutex<Option<async_channel::Receiver<()>>>>,
    stalls: bool,
}

impl Backend {
    /// Holds every list until the returned sender is dropped.
    fn hold(&self) -> async_channel::Sender<()> {
        let (release, held) = async_channel::bounded(1);
        if let Ok(mut holding) = self.held.lock() {
            *holding = Some(held);
        }
        release
    }
}

/// What the backend says to a model it does not serve, as it said it to
/// `gpt-5.3-chat-latest` on 2026-10-31.
const NOT_FOUND: &str = r#"{"detail":"The 'gpt-5.3-chat-latest' model is not supported when using Codex with a ChatGPT account.","error":{"type":"invalid_request_error","code":"model_not_found","message":"The 'gpt-5.3-chat-latest' model is not supported when using Codex with a ChatGPT account."}}"#;

impl HttpClientExt for Backend {
    fn send<T, U>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        // A credential refresh.
        let stalls = self.stalls;
        async move {
            if stalls {
                std::future::pending::<()>().await;
            }
            Err(http_client::Error::StreamEnded)
        }
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        ready(Err(http_client::Error::StreamEnded))
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let listing = request.uri().path().ends_with("/models");
        let (status, body) = match listing {
            true => {
                self.listed.fetch_add(1, Ordering::SeqCst);
                let next = self
                    .lists
                    .lock()
                    .ok()
                    .and_then(|mut lists| lists.pop_front());
                next.unwrap_or((500, "no more lists"))
            }
            false => {
                self.sent.fetch_add(1, Ordering::SeqCst);
                (400, NOT_FOUND)
            }
        };
        let held = (self.held.lock().ok())
            .and_then(|held| held.clone())
            .filter(|_| listing);
        async move {
            if let Some(held) = held {
                held.recv().await.ok();
            }
            let (sender, chunks) = async_channel::bounded(1);
            sender
                .try_send(Ok(Bytes::from_static(body.as_bytes())))
                .ok();
            let chunks: http_client::BoxedStream = Box::pin(chunks);
            Ok(Response::builder().status(status).body(chunks)?)
        }
    }
}

/// When a credential that never expires expires.
const NEVER: i64 = 4_102_444_800;

/// An app signed in to ChatGPT in a scratch home named `name` with a
/// credential that expires at `expires`, whose catalog is the plan's
/// models alone, served through `backend` once it answers `lists`, and
/// with what `setup` adds; its first frame ran. Returns its loop's wakes
/// and the agent.
fn signed_in(
    name: &str,
    (backend, lists): (&Backend, &[(u16, &'static str)]),
    expires: i64,
    setup: impl FnOnce(&mut App),
) -> (App, Receiver<()>, Entity) {
    if let Ok(mut held) = backend.lists.lock() {
        held.extend(lists.iter().copied());
    }
    let home = scratch_home(name);
    let credential = format!(
        r#"{{"access_token":"test-token","refresh_token":"test-refresh","id_token":"test-id","expires_at":{expires},"account_id":"test-account"}}"#
    );
    std::fs::create_dir_all(home.auth()).ok();
    std::fs::write(home.auth().join("chatgpt.json"), credential).ok();
    let mut catalog = Catalog::default();
    let plan = Catalog::builtin()
        .iter()
        .filter(|spec| spec.provider.vendor() == PROVIDER);
    for spec in plan {
        catalog.insert(spec.clone());
    }
    let mut connector = Connector::new(catalog);
    let http = backend.clone();
    connector.wrap_http(move |_| DynHttpClient::new(http.clone()));
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(Models(connector))
        .add_plugins((AgentPlugin, LoginPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    setup(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// The status of the plan's model `name`.
fn status(world: &World, name: &str) -> Option<ModelStatus> {
    let models = &world.resource::<Models>().0;
    let statuses = models.statuses().into_iter();
    statuses
        .filter(|(spec, _)| spec.id == name)
        .map(|(_, status)| status)
        .next()
}

/// Sends `text` to `agent` and runs `app` until the turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let world = app.world_mut();
    world.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Puts `agent` on the plan's Sol.
fn on_sol(app: &mut App, agent: Entity) {
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "chatgpt/gpt-6.1-sol".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
}

const SOL: &str =
    r#"{"models":[{"slug":"gpt-6.1-sol","visibility":"list"},{"slug":"gpt-reserve"}]}"#;
const NONE: &str = r#"{"models":[]}"#;

#[test]
fn the_plans_models_are_those_the_backend_lists_for_the_account() {
    let backend = Backend::default();
    let release = backend.hold();
    let (mut app, wakes, _) = signed_in("login-offered", (&backend, &[(200, SOL)]), NEVER, |_| {});
    let world = app.world();
    assert_eq!(status(world, "gpt-6.1-sol"), Some(ModelStatus::Checking));
    assert_eq!(
        FirstPick::of(world.resource::<Models>()),
        FirstPick::Checking
    );
    drop(release);
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") == Some(ModelStatus::Usable)
    });
    let world = app.world();
    let terra = status(world, "gpt-6.1-terra");
    assert!(
        matches!(terra, Some(ModelStatus::NotOffered(_))),
        "{terra:?}"
    );
    // The first row of the catalog is not offered; the pick skips it.
    let pick = FirstPick::of(world.resource::<Models>());
    assert_eq!(pick, FirstPick::Model("chatgpt/gpt-6.1-sol".to_owned()));
    let reachable: Vec<String> = (world.resource::<Models>().0.reachable().iter())
        .map(|spec| spec.reference())
        .collect();
    assert_eq!(reachable, ["chatgpt/gpt-6.1-sol"]);
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "tui")]
#[test]
fn an_open_model_picker_drops_the_models_the_account_is_not_offered() {
    use rig_harness::pick::Picker;

    let backend = Backend::default();
    let release = backend.hold();
    let lists = [(200, SOL)];
    let (mut app, wakes, agent) = signed_in("login-picker", (&backend, &lists), NEVER, |app| {
        app.add_plugins(crate::models::ModelsPlugin);
    });
    on_sol(&mut app, agent);
    app.world_mut()
        .trigger(RunCommand::new(agent, "model".to_owned()));
    app.update();
    let labels = |world: &mut World| -> Vec<String> {
        let mut pickers = world.query::<&Picker>();
        let items = pickers.iter(world).flat_map(|picker| picker.items.iter());
        items.map(|item| item.label.clone()).collect()
    };
    let checking = labels(app.world_mut());
    let terra = checking
        .iter()
        .find(|label| label.contains("gpt-6.1-terra"));
    assert!(
        terra.is_some_and(|label| label.ends_with("(checking your plan)")),
        "{checking:?}"
    );
    drop(release);
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") == Some(ModelStatus::Usable)
    });
    app.update();
    let listed = labels(app.world_mut());
    let plan: Vec<&String> = (listed.iter())
        .filter(|label| label.starts_with("chatgpt/"))
        .collect();
    assert_eq!(plan, ["chatgpt/gpt-6.1-sol  GPT-6.1 Sol  (ChatGPT plan)"]);
    // A picker not shown yet, as one queued behind another, opens on the
    // agent's model, wherever its line moved.
    let world = app.world_mut();
    let mut pickers = world.query::<&Picker>();
    let opens = pickers.iter(world).next();
    let opens = opens.and_then(|picker| picker.items.get(picker.selected));
    let opens = opens.map(|item| item.command.as_str());
    assert_eq!(opens, Some("model chatgpt/gpt-6.1-sol"));
}

#[test]
fn a_headless_run_with_no_model_waits_for_the_list_and_takes_an_offered_one() {
    use crate::print::AppHeadlessExt;
    use rig_harness::harness_protocol::Invocation;

    let backend = Backend::default();
    let release = backend.hold();
    let lists = [(200, SOL)];
    let (mut app, wakes, agent) = signed_in("login-headless", (&backend, &lists), NEVER, |app| {
        app.insert_resource(Invoked {
            args: Invocation::default(),
            terminal: false,
        })
        .add_plugins(rig_harness::host::exit::ExitPlugin)
        .run_headless(Some("Hello.".to_owned()));
    });
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(app.world().get::<ModelChoice>(agent), None);
    assert!(app.should_exit().is_none());
    drop(release);
    run_until(&mut app, &wakes, |world| {
        world.get::<ModelChoice>(agent).is_some()
    });
    let chosen = app.world().get::<ModelChoice>(agent).map(|it| it.0.clone());
    assert_eq!(chosen.as_deref(), Some("chatgpt/gpt-6.1-sol"));
}

#[test]
fn a_list_that_cannot_be_had_refuses_the_plans_models_and_says_why() {
    let backend = Backend::default();
    let lists = [(400, "bad request")];
    let (mut app, wakes, _) = signed_in("login-unlisted", (&backend, &lists), NEVER, |_| {});
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") != Some(ModelStatus::Checking)
    });
    let pick = FirstPick::of(app.world().resource::<Models>());
    let said = match &pick {
        FirstPick::Nothing(Some(why)) => why.starts_with("could not ask ChatGPT which models"),
        _ => false,
    };
    assert!(said, "{pick:?}");
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
}

/// A gate on the plan's GPT-6.1 Sol in `app`, whose backend is `backend`.
fn gate_on_sol(app: &App, backend: &Backend) -> Option<offered::Gate> {
    let session = app.world().resource::<super::Session>();
    let sol = Catalog::builtin().resolve("chatgpt/gpt-6.1-sol").ok()?;
    Some(offered::Gate {
        offered: Arc::clone(&session.offered),
        spec: sol.spec.clone(),
        auth: session.auth.clone(),
        http: DynHttpClient::new(backend.clone()),
    })
}

/// The account's models of `app`'s sign-in asked anew at `pace`, and the
/// status of GPT-6.1 Sol by them, waiting while it is being checked.
fn paced(
    app: &App,
    backend: &Backend,
    pace: offered::Pace,
) -> (Arc<offered::Offered>, impl Fn() -> ModelStatus) {
    let session = app.world().resource::<super::Session>();
    let offered = Arc::new(offered::Offered::paced(pace));
    let (asked, auth) = (Arc::clone(&offered), session.auth.clone());
    let http = DynHttpClient::new(backend.clone());
    let sol = Catalog::builtin().resolve("chatgpt/gpt-6.1-sol").ok();
    let sol = sol.map(|found| found.spec.clone());
    let status = move || {
        let started = std::time::Instant::now();
        loop {
            let Some(sol) = &sol else {
                return ModelStatus::NoTools;
            };
            let status = asked.status(sol, &auth, &http);
            if status != ModelStatus::Checking || started.elapsed().as_secs() > 20 {
                return status;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    };
    (offered, status)
}

/// A pace with `retries` quick retries and `cooldown` before a listing
/// asks again.
fn pace(retries: u32, cooldown: std::time::Duration) -> offered::Pace {
    let quick = std::time::Duration::from_millis(1);
    offered::Pace {
        retry: rig_ecs::retry::RetryPolicy {
            max_retries: retries,
            first_backoff: quick,
            max_backoff: quick,
            ..rig_ecs::retry::RetryPolicy::DEFAULT
        },
        cooldown,
    }
}

#[test]
fn a_list_nobody_waits_for_retries_a_transient_failure_itself() {
    let backend = Backend::default();
    let lists = [(503, "busy"), (200, SOL)];
    let (app, _, _) = signed_in("login-retried", (&backend, &lists), NEVER, |_| {});
    // A cooldown no listing outlasts: only the fetch's own retry asks again.
    let (_offered, status) = paced(&app, &backend, pace(1, std::time::Duration::from_secs(600)));
    assert_eq!(status(), ModelStatus::Usable);
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}

#[test]
fn a_sign_in_during_a_retrys_wait_stops_the_old_fetch() {
    let backend = Backend::default();
    let lists = [(503, "busy"), (200, SOL)];
    let (app, _, _) = signed_in("login-reset", (&backend, &lists), NEVER, |_| {});
    let session = app.world().resource::<super::Session>();
    let wait = std::time::Duration::from_millis(300);
    let retry = rig_ecs::retry::RetryPolicy {
        max_retries: 1,
        first_backoff: wait,
        max_backoff: wait,
        ..rig_ecs::retry::RetryPolicy::DEFAULT
    };
    let offered = Arc::new(offered::Offered::paced(offered::Pace {
        retry,
        cooldown: std::time::Duration::from_secs(600),
    }));
    let http = DynHttpClient::new(backend.clone());
    let sol = Catalog::builtin().resolve("chatgpt/gpt-6.1-sol").ok();
    assert!(sol.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(sol) = sol else { return };
    // The 503 is held until after the sign-in, so its retry's wait starts
    // with the fetch already forgotten.
    let release = backend.hold();
    assert_eq!(
        offered.status(sol.spec, &session.auth, &http),
        ModelStatus::Checking
    );
    until_listed(&backend, 1);
    offered.forget();
    drop(release);
    std::thread::sleep(wait * 2);
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
}

/// Waits, at most 10 s, until `backend` was asked for `lists` lists.
fn until_listed(backend: &Backend, lists: usize) {
    let started = std::time::Instant::now();
    while backend.listed.load(Ordering::SeqCst) < lists && started.elapsed().as_secs() < 10 {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn a_request_waiting_on_a_list_a_sign_in_dropped_asks_again() {
    let backend = Backend::default();
    let lists = [(200, SOL), (200, SOL)];
    let (app, _, _) = signed_in("login-dropped", (&backend, &lists), NEVER, |_| {});
    let gate = gate_on_sol(&app, &backend);
    assert!(gate.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(gate) = gate else { return };
    let offered = Arc::clone(&app.world().resource::<super::Session>().offered);
    let release = backend.hold();
    let proceeded = std::thread::scope(|scope| {
        let waiting = scope.spawn(|| before(&gate));
        until_listed(&backend, 1);
        offered.forget();
        drop(release);
        waiting.join().ok()
    });
    let proceeded = matches!(proceeded, Some(rig_core::serve::BeforeServe::Proceed));
    assert!(proceeded);
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}

#[test]
fn a_request_waiting_out_a_wait_asks_again_after_a_sign_in() {
    let backend = Backend::default();
    let lists = [(429, "Rate limited: try again in 30s."), (200, SOL)];
    let (app, _, _) = signed_in("login-wait-reset", (&backend, &lists), NEVER, |_| {});
    let gate = gate_on_sol(&app, &backend);
    assert!(gate.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(gate) = gate else { return };
    let offered = Arc::clone(&app.world().resource::<super::Session>().offered);
    assert!(denied(&gate).is_some());
    let started = std::time::Instant::now();
    let proceeded = std::thread::scope(|scope| {
        let waiting = scope.spawn(|| before(&gate));
        std::thread::sleep(std::time::Duration::from_millis(100));
        offered.forget();
        waiting.join().ok()
    });
    let proceeded = matches!(proceeded, Some(rig_core::serve::BeforeServe::Proceed));
    assert!(proceeded);
    // Not the 30 s the old account's backend asked for.
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}

#[test]
fn a_listing_asks_again_once_the_cooldown_after_a_failure_passed() {
    let backend = Backend::default();
    let lists = [(503, "busy"), (200, SOL)];
    let (app, _, _) = signed_in("login-relisted", (&backend, &lists), NEVER, |_| {});
    let cooldown = std::time::Duration::from_secs(1);
    let (_, status) = paced(&app, &backend, pace(0, cooldown));
    // Within the cooldown, listing says why and does not ask again.
    assert!(matches!(status(), ModelStatus::Unavailable(_)));
    assert!(matches!(status(), ModelStatus::Unavailable(_)));
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
    // After it, as when a session with no model opens `/model` again.
    std::thread::sleep(cooldown);
    assert_eq!(status(), ModelStatus::Usable);
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}

/// What `gate` does with a request for its model.
fn before(gate: &offered::Gate) -> rig_core::serve::BeforeServe {
    use rig_core::effect::{EffectId, EffectKind};
    use rig_core::serve::Intercept;

    let asking = EffectKind::Completion {
        request: Arc::new(rig_core::completion::CompletionRequest::new("Hello.")),
        stream: true,
    };
    bevy_tasks::block_on(gate.before(EffectId::from_raw(1), &asking))
}

/// The report `gate` denies a request with, if it does.
fn denied(gate: &offered::Gate) -> Option<rig_core::error::ErrorReport> {
    match before(gate) {
        rig_core::serve::BeforeServe::Deny(report) => Some(report),
        _ => None,
    }
}

#[test]
fn a_list_that_could_not_be_had_is_asked_again_by_the_next_request() {
    let backend = Backend::default();
    let lists = [(503, "busy"), (200, SOL)];
    let (mut app, wakes, agent) = signed_in("login-failed", (&backend, &lists), NEVER, |_| {});
    let gate = gate_on_sol(&app, &backend);
    assert!(gate.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(gate) = gate else { return };
    // The busy backend's refusal, with its class: it may be retried.
    let refused = denied(&gate);
    assert!(
        refused.as_ref().is_some_and(|report| report.retryable),
        "{refused:?}"
    );
    on_sol(&mut app, agent);
    run_until(&mut app, &wakes, |world| {
        world.get::<Connection>(agent).is_some()
    });
    // Listing, which follows the answer, does not ask again, and the
    // agent keeps its model.
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
    assert!(app.world().get::<Connection>(agent).is_some());
    assert!(matches!(
        before(&gate),
        rig_core::serve::BeforeServe::Proceed
    ));
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}

#[test]
fn a_list_refused_with_a_wait_is_asked_again_once_it_is_over() {
    let backend = Backend::default();
    let lists = [(429, "Rate limited: try again in 1s."), (200, SOL)];
    let (app, _, _) = signed_in("login-wait", (&backend, &lists), NEVER, |_| {});
    let gate = gate_on_sol(&app, &backend);
    assert!(gate.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(gate) = gate else { return };
    let started = std::time::Instant::now();
    let first = denied(&gate);
    let asked = first
        .as_ref()
        .and_then(|report| report.retry_after(std::time::SystemTime::now()));
    assert_eq!(asked, Some(std::time::Duration::from_secs(1)), "{first:?}");
    // The next request waits out the rest, then asks.
    assert!(matches!(
        before(&gate),
        rig_core::serve::BeforeServe::Proceed
    ));
    assert!(started.elapsed() >= std::time::Duration::from_secs(1));
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}

#[test]
fn a_list_refused_with_a_long_wait_refuses_the_next_request_at_once() {
    let backend = Backend::default();
    let lists = [(429, "Rate limited: try again in 120s."), (200, SOL)];
    let (app, _, _) = signed_in("login-long-wait", (&backend, &lists), NEVER, |_| {});
    let gate = gate_on_sol(&app, &backend);
    assert!(gate.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(gate) = gate else { return };
    assert!(denied(&gate).is_some());
    let again = denied(&gate);
    let waits = again.and_then(|report| report.retry_after(std::time::SystemTime::now()));
    assert_eq!(waits, Some(std::time::Duration::from_secs(120)));
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
}

#[test]
fn a_credential_refresh_that_never_answers_ends_with_the_lists_deadline() {
    let backend = Backend {
        stalls: true,
        ..Backend::default()
    };
    let lists = [(200, SOL)];
    // Expired: asking first refreshes it, which the backend never answers.
    let (app, _, _) = signed_in("login-stalled", (&backend, &lists), 0, |_| {});
    let gate = gate_on_sol(&app, &backend);
    assert!(gate.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(gate) = gate else { return };
    let started = std::time::Instant::now();
    let refused = denied(&gate);
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
    // A timeout, which may be retried and which signing in again would not
    // mend.
    let timed_out = refused.as_ref().is_some_and(|report| {
        report.kind == rig_core::error::ErrorKind::Timeout
            && report.retryable
            && report.message.contains("no answer in 10 s")
            && !report.message.contains("sign in")
    });
    assert!(timed_out, "{refused:?}");
    let sol = status(app.world(), "gpt-6.1-sol");
    let said = matches!(&sol, Some(ModelStatus::Unavailable(why))
        if why.contains("no answer in 10 s") && !why.contains("sign in"));
    assert!(said, "{sol:?}");
    assert_eq!(backend.listed.load(Ordering::SeqCst), 0);
}

#[test]
fn a_model_the_list_asked_again_leaves_out_is_refused_and_so_is_choosing_it_again() {
    let backend = Backend::default();
    let lists = [(200, SOL), (200, NONE)];
    let (mut app, wakes, agent) = signed_in("login-again", (&backend, &lists), NEVER, |app| {
        collect::<Notice>(app);
    });
    on_sol(&mut app, agent);
    ask(&mut app, &wakes, agent, "Hello.");
    assert_eq!(backend.sent.load(Ordering::SeqCst), 1);
    // The second list leaves Sol out: the agent's connection is refused.
    run_until(&mut app, &wakes, |world| {
        world.get::<ModelRefused>(agent).is_some()
    });
    let refused = |app: &App| {
        let refused = app.world().get::<ModelRefused>(agent);
        refused.map(|ModelRefused(why)| why.clone())
    };
    let said = "Cannot use chatgpt/gpt-6.1-sol: not offered to this ChatGPT account.";
    assert_eq!(refused(&app).as_deref(), Some(said));
    assert!(app.world().get::<Connection>(agent).is_none());
    let said = |app: &App| {
        let notices = app.world().resource::<Collected<Notice>>().0.iter();
        let refusals = notices.filter(|notice| notice.text.contains("not offered"));
        refusals.count()
    };
    app.update();
    let before = said(&app);
    // Choosing it again, as an editor may, is refused again, out loud.
    on_sol(&mut app, agent);
    app.update();
    assert!(app.world().get::<Connection>(agent).is_none());
    assert_eq!(said(&app), before + 1);
    assert_eq!(backend.sent.load(Ordering::SeqCst), 1);
}

#[test]
fn a_request_for_another_model_than_the_connected_one_is_checked_too() {
    use rig_core::completion::CompletionRequest;
    use rig_core::effect::{EffectId, EffectKind};
    use rig_core::serve::{BeforeServe, Intercept};
    use serde_json::json;

    let backend = Backend::default();
    let lists = [(200, SOL)];
    let (app, _, _) = signed_in("login-override", (&backend, &lists), NEVER, |_| {});
    let session = app.world().resource::<super::Session>();
    let sol = Catalog::builtin().resolve("chatgpt/gpt-6.1-sol").ok();
    assert!(sol.is_some(), "the catalog lists GPT-6.1 Sol");
    let Some(sol) = sol else { return };
    let gate = offered::Gate {
        offered: Arc::clone(&session.offered),
        spec: sol.spec.clone(),
        auth: session.auth.clone(),
        http: DynHttpClient::new(backend.clone()),
    };
    let asking = |model: Option<&str>| EffectKind::Completion {
        request: Arc::new(
            CompletionRequest::new("Hello.").model::<String>(model.map(str::to_owned)),
        ),
        stream: true,
    };
    // A raw `model` among the additional parameters, which the wire sends.
    let raw = EffectKind::Completion {
        request: Arc::new(
            CompletionRequest::new("Hello.").additional_params(json!({ "model": "gpt-6.1-terra" })),
        ),
        stream: true,
    };
    let before = |kind: &EffectKind| bevy_tasks::block_on(gate.before(EffectId::from_raw(1), kind));
    assert!(matches!(before(&asking(None)), BeforeServe::Proceed));
    let terra = before(&asking(Some("gpt-6.1-terra")));
    let said = match &terra {
        BeforeServe::Deny(report) => report.message.clone(),
        _ => String::new(),
    };
    assert_eq!(
        said,
        "chatgpt/gpt-6.1-terra is not offered to this ChatGPT account"
    );
    let raw = before(&raw);
    assert!(matches!(raw, BeforeServe::Deny(_)), "{raw:?}");
    // A provider's options, which the wire merges over the model: only
    // ChatGPT's Responses and shared sections are sent, the route over the
    // shared one.
    let optioned = |options: serde_json::Value| {
        let mut request = CompletionRequest::new("Hello.");
        request.provider_options = serde_json::from_value(options).unwrap_or_default();
        assert!(!request.provider_options.is_empty());
        let kind = EffectKind::Completion {
            request: Arc::new(request),
            stream: true,
        };
        matches!(before(&kind), BeforeServe::Proceed)
    };
    let terra = json!({ "model": "gpt-6.1-terra" });
    let sol = json!({ "model": "gpt-6.1-sol" });
    assert!(!optioned(
        json!({ "chatgpt": { "openai.responses": terra } })
    ));
    assert!(!optioned(json!({ "chatgpt": { "*": terra } })));
    assert!(optioned(
        json!({ "chatgpt": { "*": terra, "openai.responses": sol } })
    ));
    assert!(optioned(json!({ "chatgpt": { "openai.chat": terra } })));
    assert!(optioned(json!({ "openai": { "openai.responses": terra } })));
    // An empty override, which preparing the request drops.
    assert!(matches!(before(&asking(Some(""))), BeforeServe::Proceed));
    assert_eq!(backend.sent.load(Ordering::SeqCst), 0);
}

#[test]
fn the_list_is_asked_again_once_a_run() {
    let backend = Backend::default();
    // Were the list asked a third time, the third request would be refused
    // unsent: the third list leaves Sol out.
    let lists = [(200, SOL), (200, SOL), (200, NONE)];
    let (mut app, wakes, agent) = signed_in("login-once", (&backend, &lists), NEVER, |_| {});
    on_sol(&mut app, agent);
    ask(&mut app, &wakes, agent, "Hello.");
    run_until(&mut app, &wakes, |_| {
        backend.listed.load(Ordering::SeqCst) == 2
    });
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") == Some(ModelStatus::Usable)
    });
    ask(&mut app, &wakes, agent, "Again.");
    ask(&mut app, &wakes, agent, "Once more.");
    assert_eq!(backend.sent.load(Ordering::SeqCst), 3);
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}

#[test]
fn choosing_the_connected_model_in_the_frame_the_list_refuses_it_is_refused() {
    let backend = Backend::default();
    let lists = [(200, SOL), (200, NONE)];
    let (mut app, wakes, agent) = signed_in("login-same-frame", (&backend, &lists), NEVER, |_| {});
    on_sol(&mut app, agent);
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") == Some(ModelStatus::Usable)
    });
    // The list asked again waits until the turn ended.
    let release = backend.hold();
    ask(&mut app, &wakes, agent, "Hello.");
    drop(release);
    // No frame runs once the answer is in: only the choice itself can refuse.
    let offered = Arc::clone(&app.world().resource::<super::Session>().offered);
    let started = std::time::Instant::now();
    while offered.known_refusal("gpt-6.1-sol").is_none()
        && started.elapsed() < std::time::Duration::from_secs(10)
    {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(app.world().get::<ModelRefused>(agent).is_none());
    on_sol(&mut app, agent);
    app.world_mut().flush();
    assert!(app.world().get::<Connection>(agent).is_none());
    let refused = app.world().get::<ModelRefused>(agent);
    let said = refused.is_some_and(|ModelRefused(why)| why.contains("not offered"));
    assert!(said, "{refused:?}");
}

/// The choices taken, in order, with their kinds.
#[derive(Resource, Default)]
struct Taken(Vec<(ChoiceKind, Option<ModelChoice>)>);

#[test]
fn a_sign_in_switches_the_agent_for_that_run_only() {
    let backend = Backend::default();
    let lists = [(200, SOL)];
    let (mut app, _, agent) = signed_in("login-switch", (&backend, &lists), NEVER, |app| {
        let keep = |taken: On<ChoiceTaken>, mut kept: ResMut<Taken>| {
            kept.0.push((taken.kind, taken.model.clone()));
        };
        app.init_resource::<Taken>().add_observer(keep);
    });
    let (_, prompts) = crossbeam_channel::unbounded();
    let ended = Done(SignInEnded(Ok(())));
    app.world_mut()
        .spawn((PendingLogin { agent, prompts }, ended));
    app.update();
    let model = ModelChoice(format!("{PROVIDER}/{}", chatgpt::DEFAULT_MODEL));
    let taken = &app.world().resource::<Taken>().0;
    assert_eq!(taken, &[(ChoiceKind::ForThisRun, Some(model))]);
}
