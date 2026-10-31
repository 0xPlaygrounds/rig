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
use rig_harness_test_support::{Ended, app_in, first_agent, log_ended, run_until, scratch_home};

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

/// What the backend says to a model it does not serve.
const NOT_FOUND: &str = r#"{"error":{"message":"The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account.","type":"invalid_request_error","code":"model_not_found"}}"#;

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
        terra.is_some_and(|label| label.ends_with("(checking whether your plan offers it)")),
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
    let lists = [(503, "busy")];
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

#[test]
fn a_credential_refresh_that_never_answers_ends_with_the_lists_deadline() {
    let backend = Backend {
        stalls: true,
        ..Backend::default()
    };
    let lists = [(200, SOL)];
    // Expired: asking first refreshes it, which the backend never answers.
    let (mut app, wakes, _) = signed_in("login-stalled", (&backend, &lists), 0, |_| {});
    let started = std::time::Instant::now();
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") != Some(ModelStatus::Checking)
    });
    let sol = status(app.world(), "gpt-6.1-sol");
    let said =
        matches!(&sol, Some(ModelStatus::Unavailable(why)) if why.contains("no answer in 10 s"));
    assert!(said, "{sol:?}");
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
    assert_eq!(backend.listed.load(Ordering::SeqCst), 0);
}

#[test]
fn a_model_the_list_asked_again_leaves_out_is_refused_unsent() {
    let backend = Backend::default();
    let lists = [(200, SOL), (200, NONE)];
    let (mut app, wakes, agent) = signed_in("login-again", (&backend, &lists), NEVER, |_| {});
    on_sol(&mut app, agent);
    ask(&mut app, &wakes, agent, "Hello.");
    assert_eq!(backend.sent.load(Ordering::SeqCst), 1);
    run_until(&mut app, &wakes, |world| {
        matches!(
            status(world, "gpt-6.1-sol"),
            Some(ModelStatus::NotOffered(_))
        )
    });
    ask(&mut app, &wakes, agent, "Again.");
    let ended = &app.world().resource::<Ended>().0;
    let refused = ended.last();
    assert!(
        matches!(refused, Some(TurnOutcome::Failed(why)) if why.contains("not offered to this ChatGPT account")),
        "{ended:?}"
    );
    assert_eq!(backend.sent.load(Ordering::SeqCst), 1);
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
