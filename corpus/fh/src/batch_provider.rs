//! FH-25: model requests through Anthropic's Message Batches API, for
//! unattended runs at the batch price.
//!
//! Every Anthropic catalog model gets a twin under the vendor
//! `anthropic-batch`, at half its list price, connected to a handler that
//! serves nothing (`ErasedHandler::unavailable`). A model call to a twin is
//! held in [`GateSystems::After`] (`batch: queued`), and the request Start
//! would send it ([`Adapted::sent`]), encoded by rig-core's Messages
//! wire, is queued as a batch item. Items queued within
//! [`BatchSettings::window_ms`] go together as one batch. The batch is
//! polled with backoff ([`BatchSettings::first_poll_secs`], doubling up to
//! [`BatchSettings::max_poll_secs`]); once it ended, each item's result
//! decides its call: the reply, decoded as a live call would decode it
//! (`wire::decode_reply`) and priced at the batch price, or the provider's
//! error. An expired item is sent once more in a new batch.
//!
//! A turn's pending item is saved on its agent ([`BatchWait`]), and the
//! turn carries on across a restart ([`CarryOn`]): a quit leaves it for the
//! restart, whose restored call is held for the saved item, and polling
//! goes on; nothing is sent again. Interrupting the turn
//! cancels its batch at the provider when no other call waits for it.
//! Requests go through the app's HTTP client, so `wrap_http` wrappers see
//! them, with the key of [`BatchAccount`].

use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Catalog, ModelFacts, ModelSpec, Transport};
use rig_harness::rig_core::completion::{CompletionRequest, ModelRef};
use rig_harness::rig_core::effect::{self, FamilyDescriptor, HandlerDescriptor, model_key};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::http_client::{Bytes, DynHttpClient, HeaderMap, HttpClientExt, Request};
use rig_harness::rig_core::providers::anthropic::wire::Messages;
use rig_harness::rig_core::providers::anthropic::{self, AnthropicConfig};
use rig_harness::rig_core::providers::registry::ProviderId;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::wire::{Body, Mode, Wire, WireFrame, decode_reply};
use rig_harness::rig_ecs::session_log::Reset;
use serde::Deserialize;
use serde_json::{Value, json};

/// The batch twins' vendor.
pub const VENDOR: &str = "anthropic-batch";

/// The vendor whose models get batch twins.
const BASE: &str = anthropic::PROVIDER_NAME;

/// What a batch costs, of the list price.
pub const DISCOUNT: f64 = 0.5;

/// The most requests one batch takes.
const LIMIT: usize = 100_000;

const NAME: &str = "fh-batch-provider";

/// The account batches are sent with: `ANTHROPIC_API_KEY` and
/// `ANTHROPIC_BASE_URL` unless the app inserted its own.
#[derive(Resource, Clone, Debug)]
pub struct BatchAccount(pub AnthropicConfig);

/// The `[batch]` settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct BatchSettings {
    /// How long requests are gathered into one batch, in milliseconds.
    pub window_ms: u64,
    /// The first wait before a batch is polled, in seconds.
    pub first_poll_secs: u64,
    /// The longest wait between polls, in seconds.
    pub max_poll_secs: u64,
}

impl Default for BatchSettings {
    fn default() -> Self {
        Self {
            window_ms: 2_000,
            first_poll_secs: 10,
            max_poll_secs: 300,
        }
    }
}

/// The item an agent's turn waits for, saved with the session so a restart
/// polls it again rather than sending the request again.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Saved(Reset), Default)]
pub struct BatchWait(pub Option<PendingItem>);

/// A submitted batch item.
#[derive(Reflect, Clone, Debug, Default, PartialEq)]
pub struct PendingItem {
    /// The provider's batch id.
    pub batch: String,
    /// The item's id in it.
    pub custom_id: String,
    /// When it was first queued, in seconds since the Unix epoch.
    pub since: u64,
    /// Whether it was sent again after it expired.
    pub resubmitted: bool,
}

/// On a model call that goes through a batch.
#[derive(Component, Clone, Debug)]
#[component(storage = "SparseSet")]
pub struct Batched {
    custom_id: String,
    batch: Option<String>,
    since: SystemTime,
    resubmitted: bool,
    /// The wire bound to the batch model's facts, so its reply is priced
    /// at the batch price.
    wire: Messages,
    /// The request as sent, which its reply is decoded against.
    request: CompletionRequest,
    /// The Messages body, a batch item's `params`.
    params: Value,
    endpoint: Endpoint,
}

/// The Messages endpoint a request was encoded for, and its headers (key,
/// version, betas); the batch endpoints are under it.
#[derive(Clone, Debug)]
struct Endpoint {
    url: String,
    headers: HeaderMap,
}

/// The hold that keeps a batched call from being sent.
#[derive(Component)]
struct BatchHold;

/// Calls gathered for the next batch.
#[derive(Resource, Default)]
struct Queue(Vec<Entity>);

/// The batches being polled.
#[derive(Resource, Default)]
struct Polled(HashSet<String>);

/// A batch window: the queue is submitted when it is due.
#[derive(Component)]
struct Window;

/// A submission on its way, with the calls it carries.
#[derive(Component)]
struct Submitting {
    calls: Vec<Entity>,
    endpoint: Endpoint,
}

/// A submission's answer: the batch id.
struct Submitted(Result<String, String>);

/// A batch polled with backoff.
#[derive(Component)]
struct Poller {
    batch: String,
    endpoint: Endpoint,
    wait: Duration,
}

/// A poll's answer: each item's result by custom id once the batch ended.
struct PollAnswer(Result<Option<HashMap<String, ItemResult>>, String>);

/// One item's result, as a batch's results file has it.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ItemResult {
    Succeeded { message: Value },
    Errored { error: Value },
    Canceled,
    Expired,
}

/// Adds the batch models and sends their calls through batches.
#[derive(Default)]
pub struct BatchProviderPlugin;

/// The plugin's stage systems: in [`GateSystems::After`], so a call a gate
/// refused is never batched.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchSystems;

impl Plugin for BatchProviderPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<BatchSettings>("batch")
            .register_type::<BatchWait>()
            .register_required_components::<Agent, BatchWait>()
            .init_resource::<Queue>()
            .init_resource::<Polled>()
            .configure_sets(AgentPass, BatchSystems.in_set(GateSystems::After))
            .add_systems(AgentPass, take_over.in_set(BatchSystems))
            .add_systems(AgentPass, carry_on_waits.in_set(Stage::Gate))
            .add_observer(submit)
            .add_observer(submitted)
            .add_observer(poll)
            .add_observer(polled)
            .add_observer(settled);
        #[cfg(feature = "tui")]
        app.add_systems(
            PostUpdate,
            status::show.in_set(rig_tui::StatusSystems::Show),
        );
    }

    /// Mirrors Anthropic's models once every plugin added its own.
    fn finish(&self, app: &mut App) {
        if !app.world().contains_resource::<BatchAccount>()
            && let Ok(config) = AnthropicConfig::from_env()
        {
            app.insert_resource(BatchAccount(config));
        }
        let Ok(id) = ProviderId::declare(VENDOR) else {
            return;
        };
        let mut models = app.world_mut().get_resource_or_init::<Models>();
        let twins: Vec<ModelSpec> = models
            .0
            .catalog()
            .iter()
            .filter(|spec| spec.provider.vendor() == BASE)
            .map(|spec| twin(spec, id))
            .collect();
        for spec in twins {
            models.0.catalog_mut().insert(spec);
        }
        models
            .0
            .provide(id, |spec: &ModelSpec, _: &Catalog, _: &Transport| {
                let descriptor = HandlerDescriptor {
                    key: model_key(&spec.reference()),
                    family: FamilyDescriptor::Completion {
                        model: ModelRef::from(spec.reference()),
                        capabilities: Default::default(),
                    },
                    layers: Vec::new(),
                };
                Ok(ErasedHandler::unavailable(
                    descriptor,
                    "answered from the batch API",
                ))
            });
    }
}

/// `spec` as a model of the batch vendor `id`, at the batch price.
fn twin(spec: &ModelSpec, id: ProviderId) -> ModelSpec {
    let mut twin = spec.clone();
    twin.provider = id;
    twin.display_name = format!("{} (batch)", spec.display_name);
    twin.pricing = spec.pricing.map(|mut price| {
        price.input *= DISCOUNT;
        price.output *= DISCOUNT;
        price.cache_read = price.cache_read.map(|rate| rate * DISCOUNT);
        price.cache_write = price.cache_write.map(|rate| rate * DISCOUNT);
        price
    });
    twin
}

fn task_pool() -> &'static bevy_tasks::TaskPool {
    bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default)
}

fn failed(why: impl Into<String>) -> Outcome {
    Outcome(Err(ErrorReport::new(ErrorKind::Provider, why)))
}

/// `request` to the batch model `spec`, encoded as a live unary call to
/// Anthropic's model would send it, by a wire bound to `spec`'s facts.
fn encode(
    account: &AnthropicConfig,
    http: DynHttpClient,
    catalog: &Catalog,
    spec: &ModelSpec,
    request: CompletionRequest,
) -> Result<(Messages, Value, Endpoint), String> {
    let model = account.clone().connect(http).completion(&spec.id);
    let facts = ModelFacts::new(spec.clone()).with_catalog(catalog);
    let wire = model.wire.with_facts(facts);
    let encoded = wire
        .encode(request, Mode::Unary)
        .map_err(|error| error.to_string())?;
    let (parts, body) = encoded.request.into_parts();
    let Body::Bytes(body) = body else {
        return Err("a multipart request cannot go in a batch".to_owned());
    };
    let params = serde_json::from_slice(&body).map_err(|error| error.to_string())?;
    let endpoint = Endpoint {
        url: parts.uri.to_string(),
        headers: parts.headers,
    };
    Ok((wire, params, endpoint))
}

/// Holds each model call to a batch model and queues it, or attaches a
/// turn's call to the item its agent saved before a restart.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn take_over(
    mut calls: Query<
        (
            Entity,
            &ModelRequest,
            Option<&Adapted>,
            &Effect,
            &mut Decision,
            Option<&ForAgent>,
            Has<TurnCall>,
            Option<&CallOf>,
        ),
        (ReadyToStart, ModelCalls, Without<Batched>),
    >,
    agents: Query<&BatchWait>,
    (models, account, settings, invoked): (
        Res<Models>,
        Option<Res<BatchAccount>>,
        Res<BatchSettings>,
        Option<Res<Invoked>>,
    ),
    (mut queue, mut polled): (ResMut<Queue>, ResMut<Polled>),
    mut warned: Local<bool>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (call, model, adapted, effect, mut decision, agent, turn_call, turn) in &mut calls {
        let agent = agent.map(|agent| agent.0);
        if decision.is_decided() {
            continue;
        }
        let Some(adapted) = adapted else {
            continue;
        };
        let connection = &adapted.0;
        if connection.spec.provider.vendor() != VENDOR {
            continue;
        }
        let request = adapted.sent(model);
        let (Some(account), Some(http)) = (&account, models.0.transport().shared()) else {
            let why = "batch models need ANTHROPIC_API_KEY and an HTTP client";
            decision.decide(Decider::plugin(NAME), failed(why));
            continue;
        };
        let (wire, params, endpoint) = match encode(
            &account.0,
            http,
            models.0.catalog(),
            &connection.spec,
            request.clone(),
        ) {
            Ok(encoded) => encoded,
            Err(why) => {
                decision.decide(Decider::plugin(NAME), failed(why));
                continue;
            }
        };
        if !*warned && invoked.as_deref().is_some_and(Invoked::interactive) {
            *warned = true;
            let text =
                "Batch models answer through the provider's batch API: a reply may take hours.";
            notices.write(Notice::warn(agent, text));
        }
        let mut batched = Batched {
            custom_id: format!("rig-{}", effect.id().as_u64()),
            batch: None,
            since: SystemTime::now(),
            resubmitted: false,
            wire,
            request,
            params,
            endpoint,
        };
        let saved = agent
            .and_then(|agent| agents.get(agent).ok())
            .and_then(|wait| wait.0.clone())
            .filter(|_| turn_call);
        if let Some(saved) = saved {
            // Run again after a restart: wait for the saved item.
            batched.custom_id = saved.custom_id;
            batched.since = UNIX_EPOCH + Duration::from_secs(saved.since);
            batched.resubmitted = saved.resubmitted;
            if polled.0.insert(saved.batch.clone()) {
                let endpoint = batched.endpoint.clone();
                commands.spawn(poller(&settings, saved.batch.clone(), endpoint));
            }
            batched.batch = Some(saved.batch);
        } else {
            enqueue(&mut queue, &settings, call, &mut commands);
        }
        commands.entity(call).insert(batched);
        if let Some(&CallOf(turn)) = turn.filter(|_| turn_call) {
            commands.entity(turn).try_insert(CarryOn);
        }
        commands.spawn((
            Name::new("batch: queued"),
            BatchHold,
            HoldOf(call),
            KeepAwake(Duration::from_secs(60)),
        ));
    }
}

/// Lets each restored turn whose agent waits on a saved item carry on, so
/// [`take_over`] holds its call for that item again.
fn carry_on_waits(
    turns: Query<(Entity, &TurnOf), (With<Restored>, Without<CarryOn>)>,
    agents: Query<&BatchWait>,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent)) in &turns {
        if agents.get(agent).is_ok_and(|wait| wait.0.is_some()) {
            commands.entity(turn).insert(CarryOn);
        }
    }
}

/// Adds `call` to the next batch, opening its window, or closing it at
/// once when the batch is full.
fn enqueue(queue: &mut Queue, settings: &BatchSettings, call: Entity, commands: &mut Commands) {
    queue.0.push(call);
    if queue.0.len() == 1 {
        let window = WakeAt::after(Duration::from_millis(settings.window_ms));
        commands.spawn((Window, window));
    } else if queue.0.len() == LIMIT {
        commands.spawn((Window, WakeAt(UNIX_EPOCH)));
    }
}

fn poller(settings: &BatchSettings, batch: String, endpoint: Endpoint) -> impl Bundle {
    let wait = Duration::from_secs(settings.first_poll_secs);
    let poller = Poller {
        batch,
        endpoint,
        wait,
    };
    (poller, WakeAt::after(wait))
}

/// Sends `body` with `method` to `url` with `headers`; the reply's body,
/// or why it failed.
async fn send(
    http: DynHttpClient,
    method: &str,
    url: String,
    headers: HeaderMap,
    body: Option<Value>,
) -> Result<Bytes, String> {
    let mut request = Request::builder().method(method).uri(url);
    if let Some(slot) = request.headers_mut() {
        *slot = headers;
    }
    let body = body.map(|body| body.to_string().into_bytes());
    let request = request
        .body(body.unwrap_or_default())
        .map_err(|error| error.to_string())?;
    let response = http
        .send::<_, Bytes>(request)
        .await
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let body = response
        .into_body()
        .await
        .map_err(|error| error.to_string())?;
    if status.is_success() {
        Ok(body)
    } else {
        Err(format!("{status}: {}", String::from_utf8_lossy(&body)))
    }
}

/// A batch as the API describes it.
#[derive(Deserialize)]
struct BatchStatus {
    id: String,
    processing_status: String,
    results_url: Option<String>,
}

fn status_of(body: &[u8]) -> Result<BatchStatus, String> {
    serde_json::from_slice(body).map_err(|error| error.to_string())
}

/// Submits the queue when its window closes, up to the batch limit; the
/// rest go in the next batch at once.
fn submit(
    due: On<Add<Due>>,
    windows: Query<(), With<Window>>,
    calls: Query<&Batched>,
    models: Res<Models>,
    mut queue: ResMut<Queue>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    if !windows.contains(due.entity) {
        return;
    }
    let full = LIMIT.min(queue.0.len());
    let rest = queue.0.split_off(full);
    let taken = std::mem::replace(&mut queue.0, rest);
    if !queue.0.is_empty() {
        commands.spawn((Window, WakeAt(UNIX_EPOCH)));
    }
    let items: Vec<&Batched> = taken
        .iter()
        .filter_map(|&call| calls.get(call).ok())
        .collect();
    let (Some(first), Some(http)) = (items.first(), models.0.transport().shared()) else {
        commands.entity(due.entity).despawn();
        return;
    };
    let endpoint = first.endpoint.clone();
    let requests: Vec<Value> = items
        .iter()
        .map(|item| json!({"custom_id": item.custom_id, "params": item.params}))
        .collect();
    let sending = send(
        http,
        "POST",
        format!("{}/batches", endpoint.url),
        endpoint.headers.clone(),
        Some(json!({ "requests": requests })),
    );
    let work = async move { Submitted(sending.await.and_then(|body| Ok(status_of(&body)?.id))) };
    commands.entity(due.entity).remove::<Window>().insert((
        Submitting {
            calls: taken,
            endpoint,
        },
        Running::spawn(task_pool(), &wake, work),
    ));
}

/// Notes each call's batch, on its agent too for a turn's call, and starts
/// polling; a refused submission fails its calls. A batch whose every call
/// was stopped meanwhile is cancelled.
#[allow(clippy::type_complexity)]
fn submitted(
    done: On<Insert<Done<Submitted>>>,
    submissions: Query<(&Submitting, &Done<Submitted>)>,
    mut calls: Query<
        (
            &mut Batched,
            &mut Decision,
            Option<&ForAgent>,
            Has<TurnCall>,
        ),
        Without<Settled>,
    >,
    mut agents: Query<&mut BatchWait>,
    (models, settings, mut polled): (Res<Models>, Res<BatchSettings>, ResMut<Polled>),
    holds: Query<(Entity, &HoldOf), With<BatchHold>>,
    mut commands: Commands,
) {
    let Ok((submitting, Done(Submitted(answer)))) = submissions.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let mut waiting = false;
    for &call in &submitting.calls {
        let Ok((mut batched, mut decision, agent, turn_call)) = calls.get_mut(call) else {
            continue;
        };
        waiting = true;
        match answer {
            Ok(batch) => {
                batched.batch = Some(batch.clone());
                if let (true, Some(agent)) = (turn_call, agent)
                    && let Ok(mut wait) = agents.get_mut(agent.0)
                {
                    let since = batched.since.duration_since(UNIX_EPOCH);
                    wait.0 = Some(PendingItem {
                        batch: batch.clone(),
                        custom_id: batched.custom_id.clone(),
                        since: since.unwrap_or_default().as_secs(),
                        resubmitted: batched.resubmitted,
                    });
                }
            }
            Err(why) => {
                let why = format!("the batch was not accepted: {why}");
                decision.decide(Decider::plugin(NAME), failed(why));
                release(call, &holds, &mut commands);
            }
        }
    }
    let Ok(batch) = answer else {
        return;
    };
    if !waiting {
        cancel(&models, &submitting.endpoint, batch);
    } else if polled.0.insert(batch.clone()) {
        let endpoint = submitting.endpoint.clone();
        commands.spawn(poller(&settings, batch.clone(), endpoint));
    }
}

/// Cancels `batch` at the provider; nothing waits for the answer.
fn cancel(models: &Models, endpoint: &Endpoint, batch: &str) {
    let Some(http) = models.0.transport().shared() else {
        return;
    };
    let url = format!("{}/batches/{batch}/cancel", endpoint.url);
    let cancel = send(http, "POST", url, endpoint.headers.clone(), None);
    task_pool()
        .spawn(async move {
            cancel.await.ok();
        })
        .detach();
}

/// Lets go of a call's batch hold.
fn release(
    call: Entity,
    holds: &Query<(Entity, &HoldOf), With<BatchHold>>,
    commands: &mut Commands,
) {
    for (hold, &HoldOf(of)) in holds {
        if of == call {
            commands.entity(hold).despawn();
        }
    }
}

/// Polls a batch when its wait is over: its status, then its results once
/// it ended.
fn poll(
    due: On<Add<Due>>,
    pollers: Query<&Poller, Without<Running>>,
    models: Res<Models>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (Ok(poller), Some(http)) = (pollers.get(due.entity), models.0.transport().shared()) else {
        return;
    };
    let url = format!("{}/batches/{}", poller.endpoint.url, poller.batch);
    let headers = poller.endpoint.headers.clone();
    let work = async move {
        let answer = async {
            let status = status_of(&send(http.clone(), "GET", url, headers.clone(), None).await?)?;
            let (Some(results), "ended") = (status.results_url, status.processing_status.as_str())
            else {
                return Ok(None);
            };
            let lines = send(http, "GET", results, headers, None).await?;
            let mut items = HashMap::new();
            for line in lines.split(|&byte| byte == b'\n') {
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                #[derive(Deserialize)]
                struct Line {
                    custom_id: String,
                    result: ItemResult,
                }
                let line: Line = serde_json::from_slice(line).map_err(|error| error.to_string())?;
                items.insert(line.custom_id, line.result);
            }
            Ok(Some(items))
        };
        PollAnswer(answer.await)
    };
    commands
        .entity(due.entity)
        .insert(Running::spawn(task_pool(), &wake, work));
}

/// Once a batch ended, answers each of its calls, sending an expired one
/// again once; until then, polls again later.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn polled(
    done: On<Insert<Done<PollAnswer>>>,
    mut pollers: Query<(&mut Poller, &Done<PollAnswer>)>,
    mut calls: Query<(Entity, &mut Batched, &mut Decision, Option<&ForAgent>)>,
    holds: Query<(Entity, &HoldOf), With<BatchHold>>,
    settings: Res<BatchSettings>,
    (mut queue, mut polled): (ResMut<Queue>, ResMut<Polled>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((mut poller, Done(PollAnswer(answer)))) = pollers.get_mut(done.entity) else {
        return;
    };
    let results = match answer {
        Ok(Some(results)) => results,
        Ok(None) | Err(_) => {
            if let Err(why) = answer {
                let text = format!("Polling batch {}: {why}", poller.batch);
                notices.write(Notice::warn(None, text));
            }
            poller.wait = (poller.wait * 2).min(Duration::from_secs(settings.max_poll_secs));
            commands
                .entity(done.entity)
                .remove::<Done<PollAnswer>>()
                .insert(WakeAt::after(poller.wait));
            return;
        }
    };
    for (call, mut batched, mut decision, agent) in &mut calls {
        if batched.batch.as_ref() != Some(&poller.batch) || decision.is_decided() {
            continue;
        }
        let outcome = match results.get(&batched.custom_id) {
            Some(ItemResult::Succeeded { message }) => {
                let body = WireFrame::Bytes(message.to_string().into_bytes());
                match decode_reply(&batched.wire, &batched.request, Mode::Unary, [body]) {
                    Ok(reply) => Outcome(Ok(effect::Answered::Completion(reply.into()))),
                    Err(error) => failed(error.to_string()),
                }
            }
            Some(ItemResult::Expired) if !batched.resubmitted => {
                batched.resubmitted = true;
                batched.batch = None;
                enqueue(&mut queue, &settings, call, &mut commands);
                let text = "A batch request expired; sent once more.";
                notices.write(Notice::info(agent.map(|agent| agent.0), text));
                continue;
            }
            Some(ItemResult::Expired) => failed("the batch request expired twice"),
            Some(ItemResult::Errored { error }) => {
                let message = error
                    .pointer("/error/message")
                    .or_else(|| error.get("message"))
                    .and_then(Value::as_str);
                failed(message.map_or_else(|| error.to_string(), str::to_owned))
            }
            Some(ItemResult::Canceled) => failed("the batch request was cancelled"),
            None => failed("the batch ended without this request's result"),
        };
        decision.decide(Decider::plugin(NAME), outcome);
        release(call, &holds, &mut commands);
    }
    polled.0.remove(&poller.batch);
    commands.entity(done.entity).despawn();
}

/// Once a batched call settles: its agent waits no more, its turn no
/// longer outlives an exit, and a call stopped before its batch answered
/// (an interrupt) cancels the batch at the provider when no other call
/// waits for it; one stopped while its batch is being submitted, once the
/// batch has its id. A restart does not settle the call, so its item stays
/// saved.
fn settled(
    settled: On<Add<Settled>>,
    calls: Query<(
        &Batched,
        &Decision,
        Option<&ForAgent>,
        Option<&CallOf>,
        Has<TurnCall>,
    )>,
    others: Query<(Entity, &Batched), Without<Settled>>,
    mut agents: Query<&mut BatchWait>,
    models: Res<Models>,
    mut queue: ResMut<Queue>,
    mut commands: Commands,
) {
    let Ok((batched, decision, agent, turn, turn_call)) = calls.get(settled.entity) else {
        return;
    };
    queue.0.retain(|&call| call != settled.entity);
    if let (true, Some(agent)) = (turn_call, agent)
        && let Ok(mut wait) = agents.get_mut(agent.0)
    {
        wait.0 = None;
    }
    if let Some(&CallOf(turn)) = turn.filter(|_| turn_call) {
        commands.entity(turn).try_remove::<CarryOn>();
    }
    let Some(batch) = batched.batch.as_ref().filter(|_| !decision.is_decided()) else {
        return;
    };
    let shared = others
        .iter()
        .any(|(call, other)| call != settled.entity && other.batch.as_ref() == Some(batch));
    if !shared {
        cancel(&models, &batched.endpoint, batch);
    }
}

#[cfg(feature = "tui")]
mod status {
    //! `batch: queued 12m` in the status line of each waiting agent.

    use std::time::SystemTime;

    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, Tone};

    use super::Batched;

    /// After the status; gone before the reasoning setting.
    const BATCH: StatusItem = StatusItem::new("fh-batch-provider", Side::Left, 52, 12);

    /// Shows how long each agent's batch item has waited, in minutes.
    pub(super) fn show(
        calls: Query<(&Batched, &ForAgent), Without<Settled>>,
        mut agents: Query<(Entity, &mut StatusItems), With<Agent>>,
        mut shown: Local<std::collections::HashMap<Entity, String>>,
    ) {
        let now = SystemTime::now();
        for (agent, mut items) in &mut agents {
            let text = calls
                .iter()
                .filter(|(_, of)| of.0 == agent)
                .map(|(batched, _)| batched.since)
                .min()
                .map(|since| {
                    let waited = now.duration_since(since).unwrap_or_default();
                    format!("batch: queued {}m", waited.as_secs() / 60)
                })
                .unwrap_or_default();
            if shown.get(&agent) != Some(&text) {
                items.show(BATCH.says(text.clone(), Tone::Dim));
                shown.insert(agent, text);
            }
        }
    }
}

#[cfg(test)]
mod tests;
