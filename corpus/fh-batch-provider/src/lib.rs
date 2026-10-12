//! FH-25: model requests through a provider's batch API, for unattended
//! runs at the batch price.
//!
//! Every Anthropic and OpenAI catalog model gets a twin under the vendor
//! `anthropic-batch` or `openai-batch`, at half its list price. A model
//! call (a turn's or a plugin's) that goes to such a model is never sent
//! the usual way: a Gate system holds it (`batch: queued`) and queues it as
//! a batch item. Items queued within [`BatchSettings::window_ms`] go
//! together as one batch, up to the endpoint's limit. The batch is then
//! polled with backoff ([`BatchSettings::first_poll_secs`], doubling up to
//! [`BatchSettings::max_poll_secs`]) and each item's result decides its
//! call: a complete reply (no streaming), with its tool calls and its usage
//! priced at the batch price, or the provider's error. An expired item is
//! sent once more in a new batch.
//!
//! A turn's pending item is saved on its agent ([`BatchWait`]): after a
//! restart, the turn the kernel carries on is attached to the saved item
//! and polling goes on; nothing is sent again. Interrupting the turn
//! cancels the item at the provider. The endpoint of each batch vendor is
//! a [`BatchApi`] in [`BatchEndpoints`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::BoxFuture;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Catalog, ModelSpec, Pricing, Transport};
use rig_harness::rig_core::completion::ProviderCapabilities;
use rig_harness::rig_core::completion::{CompletionRequest, CompletionResponse, ModelRef};
use rig_harness::rig_core::effect::{
    self, EffectKind, FamilyDescriptor, HandlerDescriptor, family, model_key,
};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::providers::registry::{ConnectError, ProviderId};
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use serde::Deserialize;

/// Each base vendor and its batch twin.
pub const VENDORS: [(&str, &str); 2] =
    [("anthropic", "anthropic-batch"), ("openai", "openai-batch")];

/// What a batch costs, of the list price.
pub const DISCOUNT: f64 = 0.5;

/// One request in a batch.
#[derive(Clone, Debug)]
pub struct BatchItem {
    /// The id the provider answers it under.
    pub custom_id: String,
    /// The model of the base vendor, such as `claude-sonnet-4-5`.
    pub model: String,
    /// The request, as the agent built it.
    pub request: CompletionRequest,
}

/// Where an item of a batch stands.
#[derive(Clone, Debug)]
pub enum ItemState {
    /// Not answered yet.
    Pending,
    /// The provider's reply.
    Succeeded(Box<CompletionResponse>),
    /// The provider refused or failed it, with its message.
    Errored(String),
    /// The batch expired before the item was answered.
    Expired,
    /// The item was cancelled.
    Cancelled,
}

/// A provider's batch endpoint. Its futures run on Bevy's IO task pool.
pub trait BatchApi: Send + Sync + 'static {
    /// Submits `items` as one batch; the batch's id.
    fn submit(&self, items: Vec<BatchItem>) -> BoxFuture<'static, Result<String, String>>;
    /// Where each item of the batch `batch` stands, by custom id.
    fn poll(&self, batch: String) -> BoxFuture<'static, Result<Vec<(String, ItemState)>, String>>;
    /// Cancels the item `custom_id` of `batch`, where the provider can.
    fn cancel(&self, batch: String, custom_id: String) -> BoxFuture<'static, Result<(), String>>;
    /// How many items one batch takes.
    fn limit(&self) -> usize {
        10_000
    }
}

/// The endpoint of each batch vendor (`anthropic-batch`, `openai-batch`).
#[derive(Resource, Default, Clone)]
pub struct BatchEndpoints(pub HashMap<String, Arc<dyn BatchApi>>);

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
#[reflect(Component, Saved, Default)]
pub struct BatchWait(pub Option<PendingItem>);

/// A submitted batch item.
#[derive(Reflect, Clone, Debug, Default, PartialEq)]
pub struct PendingItem {
    /// The batch vendor, such as `anthropic-batch`.
    pub vendor: String,
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
    /// The batch vendor.
    pub vendor: String,
    /// What is sent.
    pub item: BatchItem,
    /// The batch it is in, once submitted.
    pub batch: Option<String>,
    /// When it was first queued.
    pub since: SystemTime,
    /// Whether it was sent again after it expired.
    pub resubmitted: bool,
    /// The batch price.
    pub pricing: Option<Pricing>,
}

/// The hold that keeps a batched call from being sent.
#[derive(Component)]
struct BatchHold;

/// Items gathered for the next batch of each vendor.
#[derive(Resource, Default)]
struct Queue(HashMap<String, Vec<Entity>>);

/// The batches being polled, by vendor and id.
#[derive(Resource, Default)]
struct Polled(HashSet<(String, String)>);

/// A batch window: its vendor's queue is submitted when it is due.
#[derive(Component)]
struct Window(String);

/// A submission on its way, with the calls it carries.
#[derive(Component)]
struct Submitting {
    vendor: String,
    calls: Vec<Entity>,
}

/// A submission's answer: the batch id.
struct Submitted(Result<String, String>);

/// A batch polled with backoff.
#[derive(Component)]
struct Poller {
    vendor: String,
    batch: String,
    wait: Duration,
}

/// A poll's answer.
struct PollAnswer(Result<Vec<(String, ItemState)>, String>);

/// Adds the batch models and sends their calls through batches.
#[derive(Default)]
pub struct BatchProviderPlugin;

impl Plugin for BatchProviderPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<BatchSettings>("batch")
            .register_type::<BatchWait>()
            .register_required_components::<Agent, BatchWait>()
            .init_resource::<BatchEndpoints>()
            .init_resource::<Queue>()
            .init_resource::<Polled>()
            .add_systems(AgentPass, take_over.in_set(Stage::Gate))
            .add_observer(submit)
            .add_observer(submitted)
            .add_observer(poll)
            .add_observer(polled)
            .add_observer(settled);
        #[cfg(feature = "tui")]
        app.add_systems(PostUpdate, status::show.in_set(rig_tui::StatusSystems));
    }

    /// Mirrors the base vendors' models once every plugin added its own.
    fn finish(&self, app: &mut App) {
        let mut models = app.world_mut().get_resource_or_init::<Models>();
        for (base, batch) in VENDORS {
            let Ok(id) = ProviderId::declare(batch) else {
                continue;
            };
            let twins: Vec<ModelSpec> = models
                .0
                .catalog()
                .iter()
                .filter(|spec| spec.provider.vendor() == base)
                .map(|spec| twin(spec, id))
                .collect();
            for spec in twins {
                models.0.catalog_mut().insert(spec);
            }
            models.0.provide(id, unbatched);
        }
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

/// The handler of a batch model: the plugin's Gate system answers every
/// call to it, so one that reaches it was not batched.
fn unbatched(spec: &ModelSpec, _: &Catalog, _: &Transport) -> Result<ErasedHandler, ConnectError> {
    Ok(ErasedHandler::new(Unbatched(ModelRef::from(
        spec.reference(),
    ))))
}

struct Unbatched(ModelRef);

impl Serve for Unbatched {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: model_key(self.0.as_str()),
            family: FamilyDescriptor::Completion {
                model: self.0.clone(),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        let why = "a batch model is answered only through its batch";
        Reply::Outcome(Err(ErrorReport::new(ErrorKind::HandlerUnavailable, why)))
    }
}

fn now_secs(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn task_pool() -> &'static bevy_tasks::TaskPool {
    bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default)
}

/// Holds each model call about to go to a batch model and queues it, or
/// attaches a turn's call to the item its agent saved before a restart.
#[allow(clippy::too_many_arguments)]
fn take_over(
    mut calls: Query<
        (
            Entity,
            &ModelRequest,
            &Effect,
            &mut Decision,
            Option<&ForAgent>,
            Has<TurnCall>,
        ),
        (Undecided, ModelCalls, Without<Batched>),
    >,
    agents: Query<(Option<&Connection>, &AgentId, &BatchWait)>,
    endpoints: Res<BatchEndpoints>,
    settings: Res<BatchSettings>,
    invoked: Option<Res<Invoked>>,
    (mut queue, mut polled): (ResMut<Queue>, ResMut<Polled>),
    mut warned: Local<bool>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (call, request, effect, mut decision, agent, turn_call) in &mut calls {
        if decision.is_decided() {
            continue;
        }
        let agent = agent.map(|agent| agent.0);
        let owner = agent.and_then(|agent| agents.get(agent).ok());
        let connection = request
            .connection
            .as_ref()
            .or_else(|| owner.and_then(|(connection, ..)| connection));
        let Some(connection) = connection else {
            continue;
        };
        let vendor = connection.spec.provider.vendor();
        if !VENDORS.iter().any(|(_, batch)| *batch == vendor) {
            continue;
        }
        let Some(api) = endpoints.0.get(vendor) else {
            let why = format!("no batch endpoint is set up for `{vendor}`");
            decision.decide(Outcome(Err(ErrorReport::new(ErrorKind::Provider, why))));
            continue;
        };
        if !*warned && invoked.as_deref().is_some_and(Invoked::interactive) {
            *warned = true;
            let text =
                "Batch models answer through the provider's batch API: a reply may take hours.";
            notices.write(Notice::warn(agent, text));
        }
        let saved = owner
            .and_then(|(.., wait)| wait.0.clone())
            .filter(|saved| turn_call && saved.vendor == vendor);
        let id = owner.map_or_else(|| "app".to_owned(), |(_, id, _)| id.0.clone());
        let mut item = BatchItem {
            custom_id: format!("{id}-{}", effect.id().as_u64()),
            model: connection.spec.id.clone(),
            request: request.request.clone(),
        };
        let mut batched = Batched {
            vendor: vendor.to_owned(),
            item: item.clone(),
            batch: None,
            since: SystemTime::now(),
            resubmitted: false,
            pricing: connection.spec.pricing,
        };
        if let Some(saved) = saved {
            // Carried on after a restart: wait for the saved item.
            item.custom_id.clone_from(&saved.custom_id);
            batched.item = item;
            batched.batch = Some(saved.batch.clone());
            batched.since = UNIX_EPOCH + Duration::from_secs(saved.since);
            batched.resubmitted = saved.resubmitted;
            if polled.0.insert((saved.vendor.clone(), saved.batch.clone())) {
                commands.spawn(poller(&settings, saved.vendor, saved.batch));
            }
        } else {
            enqueue(
                &mut queue,
                &settings,
                api.limit(),
                vendor,
                call,
                &mut commands,
            );
        }
        commands.entity(call).insert(batched);
        commands.spawn((
            Name::new("batch: queued"),
            BatchHold,
            HoldOf(call),
            KeepAwake(Duration::from_secs(60)),
        ));
    }
}

/// Adds `call` to its vendor's next batch, opening its window, or closing
/// it at once when the batch is full.
fn enqueue(
    queue: &mut Queue,
    settings: &BatchSettings,
    limit: usize,
    vendor: &str,
    call: Entity,
    commands: &mut Commands,
) {
    let calls = queue.0.entry(vendor.to_owned()).or_default();
    calls.push(call);
    if calls.len() == 1 {
        let window = WakeAt::after(Duration::from_millis(settings.window_ms));
        commands.spawn((Window(vendor.to_owned()), window));
    } else if calls.len() >= limit {
        commands.spawn((Window(vendor.to_owned()), WakeAt(SystemTime::UNIX_EPOCH)));
    }
}

fn poller(settings: &BatchSettings, vendor: String, batch: String) -> impl Bundle {
    let wait = Duration::from_secs(settings.first_poll_secs);
    (
        Poller {
            vendor,
            batch,
            wait,
        },
        WakeAt::after(wait),
    )
}

/// Submits a vendor's queue when its window closes.
fn submit(
    due: On<Add<Due>>,
    windows: Query<&Window>,
    calls: Query<&Batched>,
    endpoints: Res<BatchEndpoints>,
    mut queue: ResMut<Queue>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(Window(vendor)) = windows.get(due.entity) else {
        return;
    };
    let mut window = commands.entity(due.entity);
    window.remove::<Window>();
    let taken = queue.0.remove(vendor).unwrap_or_default();
    let (Some(api), false) = (endpoints.0.get(vendor), taken.is_empty()) else {
        window.despawn();
        return;
    };
    let items: Vec<BatchItem> = taken
        .iter()
        .filter_map(|&call| calls.get(call).ok())
        .map(|batched| batched.item.clone())
        .collect();
    let sending = api.submit(items);
    let work = async move { Submitted(sending.await) };
    window.insert((
        Submitting {
            vendor: vendor.clone(),
            calls: taken,
        },
        Running::spawn(task_pool(), &wake, work),
    ));
}

/// Notes each call's batch, on its agent too for a turn's call, and starts
/// polling; a refused submission fails its calls.
#[allow(clippy::too_many_arguments)]
fn submitted(
    done: On<Insert<Done<Submitted>>>,
    submissions: Query<(&Submitting, &Done<Submitted>)>,
    mut calls: Query<(
        &mut Batched,
        &mut Decision,
        Option<&ForAgent>,
        Has<TurnCall>,
    )>,
    mut agents: Query<&mut BatchWait>,
    settings: Res<BatchSettings>,
    mut polled: ResMut<Polled>,
    holds: Query<(Entity, &HoldOf), With<BatchHold>>,
    mut commands: Commands,
) {
    let Ok((submitting, Done(Submitted(answer)))) = submissions.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let vendor = &submitting.vendor;
    for &call in &submitting.calls {
        let Ok((mut batched, mut decision, agent, turn_call)) = calls.get_mut(call) else {
            continue;
        };
        match answer {
            Ok(batch) => {
                batched.batch = Some(batch.clone());
                if let (true, Some(agent)) = (turn_call, agent)
                    && let Ok(mut wait) = agents.get_mut(agent.0)
                {
                    wait.0 = Some(PendingItem {
                        vendor: vendor.clone(),
                        batch: batch.clone(),
                        custom_id: batched.item.custom_id.clone(),
                        since: now_secs(batched.since),
                        resubmitted: batched.resubmitted,
                    });
                }
            }
            Err(why) => {
                let why = format!("the batch was not accepted: {why}");
                decision.decide(Outcome(Err(ErrorReport::new(ErrorKind::Provider, why))));
                release(call, &holds, &mut commands);
            }
        }
    }
    if let Ok(batch) = answer
        && polled.0.insert((vendor.clone(), batch.clone()))
    {
        commands.spawn(poller(&settings, vendor.clone(), batch.clone()));
    }
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

/// Polls a batch when its wait is over.
fn poll(
    due: On<Add<Due>>,
    pollers: Query<&Poller, Without<Running>>,
    endpoints: Res<BatchEndpoints>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(poller) = pollers.get(due.entity) else {
        return;
    };
    let Some(api) = endpoints.0.get(&poller.vendor) else {
        return;
    };
    let asking = api.poll(poller.batch.clone());
    let work = async move { PollAnswer(asking.await) };
    commands
        .entity(due.entity)
        .insert(Running::spawn(task_pool(), &wake, work));
}

/// Answers each call whose item ended, sends an expired one again once,
/// and polls again later while any call waits for the batch.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn polled(
    done: On<Insert<Done<PollAnswer>>>,
    mut pollers: Query<(&mut Poller, &Done<PollAnswer>)>,
    mut calls: Query<(Entity, &mut Batched, &mut Decision, Option<&ForAgent>)>,
    holds: Query<(Entity, &HoldOf), With<BatchHold>>,
    endpoints: Res<BatchEndpoints>,
    settings: Res<BatchSettings>,
    (mut queue, mut polled): (ResMut<Queue>, ResMut<Polled>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((mut poller, Done(PollAnswer(answer)))) = pollers.get_mut(done.entity) else {
        return;
    };
    let states = match answer {
        Ok(states) => states.as_slice(),
        Err(why) => {
            notices.write(Notice::warn(
                None,
                format!("Polling batch {}: {why}", poller.batch),
            ));
            &[]
        }
    };
    for (custom_id, state) in states {
        let call = calls.iter_mut().find(|(_, batched, ..)| {
            batched.batch.as_ref() == Some(&poller.batch) && batched.item.custom_id == *custom_id
        });
        let Some((call, mut batched, mut decision, agent)) = call else {
            continue;
        };
        let failed = |why: String| Outcome(Err(ErrorReport::new(ErrorKind::Provider, why)));
        let outcome = match state {
            ItemState::Pending => continue,
            ItemState::Succeeded(reply) => {
                let mut reply = (**reply).clone();
                if reply.usage.cost.is_none() {
                    reply.usage.cost = batched.pricing.and_then(|price| price.cost(&reply.usage));
                }
                Outcome(Ok(effect::Outcome::Completion(reply)))
            }
            ItemState::Expired if !batched.resubmitted => {
                batched.resubmitted = true;
                batched.batch = None;
                let limit = endpoints.0.get(&poller.vendor).map_or(1, |api| api.limit());
                enqueue(
                    &mut queue,
                    &settings,
                    limit,
                    &poller.vendor,
                    call,
                    &mut commands,
                );
                let agent = agent.map(|agent| agent.0);
                notices.write(Notice::info(
                    agent,
                    "A batch request expired; sent once more.",
                ));
                continue;
            }
            ItemState::Expired => failed("the batch request expired twice".to_owned()),
            ItemState::Errored(why) => failed(why.clone()),
            ItemState::Cancelled => failed("the batch request was cancelled".to_owned()),
        };
        decision.decide(outcome);
        release(call, &holds, &mut commands);
    }
    let waiting = calls.iter().any(|(_, batched, decision, _)| {
        batched.batch.as_ref() == Some(&poller.batch) && !decision.is_decided()
    });
    let max = Duration::from_secs(settings.max_poll_secs);
    if waiting {
        poller.wait = (poller.wait * 2).min(max);
        commands
            .entity(done.entity)
            .remove::<Done<PollAnswer>>()
            .insert(WakeAt::after(poller.wait));
    } else {
        polled
            .0
            .remove(&(poller.vendor.clone(), poller.batch.clone()));
        commands.entity(done.entity).despawn();
    }
}

/// Once a batched call settles: its agent waits no more, and an item
/// stopped by an interrupt is cancelled at the provider. A restart does
/// not settle the call, so its item stays saved.
fn settled(
    settled: On<Add<Settled>>,
    calls: Query<(&Batched, &Outcome, Option<&ForAgent>, Has<TurnCall>)>,
    mut agents: Query<&mut BatchWait>,
    endpoints: Res<BatchEndpoints>,
    mut queue: ResMut<Queue>,
) {
    let Ok((batched, Outcome(outcome), agent, turn_call)) = calls.get(settled.entity) else {
        return;
    };
    if let Some(waiting) = queue.0.get_mut(&batched.vendor) {
        waiting.retain(|&call| call != settled.entity);
    }
    if let (true, Some(agent)) = (turn_call, agent)
        && let Ok(mut wait) = agents.get_mut(agent.0)
    {
        wait.0 = None;
    }
    let stopped = outcome
        .as_ref()
        .is_err_and(|report| report.kind == ErrorKind::Cancelled);
    if let (true, Some(batch), Some(api)) = (
        stopped,
        batched.batch.clone(),
        endpoints.0.get(&batched.vendor),
    ) {
        let cancel = api.cancel(batch, batched.item.custom_id.clone());
        task_pool()
            .spawn(async move {
                cancel.await.ok();
            })
            .detach();
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
    const BATCH: StatusItem = StatusItem::at(Side::Left, 52, 12);

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
