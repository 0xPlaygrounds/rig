//! The agent's systems: one per named set of [`RigSet`], in the bus
//! module's `RigSchedule`, run to quiescence by its runner.
//!
//! | set | true before | written during |
//! |---|---|---|
//! | `Advance` (first) | a `Ready` run has no phase | `open_runs`: the `Prompt` becomes the last utterance; the run is `Assembling`, or `LoadingMemory` with its `Load` effect |
//! | `Advance` | a `Ready` run in `Assembling` has no fresh turn | a turn is spawned `ChildOf` the run with its adverts and attachments, or the run fails `MaxTurns` |
//! | `Select` | a run may lack a model of its own | the agent's `UsesModel` is copied to the run |
//! | `Assemble` | a fresh turn's graph is complete | the fold spawns the turn's effect; the run is `AwaitingModel` |
//! | `Patch` | the folded effect is a `PendingEffect` | a user system may rewrite it (the second steering slot) |
//! | `Release` | a turn's tool batch is out | `release_batch` un-holds the next calls up to `ToolPolicy.concurrency` |
//! | *the bus's `Gate`, `Dispatch`, `Collect`, `Judge`* | | |
//! | `Fold` | the effect may have streamed or landed | `Outputs` on the turn, per tick |
//! | `Judge` | the turn's outputs are complete | a user system may rewrite them, or an `EffectOutcome` of a tool child |
//! | `Materialise` | a complete turn is unread, or its batch has landed | `land_batch`: the results as one user utterance, or a failure; `materialise`: the assistant utterance, the answer, a reprompt, an invalid call, the tool batch, a provider retry (`ProviderRetried`, `ProviderRetrying`, `Assembling`), or a failure |
//! | `Settle` | a run settled or failed this pass | nothing yet (observers fire on `Settled`/`Failed`) |
//!
//! The first steering slot is any system before `Assemble`: it edits the
//! graph (utterances, documents, grants, settings).

use crate::agent::checkpoint::{
    ToolTurnCommit, ToolTurnCommitted, ToolTurnHolds, TurnAssistant, TurnResults,
};
use crate::agent::content::{
    binary::BinaryAssets,
    cache::{AssemblyStats, Cached, CachedMessage},
    parts::{
        ContentError, ContentGraph, ToolResultLimit, ToolResultStatus, replace_deferred,
        spawn_deferred, spawn_deferred_with, write_message,
    },
};

use crate::agent::content::parts::{EditTarget, RequestPartEdit};
use bevy_ecs::{prelude::*, query::QueryData, system::SystemParam};
use rig_core::{
    completion::message::{
        AssistantContent, ToolChoice, ToolResultContent, UserContent, canonical_streamed_choice,
        turn_delivered_no_answer,
    },
    effect::{EffectKind, FamilyDescriptor, Outcome},
    error::ErrorKind,
};

use crate::{
    agent::{
        AdditionalParams, Advert, Assembling, Attachment, AwaitingModel, Batch, Cancelled,
        Completion, Context, Conversation, Cursor, DEFAULT_PROVIDER_RETRIES, DocumentId,
        DocumentProps, DocumentText, Failed, Failure, Grant, InvalidCall, InvalidCalls,
        InvalidRetries, LoadingMemory, MaxTokens, MaxTurns, MemoryAppendScheduled, MessageParts,
        Order, OrderCounter, Output, OutputKind, OutputRetries, OutputToolConfig, OutputToolName,
        Outputs, Preamble, Prompt, ProviderRetried, ProviderRetries, ProviderRetrying, Remembered,
        Remembering, Remembers, Reprompt, RequestPatch, Resolution, ResolvingTools, Retrievable,
        Retrieval, RetrievalKind, Retrieves, Retrieving, Retry, Run, RunCounter, RunOf, RunResult,
        RunSeq, Settled, StreamRequested, Temperature, ToolAccess, ToolCallSlot, ToolChoiceSpec,
        ToolContextSpec, ToolPolicy, Turn, Unhandled, Usage, UsesModel, Utterance,
    },
    bus::{
        Bound, BusSet, EffectOutcome, Issued, PendingEffect, Progress, RigSchedule, Scope,
        Streamed as BusStreamed, ToolInputs,
    },
    policy::{self, RequestGraph},
};

mod run_config;
pub use run_config::RunConfig;
mod stream_invalid;
pub mod witness;
pub use stream_invalid::discover_streamed_invalid_calls;

/// The agent's sets, in order, around the bus module's.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RigSet {
    /// A run that wants a turn gets one, or fails its budget.
    Advance,
    /// A run without a model of its own takes the agent's.
    Select,
    /// The fold: the turn's graph becomes the turn's effect.
    Assemble,
    /// The second steering slot: the folded effect, before the bus.
    Patch,
    /// A turn's tool batch is released up to its concurrency.
    Release,
    /// The turn's outputs from the effect's stream or answer.
    Fold,
    /// The agent's judge: the turn's outputs, before they are read.
    Judge,
    /// The turn is read into the graph.
    Materialise,
    /// Committed graph writes are visible; hosts may inspect or save before advancement.
    Checkpoint,
    /// A run settled or failed.
    Settle,
}

/// A run that wants a turn: `Ready`, `Assembling`, not failed.
pub type Wanting = (With<Assembling>, With<crate::agent::Ready>, Without<Failed>);
/// A `Ready` run that has neither a phase nor an ending: `open_runs` opens it.
pub type Unopened = (
    With<Run>,
    With<crate::agent::Ready>,
    Without<Failed>,
    Without<Settled>,
    Without<Assembling>,
    Without<LoadingMemory>,
    Without<AwaitingModel>,
    Without<ResolvingTools>,
);
/// A run with no model of its own yet.
pub type Unselected = (With<Run>, Without<UsesModel>);
/// What `Fold` reads of a completed model effect.
type EffectView = (
    &'static ChildOf,
    Option<&'static BusStreamed>,
    &'static EffectOutcome,
);
/// An invalid call nothing resolved.
pub type Unresolved = (With<InvalidCall>, Without<Resolution>);
/// A turn `Materialise` has not read.
pub type Unread = (With<Turn>, Without<Materialised>);
#[derive(QueryData)]
struct AssemblingView {
    agent: &'static RunOf,
    seq: &'static RunSeq,
    stream: &'static StreamRequested,
    model: Option<&'static UsesModel>,
    minted: &'static OutputToolName,
}
/// The request settings `assemble` resolves, the run's over the agent's.
#[derive(bevy_ecs::system::SystemParam)]
struct Settings<'w, 's> {
    /// The preamble.
    pub preambles: Query<'w, 's, &'static Preamble>,
    /// The temperature.
    pub temperatures: Query<'w, 's, &'static Temperature>,
    /// The token budget.
    pub max_tokens: Query<'w, 's, &'static MaxTokens>,
    /// The provider's extra parameters.
    pub params: Query<'w, 's, &'static AdditionalParams>,
    /// The tool choice.
    pub choices: Query<'w, 's, &'static ToolChoiceSpec>,
    /// The output mode.
    pub outputs: Query<'w, 's, &'static Output>,
    /// The output tool's reserved name, description and preamble behavior.
    pub output_tools: Query<'w, 's, &'static OutputToolConfig>,
    /// Execution bindings and permissions, separate from advertisements.
    pub tool_access: Query<'w, 's, &'static ToolAccess>,
    /// The request-time size policy for tool-result text.
    pub tool_result_limits: Query<'w, 's, &'static ToolResultLimit>,
}
#[derive(QueryData)]
struct FreshView {
    entity: Entity,
    parent: &'static ChildOf,
    patch: Option<&'static RequestPatch>,
    retrieving: Has<Retrieving>,
}
/// A fresh turn whose retrievals are out.
pub type RetrievingTurn = (With<Fresh>, With<Retrieving>);
/// A remembering run whose persisted finalization has not scheduled an append.
pub type NeedsMemoryAppend = (
    With<Settled>,
    With<Remembering>,
    Without<MemoryAppendScheduled>,
);
type CompletionEffect = (With<PendingEffect>, With<Completion>);
/// What `materialise` reads besides the graph: the tool choice and access
/// settings, retry budgets, and witness. Separate from graph mutation access.
#[derive(bevy_ecs::system::SystemParam)]
struct MaterialiseReads<'w, 's> {
    /// The tool choice, the run's over the agent's.
    pub choices: Query<'w, 's, &'static ToolChoiceSpec>,
    /// The tool access spec.
    pub access: Query<'w, 's, &'static ToolAccess>,
    /// The provider-retry budget, the run's over the agent's.
    pub provider_retries: Query<'w, 's, &'static ProviderRetries>,
    /// Subjects for the witness.
    pub subjects: crate::bus::Subjects<'w, 's>,
    outputs: Query<'w, 's, &'static Output>,
    max_turns: Query<'w, 's, &'static MaxTurns>,
    policies: Query<'w, 's, &'static InvalidCalls>,
    tool_policies: Query<'w, 's, &'static ToolPolicy>,
    contexts: Query<'w, 's, &'static ToolContextSpec>,
    /// The witness, if the world has one.
    pub witness: Option<Res<'w, crate::bus::Witnessing>>,
}

#[derive(QueryData)]
struct AwaitingView {
    agent: &'static RunOf,
    cursor: &'static Cursor,
    retries: &'static OutputRetries,
    invalid_retries: &'static InvalidRetries,
    minted: &'static OutputToolName,
    usage: &'static Usage,
    provider_retried: &'static ProviderRetried,
    seq: &'static RunSeq,
}

#[derive(QueryData)]
#[query_data(mutable)]
struct MaterialiseTurn {
    entity: Entity,
    parent: &'static ChildOf,
    outputs: &'static mut Outputs,
    mode: &'static Folded,
    retry: Option<&'static Retry>,
}

/// What the cancel observer reads of a run: awaiting its model, resolving
/// its tools, already ended.
pub type RunPhase = (Has<AwaitingModel>, Has<ResolvingTools>, Has<Failed>);
/// What the cancel observer reads of a turn: its run, whether it was
/// read, whether its batch is out.
pub type TurnState = (&'static ChildOf, Has<Materialised>, Has<Batch>);

/// A fresh turn: spawned by `Advance`, not yet folded by `Assemble`.
#[derive(Component, Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Fresh;

/// The output mode the turn was folded under, pinned.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Folded(pub OutputKind);

/// A turn `Materialise` has read.
#[derive(Component, Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Materialised;

#[derive(Resource)]
struct AgentInstalled;

/// Install the agent runtime once. The bus must be installed first; repeated
/// calls preserve its schedule, policy, counters, and the existing observers.
pub fn install_agent(world: &mut World) {
    if world.contains_resource::<AgentInstalled>() {
        return;
    }
    assert!(
        world.contains_resource::<crate::bus::Policy>(),
        "install_agent needs the bus installed first: it runs in the bus's RigSchedule"
    );
    assert!(
        world.resource::<Schedules>().contains(RigSchedule),
        "install_agent needs RigSchedule installed"
    );
    world.insert_resource(AgentInstalled);
    world.init_resource::<OrderCounter>();
    world.init_resource::<BinaryAssets>();
    world.init_resource::<RunCounter>();
    world.init_resource::<AssemblyStats>();
    world.add_observer(effect_cancelled);
    world.add_observer(run_cancelled);
    world.add_observer(batch_marker_follows_the_hold);
    witness::install(world);
    let mut schedules = world.resource_mut::<Schedules>();
    #[expect(
        clippy::expect_used,
        reason = "removing the required schedule during observer installation must fail loudly"
    )]
    let schedule = schedules
        .get_mut(RigSchedule)
        .expect("RigSchedule must remain installed while adding agent observers");
    schedule.configure_sets(
        (
            RigSet::Advance,
            RigSet::Select,
            RigSet::Assemble,
            RigSet::Patch,
            RigSet::Release,
        )
            .chain()
            .before(BusSet::Gate),
    );
    schedule.configure_sets(
        (
            RigSet::Fold,
            RigSet::Judge,
            RigSet::Materialise,
            RigSet::Checkpoint,
            RigSet::Settle,
        )
            .chain()
            .after(BusSet::Judge),
    );
    schedule.add_systems((
        (open_runs, advance).chain().in_set(RigSet::Advance),
        attach_retrieved
            .after(RigSet::Advance)
            .before(RigSet::Select),
        select.in_set(RigSet::Select),
        assemble.in_set(RigSet::Assemble),
        release_batch.in_set(RigSet::Release),
        (
            (fold_streamed, fold).chain_ignore_deferred(),
            discover_streamed_invalid_calls,
        )
            .chain()
            .in_set(RigSet::Fold),
        (
            land_memory,
            resolve_invalid_defaults,
            land_batch,
            materialise,
        )
            .chain()
            .in_set(RigSet::Materialise),
        append_memory.in_set(RigSet::Settle),
    ));
}

#[derive(SystemParam)]
struct GraphWrites<'w, 's> {
    commands: Commands<'w, 's>,
    assets: ResMut<'w, BinaryAssets>,
    orders: ResMut<'w, OrderCounter>,
    progress: ResMut<'w, Progress>,
}

impl GraphWrites<'_, '_> {
    fn say(&mut self, run: Entity, parts: MessageParts) -> Result<Entity, ContentError> {
        spawn_deferred(
            &mut self.commands,
            &mut self.assets,
            run,
            parts,
            next_order_in(&mut self.orders),
        )
    }

    fn results(
        &mut self,
        run: Entity,
        parts: MessageParts,
        statuses: Vec<ToolResultStatus>,
    ) -> Result<Entity, ContentError> {
        spawn_deferred_with(
            &mut self.commands,
            &mut self.assets,
            run,
            parts,
            next_order_in(&mut self.orders),
            statuses,
        )
    }

    fn replace(&mut self, entity: Entity, parts: MessageParts) -> Result<(), ContentError> {
        replace_deferred(&mut self.commands, &mut self.assets, entity, parts)
    }

    fn finish(&mut self, run: Entity, result: Result<(), ContentError>) {
        if let Err(error) = result {
            fail_content(&mut self.commands, run, error);
            self.progress.mark();
        }
    }
}

type Phases = (
    Assembling,
    AwaitingModel,
    LoadingMemory,
    ResolvingTools,
    Settled,
    Failed,
);

// Removal observers see the outgoing state, insertion observers see only the new
// phase. Re-fetch after removal: callbacks may despawn or cancel the run. Public
// component writes remain public; this is the supported runtime transition path.
fn transition_run(world: &mut World, run: Entity, next: impl Bundle) {
    let Some(entity) = world.get_entity(run).ok() else {
        return;
    };
    if entity.contains::<Failed>() {
        return;
    }
    let cancelled = entity.get::<Cancelled>().map(|reason| reason.0.clone());
    world.entity_mut(run).remove::<Phases>();
    let Ok(mut entity) = world.get_entity_mut(run) else {
        return;
    };
    if entity.contains::<Failed>() {
        return;
    }
    if let Some(reason) = entity
        .get::<Cancelled>()
        .map(|reason| reason.0.clone())
        .or(cancelled)
    {
        entity.insert(Failed(Failure::Cancelled(
            rig_core::error::ErrorReport::new(ErrorKind::Cancelled, reason),
        )));
    } else {
        entity.insert(next);
    }
}

trait RunTransition {
    fn transition(&mut self, next: impl Bundle) -> &mut Self;
}

impl RunTransition for bevy_ecs::system::EntityCommands<'_> {
    fn transition(&mut self, next: impl Bundle) -> &mut Self {
        let run = self.id();
        self.commands()
            .queue(move |world: &mut World| transition_run(world, run, next));
        self
    }
}

fn fail_content(commands: &mut Commands, run: Entity, error: ContentError) {
    commands
        .entity(run)
        .transition(Failed(Failure::Content(error)));
}

/// Why a despawn left a run in the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunBusy {
    /// The entity is not a run.
    NotARun,
    /// The run has not ended: it has no [`Settled`] and no [`Failed`].
    Unsettled,
    /// An effect of the run is still in flight or waiting to dispatch.
    /// Cancel the run and let it drain first.
    InFlight,
}

/// A queued `despawn_run` ([`RunCommands`] on `Commands`) refused the run:
/// triggered on the run entity when the command applies, with the reason
/// the exclusive form would have returned. The run is left as it was.
#[derive(EntityEvent, Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunDespawnRefused {
    /// The run that stays.
    pub entity: Entity,
    /// Why.
    pub reason: RunBusy,
}

/// The components every run is made of: what [`RunCommands::spawn_run`]
/// spawns, for a host that assembles a run by hand — `world.spawn((RunBundle::new(world, agent, false), Prompt::from("…")))`,
/// its history utterances `ChildOf` the run in `Order`, an optional
/// [`MaxTurns`], and last [`Ready`](crate::agent::Ready). [`RunSeq`] is the world's next run
/// number and [`Scope`] is `{owner}/run#{seq}`, both taken from the world
/// by [`RunBundle::new`].
#[derive(Bundle, Debug, Clone)]
pub struct RunBundle {
    /// The run marker.
    pub run: Run,
    /// The run's agent.
    pub run_of: RunOf,
    /// The run's number in the world.
    pub seq: RunSeq,
    /// Whether the model is asked for a stream.
    pub streamed: StreamRequested,
    /// The witness scope: `{owner}/run#{seq}`.
    pub scope: Scope,
}

impl RunBundle {
    /// A fresh run of `agent`: takes the next [`RunSeq`] from the world's
    /// [`RunCounter`] and the agent's `Owner` for the [`Scope`].
    pub fn new(world: &mut World, agent: Entity, streamed: bool) -> Self {
        let seq = {
            let mut counter = world.resource_mut::<RunCounter>();
            let seq = counter.0;
            counter.0 += 1;
            seq
        };
        let owner = world
            .get::<crate::agent::Owner>(agent)
            .map(|owner| owner.0.clone())
            .unwrap_or_default();
        Self {
            run: Run,
            run_of: RunOf(agent),
            seq: RunSeq(seq),
            streamed: StreamRequested(streamed),
            scope: Scope(format!("{owner}/run#{seq}")),
        }
    }
}

/// The run entry points, on `Commands` and on `World`. The `Commands`
/// form queues the work and reserves the entity: the run exists — its
/// bundle, its utterances, its `Ready` — once the commands apply, and
/// `Advance` sees it on the first schedule pass after that; a run spawned
/// in a system is visible to the runtime only after that system's
/// commands are flushed. The `World` form does the same work at once.
/// Commands queued in order apply in order: a `cancel_run` or
/// `despawn_run` queued after a `spawn_run` finds the run.
pub trait RunCommands {
    /// What `despawn_run` reports: the refusal, on `World`; nothing on
    /// `Commands`, which triggers [`RunDespawnRefused`] on the run instead.
    type Despawned;

    /// Spawn a run of `agent` with `prompt` as its first utterance, after
    /// `config.history`: the host's one entry point. The prompt is a user
    /// message's parts (`&str` text, or text and images kept in their
    /// given order, [`Prompt`]). Returns the run entity. On `Commands`
    /// the entity is reserved at once and populated when the command
    /// applies; a run despawned before or while it is populated (a host
    /// `Add<Run>` observer that refuses it, say) is simply gone.
    fn spawn_run(
        &mut self,
        agent: Entity,
        prompt: impl Into<Prompt>,
        config: RunConfig<'_>,
    ) -> Entity;

    /// Stop `run` with `reason` (CONTRACT §9.1): `Cancelled(reason)` on the
    /// run. A run that ended keeps its ending; an entity that is not a run
    /// is left alone. A run cancelled before it opened (`Ready` written by
    /// hand, not yet seen by `Advance`) fails with its unread [`Prompt`]
    /// still on it, and a scene saves the prompt with the failed run.
    fn cancel_run(&mut self, run: Entity, reason: impl Into<String>);

    /// Despawn an ended run and everything that is its: turns, utterances,
    /// adverts, attachments and the settled effects under them (`ChildOf`
    /// is linked, so the despawn is deep), a `Streamed` fold included.
    /// The world keeps nothing of a run by itself — a host that runs for
    /// long must despawn the runs it is done reading, or their graphs and
    /// folds accumulate for the life of the world. Refused while the run
    /// has not ended or an effect of it is still in flight; nothing is
    /// despawned then.
    fn despawn_run(&mut self, run: Entity) -> Self::Despawned;
}

impl RunCommands for World {
    type Despawned = Result<(), RunBusy>;

    fn spawn_run(
        &mut self,
        agent: Entity,
        prompt: impl Into<Prompt>,
        config: RunConfig<'_>,
    ) -> Entity {
        let run = self.spawn_empty().id();
        spawn_run_at(
            self,
            run,
            agent,
            config.history,
            prompt.into(),
            config.streamed,
            config.max_turns,
        );
        run
    }

    fn cancel_run(&mut self, run: Entity, reason: impl Into<String>) {
        cancel_run_in(self, run, reason.into());
    }

    fn despawn_run(&mut self, run: Entity) -> Result<(), RunBusy> {
        despawn_run_in(self, run)
    }
}

impl RunCommands for Commands<'_, '_> {
    type Despawned = ();

    fn spawn_run(
        &mut self,
        agent: Entity,
        prompt: impl Into<Prompt>,
        config: RunConfig<'_>,
    ) -> Entity {
        let run = self.spawn_empty().id();
        let history = config.history.to_vec();
        let prompt = prompt.into();
        let streamed = config.streamed;
        let max_turns = config.max_turns;
        self.queue(move |world: &mut World| {
            spawn_run_at(world, run, agent, &history, prompt, streamed, max_turns);
        });
        run
    }

    fn cancel_run(&mut self, run: Entity, reason: impl Into<String>) {
        let reason = reason.into();
        self.queue(move |world: &mut World| cancel_run_in(world, run, reason));
    }

    fn despawn_run(&mut self, run: Entity) {
        self.queue(move |world: &mut World| {
            if let Err(reason) = despawn_run_in(world, run) {
                world.trigger(RunDespawnRefused {
                    entity: run,
                    reason,
                });
            }
        });
    }
}

/// The run on the reserved `run` entity: the bundle, the prompt, the
/// history utterances, `Ready`, and the opening at once, so the world form
/// hands back a run whose utterances are in the graph. Every step looks
/// the run up afresh: a run despawned before it was populated, or by a
/// host observer while it was (an `Add<Run>` observer that refuses it,
/// say), is simply gone — the steps after the despawn do nothing, and no
/// utterance is spawned under a run that is not there.
fn spawn_run_at(
    world: &mut World,
    run: Entity,
    agent: Entity,
    history: &[MessageParts],
    prompt: Prompt,
    streamed: bool,
    max_turns: Option<usize>,
) {
    if world.get_entity(run).is_err() {
        return;
    }
    let bundle = RunBundle::new(world, agent, streamed);
    let Ok(mut entity) = world.get_entity_mut(run) else {
        return;
    };
    entity.insert((bundle, prompt));
    if let Some(limit) = max_turns {
        let Ok(mut entity) = world.get_entity_mut(run) else {
            return;
        };
        entity.insert(MaxTurns(limit));
    }
    for parts in history.iter().cloned() {
        if world.get_entity(run).is_err() {
            return;
        }
        if let Err(error) = spawn_utterance(world, run, parts) {
            if let Ok(mut entity) = world.get_entity_mut(run) {
                entity.remove::<Prompt>();
            }
            transition_run(world, run, Failed(Failure::Content(error)));
            return;
        }
    }
    let Ok(mut entity) = world.get_entity_mut(run) else {
        return;
    };
    entity.insert(crate::agent::Ready);
    open_run(world, run);
}

fn cancel_run_in(world: &mut World, run: Entity, reason: String) {
    if let Ok(mut entity) = world.get_entity_mut(run)
        && entity.contains::<Run>()
    {
        entity.insert(Cancelled(reason));
    }
}

fn despawn_run_in(world: &mut World, run: Entity) -> Result<(), RunBusy> {
    if world.get::<Run>(run).is_none() {
        return Err(RunBusy::NotARun);
    }
    if world.get::<Settled>(run).is_none() && world.get::<Failed>(run).is_none() {
        return Err(RunBusy::Unsettled);
    }
    let mut stack = vec![run];
    while let Some(entity) = stack.pop() {
        if world.get::<crate::bus::InFlight>(entity).is_some()
            || (world.get::<PendingEffect>(entity).is_some()
                && world.get::<EffectOutcome>(entity).is_none())
        {
            return Err(RunBusy::InFlight);
        }
        if let Some(children) = world.get::<Children>(entity) {
            stack.extend(children.iter());
        }
    }
    world.entity_mut(run).despawn();
    Ok(())
}

/// First in `RigSet::Advance`: every [`Ready`](crate::agent::Ready) run without a phase and
/// without an ending is opened — its [`Prompt`] spawned as its last
/// utterance and taken off, then its first phase: `Assembling`, or, for
/// an agent that `Remembers` given no history, `LoadingMemory` with the
/// conversation's `Load` effect (CONTRACT §11). A run the host populates
/// by hand starts here, the pass after it writes `Ready`; a run
/// `spawn_run` made was opened at once. A `Prompt` a hand-made run gives
/// before `Ready` is never read before the run opens.
pub fn open_runs(mut commands: Commands, runs: Query<Entity, Unopened>) {
    // Queued: the opening writes the graph, and the chain's sync point
    // applies it before `advance` reads.
    for run in &runs {
        commands.queue(move |world: &mut World| open_run(world, run));
    }
}

/// Open `run` (see [`open_runs`]): a no-op unless the run is `Ready`,
/// phaseless and not ended.
fn open_run(world: &mut World, run: Entity) {
    let Some(entity) = world.get_entity(run).ok() else {
        return;
    };
    if !entity.contains::<Run>()
        || !entity.contains::<crate::agent::Ready>()
        || entity.contains::<Failed>()
        || entity.contains::<Settled>()
        || entity.contains::<Assembling>()
        || entity.contains::<LoadingMemory>()
        || entity.contains::<AwaitingModel>()
        || entity.contains::<ResolvingTools>()
    {
        return;
    }
    let agent = entity.get::<RunOf>().map(|run_of| run_of.0);
    let had_history = entity.get::<Children>().is_some_and(|children| {
        children
            .iter()
            .any(|child| world.get::<Utterance>(child).is_some())
    });
    if let Some(Prompt(content)) = world.entity_mut(run).take::<Prompt>()
        && let Err(error) = spawn_utterance(world, run, MessageParts::User { content })
    {
        transition_run(world, run, Failed(Failure::Content(error)));
        world.resource_mut::<Progress>().mark();
        return;
    }
    // An agent that remembers, and a run given no history: the conversation
    // is loaded before the first turn (CONTRACT §11).
    let memory = (!had_history)
        .then(|| {
            let agent = agent?;
            let handler = world.get::<Remembers>(agent).map(|remembers| remembers.0)?;
            let conversation = world
                .get::<Conversation>(agent)
                .map(|conversation| conversation.0.clone())?;
            let key = world.get::<Bound>(handler).map(|bound| bound.key.clone())?;
            Some((key, conversation))
        })
        .flatten();
    match memory {
        Some((key, conversation)) => {
            transition_run(
                world,
                run,
                (
                    LoadingMemory,
                    Remembering,
                    Conversation(conversation.clone()),
                ),
            );
            if world.get::<LoadingMemory>(run).is_none() {
                world.resource_mut::<Progress>().mark();
                return;
            }
            world.spawn((
                PendingEffect::new(
                    key,
                    EffectKind::Memory {
                        op: rig_core::effect::MemoryOp::Load {
                            conversation: rig_core::id::ConversationId::from(conversation.as_str()),
                        },
                    },
                ),
                ChildOf(run),
            ));
        }
        None => {
            transition_run(world, run, Assembling);
        }
    }
    world.resource_mut::<Progress>().mark();
}

/// Spawn one utterance `ChildOf` `run`, next in order.
pub fn spawn_utterance(
    world: &mut World,
    run: Entity,
    parts: MessageParts,
) -> Result<Entity, ContentError> {
    let order = next_order(world);
    let entity = world.spawn((Utterance, order, ChildOf(run))).id();
    if let Err(error) = write_message(world, entity, parts) {
        world.despawn(entity);
        return Err(error);
    }
    Ok(entity)
}

/// The next [`Order`].
pub(crate) fn next_order(world: &mut World) -> Order {
    let mut counter = world.resource_mut::<OrderCounter>();
    let order = Order(counter.0);
    counter.0 += 1;
    order
}

pub(crate) fn next_order_in(counter: &mut OrderCounter) -> Order {
    let order = Order(counter.0);
    counter.0 += 1;
    order
}

/// A run's effective setting: its own component, else its agent's.
fn setting<'a, C: Component>(run: Entity, agent: Entity, query: &'a Query<&C>) -> Option<&'a C> {
    query.get(run).ok().or_else(|| query.get(agent).ok())
}

/// The links of one kind under `owner`, in order.
fn links_in_order<'a, L: Component, F: bevy_ecs::query::QueryFilter>(
    owner: Entity,
    children: &Query<&Children>,
    links: &'a Query<(&L, &Order), F>,
) -> impl Iterator<Item = &'a L> {
    let mut found: Vec<(&Order, &L)> = children
        .get(owner)
        .map(|children| {
            links
                .iter_many(children.iter())
                .map(|(link, order)| (order, link))
                .collect()
        })
        .unwrap_or_default();
    found.sort_by_key(|(order, _)| **order);
    found.into_iter().map(|(_, link)| link)
}

/// `RigSet::Advance`: a `Ready` run in `Assembling` with no fresh turn gets one —
/// `ChildOf` the run, with an advert per grant and an attachment per
/// context link, in the agent's order (an agent with `Retrieves` links
/// gets a `Retrieving` turn instead: the adverts and attachments come with
/// the results) — or, at its budget, fails `MaxTurns`.
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn advance(
    mut commands: Commands,
    runs: Query<(Entity, &RunOf, &Cursor, &RunSeq), Wanting>,
    fresh: Query<&ChildOf, With<Fresh>>,
    children: Query<&Children>,
    grants: Query<(&Grant, &Order), Without<Retrievable>>,
    contexts: Query<(&Context, &Order)>,
    retrievals: Query<(), With<Retrieves>>,
    max_turns: Query<&MaxTurns>,
    retrying: Query<(), With<ProviderRetrying>>,
    holds: Query<&ToolTurnHolds>,
    commits: Query<(&ChildOf, &ToolTurnCommit)>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    let mut runs: Vec<_> = runs.iter().collect();
    runs.sort_by_key(|(_, _, _, seq)| **seq);
    for (run, RunOf(agent), cursor, _) in runs {
        if children
            .get(run)
            .is_ok_and(|owned| fresh.iter_many(owned.iter()).next().is_some())
        {
            continue;
        }
        // A retried attempt re-issues a turn the cursor already counted
        // (CONTRACT §5): it neither checks nor spends the model-call budget.
        let retrying = retrying.get(run).is_ok();
        let limit = setting(run, *agent, &max_turns).map_or(1, |limit| limit.0);
        if !retrying && cursor.turn >= limit {
            commands
                .entity(run)
                .transition(Failed(Failure::MaxTurns { limit }));
            progress.mark();
            continue;
        }
        if holds.get(run).is_ok_and(|holds| {
            children.get(run).is_ok_and(|owned| {
                commits
                    .iter_many(owned.iter())
                    .any(|(_, commit)| holds.blocks(commit.turn))
            })
        }) {
            continue;
        }
        let turn = commands
            .spawn((Turn, Fresh, next_order_in(&mut orders), ChildOf(run)))
            .id();
        let retrieves = children
            .get(*agent)
            .map(|children| retrievals.iter_many(children.iter()).next().is_some())
            .unwrap_or(false);
        if retrieves {
            // Retrieval first (CONTRACT §12): `assemble` spawns the effects
            // on its first pass over the turn; the adverts and attachments
            // come with the results (`attach_retrieved`).
            commands.entity(turn).insert(Retrieving);
        } else {
            for Grant(tool) in links_in_order(*agent, &children, &grants) {
                commands.spawn((Advert(*tool), next_order_in(&mut orders), ChildOf(turn)));
            }
            for Context(document) in links_in_order(*agent, &children, &contexts) {
                commands.spawn((
                    Attachment(*document),
                    next_order_in(&mut orders),
                    ChildOf(turn),
                ));
            }
        }
        if retrying {
            commands.entity(run).remove::<ProviderRetrying>();
        } else {
            commands.entity(run).insert(Cursor {
                turn: cursor.turn + 1,
            });
        }
        progress.mark();
    }
}

/// A fresh turn whose retrievals landed gets its adverts and attachments
/// (CONTRACT §12): the retrieved tools first, in result order, then the
/// static grants; the static attachments, then one document entity per
/// result (an existing entity with that id reused). Runs after `Advance`
/// and before `Select`; `assemble` waits for it.
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn attach_retrieved(
    mut commands: Commands,
    turns: Query<(Entity, &ChildOf), RetrievingTurn>,
    runs: Query<&RunOf>,
    children: Query<&Children>,
    retrievals: Query<(&PendingEffect, &Retrieval, Option<&EffectOutcome>)>,
    grants: Query<(&Grant, &Order, Has<Retrievable>)>,
    contexts: Query<(&Context, &Order)>,
    bound: Query<&Bound>,
    indexes: Query<&Retrieves, (With<Retrieval>, With<Order>)>,
    documents: Query<(Entity, &DocumentId)>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    if turns.is_empty() {
        return;
    }
    // Preserve the first committed query match; reservations are shared by all
    // turns. Borrow committed IDs instead of copying the world's document names.
    let mut documents_by_id = std::collections::HashMap::new();
    for (entity, id) in &documents {
        documents_by_id
            .entry(std::borrow::Cow::Borrowed(id.0.as_str()))
            .or_insert(entity);
    }
    for (turn, turn_of) in &turns {
        let run = turn_of.parent();
        let Ok(RunOf(agent)) = runs.get(run) else {
            continue;
        };
        let effects: Vec<(&Retrieval, Option<&EffectOutcome>)> = children
            .get(turn)
            .map(|children| {
                retrievals
                    .iter_many(children.iter())
                    .map(|(_, retrieval, outcome)| (retrieval, outcome))
                    .collect()
            })
            .unwrap_or_default();
        // Wait for an available index's effects to be spawned later this
        // tick. With no available index, still attach the static links here.
        let available_index = children.get(*agent).is_ok_and(|children| {
            children.iter().any(|child| {
                indexes
                    .get(child)
                    .is_ok_and(|Retrieves(index)| bound.get(*index).is_ok())
            })
        });
        if (effects.is_empty() && available_index)
            || effects.iter().any(|(_, outcome)| outcome.is_none())
        {
            continue;
        }
        let mut retrieved_tools: Vec<String> = Vec::new();
        let mut retrieved_documents: Vec<(String, String)> = Vec::new();
        for (retrieval, outcome) in &effects {
            let Some(EffectOutcome(Ok(Outcome::Documents(results)))) = outcome else {
                continue;
            };
            match (retrieval.what, results) {
                (RetrievalKind::Tools, rig_core::effect::RetrievedDocuments::Ids(ids)) => {
                    retrieved_tools.extend(ids.iter().map(|(_, id)| id.clone()));
                }
                (
                    RetrievalKind::Documents,
                    rig_core::effect::RetrievedDocuments::Scored(scored),
                ) => {
                    retrieved_documents.extend(scored.iter().map(|(_, id, value)| {
                        (
                            id.clone(),
                            serde_json::to_string_pretty(value)
                                .unwrap_or_else(|_| value.to_string()),
                        )
                    }));
                }
                (RetrievalKind::Tools, rig_core::effect::RetrievedDocuments::Scored(_))
                | (RetrievalKind::Documents, rig_core::effect::RetrievedDocuments::Ids(_)) => {}
            }
        }
        let mut links: Vec<(&Grant, &Order, bool)> = children
            .get(*agent)
            .map(|children| grants.iter_many(children.iter()).collect())
            .unwrap_or_default();
        links.sort_by_key(|(_, order, _)| **order);
        let tool_named = |name: &str| -> Option<Entity> {
            links.iter().find_map(|(Grant(tool), _, _)| {
                bound
                    .get(*tool)
                    .ok()
                    .and_then(|bound| match &bound.descriptor.family {
                        FamilyDescriptor::Tool {
                            name: bound_name, ..
                        } if bound_name == name => Some(*tool),
                        FamilyDescriptor::Tool { .. }
                        | FamilyDescriptor::Completion { .. }
                        | FamilyDescriptor::Embed { .. }
                        | FamilyDescriptor::Rerank { .. }
                        | FamilyDescriptor::Memory { .. }
                        | FamilyDescriptor::Retrieve { .. }
                        | FamilyDescriptor::Custom { .. } => None,
                    })
            })
        };
        for name in &retrieved_tools {
            if let Some(tool) = tool_named(name) {
                commands.spawn((Advert(tool), next_order_in(&mut orders), ChildOf(turn)));
            }
        }
        for (Grant(tool), _, retrievable) in &links {
            if !retrievable {
                commands.spawn((Advert(*tool), next_order_in(&mut orders), ChildOf(turn)));
            }
        }
        for Context(document) in links_in_order(*agent, &children, &contexts) {
            commands.spawn((
                Attachment(*document),
                next_order_in(&mut orders),
                ChildOf(turn),
            ));
        }
        for (id, text) in retrieved_documents {
            let document = if let Some(entity) = documents_by_id.get(id.as_str()) {
                *entity
            } else {
                let entity = commands
                    .spawn((DocumentId(id.clone()), DocumentText(text)))
                    .id();
                documents_by_id.insert(std::borrow::Cow::Owned(id), entity);
                entity
            };
            commands.spawn((
                Attachment(document),
                next_order_in(&mut orders),
                ChildOf(turn),
            ));
        }
        commands.entity(turn).remove::<Retrieving>();
        progress.mark();
    }
}

/// `RigSet::Materialise`, first: a run whose memory load landed reads it —
/// the loaded messages become utterances before the prompt, each
/// `Remembered`, and the run is `Assembling`; a failed load fails the run
/// (CONTRACT §11).
#[allow(
    clippy::too_many_arguments,
    reason = "one system pass reads the graph and its memory effect state"
)]
pub fn land_memory(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    runs: Query<Entity, (With<LoadingMemory>, Without<Failed>)>,
    children: Query<&Children>,
    loads: Query<(&PendingEffect, &EffectOutcome)>,
    utterances: Query<(Entity, &Order), With<Utterance>>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    for run in &runs {
        let result = (|| -> Result<(), ContentError> {
            let Some(outcome) = children.get(run).ok().and_then(|children| {
                children.iter().find_map(|child| {
                    loads
                        .get(child)
                        .ok()
                        .and_then(|(effect, outcome)| match &effect.kind {
                            EffectKind::Memory {
                                op: rig_core::effect::MemoryOp::Load { .. },
                            } => Some(&outcome.0),
                            EffectKind::Memory { .. }
                            | EffectKind::Completion { .. }
                            | EffectKind::ToolCall { .. }
                            | EffectKind::Embed { .. }
                            | EffectKind::Rerank { .. }
                            | EffectKind::Retrieve { .. }
                            | EffectKind::Custom { .. } => None,
                        })
                })
            }) else {
                return Ok(());
            };
            match outcome {
                Ok(Outcome::Memory(rig_core::effect::MemoryOutcome::Loaded { messages })) => {
                    for message in messages {
                        if let Some(parts) = MessageParts::from_message(message) {
                            let utterance = spawn_deferred(
                                &mut commands,
                                &mut assets,
                                run,
                                parts,
                                next_order_in(&mut orders),
                            )?;
                            commands.entity(utterance).insert(Remembered);
                        }
                    }
                    // The prompt (and any history given) comes after what was
                    // loaded: its order is re-stamped past the loaded ones.
                    let mut existing: Vec<(Entity, Order)> = children
                        .get(run)
                        .map(|children| {
                            utterances
                                .iter_many(children.iter())
                                .map(|(entity, order)| (entity, *order))
                                .collect()
                        })
                        .unwrap_or_default();
                    existing.sort_by_key(|(_, order)| *order);
                    for (entity, _) in existing {
                        commands.entity(entity).insert(next_order_in(&mut orders));
                    }
                    commands.entity(run).transition(Assembling);
                }
                Ok(other) => {
                    commands.entity(run).transition(Failed(Failure::Memory(
                        rig_core::error::ErrorReport::new(
                            ErrorKind::Internal,
                            format!(
                                "the memory handler answered a load with a {} outcome",
                                other.family()
                            ),
                        ),
                    )));
                }
                Err(report) => {
                    commands
                        .entity(run)
                        .transition(Failed(Failure::Memory(report.clone())));
                }
            }
            progress.mark();

            Ok(())
        })();
        if let Err(error) = result {
            fail_content(&mut commands, run, error);
            progress.mark();
        }
    }
}

/// `RigSet::Settle`: a run that loaded its conversation appends what it
/// said — every utterance not `Remembered`, in order — when it settles
/// (CONTRACT §11). The persisted marker distinguishes new work from scene
/// rehydration; Bevy change-detection ticks are not durable transitions.
#[allow(
    clippy::too_many_arguments,
    reason = "one system pass reads the graph and its memory effect state"
)]
pub fn append_memory(
    mut commands: Commands,
    settled: Query<(Entity, &RunOf, &Conversation), NeedsMemoryAppend>,
    memories: Query<&Remembers>,
    bound: Query<&Bound>,
    children: Query<&Children>,
    utterances: Query<(Entity, &Order, Has<Remembered>), With<Utterance>>,
    content: ContentGraph,
    mut progress: ResMut<Progress>,
) {
    for (run, RunOf(agent), Conversation(conversation)) in &settled {
        let result = (|| -> Result<(), ContentError> {
            let Some(key) = memories
                .get(*agent)
                .ok()
                .and_then(|Remembers(memory)| bound.get(*memory).ok())
                .map(|bound| bound.key.clone())
            else {
                return Ok(());
            };
            let said: Result<Vec<_>, ContentError> = children
                .get(run)
                .map(|children| {
                    utterances
                        .iter_many(children.iter())
                        .filter(|(_, _, remembered)| !*remembered)
                        .map(|(entity, order, _)| {
                            content.message(entity).map(|message| (*order, message))
                        })
                        .collect()
                })
                .unwrap_or_else(|_| Ok(Vec::new()));
            let mut said = said?;
            said.sort_by_key(|(order, _)| *order);
            commands.spawn((
                PendingEffect::new(
                    key,
                    EffectKind::Memory {
                        op: rig_core::effect::MemoryOp::Append {
                            conversation: rig_core::id::ConversationId::from(conversation.as_str()),
                            messages: said
                                .into_iter()
                                .map(|(_, parts)| parts.to_message())
                                .collect(),
                        },
                    },
                ),
                ChildOf(run),
            ));
            commands.entity(run).insert(MemoryAppendScheduled);

            Ok(())
        })();
        if let Err(error) = result {
            fail_content(&mut commands, run, error);
            progress.mark();
        }
    }
}

/// `RigSet::Select`: a run without a model of its own takes its agent's.
/// A routing system before this one gives the run another.
pub fn select(
    mut commands: Commands,
    runs: Query<(Entity, &RunOf), Unselected>,
    models: Query<&UsesModel>,
) {
    for (run, RunOf(agent)) in &runs {
        if let Ok(UsesModel(model)) = models.get(*agent) {
            commands.entity(run).insert(UsesModel(*model));
        }
    }
}

/// Ordered request edit links, retaining malformed links for validation.
#[derive(QueryData)]
struct PartEditView {
    entity: Entity,
    order: Option<&'static Order>,
    target: Option<&'static EditTarget>,
    edit: &'static RequestPartEdit,
}

/// `RigSet::Assemble`: for every fresh turn, in run order, gather the
/// graph — the run's settings over the agent's, the utterances in order,
/// the attachments in order, the adverts in order, the model's descriptor
/// — resolve the output mode, mint the output tool's name once per run,
/// fold, and spawn the effect `ChildOf` the turn. The run is then
/// `AwaitingModel`.
/// An utterance is read from its `CachedMessage` when it holds one and
/// nothing of it changed since this system last ran (CONTRACT §1); else
/// it is rendered and the render cached for the next turn. An utterance a
/// `RequestPartEdit` targets is rendered with the edit, uncached.
/// A missing selected model or non-completion binding instead terminates the
/// run with a provider `HandlerUnavailable` report; it never silently waits.
#[derive(SystemParam)]
struct RequestHistory<'w, 's> {
    children: Query<'w, 's, &'static Children>,
    utterances: Query<'w, 's, (Entity, &'static Order), With<Utterance>>,
    content: ContentGraph<'w, 's>,
    part_edits: Query<'w, 's, PartEditView>,
    cached: Cached<'w, 's>,
}

#[derive(SystemParam)]
struct RequestBindings<'w, 's> {
    retrievals: Query<'w, 's, (&'static Retrieves, &'static Order, &'static Retrieval)>,
    retrieving: Query<'w, 's, (), With<Retrieval>>,
    adverts: Query<'w, 's, (&'static Advert, &'static Order)>,
    attachments: Query<'w, 's, (&'static Attachment, &'static Order)>,
    documents: Query<
        'w,
        's,
        (
            &'static DocumentId,
            &'static DocumentText,
            Option<&'static DocumentProps>,
        ),
    >,
    bound: Query<'w, 's, &'static Bound>,
}

#[derive(SystemParam)]
struct Assembly<'w, 's> {
    history: RequestHistory<'w, 's>,
    bindings: RequestBindings<'w, 's>,
    settings: Settings<'w, 's>,
}

fn assemble(
    mut commands: Commands,
    fresh: Query<FreshView, With<Fresh>>,
    runs: Query<AssemblingView, (With<Run>, Without<Failed>)>,
    mut assembly: Assembly,
    mut progress: ResMut<Progress>,
) {
    // The cache reader owns invalidation, even when there are no fresh turns.
    let stale = assembly.history.cached.cache.stale();
    for utterance in &stale {
        if assembly.history.cached.cache.holds(*utterance) {
            commands.entity(*utterance).remove::<CachedMessage>();
            assembly.history.cached.stats.evictions += 1;
        }
    }
    let mut turns: Vec<_> = fresh
        .iter()
        .filter_map(|turn| runs.get(turn.parent.parent()).ok().map(|run| (turn, run)))
        .collect();
    turns.sort_by_key(|(_, run)| *run.seq);
    // Do not split retrieval/completion spawning across runs: PendingEffect's
    // Add stamps Seq, so all decisions of the earlier RunSeq stay together.
    for (turn, run) in turns {
        let owner = turn.parent.parent();
        if let Err(error) = assembly.turn(&mut commands, &mut progress, turn, run, &stale) {
            fail_content(&mut commands, owner, error);
            progress.mark();
        }
    }
}

struct RequestEdits {
    edits: std::collections::BTreeMap<Entity, RequestPartEdit>,
    consumed: Vec<Entity>,
    edited: std::collections::HashSet<Entity>,
}

impl RequestHistory<'_, '_> {
    fn edits(
        &self,
        turn: Entity,
        run: Entity,
        patch: Option<&RequestPatch>,
    ) -> Result<RequestEdits, ContentError> {
        let Self {
            children,
            content,
            part_edits,
            ..
        } = self;
        let mut links: Vec<_> = part_edits
            .iter_many(
                children
                    .get(turn)
                    .into_iter()
                    .flat_map(|owned| owned.iter()),
            )
            .collect();
        if links
            .iter()
            .any(|link| link.order.is_none() || link.target.is_none())
        {
            return Err(ContentError::Missing);
        }
        links.sort_by_key(|link| link.order.copied());
        if links
            .windows(2)
            .any(|pair| pair.first().map(|link| link.order) == pair.get(1).map(|link| link.order))
        {
            return Err(ContentError::DuplicateOrder);
        }
        let mut edits = std::collections::BTreeMap::new();
        let mut consumed = Vec::new();
        let mut edited = std::collections::HashSet::new();
        for link in links {
            let target = link.target.ok_or(ContentError::Missing)?;
            let utterance = content.target_utterance(target.0)?;
            edited.insert(utterance);
            if !children
                .get(run)
                .is_ok_and(|owned| owned.contains(&utterance))
            {
                return Err(ContentError::Shape);
            }
            // A replacement history has no stable entity identity. Reject
            // conflicting operations instead of silently dropping an edit.
            if patch.is_some_and(|patch| patch.history.is_some()) {
                return Err(ContentError::Shape);
            }
            edits.insert(target.0, link.edit.clone());
            consumed.push(link.entity);
        }
        Ok(RequestEdits {
            edits,
            consumed,
            edited,
        })
    }
}

impl RequestBindings<'_, '_> {
    fn retrieve(
        &self,
        commands: &mut Commands,
        progress: &mut Progress,
        children: &Query<&Children>,
        fresh: &FreshViewItem<'_, '_>,
        agent: Entity,
        history: &[(Order, std::borrow::Cow<'_, MessageParts>)],
    ) {
        let Self {
            retrievals,
            retrieving,
            bound,
            ..
        } = self;
        let turn = fresh.entity;
        // The first pass over a retrieving turn (CONTRACT §12): one
        // `Retrieve` effect per index, in link order, `ChildOf` the
        // turn; the fold waits for `attach_retrieved`.
        let spawned = children
            .get(turn)
            .map(|children| retrieving.iter_many(children.iter()).next().is_some())
            .unwrap_or(false);
        if spawned {
            return;
        }
        let query = policy::retrieval_query(
            &history
                .iter()
                .map(|(_, parts)| parts.as_ref().clone())
                .collect::<Vec<_>>(),
        );
        let mut indexes: Vec<(&Retrieves, &Order, &Retrieval)> = children
            .get(agent)
            .map(|children| retrievals.iter_many(children.iter()).collect())
            .unwrap_or_default();
        indexes.sort_by_key(|(_, order, _)| **order);
        let mut spawned = 0usize;
        for (Retrieves(index), _, retrieval) in indexes {
            let Ok(index) = bound.get(*index) else {
                continue;
            };
            spawned += 1;
            let request = rig_core::vector_store::request::VectorSearchRequest::builder()
                .query(query.clone())
                .samples(retrieval.samples)
                .build()
                .map_filter(rig_core::vector_store::request::Filter::interpret);
            let query = match retrieval.what {
                RetrievalKind::Documents => rig_core::effect::RetrieveQuery::TopN { req: request },
                RetrievalKind::Tools => rig_core::effect::RetrieveQuery::TopNIds { req: request },
            };
            commands.spawn((
                PendingEffect::new(index.key.clone(), EffectKind::Retrieve { query }),
                *retrieval,
                ChildOf(turn),
            ));
        }
        // If every index disappeared since attachment ran, leave the
        // marker for the next attachment pass to restore static links.
        if spawned > 0 {
            progress.mark();
        }
    }
}
impl Assembly<'_, '_> {
    fn turn(
        &mut self,
        commands: &mut Commands,
        progress: &mut Progress,
        fresh: FreshViewItem<'_, '_>,
        run_view: AssemblingViewItem<'_, '_>,
        stale: &std::collections::HashSet<Entity>,
    ) -> Result<(), ContentError> {
        let Self {
            history,
            bindings,
            settings,
        } = self;
        let turn = fresh.entity;
        let run = fresh.parent.parent();
        let patch = fresh.patch;
        let RequestBindings {
            adverts,
            attachments,
            documents,
            bound,
            ..
        } = &*bindings;
        let Settings {
            preambles,
            temperatures,
            max_tokens,
            params,
            choices,
            outputs,
            output_tools,
            tool_access,
            tool_result_limits,
        } = settings;
        let is_retrieving = fresh.retrieving;
        let agent = run_view.agent.0;
        let stream = run_view.stream.0;
        let model = run_view.model;
        let minted = run_view.minted;
        let model_bound = model.and_then(|UsesModel(model)| bound.get(*model).ok());
        let Some(model_bound) = model_bound else {
            commands.entity(run).transition(Failed(Failure::Provider(
                rig_core::error::ErrorReport::new(rig_core::error::ErrorKind::HandlerUnavailable,
                    "the run has no bound completion model; its selected model or its agent's binding was removed"),
            )));
            commands.entity(turn).remove::<Fresh>();
            progress.mark();
            return Ok(());
        };
        let composes = match &model_bound.descriptor.family {
            FamilyDescriptor::Completion { capabilities, .. } => {
                capabilities.composes_native_output_with_tools
            }
            FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Memory { .. }
            | FamilyDescriptor::Retrieve { .. }
            | FamilyDescriptor::Custom { .. } => {
                commands.entity(run).transition(Failed(Failure::Provider(
                    rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::HandlerUnavailable,
                        format!(
                            "selected model `{}` does not serve completions",
                            model_bound.key
                        ),
                    ),
                )));
                commands.entity(turn).remove::<Fresh>();
                progress.mark();
                return Ok(());
            }
        };

        let RequestEdits {
            edits,
            consumed: consumed_edits,
            edited,
        } = history.edits(turn, run, patch)?;
        let RequestHistory {
            children,
            utterances,
            content,
            cached,
            ..
        } = history;
        let Cached { cache, stats } = cached;
        let cache = &*cache;
        let assets_generation = cache.assets_generation();

        let history: Result<Vec<(Order, std::borrow::Cow<'_, MessageParts>)>, ContentError> =
            children
                .get(run)
                .map(|children| {
                    utterances
                        .iter_many(children.iter())
                        .map(|(entity, order)| {
                            let parts = if edited.contains(&entity) {
                                // The turn's edit: rendered with it, kept
                                // out of the cache (the view is verbatim).
                                stats.renders += 1;
                                content
                                    .message_with(entity, &edits)
                                    .map(std::borrow::Cow::Owned)
                            } else if let Some(view) = cache.view(entity, stale) {
                                stats.hits += 1;
                                Ok(std::borrow::Cow::Borrowed(view))
                            } else {
                                stats.renders += 1;
                                content.message(entity).map(|parts| {
                                    commands.entity(entity).insert(CachedMessage::new(
                                        parts.clone(),
                                        assets_generation,
                                    ));
                                    std::borrow::Cow::Owned(parts)
                                })
                            };
                            parts.map(|parts| (*order, parts))
                        })
                        .collect()
                })
                .unwrap_or_else(|_| Ok(Vec::new()));
        let mut history = history?;
        history.sort_by_key(|(order, _)| *order);

        if is_retrieving {
            bindings.retrieve(commands, progress, children, &fresh, agent, &history);
            return Ok(());
        }

        // The size policy (CONTRACT §8.1): the request's tool-result text,
        // after the part edits `message_with` applied and before the fold;
        // the graph keeps the full text.
        if let Some(limit) = setting(run, agent, tool_result_limits) {
            for (_, parts) in &mut history {
                if policy::tool_results_exceed(parts, limit) {
                    policy::limit_tool_results(parts.to_mut(), limit);
                }
            }
        }

        // The turn's patch (CONTRACT §9.3), folded in as `prepare_request`
        // folded a completion-call hook's.
        let mut tool_links: Vec<_> = children
            .get(turn)
            .into_iter()
            .flat_map(|children| children.iter())
            .filter_map(|link| {
                let (Advert(tool), order) = adverts.get(link).ok()?;
                Some((*order, link, bound.get(*tool).ok()?))
            })
            .collect();
        tool_links.sort_by_key(|(order, _, _)| *order);
        let tools: Vec<&Bound> = tool_links
            .into_iter()
            .filter_map(|(_, link, bound)| {
                let allowed = match (
                    patch.and_then(|p| p.active_tools.as_ref()),
                    &bound.descriptor.family,
                ) {
                    (Some(allowed), FamilyDescriptor::Tool { name, .. }) => allowed.contains(name),
                    (Some(_), FamilyDescriptor::Completion { .. })
                    | (Some(_), FamilyDescriptor::Embed { .. })
                    | (Some(_), FamilyDescriptor::Rerank { .. })
                    | (Some(_), FamilyDescriptor::Memory { .. })
                    | (Some(_), FamilyDescriptor::Retrieve { .. })
                    | (Some(_), FamilyDescriptor::Custom { .. })
                    | (None, _) => true,
                };
                if allowed {
                    Some(bound)
                } else {
                    // The request and executable grant set must agree. Keep
                    // that decision on the turn so materialisation, invalid
                    // call repair and scene continuation see the same surface.
                    commands.entity(link).despawn();
                    None
                }
            })
            .collect();
        let mut attached: Vec<rig_core::completion::Document> =
            links_in_order(turn, children, attachments)
                .filter_map(|Attachment(document)| documents.get(*document).ok())
                .map(|(id, text, props)| rig_core::completion::Document {
                    id: id.0.clone(),
                    text: text.0.clone(),
                    additional_props: props.map(|props| props.0.clone()).unwrap_or_default(),
                })
                .collect();
        if let Some(patch) = patch {
            attached.extend(patch.extra_context.iter().cloned());
        }
        // A patched history replaces the prior utterances; the prompt — the
        // run's last utterance — is still what the turn asks.
        let patched_history: Option<Vec<MessageParts>> =
            patch.and_then(|p| p.history.as_ref()).map(|messages| {
                messages
                    .iter()
                    .cloned()
                    .chain(history.last().map(|(_, parts)| parts.as_ref().clone()))
                    .collect()
            });
        let merged_params: Option<serde_json::Value> = match (
            setting(run, agent, params).and_then(|p| p.0.clone()),
            patch.and_then(|p| p.additional_params.clone()),
        ) {
            (Some(base), Some(patched)) if base.is_object() && patched.is_object() => {
                Some(rig_core::json_utils::merge(base, patched))
            }
            (base, patched) => patched.or(base),
        };

        let preamble = patch
            .and_then(|p| p.preamble.as_deref())
            .or_else(|| setting(run, agent, preambles).and_then(|preamble| preamble.0.as_deref()));
        let temperature = patch
            .and_then(|p| p.temperature)
            .or_else(|| setting(run, agent, temperatures).and_then(|t| t.0));
        let max_tokens = patch
            .and_then(|p| p.max_tokens)
            .or_else(|| setting(run, agent, max_tokens).and_then(|m| m.0));
        let additional_params = merged_params.as_ref();
        let tool_choice = patch
            .and_then(|p| p.tool_choice.as_ref())
            .or_else(|| setting(run, agent, choices).and_then(|c| c.0.as_ref()));
        let output = setting(run, agent, outputs).cloned().unwrap_or_default();
        let output_tool_config = setting(run, agent, output_tools);
        let reserved_name = output_tool_config.and_then(|config| config.name.as_deref());

        let mut access = setting(run, agent, tool_access)
            .cloned()
            .unwrap_or_default();
        let executable = access.executable.get_or_insert_with(|| {
            tools
                .iter()
                .filter_map(|bound| match &bound.descriptor.family {
                    FamilyDescriptor::Tool { name, .. } => Some((name.clone(), bound.key.clone())),
                    _ => None,
                })
                .collect()
        });
        if access.allowed.is_none() {
            access.allowed = Some(executable.keys().cloned().collect());
        }

        let granted_names: Vec<&str> = tools
            .iter()
            .filter_map(|bound| match &bound.descriptor.family {
                FamilyDescriptor::Tool { name, .. } => Some(name.as_str()),
                FamilyDescriptor::Completion { .. }
                | FamilyDescriptor::Embed { .. }
                | FamilyDescriptor::Rerank { .. }
                | FamilyDescriptor::Memory { .. }
                | FamilyDescriptor::Retrieve { .. }
                | FamilyDescriptor::Custom { .. } => None,
            })
            .collect();
        let occupied_names: Vec<&str> = granted_names
            .iter()
            .copied()
            .chain(executable.keys().map(String::as_str))
            .collect();
        let output_tool = minted
            .0
            .clone()
            .or_else(|| reserved_name.map(str::to_owned))
            .unwrap_or_else(|| policy::output_tool_name(&occupied_names));
        let callable = policy::output_tool_callable(tool_choice, &output_tool);
        // A committed output tool (minted on an earlier turn) stays the
        // mode whatever this turn's choice says (CONTRACT §9.3).
        let resolved = if minted.0.is_some() || (reserved_name.is_some() && output.schema.is_some())
        {
            OutputKind::Tool
        } else {
            policy::resolve_output(
                output.mode,
                output.schema.is_some(),
                granted_names.len(),
                callable,
                composes,
            )
        };
        if resolved == OutputKind::Tool && occupied_names.contains(&output_tool.as_str()) {
            // A reserved or already minted name must not also advertise a
            // granted tool: refuse the ambiguous request before dispatch.
            commands.entity(turn).remove::<Fresh>();
            commands
                .entity(run)
                .transition(Failed(Failure::OutputToolCollision {
                    name: output_tool.clone(),
                }));
            progress.mark();
            return Ok(());
        }
        if resolved == OutputKind::Tool && minted.0.is_none() {
            commands
                .entity(run)
                .insert(OutputToolName(Some(output_tool.clone())));
        }

        let graph = RequestGraph {
            preamble,
            utterances: match &patched_history {
                Some(patched) => patched.iter().collect(),
                None => history.iter().map(|(_, parts)| parts.as_ref()).collect(),
            },
            documents: attached,
            tools: tools.iter().map(|bound| &bound.descriptor).collect(),
            temperature,
            max_tokens,
            additional_params,
            tool_choice,
            output: resolved,
            schema: output.schema.as_ref(),
            output_tool: (resolved == OutputKind::Tool).then_some(output_tool.as_str()),
            output_tool_config,
        };
        let request = policy::fold_request(&graph);
        stats.assemblies += 1;
        for link in consumed_edits {
            commands.entity(link).despawn();
        }
        commands.entity(turn).insert(access);
        commands.spawn((
            PendingEffect::new(
                model_bound.key.clone(),
                EffectKind::Completion { request, stream },
            ),
            Completion,
            ChildOf(turn),
        ));
        commands
            .entity(turn)
            .remove::<(Fresh, RequestPatch)>()
            .insert((Folded(resolved), Outputs::default()));
        commands.entity(run).transition(AwaitingModel);
        progress.mark();
        Ok(())
    }
}

/// `RigSet::Fold`: commit completed model outputs. The preceding streamed
/// preview stage publishes text deltas without marking progress; this stage
/// commits the canonical answer and marks progress once.
pub fn fold(
    effects: Query<EffectView, CompletionEffect>,
    mut turns: Query<&mut Outputs, With<Turn>>,
    mut progress: ResMut<Progress>,
) {
    for (child_of, streamed, outcome) in &effects {
        let Ok(mut outputs) = turns.get_mut(child_of.parent()) else {
            continue;
        };
        if outputs.done {
            continue;
        }
        match outcome {
            EffectOutcome(Ok(Outcome::Completion(response))) => {
                // A streamed turn is committed in the canonical order every
                // driver commits one in (reasoning, text, calls): the fold's
                // arrival order is the wire's, and a wire that delivers a
                // reasoning part last (Gemini's thought signature) would
                // otherwise commit a turn no other driver commits.
                outputs.content = if streamed.is_some() {
                    canonical_streamed_choice(response.choice.clone())
                } else {
                    response.choice.clone()
                };
                outputs.message_id = response.message_id.clone();
                outputs.done = true;
                progress.mark();
            }
            EffectOutcome(Ok(_)) | EffectOutcome(Err(_)) => {
                outputs.done = true;
                progress.mark();
            }
        }
    }
}
// A preview cannot affect the final fold: only effects without an outcome are
// eligible. Both stages write their own turn's Outputs directly; neither queues
// commands, allocates entities, invokes observers, or consumes semantic order.
// Only the final fold marks Progress. No host slot lies between the stages;
// final folds still precede invalid-name discovery.
fn fold_streamed(
    effects: Query<(&ChildOf, &BusStreamed), (CompletionEffect, Without<EffectOutcome>)>,
    mut turns: Query<&mut Outputs, With<Turn>>,
) {
    for (parent, stream) in &effects {
        let Ok(mut outputs) = turns.get_mut(parent.parent()) else {
            continue;
        };
        if !outputs.done
            && !stream.text.is_empty()
            && policy::answer_text(&outputs.content) != stream.text
        {
            outputs.content = vec![AssistantContent::text(&stream.text)];
        }
    }
}

/// The default policy for an invalid call nothing resolved: the run's
/// `InvalidCalls.unhandled`, written at the head of `Materialise`'s chain
/// (after `land_memory`), so a user system before the set wins.
pub fn resolve_invalid_defaults(
    mut commands: Commands,
    calls: Query<(Entity, &ChildOf), Unresolved>,
    turns: Query<&ChildOf, With<Turn>>,
    runs: Query<&RunOf>,
    policies: Query<&InvalidCalls>,
) {
    for (call, turn_of) in &calls {
        let Ok(run_of) = turns.get(turn_of.parent()) else {
            continue;
        };
        let run = run_of.parent();
        let Ok(RunOf(agent)) = runs.get(run) else {
            continue;
        };
        let unhandled = setting(run, *agent, &policies).map_or(Unhandled::Fail, |p| p.unhandled);
        commands.entity(call).insert(match unhandled {
            Unhandled::Fail => Resolution::Fail,
            Unhandled::Ignore => Resolution::Ignore,
        });
    }
}

/// What `Fold` reads of a tool child of a turn: which call it is, whether
/// it was issued, its outcome, whether the batch's own hold is on it.
pub type ToolChildView = (
    Entity,
    &'static ToolCallSlot,
    Option<&'static Issued>,
    Option<&'static EffectOutcome>,
    Has<BatchHeld>,
);

/// The runtime's own hold on a tool child beyond the run's concurrency,
/// placed beside `Held` at spawn and lifted by `release_batch` in call
/// order as earlier calls land. The marker says whose hold it is: a
/// `Gate` policy's `Held` is not the runtime's to lift, so a call a policy
/// holds stays held until that policy releases it, and a call the batch
/// holds is released by the batch alone.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct BatchHeld;

/// The batch's marker and its ownership go with the hold. A host may
/// approve a call the batch holds by any route the bus documents —
/// removing `Held` (which bypasses every owner), or releasing the
/// `rig-ecs/batch` owner — and the call then dispatches; were the marker
/// or the owner entry left behind, the batch would keep counting the call
/// as waiting (its slot accounting silently exceeded) and every later
/// scene would save a batch owner with no barrier, which no world loads.
pub fn batch_marker_follows_the_hold(
    released: On<Remove, crate::bus::Held>,
    mut commands: Commands,
) {
    let entity = released.event().entity;
    commands.queue(move |world: &mut World| {
        let Ok(mut effect) = world.get_entity_mut(entity) else {
            return;
        };
        effect.remove::<BatchHeld>();
        crate::bus::hold::forget_owner(world, entity, "rig-ecs/batch");
    });
}

/// One tool child of a batch: the entity, the slot, whether issued, the
/// outcome, whether the batch holds it.
type BatchChild<'a> = (
    Entity,
    &'a ToolCallSlot,
    bool,
    Option<&'a EffectOutcome>,
    bool,
);

/// The tool children of `turn`, by call index.
fn batch_children<'a>(
    turn: Entity,
    children: &Query<&Children>,
    tools: &'a Query<ToolChildView>,
) -> Vec<BatchChild<'a>> {
    let mut found: Vec<_> = children
        .get(turn)
        .map(|children| {
            tools
                .iter_many(children.iter())
                .map(|(entity, slot, issued, outcome, batch_held)| {
                    (entity, slot, issued.is_some(), outcome, batch_held)
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort_by_key(|(_, slot, _, _, _)| slot.index);
    found
}

/// `RigSet::Release`: a turn's batch is let through up to the run's
/// `ToolPolicy.concurrency` — every call beyond it was spawned `Held` with
/// the batch's own [`BatchHeld`], and is released in call order as earlier
/// ones land. Only the batch's holds are lifted: a hold a `Gate` policy
/// wrote is that policy's, and a call under one occupies its slot until the
/// policy releases it. Once a landed outcome is one the run fails on,
/// nothing more is released (fail-fast: in-flight calls drain, unstarted
/// ones never start).
pub fn release_batch(
    mut commands: Commands,
    turns: Query<(Entity, &ChildOf), With<Batch>>,
    runs: Query<&RunOf>,
    policies: Query<&ToolPolicy>,
    children: Query<&Children>,
    tools: Query<ToolChildView>,
) {
    for (turn, turn_of) in &turns {
        let run = turn_of.parent();
        let Ok(RunOf(agent)) = runs.get(run) else {
            continue;
        };
        let concurrency = setting(run, *agent, &policies)
            .map_or(1, |policy| policy.concurrency)
            .max(1);
        let batch = batch_children(turn, &children, &tools);
        if batch.iter().any(|(_, _, _, outcome, _)| {
            outcome.is_some_and(|o| policy::tool_failure(&o.0).is_some())
        }) {
            continue;
        }
        // Let through by the batch and not landed — in flight, about to be,
        // or waiting on a policy's own hold: a slot is a slot.
        let active = batch
            .iter()
            .filter(|(_, _, _, outcome, batch_held)| !*batch_held && outcome.is_none())
            .count();
        let mut free = concurrency.saturating_sub(active);
        for (entity, _, issued, _, batch_held) in &batch {
            if free == 0 {
                break;
            }
            if *batch_held && !issued {
                let entity = *entity;
                commands.queue(move |world: &mut World| {
                    if let Ok(mut effect) = world.get_entity_mut(entity) {
                        effect.remove::<BatchHeld>();
                    }
                    // A hold the batch never took (the effect was denied or
                    // despawned meanwhile) has nothing to release.
                    let _ = crate::bus::release_hold(world, entity, "rig-ecs/batch");
                });
                free -= 1;
            }
        }
    }
}

/// `RigSet::Materialise`, before `materialise`: a turn whose batch has
/// landed becomes graph — one user utterance of the results in call order
/// (CONTRACT §8.1), and the run is `Assembling` again; or, when a landed
/// outcome is one the run fails on and every started call has landed, the
/// run fails and the calls never started are despawned (never dispatched,
/// no record). A call to the output tool beside the batch settles the run
/// with its arguments once the results are history (unpinned).
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn land_batch(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    turns: Query<(Entity, &ChildOf, &Batch, &Outputs)>,
    runs: Query<(&OutputToolName, &RunSeq, &Cursor), With<ResolvingTools>>,
    children: Query<&Children>,
    tools: Query<ToolChildView>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    let mut turns: Vec<_> = turns.iter().collect();
    turns.sort_by_key(|(_, turn_of, _, _)| runs.get(turn_of.parent()).map(|(_, seq, _)| *seq).ok());
    for (turn, turn_of, batch, outs) in turns {
        let run = turn_of.parent();
        let result = (|| -> Result<(), ContentError> {
            let Ok((minted, _, cursor)) = runs.get(run) else {
                return Ok(());
            };
            let calls = batch_children(turn, &children, &tools);
            let failure = calls
                .iter()
                .find_map(|(_, _, _, outcome, _)| outcome.and_then(|o| policy::tool_failure(&o.0)));
            let started_landed = calls
                .iter()
                .all(|(_, _, issued, outcome, _)| !*issued || outcome.is_some());
            if let Some(failure) = failure {
                if !started_landed {
                    return Ok(());
                }
                // The ending first, so the despawns' observer finds the run
                // ended with this failure and leaves it.
                commands.entity(turn).remove::<Batch>();
                commands.entity(run).transition(Failed(failure));
                for (entity, _, issued, outcome, _) in &calls {
                    if !*issued && outcome.is_none() {
                        commands.entity(*entity).despawn();
                    }
                }
                progress.mark();
                return Ok(());
            }
            if calls.len() < batch.calls
                || calls.iter().any(|(_, _, _, outcome, _)| outcome.is_none())
            {
                return Ok(());
            }
            let mut parts = Vec::with_capacity(calls.len());
            let mut statuses = Vec::with_capacity(calls.len());
            let mut failed = None;
            for (_, slot, _, outcome, _) in &calls {
                let Some(EffectOutcome(outcome)) = outcome else {
                    continue;
                };
                match policy::tool_result_part(
                    slot.id.clone(),
                    slot.provider.clone(),
                    slot.name.clone(),
                    outcome,
                ) {
                    Ok((part, status)) => {
                        parts.push(part);
                        statuses.push(status);
                    }
                    Err(failure) => {
                        failed = Some(failure);
                        break;
                    }
                }
            }
            commands.entity(turn).remove::<Batch>();
            if let Some(failure) = failed {
                commands.entity(run).transition(Failed(failure));
                progress.mark();
                return Ok(());
            }
            let results = MessageParts::User { content: parts };
            let results_entity = spawn_deferred_with(
                &mut commands,
                &mut assets,
                run,
                results,
                next_order_in(&mut orders),
                statuses,
            )?;
            commands.entity(turn).insert((
                ToolTurnCommit { turn: cursor.turn },
                TurnResults(results_entity),
            ));
            let output_call = minted.0.as_deref().and_then(|name| {
                outs.content.iter().find_map(|part| match part {
                    AssistantContent::ToolCall(call) if call.function.name == name => {
                        Some(call.function.arguments.to_string())
                    }
                    AssistantContent::ToolCall(_)
                    | AssistantContent::Text(_)
                    | AssistantContent::Reasoning(_)
                    | AssistantContent::Image(_) => None,
                })
            });
            match output_call {
                Some(arguments) => {
                    commands
                        .entity(run)
                        .transition((RunResult(arguments), Settled));
                }
                None => {
                    commands.entity(run).transition(Assembling);
                }
            }
            commands.queue(move |world: &mut World| {
                // A terminal observer may already have removed the run. Do not
                // publish a live notification pointing at a graph it deleted.
                if world.get::<Run>(run).is_some()
                    && world.get::<ToolTurnCommit>(turn).is_some()
                    && world
                        .get::<ChildOf>(turn)
                        .is_some_and(|parent| parent.parent() == run)
                {
                    world.trigger(ToolTurnCommitted { run, turn });
                }
            });
            progress.mark();

            Ok(())
        })();
        if let Err(error) = result {
            fail_content(&mut commands, run, error);
            progress.mark();
        }
    }
}

/// What the pending invalid calls of a turn amount to, in precedence:
/// a `Fail` fails the run; else a `Retry` retries the turn; else a `Skip`
/// answers the call and skips the turn; else repairs and ignores edit the
/// turn's content and the turn goes on.
enum InvalidVerdict {
    Fail(InvalidCall),
    Retry(InvalidCall, String),
    Skip(InvalidCall, String),
    Edit,
}

fn invalid_verdict(pending: &[(Entity, InvalidCall, Resolution)]) -> InvalidVerdict {
    if let Some((_, call, _)) = pending
        .iter()
        .find(|(_, _, resolution)| matches!(resolution, Resolution::Fail))
    {
        return InvalidVerdict::Fail(call.clone());
    }
    if let Some((_, call, Resolution::Retry { feedback })) = pending
        .iter()
        .find(|(_, _, resolution)| matches!(resolution, Resolution::Retry { .. }))
    {
        return InvalidVerdict::Retry(call.clone(), feedback.clone());
    }
    if let Some((_, call, Resolution::Skip { reason })) = pending
        .iter()
        .find(|(_, _, resolution)| matches!(resolution, Resolution::Skip { .. }))
    {
        return InvalidVerdict::Skip(call.clone(), reason.clone());
    }
    InvalidVerdict::Edit
}

/// `RigSet::Materialise`: a complete, unread turn becomes graph — the
/// assistant utterance (unless the turn is empty), the answer and
/// `Settled`, a reprompt and another turn, an invalid call awaiting its
/// resolution, the tool batch (one effect per call to a granted tool,
/// `ChildOf` the turn; the run is `ResolvingTools`), or `Failed`.
#[derive(SystemParam)]
struct MaterialiseGraph<'w, 's> {
    children: Query<'w, 's, &'static Children>,
    adverts: Query<'w, 's, (&'static Advert, &'static Order)>,
    bound: Query<'w, 's, &'static Bound>,
    effects: Query<
        'w,
        's,
        (
            &'static ChildOf,
            &'static EffectOutcome,
            Option<&'static BusStreamed>,
        ),
        CompletionEffect,
    >,
    invalid_calls: Query<
        'w,
        's,
        (
            Entity,
            &'static ChildOf,
            &'static InvalidCall,
            &'static Resolution,
        ),
    >,
}

fn materialise(
    mut turns: Query<MaterialiseTurn, Unread>,
    runs: Query<AwaitingView, With<AwaitingModel>>,
    graph: MaterialiseGraph,
    reads: MaterialiseReads,
    mut writes: GraphWrites,
) {
    // Keep existing query-order precedence between same-verdict invalid calls,
    // including calls in different Resolution archetypes, without per-turn scans.
    let invalid_order: bevy_ecs::entity::EntityHashMap<usize> = graph
        .invalid_calls
        .iter()
        .enumerate()
        .map(|(rank, (entity, ..))| (entity, rank))
        .collect();
    let mut turns: Vec<_> = turns.iter_mut().collect();
    turns.sort_by_key(|turn| runs.get(turn.parent.parent()).map(|run| *run.seq).ok());
    for turn in turns {
        let run = turn.parent.parent();
        let Ok(run_view) = runs.get(run) else {
            continue;
        };
        let result = materialise_turn(turn, run_view, &graph, &reads, &invalid_order, &mut writes);
        writes.finish(run, result);
    }
}

fn materialise_turn(
    mut current: MaterialiseTurnItem<'_, '_>,
    run_view: AwaitingViewItem<'_, '_>,
    graph: &MaterialiseGraph,
    reads: &MaterialiseReads,
    invalid_order: &bevy_ecs::entity::EntityHashMap<usize>,
    writes: &mut GraphWrites,
) -> Result<(), ContentError> {
    let turn = current.entity;
    let run = current.parent.parent();
    let agent = run_view.agent.0;
    let cursor = run_view.cursor;
    let retries = run_view.retries;
    let minted = run_view.minted;
    let usage = run_view.usage;
    let provider_retried = run_view.provider_retried;
    let mode = current.mode.0;
    let retry = current.retry;
    let MaterialiseGraph {
        children,
        adverts,
        bound,
        effects,
        ..
    } = graph;
    let MaterialiseReads {
        access,
        provider_retries,
        subjects,
        witness,
        outputs,
        max_turns,
        tool_policies,
        contexts,
        ..
    } = reads;
    let completion = effects
        .iter_many(
            children
                .get(turn)
                .into_iter()
                .flat_map(|owned| owned.iter()),
        )
        .next();
    let outs = &mut current.outputs;

    // The tools this turn advertised, by name, with their keys.
    let mut granted: Vec<(String, rig_core::effect::HandlerKey)> =
        links_in_order(turn, children, adverts)
            .filter_map(|Advert(tool)| bound.get(*tool).ok())
            .filter_map(|bound| match &bound.descriptor.family {
                FamilyDescriptor::Tool { name, .. } => Some((name.clone(), bound.key.clone())),
                FamilyDescriptor::Completion { .. }
                | FamilyDescriptor::Embed { .. }
                | FamilyDescriptor::Rerank { .. }
                | FamilyDescriptor::Memory { .. }
                | FamilyDescriptor::Retrieve { .. }
                | FamilyDescriptor::Custom { .. } => None,
            })
            .collect();
    let access = access.get(turn).ok();
    if let Some(executable) = access.and_then(|access| access.executable.as_ref()) {
        granted = executable
            .iter()
            .map(|(name, key)| (name.clone(), key.clone()))
            .collect();
    }
    let output_tool = minted.0.as_deref();

    // An early decision can outlive the stream. Count the actual completed
    // response once, before consuming a deferred skip/repair decision.
    if !outs.usage_recorded
        && let Some((_, EffectOutcome(Ok(Outcome::Completion(response))), _)) = completion
    {
        writes
            .commands
            .entity(run)
            .insert(Usage(usage.0 + response.usage));
        outs.usage_recorded = true;
    }

    if reads.resolve_invalid(
        &mut current,
        &run_view,
        graph,
        &granted,
        invalid_order,
        writes,
    )? {
        return Ok(());
    }
    let outs = &mut current.outputs;

    if !outs.done {
        return Ok(());
    }
    let Some((_, EffectOutcome(outcome), _)) = completion else {
        return Ok(());
    };
    let response = match outcome {
        Ok(Outcome::Completion(response)) => response,
        Ok(other) => {
            writes.commands.entity(turn).insert(Materialised);
            writes
                .commands
                .entity(run)
                .transition(Failed(Failure::Unsupported(format!(
                    "a {} answer to a completion",
                    other.family()
                ))));
            writes.progress.mark();
            return Ok(());
        }
        Err(report) => {
            writes.commands.entity(turn).insert(Materialised);
            // A retryable provider failure with budget left re-issues
            // the same request over the same history (CONTRACT §5):
            // the lost turn is read and leaves nothing; the run wants
            // a turn again, marked so `Advance` does not count it.
            let budget = setting(run, agent, provider_retries)
                .map_or(DEFAULT_PROVIDER_RETRIES, |retries| retries.0);
            if report.kind != ErrorKind::Cancelled
                && report.retryable
                && provider_retried.0 < budget
            {
                let attempt = provider_retried.0 + 1;
                writes.commands.entity(run).transition((
                    ProviderRetried(attempt),
                    ProviderRetrying,
                    Assembling,
                ));
                if let Some(witness) = witness.as_deref() {
                    witness::observe_provider_retry(
                        witness,
                        subjects.of_scope(run),
                        attempt,
                        budget,
                        report,
                    );
                }
                writes.progress.mark();
                return Ok(());
            }
            let failure = if report.kind == ErrorKind::Cancelled {
                Failure::Cancelled(report.clone())
            } else {
                Failure::Provider(report.clone())
            };
            writes.commands.entity(run).transition(Failed(failure));
            writes.progress.mark();
            return Ok(());
        }
    };
    let content = outs.content.clone();

    // An answerless turn the provider cut short is a lost turn, not an
    // empty answer (rig#2322; rig-agent's rule, CONTRACT §4): the run
    // fails as a response error naming the finish reason. A turn that
    // delivered text or a call, however it stopped, is read as usual.
    if turn_delivered_no_answer(&content)
        && let Some(reason) = response
            .finish_reason()
            .filter(|reason| reason.truncated_output())
    {
        writes.commands.entity(turn).insert(Materialised);
        let report = rig_core::error::ErrorReport::from(
            &rig_core::completion::CompletionError::ResponseError(reason.no_answer_message()),
        );
        writes
            .commands
            .entity(run)
            .transition(Failed(Failure::Provider(report)));
        writes.progress.mark();
        return Ok(());
    }

    // An empty turn is not history, and answers nothing. A retry
    // written on it (CONTRACT §9.4) still asks again: the feedback
    // becomes history, the empty turn does not, and another turn
    // begins; without one, the run settles on the empty answer.
    if policy::turn_is_empty(&content) {
        writes.commands.entity(turn).insert(Materialised);
        if let Some(Retry { feedback }) = retry {
            writes.commands.entity(turn).remove::<Retry>();
            if let Some(feedback) = feedback {
                let user = MessageParts::User {
                    content: vec![UserContent::text(feedback)],
                };
                writes.say(run, user)?;
            }
            writes.commands.entity(run).transition(Assembling);
            writes.progress.mark();
            return Ok(());
        }
        writes
            .commands
            .entity(run)
            .transition((RunResult(String::new()), Settled));
        writes.progress.mark();
        return Ok(());
    }

    let schema = setting(run, agent, outputs).and_then(|output| output.schema.clone());
    let limit = setting(run, agent, max_turns).map_or(1, |limit| limit.0);

    let calls: Vec<&rig_core::completion::message::ToolCall> = content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call),
            AssistantContent::Text(_)
            | AssistantContent::Reasoning(_)
            | AssistantContent::Image(_) => None,
        })
        .collect();

    // Invalid calls: tools neither granted nor the output tool. They
    // become entities awaiting a resolution; the turn stays unread
    // until then.
    let invalid: Vec<&rig_core::completion::message::ToolCall> = calls
        .iter()
        .copied()
        .filter(|call| {
            (!granted.iter().any(|(name, _)| *name == call.function.name)
                || access
                    .and_then(|access| access.allowed.as_ref())
                    .is_some_and(|allowed| !allowed.contains(&call.function.name)))
                && output_tool != Some(call.function.name.as_str())
        })
        .collect();
    if !invalid.is_empty() {
        for call in invalid {
            writes.commands.spawn((
                InvalidCall {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                    prefix: Vec::new(),
                    stream_offset: None,
                },
                ChildOf(turn),
            ));
        }
        writes.progress.mark();
        return Ok(());
    }

    writes.commands.entity(turn).insert(Materialised);

    // A retry written on the turn (CONTRACT §9.4): tool-free turns only.
    if let Some(Retry { feedback }) = retry {
        writes.commands.entity(turn).remove::<Retry>();
        if !calls.is_empty() {
            writes
                .commands
                .entity(run)
                .transition(Failed(Failure::Unsupported(
                    "a retry of a tool-bearing turn: steer the tool calls instead".to_owned(),
                )));
            writes.progress.mark();
            return Ok(());
        }
        if let Some(feedback) = feedback {
            let assistant = MessageParts::Assistant {
                id: response.message_id.clone(),
                content: content.clone(),
            };
            writes.say(run, assistant)?;
            let user = MessageParts::User {
                content: vec![UserContent::text(feedback)],
            };
            writes.say(run, user)?;
        }
        writes.commands.entity(run).transition(Assembling);
        writes.progress.mark();
        return Ok(());
    }

    // The assistant turn is history.
    let assistant = MessageParts::Assistant {
        id: response.message_id.clone(),
        content: content.clone(),
    };
    let assistant_entity = writes.say(run, assistant)?;

    // Calls to granted tools: the batch, one effect per call `ChildOf`
    // the turn, in call order, held beyond the concurrency.
    let batch: Vec<(
        usize,
        &rig_core::completion::message::ToolCall,
        rig_core::effect::HandlerKey,
    )> = calls
        .iter()
        .filter_map(|call| {
            granted
                .iter()
                .find(|(name, _)| *name == call.function.name)
                .map(|(_, key)| (*call, key.clone()))
        })
        .enumerate()
        .map(|(index, (call, key))| (index, call, key))
        .collect();
    if !batch.is_empty() {
        let concurrency = setting(run, agent, tool_policies)
            .map_or(1, |policy| policy.concurrency)
            .max(1);
        let inputs = setting(run, agent, contexts)
            .map(|spec| spec.0.for_dispatch())
            .unwrap_or_default();
        let count = batch.len();
        for (index, call, key) in batch {
            let mut effect = writes.commands.spawn((
                PendingEffect::new(
                    key,
                    EffectKind::ToolCall {
                        name: call.function.name.clone(),
                        args: call.function.arguments.to_string(),
                    },
                ),
                ToolInputs(inputs.clone()),
                ToolCallSlot {
                    index,
                    id: call.id.clone(),
                    provider: call.provider.clone(),
                    name: call.function.name.clone(),
                },
                ChildOf(turn),
            ));
            if index >= concurrency {
                effect.insert(BatchHeld);
                let entity = effect.id();
                writes.commands.queue(move |world: &mut World| {
                    // Refused only when the effect settled or dispatched
                    // between the spawn and this command: then the batch
                    // bound no longer applies to it.
                    let _ = crate::bus::acquire_hold(
                        world,
                        entity,
                        rig_core::observe::Emitter::versioned(
                            "rig-ecs/batch",
                            env!("CARGO_PKG_VERSION"),
                        ),
                    );
                });
            }
        }
        writes
            .commands
            .entity(turn)
            .insert((Batch { calls: count }, TurnAssistant(assistant_entity)));
        writes.commands.entity(run).transition(ResolvingTools);
        writes.progress.mark();
        return Ok(());
    }

    match (mode, output_tool) {
        (OutputKind::Tool, Some(name)) => {
            let output_call = calls.iter().find(|call| call.function.name == name);
            let can_reprompt = retries.0 < 1 && cursor.turn < limit;
            match output_call {
                Some(call) => {
                    let missing = schema
                        .as_ref()
                        .map(|schema| {
                            policy::missing_required_fields(schema, &call.function.arguments)
                        })
                        .unwrap_or_default();
                    if missing.is_empty() || !can_reprompt {
                        // Match rig-agent's reusable history: the record
                        // retains the output call, but its committed answer
                        // is JSON text with all reasoning preserved.
                        let output = call.function.arguments.to_string();
                        let mut final_content: Vec<_> = content
                            .iter()
                            .filter(|part| !matches!(part, AssistantContent::ToolCall(_)))
                            .cloned()
                            .collect();
                        final_content.push(AssistantContent::text(output.clone()));
                        writes.replace(
                            assistant_entity,
                            MessageParts::Assistant {
                                id: response.message_id.clone(),
                                content: final_content,
                            },
                        )?;
                        writes
                            .commands
                            .entity(run)
                            .transition((RunResult(output), Settled));
                    } else {
                        let feedback = policy::reprompt_missing_fields(name, &missing);
                        let reprompt = MessageParts::User {
                            content: vec![UserContent::ToolResult(
                                rig_core::completion::message::ToolResult {
                                    call: call.id.clone(),
                                    provider: call.provider.clone(),
                                    name: name.to_owned(),
                                    content: vec![ToolResultContent::text(feedback)],
                                },
                            )],
                        };
                        writes
                            .commands
                            .entity(turn)
                            .insert(Reprompt(reprompt.to_message()));
                        writes.results(run, reprompt, vec![ToolResultStatus::Skipped])?;
                        writes
                            .commands
                            .entity(run)
                            .transition((OutputRetries(retries.0 + 1), Assembling));
                    }
                }
                // A text that already is the structured output settles the
                // run (CONTRACT §4); one that is not is reprompted while
                // the budget lasts.
                None if can_reprompt
                    && !policy::text_satisfies_schema(
                        schema.as_ref(),
                        &policy::answer_text(&content),
                    ) =>
                {
                    let reprompt = MessageParts::User {
                        content: vec![UserContent::text(policy::text::reprompt_text_answer(name))],
                    };
                    writes
                        .commands
                        .entity(turn)
                        .insert(Reprompt(reprompt.to_message()));
                    writes.say(run, reprompt)?;
                    writes
                        .commands
                        .entity(run)
                        .transition((OutputRetries(retries.0 + 1), Assembling));
                }
                None => {
                    writes
                        .commands
                        .entity(run)
                        .transition((RunResult(policy::answer_text(&content)), Settled));
                }
            }
        }
        (OutputKind::Tool, None)
        | (OutputKind::Auto | OutputKind::Native | OutputKind::Prompted, _) => {
            writes
                .commands
                .entity(run)
                .transition((RunResult(policy::answer_text(&content)), Settled));
        }
    }
    writes.progress.mark();
    Ok(())
}

impl MaterialiseReads<'_, '_> {
    fn resolve_invalid(
        &self,
        current: &mut MaterialiseTurnItem<'_, '_>,
        run_view: &AwaitingViewItem<'_, '_>,
        graph: &MaterialiseGraph,
        granted: &[(String, rig_core::effect::HandlerKey)],
        invalid_order: &bevy_ecs::entity::EntityHashMap<usize>,
        writes: &mut GraphWrites,
    ) -> Result<bool, ContentError> {
        let turn = current.entity;
        let run = current.parent.parent();
        let agent = run_view.agent.0;
        let invalid_retries = run_view.invalid_retries;
        let output_tool = run_view.minted.0.as_deref();
        let tool_choice = setting(run, agent, &self.choices).and_then(|c| c.0.clone());
        let policies = &self.policies;
        let MaterialiseGraph {
            children,
            effects,
            invalid_calls,
            ..
        } = graph;
        let completion = effects
            .iter_many(
                children
                    .get(turn)
                    .into_iter()
                    .flat_map(|owned| owned.iter()),
            )
            .next();
        let outs = &mut current.outputs;
        // Pending invalid calls of this turn: consumed first.
        let mut pending: Vec<(Entity, InvalidCall, Resolution)> = invalid_calls
            .iter_many(
                children
                    .get(turn)
                    .into_iter()
                    .flat_map(|owned| owned.iter()),
            )
            .map(|(entity, _, call, resolution)| {
                let mut call = call.clone();
                if let Some(offset) = call.stream_offset
                    && let Some((_, _, Some(stream))) = completion
                    && let Some(id) = stream_invalid::completed_call_id(&stream.events, offset)
                {
                    call.id = id;
                }
                (entity, call, resolution.clone())
            })
            .collect();
        pending.sort_by_key(|(entity, _, _)| invalid_order.get(entity).copied());
        if !pending.is_empty() {
            let budget = setting(run, agent, policies).map_or(0, |p| p.retries);
            let verdict = match invalid_verdict(&pending) {
                InvalidVerdict::Retry(call, _) if invalid_retries.0 >= budget => {
                    InvalidVerdict::Fail(call)
                }
                InvalidVerdict::Skip(call, _) if matches!(tool_choice, Some(ToolChoice::None)) => {
                    InvalidVerdict::Fail(call)
                }
                verdict => verdict,
            };
            // Effective failure (including an exhausted retry) is immediate.
            // Keep edits until final folding and real usage arrive.
            if !outs.done && !matches!(verdict, InvalidVerdict::Fail(_)) {
                return Ok(true);
            }
            // EOF can arrive with more name events than this pass judged.
            // Retain earlier edits while discovery visits that delivered tail,
            // so the next decision sees the repaired/ignored prefix in order.
            if matches!(verdict, InvalidVerdict::Edit)
                && completion.is_some_and(|(_, _, stream)| {
                    stream.is_some_and(|stream| {
                        outs.stream_validated < stream_invalid::validation_len(stream)
                    })
                })
            {
                return Ok(true);
            }
            for (entity, _, _) in &pending {
                writes.commands.entity(*entity).despawn();
            }
            match verdict {
                InvalidVerdict::Fail(call) => {
                    if !call.prefix.is_empty() {
                        let assistant = MessageParts::Assistant {
                            id: outs.message_id.clone(),
                            content: call.prefix.clone(),
                        };
                        writes.say(run, assistant)?;
                    }
                    writes.commands.entity(turn).insert(Materialised);
                    writes
                        .commands
                        .entity(run)
                        .transition(Failed(Failure::UnknownToolCall { name: call.name }));
                    writes.progress.mark();
                    return Ok(true);
                }
                InvalidVerdict::Retry(call, feedback) | InvalidVerdict::Skip(call, feedback) => {
                    let retried = matches!(invalid_verdict(&pending), InvalidVerdict::Retry(..));
                    // A streamed turn is abandoned where the call surfaced.
                    let events = completion
                        .and_then(|(_, _, streamed)| streamed)
                        .map(|streamed| streamed.events.as_slice());
                    let allowed_names: Vec<String> = granted
                        .iter()
                        .map(|(name, _)| name.clone())
                        .chain(output_tool.map(str::to_owned))
                        .collect();
                    let (content, diagnostic_id) =
                        if let Some(AssistantContent::ToolCall(diagnostic)) = call.prefix.last() {
                            (call.prefix.clone(), diagnostic.id.clone())
                        } else {
                            policy::partial_turn_at(&outs.content, events, &call.id, &allowed_names)
                        };
                    let assistant = MessageParts::Assistant {
                        id: outs.message_id.clone(),
                        content: content.clone(),
                    };
                    writes.say(run, assistant)?;
                    let results = policy::invalid_peer_results(&content, &diagnostic_id, &feedback);
                    let skipped = match &results {
                        MessageParts::User { content } => {
                            vec![ToolResultStatus::Skipped; content.len()]
                        }
                        MessageParts::Assistant { .. } => Vec::new(),
                    };
                    writes.results(run, results, skipped)?;
                    writes.commands.entity(turn).insert(Materialised);
                    let mut run_commands = writes.commands.entity(run);
                    run_commands.transition(Assembling);
                    if retried {
                        run_commands.insert(InvalidRetries(invalid_retries.0 + 1));
                    }
                    writes.progress.mark();
                    return Ok(true);
                }
                InvalidVerdict::Edit => {
                    // Repairs rename their call; ignores drop theirs. What is
                    // left is the turn.
                    let mut content = outs.content.clone();
                    for (_, call, resolution) in &pending {
                        match resolution {
                            Resolution::Repair { to } => {
                                for part in &mut content {
                                    if let AssistantContent::ToolCall(tool_call) = part
                                        && tool_call.id == call.id
                                    {
                                        tool_call.function.name = to.clone();
                                    }
                                }
                            }
                            Resolution::Ignore => {
                                content.retain(|part| match part {
                                    AssistantContent::ToolCall(tool_call) => {
                                        tool_call.id != call.id
                                    }
                                    AssistantContent::Text(_)
                                    | AssistantContent::Reasoning(_)
                                    | AssistantContent::Image(_) => true,
                                });
                            }
                            Resolution::Fail
                            | Resolution::Retry { .. }
                            | Resolution::Skip { .. } => {}
                        }
                    }
                    if outs.content != content {
                        outs.content = content;
                    }
                }
            }
        }
        Ok(false)
    }
}

/// An effect despawned while its turn was unread — a system in `Patch`
/// stopping the run, a host cancelling — ends the run `Cancelled`: the
/// record says so (the bus's cancel observer), and so does the run.
pub fn effect_cancelled(
    removed: On<bevy_ecs::lifecycle::Remove, PendingEffect>,
    effects: Query<(&ChildOf, Has<ToolCallSlot>), With<PendingEffect>>,
    turns: Query<TurnState, With<Turn>>,
    runs: Query<RunPhase, With<Run>>,
    mut commands: Commands,
) {
    let effect = removed.event().entity;
    let Ok((turn_of, is_tool_call)) = effects.get(effect) else {
        return;
    };
    let turn = turn_of.parent();
    let Ok((run_of, materialised, batched)) = turns.get(turn) else {
        return;
    };
    let run = run_of.parent();
    let Ok((awaiting, resolving, failed)) = runs.get(run) else {
        return;
    };
    // A run already ended keeps its ending: `run_cancelled` writes the
    // reason before it despawns what was pending.
    if failed {
        return;
    }
    let cancelled = Failed(Failure::Cancelled(rig_core::serve::cancelled()));
    if is_tool_call && batched && resolving {
        // A tool child despawned while its batch was out: the run ends
        // here, the batch with it.
        commands.entity(turn).remove::<Batch>();
        commands.entity(run).transition(cancelled);
    } else if !is_tool_call && !materialised && awaiting {
        commands.entity(turn).insert(Materialised);
        commands.entity(run).transition(cancelled);
    }
}

/// `Cancelled(reason)` written on a run (CONTRACT §9.1): the run ends
/// `Failed(Cancelled)` with the reason, its current turn is read, and every
/// effect of the run never issued — a completion folded and not yet
/// dispatched, a tool child held or pending, a hook's own dispatch — is
/// despawned before the bus sees it (no record). An effect in flight is
/// left to its handler: the record is the handler's.
pub fn run_cancelled(
    added: On<Add, Cancelled>,
    reasons: Query<(&Cancelled, Has<Settled>, Has<Failed>)>,
    children: Query<&Children>,
    turns: Query<(), With<Turn>>,
    effects: Query<Has<Issued>, With<PendingEffect>>,
    mut commands: Commands,
) {
    let run = added.event().entity;
    let Ok((Cancelled(reason), settled, failed)) = reasons.get(run) else {
        return;
    };
    // A run that ended keeps its ending: a cancel after the fact is a
    // no-op, never a second ending.
    if settled || failed {
        return;
    }
    let mut pending: Vec<Entity> = Vec::new();
    let mut unread: Vec<Entity> = Vec::new();
    for child in children.get(run).into_iter().flat_map(|owned| owned.iter()) {
        if let Ok(issued) = effects.get(child) {
            if !issued {
                pending.push(child);
            }
            continue;
        }
        if turns.get(child).is_ok() {
            unread.push(child);
            for effect in children
                .get(child)
                .into_iter()
                .flat_map(|owned| owned.iter())
            {
                if let Ok(false) = effects.get(effect) {
                    pending.push(effect);
                }
            }
        }
    }
    // The ending first, so the despawns' observer (`effect_cancelled`)
    // finds the run ended with this reason and leaves it.
    commands.entity(run).transition(Failed(Failure::Cancelled(
        rig_core::error::ErrorReport::new(ErrorKind::Cancelled, reason.clone()),
    )));
    for effect in pending {
        commands.entity(effect).despawn();
    }
    for turn in unread {
        commands
            .entity(turn)
            .insert(Materialised)
            .remove::<(Batch, Fresh, RequestPatch)>();
    }
}
