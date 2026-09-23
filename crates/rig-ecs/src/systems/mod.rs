//! Agent run lifecycle and request-processing systems in [`RigSchedule`].
//!
//! Hosts edit the graph before [`RigSet::Assemble`], folded effects in
//! [`RigSet::Patch`], and outputs in [`RigSet::Judge`].
//!
//! ```
//! use bevy_ecs::schedule::IntoScheduleConfigs;
//! use rig_ecs::{RigPlugin, bus::RigSchedule, systems::RigSet};
//! let mut app = bevy_app::App::new();
//! app.add_plugins(RigPlugin::default());
//! app.add_systems(RigSchedule, (|| {}).in_set(RigSet::Checkpoint));
//! ```

use crate::agent::checkpoint::{
    ToolTurnCommit, ToolTurnCommitted, ToolTurnHolds, TurnAssistant, TurnResults,
};
use crate::agent::content::{
    binary::BinaryAssets,
    parts::{
        ContentError, ContentGraph, ToolResultLimit, ToolResultStatus, replace_deferred,
        spawn_deferred, spawn_deferred_with, write_message,
    },
};
use bevy_reflect::Reflect;

use crate::agent::content::parts::{EditTarget, RequestPartEdit};
use bevy_ecs::{
    prelude::*,
    query::{QueryData, QueryFilter},
};
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
        AdditionalParams, Advert, Attachment, Batch, Cancelled, Context, Conversation, Cursor,
        DEFAULT_PROVIDER_RETRIES, DocumentId, DocumentProps, DocumentText, Failed, Failure, Grant,
        InvalidCall, InvalidCalls, InvalidRetries, MaxTokens, MaxTurns, MemoryAppendScheduled,
        MessageParts, Output, OutputKind, OutputRetries, OutputToolConfig, OutputToolName, Outputs,
        Preamble, Prompt, ProviderRetried, ProviderRetries, ProviderRetrying, Remembered,
        Remembering, Remembers, Reprompt, RequestPatch, Resolution, Retrievable, Retrieval,
        RetrievalKind, Retrieves, Retrieving, Retry, Run, RunCounter, RunOf, RunPhase, RunResult,
        RunSeq, Settled, StreamRequested, Temperature, ToolAccess, ToolCallSlot, ToolChoiceSpec,
        ToolContextSpec, ToolPolicy, Turn, Unhandled, Usage, UsesModel, Utterance,
    },
    bus::{
        Bound, BusSet, EffectOutcome, Issued, PendingEffect, RigSchedule, ServedBy,
        Streamed as BusStreamed, ToolInputs,
    },
    policy::{self, RequestGraph},
};

pub mod backoff;
pub mod diagnostics;
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

/// A run that may want a turn: `Ready`, phased, not failed (`Assembling`
/// is read off the phase).
#[derive(QueryFilter)]
pub struct Wanting {
    phased: With<RunPhase>,
    ready: With<crate::agent::Ready>,
    live: Without<Failed>,
}
/// A `Ready` run that has neither a phase nor an ending: `open_runs` opens it.
#[derive(QueryFilter)]
pub struct Unopened {
    run: With<Run>,
    ready: With<crate::agent::Ready>,
    not_failed: Without<Failed>,
    not_settled: Without<Settled>,
    phaseless: Without<RunPhase>,
}
/// A run with no model of its own yet.
#[derive(QueryFilter)]
pub struct Unselected {
    run: With<Run>,
    unselected: Without<UsesModel>,
}
/// A run that has not failed.
#[derive(QueryFilter)]
pub struct LiveRun {
    run: With<Run>,
    live: Without<Failed>,
}
/// What `Fold` reads of an effect: its turn, its stream so far, its
/// outcome once landed.
#[derive(QueryData)]
pub struct EffectView {
    /// The turn the effect is `ChildOf`.
    pub turn_of: &'static ChildOf,
    /// The stream so far, for a streamed effect.
    pub streamed: Option<&'static BusStreamed>,
    /// The outcome, once landed.
    pub outcome: Option<&'static EffectOutcome>,
}
/// What `Materialise` reads of a landed completion effect: its turn, its
/// outcome, its stream if it streamed.
#[derive(QueryData)]
pub struct LandedEffect {
    /// The turn the effect is `ChildOf`.
    pub turn_of: &'static ChildOf,
    /// The outcome.
    pub outcome: &'static EffectOutcome,
    /// The stream, for a streamed effect.
    pub streamed: Option<&'static BusStreamed>,
}
/// An invalid call nothing resolved.
#[derive(QueryFilter)]
pub struct Unresolved {
    invalid: With<InvalidCall>,
    unresolved: Without<Resolution>,
}
/// A turn `Materialise` has not read.
#[derive(QueryFilter)]
pub struct Unread {
    turn: With<Turn>,
    unread: Without<Materialised>,
}
/// What `gather_turn` reads of a run.
#[derive(QueryData)]
pub struct AssemblingRun {
    /// The agent.
    pub run_of: &'static RunOf,
    /// The run's order among runs.
    pub seq: &'static RunSeq,
    /// Whether the run streams.
    pub stream: &'static StreamRequested,
    /// The run's model, once selected.
    pub model: Option<&'static UsesModel>,
    /// The output tool's name, once minted.
    pub minted: &'static OutputToolName,
}
/// The request settings `gather_turn` resolves, the run's over the agent's.
#[derive(bevy_ecs::system::SystemParam)]
pub struct Settings<'w, 's> {
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
/// What `gather_turn` reads of a fresh turn: its run, its patch, whether
/// it is retrieving.
#[derive(QueryData)]
pub struct FreshTurn {
    /// The turn.
    pub entity: Entity,
    /// The run the turn is `ChildOf`.
    pub turn_of: &'static ChildOf,
    /// The turn's request patch, if a system wrote one.
    pub patch: Option<&'static RequestPatch>,
    /// Whether the turn retrieves before it folds.
    pub retrieving: Has<Retrieving>,
}
/// A fresh turn whose retrievals are out.
#[derive(QueryFilter)]
pub struct RetrievingTurn {
    fresh: With<Fresh>,
    retrieving: With<Retrieving>,
}
/// A remembering run whose persisted finalization has not scheduled an append.
#[derive(QueryFilter)]
pub struct NeedsMemoryAppend {
    settled: With<Settled>,
    remembering: With<Remembering>,
    unscheduled: Without<MemoryAppendScheduled>,
}
/// A completion effect: any effect of a turn that is not a retrieval.
#[derive(QueryFilter)]
pub struct NotRetrieval {
    effect: With<PendingEffect>,
    completion: Without<Retrieval>,
}
/// What the cancel observer reads of a run: its phase, whether ended.
#[derive(QueryData)]
pub struct RunState {
    /// The phase, while the run has one.
    pub phase: Option<&'static RunPhase>,
    /// Whether the run failed.
    pub failed: Has<Failed>,
}
/// What the cancel observer reads of a turn: its run, whether it was
/// read, whether its batch is out.
#[derive(QueryData)]
pub struct TurnState {
    /// The run the turn is `ChildOf`.
    pub run_of: &'static ChildOf,
    /// Whether `Materialise` read the turn.
    pub materialised: Has<Materialised>,
    /// Whether the turn's batch is out.
    pub batched: Has<Batch>,
}

/// A fresh turn: spawned by `Advance`, not yet folded by `Assemble`.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(Component)]
pub struct Fresh;

/// The output mode the turn was folded under, pinned.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component)]
pub struct Folded(pub OutputKind);

/// A turn `Materialise` has read.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(Component)]
pub struct Materialised;

/// A tool a turn may call: its name, its handler key, and its handler
/// entity; `None` when the run's `ToolAccess.executable` named it rather
/// than an advert (the bus routes the key).
#[derive(Debug, Clone)]
pub struct GrantedTool {
    /// The tool's name, as the model calls it.
    pub name: String,
    /// The handler key the call dispatches to.
    pub key: rig_core::effect::HandlerKey,
    /// The advertised handler entity, when an advert granted the tool.
    pub handler: Option<Entity>,
}

/// Complete turn content after judgement and invalid-call edits, with executable
/// tools and the materialised assistant entity. Transient within one
/// [`RigSet::Materialise`] pass; removed when the turn finishes processing.
#[derive(Component, Debug, Clone)]
pub struct TurnRead {
    /// The provider's message id, when the answer carried one.
    pub message_id: Option<String>,
    /// The turn's parts as read.
    pub content: Vec<AssistantContent>,
    /// The tools the turn may call, in advertisement order.
    pub granted: Vec<GrantedTool>,
    /// The assistant utterance, once spawned.
    pub assistant: Option<Entity>,
}

impl TurnRead {
    /// The turn's tool calls, in order.
    fn calls(&self) -> impl Iterator<Item = &rig_core::completion::message::ToolCall> {
        self.content.iter().filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call),
            AssistantContent::Text(_)
            | AssistantContent::Reasoning(_)
            | AssistantContent::Image(_) => None,
        })
    }
}

/// What `gather_turn` gathered of a fresh turn's graph, for `fold_turn`:
/// everything the fold reads, owned, the settings resolved (the patch
/// over the run's over the agent's), the history after the turn's part
/// edits and the size policy, the output mode resolved and the output
/// tool's name minted. Lives one pass: `fold_turn` removes it.
#[derive(Component, Debug, Clone)]
pub struct AssemblyInputs {
    /// The run's bound completion model.
    pub model: Entity,
    /// Whether the run streams.
    pub stream: bool,
    /// The preamble.
    pub preamble: Option<String>,
    /// The utterances in order, as the request sees them.
    pub utterances: Vec<MessageParts>,
    /// The documents attached, in order, then the patch's extra context.
    pub documents: Vec<rig_core::completion::Document>,
    /// The tool handler entities advertised and allowed, in advert order.
    pub tools: Vec<Entity>,
    /// Sampling.
    pub temperature: Option<f64>,
    /// The token budget.
    pub max_tokens: Option<u64>,
    /// Provider parameters, the patch's merged over the setting's.
    pub additional_params: Option<serde_json::Value>,
    /// The tool choice.
    pub tool_choice: Option<ToolChoice>,
    /// The output mode, resolved.
    pub output: OutputKind,
    /// The output schema, if any.
    pub schema: Option<serde_json::Value>,
    /// The output tool's name, when the mode is `Tool`.
    pub output_tool: Option<String>,
    /// The output tool's description and preamble behavior.
    pub output_tool_config: Option<OutputToolConfig>,
}

/// The agent runtime: the sets, the counters, the observers and the systems,
/// in the bus's [`RigSchedule`]. Requires [`crate::bus::BusPlugin`] first.
#[derive(Debug, Default, Clone, Copy)]
pub struct AgentPlugin;

impl bevy_app::Plugin for AgentPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        Self::install(app.world_mut());
        if app
            .world()
            .contains_resource::<bevy_diagnostic::DiagnosticsStore>()
        {
            diagnostics::register(app);
            app.add_systems(RigSchedule, diagnostics::measure.in_set(RigSet::Settle));
        }
    }
}

impl AgentPlugin {
    /// Install agent systems, counters, and observers for a host-driven schedule.
    /// Panics unless the bus policy and schedules have already been installed.
    pub fn install(world: &mut World) {
        install_agent(world);
    }
}

fn install_agent(world: &mut World) {
    assert!(
        world.contains_resource::<crate::bus::Policy>(),
        "AgentPlugin needs BusPlugin first: it runs in the bus's RigSchedule"
    );
    world.init_resource::<BinaryAssets>();
    world.init_resource::<RunCounter>();
    world.add_observer(effect_cancelled);
    world.add_observer(run_cancelled);
    world.add_observer(batch_marker_follows_the_hold);
    witness::install(world);
    let mut schedules = world.resource_mut::<Schedules>();
    let Some(schedule) = schedules.get_mut(RigSchedule) else {
        return;
    };
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
        (gather_turn, fold_turn).chain().in_set(RigSet::Assemble),
        backoff::hold_retries
            .after(RigSet::Patch)
            .before(RigSet::Release),
        release_batch.in_set(RigSet::Release),
        (fold, discover_streamed_invalid_calls)
            .chain()
            .in_set(RigSet::Fold),
        (
            land_memory,
            resolve_invalid_defaults,
            land_batch,
            record_usage,
            judge_invalid_calls,
            read_turn,
            materialise_assistant,
            materialise_batch,
            materialise_reprompt,
            materialise_answer,
        )
            .chain()
            .in_set(RigSet::Materialise),
        append_memory.in_set(RigSet::Settle),
    ));
}

/// The run's phase changes: the next phase, or an ending in its place.
trait PhaseCommands {
    fn phase(&mut self, next: RunPhase) -> &mut Self;
    fn end(&mut self, ending: impl Bundle) -> &mut Self;
}

impl PhaseCommands for bevy_ecs::system::EntityCommands<'_> {
    fn phase(&mut self, next: RunPhase) -> &mut Self {
        self.insert(next)
    }
    fn end(&mut self, ending: impl Bundle) -> &mut Self {
        self.remove::<RunPhase>().insert(ending)
    }
}

/// Fail only the affected run with a content error, removing any settled marker.
fn fail_content(commands: &mut Commands, run: Entity, error: ContentError) {
    commands
        .entity(run)
        .remove::<(RunPhase, Settled)>()
        .insert(Failed(Failure::Content(error)));
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

/// Spawn, cancel, and remove runs through a world or deferred commands.
/// World operations apply immediately; commands reserve spawn IDs immediately
/// but populate runs only when flushed. Ordered commands preserve dependencies
/// between spawning, cancelling, and removing a run.
pub trait RunCommands {
    /// What `despawn_run` reports: the refusal, on `World`; nothing on
    /// `Commands`, which triggers [`RunDespawnRefused`] on the run instead.
    type Despawned;

    /// Spawn a run of `agent` with `prompt` as its first utterance, after
    /// `history`: the host's one entry point. The prompt is a user
    /// message's parts (`&str` text, or text and images kept in their
    /// given order, [`Prompt`]). Returns the run entity. On `Commands`
    /// the entity is reserved at once and populated when the command
    /// applies; a run despawned before or while it is populated (a host
    /// `Add<Run>` observer that refuses it, say) is simply gone.
    fn spawn_run(
        &mut self,
        agent: Entity,
        history: &[MessageParts],
        prompt: impl Into<Prompt>,
        streamed: bool,
        max_turns: Option<usize>,
    ) -> Entity;

    /// Stop `run` with `reason` (CONTRACT §9.1): `Cancelled(reason)` on the
    /// run. A run that ended keeps its ending; an entity that is not a run
    /// is left alone. A run cancelled before it opened (`Ready` written by
    /// hand, not yet seen by `Advance`) fails with its unread [`Prompt`]
    /// still on it, and a scene saves the prompt with the failed run.
    fn cancel_run(&mut self, run: Entity, reason: impl Into<String>);

    /// Despawn an ended run and its linked descendants, including stream folds.
    /// Refuses without mutation if the entity is not a run, has not ended, or
    /// has pending or in-flight effects. Hosts must remove finished runs to
    /// reclaim their graphs; the runtime does not remove them automatically.
    fn despawn_run(&mut self, run: Entity) -> Self::Despawned;
}

impl RunCommands for World {
    type Despawned = Result<(), RunBusy>;

    fn spawn_run(
        &mut self,
        agent: Entity,
        history: &[MessageParts],
        prompt: impl Into<Prompt>,
        streamed: bool,
        max_turns: Option<usize>,
    ) -> Entity {
        let run = self.spawn_empty().id();
        spawn_run_at(
            self,
            run,
            agent,
            history,
            prompt.into(),
            streamed,
            max_turns,
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
        history: &[MessageParts],
        prompt: impl Into<Prompt>,
        streamed: bool,
        max_turns: Option<usize>,
    ) -> Entity {
        let run = self.spawn_empty().id();
        let history = history.to_vec();
        let prompt = prompt.into();
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

/// Populate and open a reserved run with its history and prompt. Stop if a host
/// observer removes the run during insertion; content errors mark the run failed.
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
    let Ok(mut entity) = world.get_entity_mut(run) else {
        return;
    };
    entity.insert((Run, RunOf(agent), StreamRequested(streamed), prompt));
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
                entity
                    .remove::<Prompt>()
                    .insert(Failed(Failure::Content(error)));
            }
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

/// Open ready, phaseless, nonterminal runs before advancement. Consumes their
/// prompts into history and starts assembly, or loads memory first when the agent
/// remembers and no history was supplied. Content errors fail the affected run.
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
        || entity.contains::<RunPhase>()
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
        world
            .entity_mut(run)
            .insert(Failed(Failure::Content(error)));
        return;
    }
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
            world.entity_mut(run).insert((
                RunPhase::LoadingMemory,
                Remembering,
                Conversation(conversation.clone()),
            ));
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
            world.entity_mut(run).insert(RunPhase::Assembling);
        }
    }
}

/// Spawn one utterance `ChildOf` `run`, last among its siblings.
/// Returns content conversion errors and removes the new utterance on failure.
pub fn spawn_utterance(
    world: &mut World,
    run: Entity,
    parts: MessageParts,
) -> Result<Entity, ContentError> {
    let entity = world.spawn((Utterance, ChildOf(run))).id();
    if let Err(error) = write_message(world, entity, parts) {
        world.despawn(entity);
        return Err(error);
    }
    Ok(entity)
}

/// A run's effective setting: its own component, else its agent's.
fn setting<'a, C: Component>(run: Entity, agent: Entity, query: &'a Query<&C>) -> Option<&'a C> {
    query.get(run).ok().or_else(|| query.get(agent).ok())
}

/// The links of one kind under `owner`, in sibling (`Children`) order.
fn links_in_order<'a, L: Component, F: bevy_ecs::query::QueryFilter>(
    owner: Entity,
    children: &Query<&Children>,
    links: &'a Query<&L, F>,
) -> Vec<&'a L> {
    children
        .get(owner)
        .map(|children| {
            children
                .iter()
                .filter_map(|child| links.get(child).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Advance ready, assembling runs in sequence order, failing exhausted turn budgets.
/// Creates a fresh turn with ordered grants and context, or a retrieving turn
/// whose links are populated after retrieval. Commit holds postpone advancement;
/// provider retries do not consume the turn budget.
pub fn advance(
    mut commands: Commands,
    runs: Query<(Entity, &RunOf, &Cursor, &RunSeq, &RunPhase), Wanting>,
    fresh: Query<&ChildOf, With<Fresh>>,
    children: Query<&Children>,
    grants: Query<&Grant, Without<Retrievable>>,
    contexts: Query<&Context>,
    retrievals: Query<(), With<Retrieves>>,
    max_turns: Query<&MaxTurns>,
    retrying: Query<(), With<ProviderRetrying>>,
    provider_retried: Query<&ProviderRetried>,
    holds: Query<&ToolTurnHolds>,
    commits: Query<(&ChildOf, &ToolTurnCommit)>,
) {
    let mut runs: Vec<_> = runs
        .iter()
        .filter(|(_, _, _, _, phase)| **phase == RunPhase::Assembling)
        .collect();
    runs.sort_by_key(|(_, _, _, seq, _)| **seq);
    for (run, RunOf(agent), cursor, _, _) in runs {
        if fresh.iter().any(|child_of| child_of.parent() == run) {
            continue;
        }
        // A retried attempt re-issues a turn the cursor already counted
        // (CONTRACT §5): it neither checks nor spends the model-call budget.
        let retrying = retrying.get(run).is_ok();
        let limit = setting(run, *agent, &max_turns).map_or(1, |limit| limit.0);
        if !retrying && cursor.turn >= limit {
            commands
                .entity(run)
                .end(Failed(Failure::MaxTurns { limit }));
            continue;
        }
        if holds.get(run).is_ok_and(|holds| {
            commits
                .iter()
                .any(|(parent, commit)| parent.parent() == run && holds.blocks(commit.turn))
        }) {
            continue;
        }
        let turn = commands.spawn((Turn, Fresh, ChildOf(run))).id();
        let retrieves = children
            .get(*agent)
            .map(|children| children.iter().any(|child| retrievals.get(child).is_ok()))
            .unwrap_or(false);
        if retrieves {
            commands.entity(turn).insert(Retrieving);
        } else {
            for Grant(tool) in links_in_order(*agent, &children, &grants) {
                commands.spawn((Advert(*tool), ChildOf(turn)));
            }
            for Context(document) in links_in_order(*agent, &children, &contexts) {
                commands.spawn((Attachment(*document), ChildOf(turn)));
            }
        }
        if retrying {
            commands.entity(run).remove::<ProviderRetrying>();
            commands.entity(turn).insert(backoff::RetryAttempt(
                provider_retried.get(run).map_or(1, |n| n.0),
            ));
        } else {
            commands.entity(run).insert(Cursor {
                turn: cursor.turn + 1,
            });
        }
    }
}

/// A fresh turn whose retrievals landed gets its adverts and attachments
/// (CONTRACT §12): the retrieved tools first, in result order, then the
/// static grants; the static attachments, then one document entity per
/// result (an existing entity with that id reused). Runs after `Advance`
/// and before `Select`; `gather_turn` waits for it.
pub fn attach_retrieved(
    mut commands: Commands,
    turns: Query<(Entity, &ChildOf), RetrievingTurn>,
    runs: Query<&RunOf>,
    children: Query<&Children>,
    retrievals: Query<(&PendingEffect, &Retrieval, Option<&EffectOutcome>)>,
    grants: Query<(&Grant, Has<Retrievable>)>,
    contexts: Query<&Context>,
    bound: Query<&Bound>,
    indexes: Query<&Retrieves, With<Retrieval>>,
    documents: Query<(Entity, &DocumentId)>,
) {
    for (turn, turn_of) in &turns {
        let run = turn_of.parent();
        let Ok(RunOf(agent)) = runs.get(run) else {
            continue;
        };
        let effects: Vec<(&Retrieval, Option<&EffectOutcome>)> = children
            .get(turn)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| retrievals.get(child).ok())
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
        let links: Vec<(&Grant, bool)> = children
            .get(*agent)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| grants.get(child).ok())
                    .collect()
            })
            .unwrap_or_default();
        let tool_named = |name: &str| -> Option<Entity> {
            links.iter().find_map(|(Grant(tool), _)| {
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
                commands.spawn((Advert(tool), ChildOf(turn)));
            }
        }
        for (Grant(tool), retrievable) in &links {
            if !retrievable {
                commands.spawn((Advert(*tool), ChildOf(turn)));
            }
        }
        for Context(document) in links_in_order(*agent, &children, &contexts) {
            commands.spawn((Attachment(*document), ChildOf(turn)));
        }
        for (id, text) in retrieved_documents {
            let document = documents
                .iter()
                .find(|(_, existing)| existing.0 == id)
                .map(|(entity, _)| entity)
                .unwrap_or_else(|| commands.spawn((DocumentId(id), DocumentText(text))).id());
            commands.spawn((Attachment(document), ChildOf(turn)));
        }
        commands.entity(turn).remove::<Retrieving>();
    }
}

/// Materialise loaded memory before the prompt, marking imported utterances
/// remembered and starting assembly. Failed loads, wrong outcome families, and
/// content conversion errors fail the affected run.
pub fn land_memory(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    runs: Query<(Entity, &RunPhase), Without<Failed>>,
    children: Query<&Children>,
    loads: Query<(&PendingEffect, &EffectOutcome)>,
) {
    for (run, _) in runs
        .iter()
        .filter(|(_, phase)| **phase == RunPhase::LoadingMemory)
    {
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
            continue;
        };
        match outcome {
            Ok(Outcome::Memory(rig_core::effect::MemoryOutcome::Loaded { messages })) => {
                if let Err(error) = land_loaded(&mut commands, &mut assets, run, messages) {
                    fail_content(&mut commands, run, error);
                }
            }
            Ok(other) => {
                commands.entity(run).end(Failed(Failure::Memory(
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
                    .end(Failed(Failure::Memory(report.clone())));
            }
        }
    }
}

/// The loaded messages become `run`'s first utterances, each `Remembered`,
/// and the run is `Assembling`.
fn land_loaded(
    commands: &mut Commands,
    assets: &mut BinaryAssets,
    run: Entity,
    messages: &[rig_core::completion::Message],
) -> Result<(), ContentError> {
    let mut loaded = Vec::with_capacity(messages.len());
    for message in messages {
        if let Some(parts) = MessageParts::from_message(message) {
            let utterance = spawn_deferred(commands, assets, run, parts)?;
            commands.entity(utterance).insert(Remembered);
            loaded.push(utterance);
        }
    }
    if !loaded.is_empty() {
        commands.entity(run).insert_children(0, &loaded);
    }
    commands.entity(run).phase(RunPhase::Assembling);
    Ok(())
}

/// Schedule a settled remembering run's ordered, nonremembered utterances for
/// memory append. Content errors fail the run. A persisted marker prevents
/// scheduling a second append after checkpoint restoration.
pub fn append_memory(
    mut commands: Commands,
    settled: Query<(Entity, &RunOf, &Conversation), NeedsMemoryAppend>,
    memories: Query<&Remembers>,
    bound: Query<&Bound>,
    children: Query<&Children>,
    utterances: Query<(Entity, Has<Remembered>), With<Utterance>>,
    content: ContentGraph,
) {
    for (run, RunOf(agent), Conversation(conversation)) in &settled {
        let Some(key) = memories
            .get(*agent)
            .ok()
            .and_then(|Remembers(memory)| bound.get(*memory).ok())
            .map(|bound| bound.key.clone())
        else {
            continue;
        };
        let said: Result<Vec<MessageParts>, ContentError> = children
            .get(run)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| utterances.get(child).ok())
                    .filter(|(_, remembered)| !*remembered)
                    .map(|(entity, _)| content.message(entity))
                    .collect()
            })
            .unwrap_or_else(|_| Ok(Vec::new()));
        let said = match said {
            Ok(said) => said,
            Err(error) => {
                fail_content(&mut commands, run, error);
                continue;
            }
        };
        commands.spawn((
            PendingEffect::new(
                key,
                EffectKind::Memory {
                    op: rig_core::effect::MemoryOp::Append {
                        conversation: rig_core::id::ConversationId::from(conversation.as_str()),
                        messages: said.into_iter().map(|parts| parts.to_message()).collect(),
                    },
                },
            ),
            ChildOf(run),
        ));
        commands.entity(run).insert(MemoryAppendScheduled);
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

/// A request edit link, including a malformed link for explicit validation;
/// the links apply in the turn's sibling (`Children`) order.
#[derive(QueryData)]
pub struct PartEdit {
    /// The link entity.
    pub link: Entity,
    /// The part the edit targets; a link without one is malformed.
    pub target: Option<&'static EditTarget>,
    /// The edit.
    pub edit: &'static RequestPartEdit,
}

/// A turn's request part edits, gathered: by target, the links consumed,
/// the utterances they touch.
struct PartEdits {
    edits: std::collections::BTreeMap<Entity, RequestPartEdit>,
    consumed: Vec<Entity>,
    edited: std::collections::HashSet<Entity>,
}

/// The part edits linked to `turn`, in link order. A link without a
/// target is `Missing`; a target outside `run`, or any edit beside a
/// patched history (a replacement history has no stable entity identity:
/// conflicting operations are refused, never silently dropped), is `Shape`.
fn part_edits_of(
    turn: Entity,
    run: Entity,
    patch: Option<&RequestPatch>,
    children: &Query<&Children>,
    part_edits: &Query<PartEdit>,
    content: &ContentGraph,
) -> Result<PartEdits, ContentError> {
    let mut gathered = PartEdits {
        edits: std::collections::BTreeMap::new(),
        consumed: Vec::new(),
        edited: std::collections::HashSet::new(),
    };
    let links = children
        .get(turn)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter_map(|link| part_edits.get(link).ok());
    for PartEditItem { link, target, edit } in links {
        let target = target.ok_or(ContentError::Missing)?;
        let utterance = content.target_utterance(target.0)?;
        gathered.edited.insert(utterance);
        if !children
            .get(run)
            .is_ok_and(|owned| owned.contains(&utterance))
            || patch.is_some_and(|patch| patch.history.is_some())
        {
            return Err(ContentError::Shape);
        }
        gathered.edits.insert(target.0, edit.clone());
        gathered.consumed.push(link);
    }
    Ok(gathered)
}

/// `run`'s utterances in order, as DTOs, under the part edits linked to
/// `turn` ([`part_edits_of`]): an edited one rendered with its edit, any
/// other rendered plain. With them, the edit links consumed.
fn render_history(
    turn: Entity,
    run: Entity,
    patch: Option<&RequestPatch>,
    children: &Query<&Children>,
    utterances: &Query<Entity, With<Utterance>>,
    part_edits: &Query<PartEdit>,
    content: &ContentGraph,
) -> Result<(Vec<MessageParts>, Vec<Entity>), ContentError> {
    let edits = part_edits_of(turn, run, patch, children, part_edits, content)?;
    let history = children
        .get(run)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter_map(|child| utterances.get(child).ok())
        .map(|entity| {
            if edits.edited.contains(&entity) {
                content.message_with(entity, &edits.edits)
            } else {
                content.message(entity)
            }
        })
        .collect::<Result<_, _>>()?;
    Ok((history, edits.consumed))
}

/// The first pass over a retrieving turn (CONTRACT §12): one `Retrieve`
/// effect per index of the agent (`indexes`, in link order), `ChildOf`
/// the turn; the fold waits for `attach_retrieved`.
fn spawn_retrievals<'a>(
    commands: &mut Commands,
    turn: Entity,
    history: &[MessageParts],
    indexes: impl Iterator<Item = (&'a Retrieves, &'a Retrieval)>,
    bound: &Query<&Bound>,
) {
    let query = policy::retrieval_query(history);
    for (Retrieves(index), retrieval) in indexes {
        let Ok(index) = bound.get(*index) else {
            continue;
        };
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
}

/// The tools `turn` advertises that the patch allows, in advert order, as
/// handler entities with their bindings. An advert the patch's
/// `active_tools` excludes is despawned: the request and the executable
/// grant set must agree, so materialisation, invalid-call repair and scene
/// continuation see the same surface.
fn allowed_tools<'a>(
    commands: &mut Commands,
    turn: Entity,
    patch: Option<&RequestPatch>,
    children: &Query<&Children>,
    adverts: &Query<&Advert>,
    bound: &'a Query<&Bound>,
) -> Vec<(Entity, &'a Bound)> {
    children
        .get(turn)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter_map(|link| {
            let Advert(tool) = adverts.get(link).ok()?;
            Some((link, *tool, bound.get(*tool).ok()?))
        })
        .filter_map(|(link, tool, bound)| {
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
                Some((tool, bound))
            } else {
                commands.entity(link).despawn();
                None
            }
        })
        .collect()
}

/// The run's bound completion model, or why the run cannot fold: no
/// selected model, no binding, or a binding that does not serve
/// completions (a provider `HandlerUnavailable` report: the run never
/// silently waits).
fn completion_model(
    model: Option<&UsesModel>,
    bound: &Query<&Bound>,
) -> Result<(Entity, bool), Failure> {
    let unavailable = |message: String| {
        Failure::Provider(rig_core::error::ErrorReport::new(
            rig_core::error::ErrorKind::HandlerUnavailable,
            message,
        ))
    };
    let Some((model, model_bound)) =
        model.and_then(|UsesModel(model)| bound.get(*model).ok().map(|bound| (*model, bound)))
    else {
        return Err(unavailable(
            "the run has no bound completion model; its selected model or its agent's binding was removed".to_owned(),
        ));
    };
    match &model_bound.descriptor.family {
        FamilyDescriptor::Completion { capabilities, .. } => {
            Ok((model, capabilities.composes_native_output_with_tools))
        }
        FamilyDescriptor::Tool { .. }
        | FamilyDescriptor::Embed { .. }
        | FamilyDescriptor::Rerank { .. }
        | FamilyDescriptor::Memory { .. }
        | FamilyDescriptor::Retrieve { .. }
        | FamilyDescriptor::Custom { .. } => Err(unavailable(format!(
            "selected model `{}` does not serve completions",
            model_bound.key
        ))),
    }
}

/// The output mode of a turn, resolved, with the output tool's name: a
/// committed name (minted on an earlier turn) stays the mode whatever this
/// turn's choice says (CONTRACT §9.3); a reserved name with a schema is
/// `Tool`; else the policy decides from the mode, the schema, the tools,
/// the choice and the model's capabilities. `Err(name)` is a collision:
/// the output tool's name is also a granted tool's.
fn resolve_output_tool(
    minted: &OutputToolName,
    reserved_name: Option<&str>,
    output: &Output,
    tool_choice: Option<&ToolChoice>,
    tools: &[(Entity, &Bound)],
    access: &ToolAccess,
    composes: bool,
) -> Result<(OutputKind, String), String> {
    let granted_names: Vec<&str> = tools
        .iter()
        .filter_map(|(_, bound)| match &bound.descriptor.family {
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
        .chain(
            access
                .executable
                .iter()
                .flat_map(|executable| executable.keys().map(String::as_str)),
        )
        .collect();
    let output_tool = minted
        .0
        .clone()
        .or_else(|| reserved_name.map(str::to_owned))
        .unwrap_or_else(|| policy::output_tool_name(&occupied_names));
    let callable = policy::output_tool_callable(tool_choice, &output_tool);
    let resolved = if minted.0.is_some() || (reserved_name.is_some() && output.schema.is_some()) {
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
        return Err(output_tool);
    }
    Ok((resolved, output_tool))
}

/// The documents attached to `turn`, in link order, then the patch's
/// extra context.
fn attached_documents(
    turn: Entity,
    patch: Option<&RequestPatch>,
    children: &Query<&Children>,
    attachments: &Query<&Attachment>,
    documents: &Query<(&DocumentId, &DocumentText, Option<&DocumentProps>)>,
) -> Vec<rig_core::completion::Document> {
    links_in_order(turn, children, attachments)
        .into_iter()
        .filter_map(|Attachment(document)| documents.get(*document).ok())
        .map(|(id, text, props)| rig_core::completion::Document {
            id: id.0.clone(),
            text: text.0.clone(),
            additional_props: props.map(|props| props.0.clone()).unwrap_or_default(),
        })
        .chain(
            patch
                .into_iter()
                .flat_map(|patch| patch.extra_context.iter().cloned()),
        )
        .collect()
}

/// The run's `ToolAccess` (else the agent's, else none) completed for the
/// turn: `executable` defaults to the advertised tools by name, `allowed`
/// to every executable name.
fn tool_access_for(
    run: Entity,
    agent: Entity,
    tools: &[(Entity, &Bound)],
    tool_access: &Query<&ToolAccess>,
) -> ToolAccess {
    let mut access = setting(run, agent, tool_access)
        .cloned()
        .unwrap_or_default();
    let executable = access.executable.get_or_insert_with(|| {
        tools
            .iter()
            .filter_map(|(_, bound)| match &bound.descriptor.family {
                FamilyDescriptor::Tool { name, .. } => Some((name.clone(), bound.key.clone())),
                _ => None,
            })
            .collect()
    });
    if access.allowed.is_none() {
        access.allowed = Some(executable.keys().cloned().collect());
    }
    access
}

/// The request settings of one turn, resolved: the patch's over the
/// run's over the agent's.
struct Resolved {
    preamble: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
    tool_choice: Option<ToolChoice>,
    output: Output,
    output_tool_config: Option<OutputToolConfig>,
}

impl Settings<'_, '_> {
    /// The settings for `run`'s turn under `patch` (CONTRACT §9.3): the
    /// patch's field where it has one (an object `additional_params`
    /// merges over the setting's), else the run's, else the agent's.
    fn resolve(&self, run: Entity, agent: Entity, patch: Option<&RequestPatch>) -> Resolved {
        let additional_params = match (
            setting(run, agent, &self.params).and_then(|p| p.0.clone()),
            patch.and_then(|p| p.additional_params.clone()),
        ) {
            (Some(base), Some(patched)) if base.is_object() && patched.is_object() => {
                Some(rig_core::json_utils::merge(base, patched))
            }
            (base, patched) => patched.or(base),
        };
        Resolved {
            preamble: patch
                .and_then(|p| p.preamble.clone())
                .or_else(|| setting(run, agent, &self.preambles).and_then(|p| p.0.clone())),
            temperature: patch
                .and_then(|p| p.temperature)
                .or_else(|| setting(run, agent, &self.temperatures).and_then(|t| t.0)),
            max_tokens: patch
                .and_then(|p| p.max_tokens)
                .or_else(|| setting(run, agent, &self.max_tokens).and_then(|m| m.0)),
            additional_params,
            tool_choice: patch
                .and_then(|p| p.tool_choice.clone())
                .or_else(|| setting(run, agent, &self.choices).and_then(|c| c.0.clone())),
            output: setting(run, agent, &self.outputs)
                .cloned()
                .unwrap_or_default(),
            output_tool_config: setting(run, agent, &self.output_tools).cloned(),
        }
    }
}

/// Gather fresh turns in run order into [`AssemblyInputs`] and retained tool
/// access snapshots. Applies request edits, size limits, settings, and output
/// policy; retrieving turns instead dispatch their initial retrievals.
/// Missing or incompatible models, output-tool collisions, and content errors
/// fail the affected run.
pub fn gather_turn(
    mut commands: Commands,
    fresh: Query<FreshTurn, With<Fresh>>,
    runs: Query<AssemblingRun, LiveRun>,
    children: Query<&Children>,
    utterances: Query<Entity, With<Utterance>>,
    content: ContentGraph,
    part_edits: Query<PartEdit>,
    retrievals: Query<(&Retrieves, &Retrieval)>,
    retrieving: Query<(), With<Retrieval>>,
    adverts: Query<&Advert>,
    attachments: Query<&Attachment>,
    documents: Query<(&DocumentId, &DocumentText, Option<&DocumentProps>)>,
    bound: Query<&Bound>,
    settings: Settings,
) {
    let mut turns: Vec<_> = fresh
        .iter()
        .filter_map(|turn| {
            let run = turn.turn_of.parent();
            runs.get(run).ok().map(|view| (view.seq.0, turn, run))
        })
        .collect();
    turns.sort_by_key(|(seq, _, _)| *seq);
    for (_, turn, run) in turns {
        let Ok(view) = runs.get(run) else {
            continue;
        };
        let agent = view.run_of.0;
        let (model, composes) = match completion_model(view.model, &bound) {
            Ok(model) => model,
            Err(failure) => {
                commands.entity(run).end(Failed(failure));
                commands.entity(turn.entity).remove::<Fresh>();
                continue;
            }
        };
        let patch = turn.patch;
        let rendered = render_history(
            turn.entity,
            run,
            patch,
            &children,
            &utterances,
            &part_edits,
            &content,
        );
        let (mut history, consumed_edits) = match rendered {
            Ok(rendered) => rendered,
            Err(error) => {
                fail_content(&mut commands, run, error);
                continue;
            }
        };
        if turn.retrieving {
            let spawned = children
                .get(turn.entity)
                .is_ok_and(|children| children.iter().any(|child| retrieving.get(child).is_ok()));
            if !spawned {
                let indexes = children
                    .get(agent)
                    .into_iter()
                    .flat_map(|children| children.iter())
                    .filter_map(|child| retrievals.get(child).ok());
                spawn_retrievals(&mut commands, turn.entity, &history, indexes, &bound);
            }
            continue;
        }
        // Limit only rendered request text so persisted history remains lossless.
        if let Some(limit) = setting(run, agent, &settings.tool_result_limits) {
            for parts in &mut history {
                if policy::tool_results_exceed(parts, limit) {
                    policy::limit_tool_results(parts, limit);
                }
            }
        }
        let tools = allowed_tools(
            &mut commands,
            turn.entity,
            patch,
            &children,
            &adverts,
            &bound,
        );
        let attached = attached_documents(turn.entity, patch, &children, &attachments, &documents);
        // History replacement must retain the final utterance as the current prompt.
        if let Some(patched) = patch.and_then(|p| p.history.as_ref()) {
            let prompt = history.pop();
            history = patched.iter().cloned().chain(prompt).collect();
        }
        let resolved = settings.resolve(run, agent, patch);
        let reserved_name = resolved
            .output_tool_config
            .as_ref()
            .and_then(|config| config.name.as_deref());
        let access = tool_access_for(run, agent, &tools, &settings.tool_access);
        let (mode, output_tool) = match resolve_output_tool(
            view.minted,
            reserved_name,
            &resolved.output,
            resolved.tool_choice.as_ref(),
            &tools,
            &access,
            composes,
        ) {
            Ok(mode) => mode,
            Err(name) => {
                commands.entity(turn.entity).remove::<Fresh>();
                commands
                    .entity(run)
                    .end(Failed(Failure::OutputToolCollision { name }));
                continue;
            }
        };
        if mode == OutputKind::Tool && view.minted.0.is_none() {
            commands
                .entity(run)
                .insert(OutputToolName(Some(output_tool.clone())));
        }
        for link in consumed_edits {
            commands.entity(link).despawn();
        }
        let Resolved {
            preamble,
            temperature,
            max_tokens,
            additional_params,
            tool_choice,
            output,
            output_tool_config,
        } = resolved;
        commands.entity(turn.entity).insert((
            access,
            AssemblyInputs {
                model,
                stream: view.stream.0,
                preamble,
                utterances: history,
                documents: attached,
                tools: tools.into_iter().map(|(tool, _)| tool).collect(),
                temperature,
                max_tokens,
                additional_params,
                tool_choice,
                output: mode,
                schema: output.schema,
                output_tool: (mode == OutputKind::Tool).then_some(output_tool),
                output_tool_config,
            },
        ));
    }
}

/// Fold gathered inputs in run order into completion effects owned by their turns.
/// Consumes fresh-turn inputs and patches, initializes outputs, and moves runs
/// to `AwaitingModel`. Turns whose model binding is absent remain unchanged.
pub fn fold_turn(
    mut commands: Commands,
    mut turns: Query<(Entity, &ChildOf, &mut AssemblyInputs), With<Fresh>>,
    runs: Query<&RunSeq, LiveRun>,
    bound: Query<&Bound>,
) {
    let mut turns: Vec<_> = turns
        .iter_mut()
        .filter_map(|(turn, turn_of, inputs)| {
            let run = turn_of.parent();
            runs.get(run).ok().map(|seq| (seq.0, turn, run, inputs))
        })
        .collect();
    turns.sort_by_key(|(seq, _, _, _)| *seq);
    for (_, turn, run, mut inputs) in turns {
        let Ok(model) = bound.get(inputs.model) else {
            continue;
        };
        let documents = std::mem::take(&mut inputs.documents);
        let graph = RequestGraph {
            preamble: inputs.preamble.as_deref(),
            utterances: inputs.utterances.iter().collect(),
            documents,
            tools: inputs
                .tools
                .iter()
                .filter_map(|tool| bound.get(*tool).ok())
                .map(|bound| &bound.descriptor)
                .collect(),
            temperature: inputs.temperature,
            max_tokens: inputs.max_tokens,
            additional_params: inputs.additional_params.as_ref(),
            tool_choice: inputs.tool_choice.as_ref(),
            output: inputs.output,
            schema: inputs.schema.as_ref(),
            output_tool: inputs.output_tool.as_deref(),
            output_tool_config: inputs.output_tool_config.as_ref(),
        };
        let request = policy::fold_request(&graph);
        commands.spawn((
            PendingEffect::new(
                model.key.clone(),
                EffectKind::Completion {
                    request,
                    stream: inputs.stream,
                },
            ),
            ServedBy(inputs.model),
            ChildOf(turn),
        ));
        commands
            .entity(turn)
            .remove::<(AssemblyInputs, Fresh, RequestPatch)>()
            .insert((Folded(inputs.output), Outputs::default()));
        commands.entity(run).phase(RunPhase::AwaitingModel);
    }
}

/// Update turn outputs from streamed text or final outcomes. Changed outputs
/// signal progress; completed streamed responses use canonical content order.
pub fn fold(effects: Query<EffectView, NotRetrieval>, mut turns: Query<&mut Outputs, With<Turn>>) {
    for EffectViewItem {
        turn_of,
        streamed,
        outcome,
    } in &effects
    {
        let Ok(mut outputs) = turns.get_mut(turn_of.parent()) else {
            continue;
        };
        if outputs.done {
            continue;
        }
        match outcome {
            Some(EffectOutcome(Ok(Outcome::Completion(response)))) => {
                // Wire arrival order is not conversation order; commit reasoning,
                // text, and calls in the canonical sequence.
                outputs.content = if streamed.is_some() {
                    canonical_streamed_choice(response.choice.clone())
                } else {
                    response.choice.clone()
                };
                outputs.message_id = response.message_id.clone();
                outputs.done = true;
            }
            Some(EffectOutcome(Ok(_))) | Some(EffectOutcome(Err(_))) => {
                outputs.done = true;
            }
            None => {
                if let Some(streamed) = streamed
                    && !streamed.text.is_empty()
                {
                    let current = policy::answer_text(&outputs.content);
                    if current != streamed.text {
                        outputs.content = vec![AssistantContent::text(&streamed.text)];
                    }
                }
            }
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

/// What the batch's systems read of a tool child of a turn: which call it
/// is, whether it was issued, its outcome, whether the batch's own hold
/// is on it.
#[derive(QueryData)]
pub struct ToolChild {
    /// The effect entity.
    pub entity: Entity,
    /// Which call of the batch it is.
    pub slot: &'static ToolCallSlot,
    /// Whether the bus issued it.
    pub issued: Has<Issued>,
    /// The outcome, once landed.
    pub outcome: Option<&'static EffectOutcome>,
    /// Whether the batch's own hold is on it.
    pub batch_held: Has<BatchHeld>,
}

/// The runtime's own hold on a tool child beyond the run's concurrency,
/// placed beside `Held` at spawn and lifted by `release_batch` in call
/// order as earlier calls land. The marker says whose hold it is: a
/// `Gate` policy's `Held` is not the runtime's to lift, so a call a policy
/// holds stays held until that policy releases it, and a call the batch
/// holds is released by the batch alone.
#[derive(Component, Debug, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct BatchHeld;

/// Remove batch ownership and its accounting marker when a hold is removed.
/// Hosts removing `Held` bypass all owners; clearing the marker keeps slot
/// accounting and checkpoint state consistent with that release.
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

/// The tool children of `turn`, by call index.
fn batch_children<'a>(
    turn: Entity,
    children: &Query<&Children>,
    tools: &'a Query<ToolChild>,
) -> Vec<ToolChildItem<'a, 'a>> {
    let mut found: Vec<_> = children
        .get(turn)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter_map(|child| tools.get(child).ok())
        .collect();
    found.sort_by_key(|child| child.slot.index);
    found
}

/// Release batch-owned holds in call order up to the run's tool concurrency.
/// Other policies' holds remain intact and consume slots. A terminal tool
/// failure prevents further releases while already-started calls drain.
pub fn release_batch(
    mut commands: Commands,
    turns: Query<(Entity, &ChildOf), With<Batch>>,
    runs: Query<&RunOf>,
    policies: Query<&ToolPolicy>,
    children: Query<&Children>,
    tools: Query<ToolChild>,
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
        if batch.iter().any(|child| {
            child
                .outcome
                .is_some_and(|o| policy::tool_failure(&o.0).is_some())
        }) {
            continue;
        }
        // Other policies' holds still occupy slots once the batch releases its hold.
        let active = batch
            .iter()
            .filter(|child| !child.batch_held && child.outcome.is_none())
            .count();
        let mut free = concurrency.saturating_sub(active);
        for child in &batch {
            if free == 0 {
                break;
            }
            if child.batch_held && !child.issued {
                let entity = child.entity;
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

/// Commit a completed tool batch as one user utterance in call order, then
/// resume assembly or settle an accompanying output-tool call with its arguments.
/// Terminal failures wait for started calls to drain, fail the run, and despawn
/// unstarted calls without dispatch or records. Content errors fail the run.
pub fn land_batch(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    turns: Query<(Entity, &ChildOf, &Batch, &Outputs)>,
    runs: Query<(&OutputToolName, &RunSeq, &Cursor, &RunPhase)>,
    children: Query<&Children>,
    tools: Query<ToolChild>,
) {
    let mut turns: Vec<_> = turns.iter().collect();
    turns.sort_by_key(|(_, turn_of, _, _)| {
        runs.get(turn_of.parent()).map(|(_, seq, _, _)| *seq).ok()
    });
    for (turn, turn_of, batch, outs) in turns {
        let run = turn_of.parent();
        let Ok((minted, _, cursor, &RunPhase::ResolvingTools)) = runs.get(run) else {
            continue;
        };
        let calls = batch_children(turn, &children, &tools);
        let failure = calls
            .iter()
            .find_map(|child| child.outcome.and_then(|o| policy::tool_failure(&o.0)));
        let started_landed = calls
            .iter()
            .all(|child| !child.issued || child.outcome.is_some());
        if let Some(failure) = failure {
            if !started_landed {
                continue;
            }
            // The ending first, so the despawns' observer finds the run
            // ended with this failure and leaves it.
            commands.entity(turn).remove::<Batch>();
            commands.entity(run).end(Failed(failure));
            for child in &calls {
                if !child.issued && child.outcome.is_none() {
                    commands.entity(child.entity).despawn();
                }
            }
            continue;
        }
        if calls.len() < batch.calls || calls.iter().any(|child| child.outcome.is_none()) {
            continue;
        }
        let mut parts = Vec::with_capacity(calls.len());
        let mut statuses = Vec::with_capacity(calls.len());
        let mut failed = None;
        for child in &calls {
            let Some(EffectOutcome(outcome)) = child.outcome else {
                continue;
            };
            let slot = child.slot;
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
            commands.entity(run).end(Failed(failure));
            continue;
        }
        let results = MessageParts::User { content: parts };
        let results_entity =
            match spawn_deferred_with(&mut commands, &mut assets, run, results, statuses) {
                Ok(results) => results,
                Err(error) => {
                    fail_content(&mut commands, run, error);
                    continue;
                }
            };
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
                commands.entity(run).end((RunResult(arguments), Settled));
            }
            None => {
                commands.entity(run).phase(RunPhase::Assembling);
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

/// The landed completion effect of `turn`, if any.
fn landed_effect<'a>(
    turn: Entity,
    effects: &'a Query<LandedEffect, NotRetrieval>,
) -> Option<LandedEffectItem<'a, 'a>> {
    effects
        .iter()
        .find(|effect| effect.turn_of.parent() == turn)
}

/// The tools `turn` may call: the run's `ToolAccess.executable` when the
/// turn was folded under one, else the tools it advertised, by name, with
/// their keys and their handler entities.
fn granted_tools(
    turn: Entity,
    children: &Query<&Children>,
    adverts: &Query<&Advert>,
    bound: &Query<&Bound>,
    access: Option<&ToolAccess>,
) -> Vec<GrantedTool> {
    if let Some(executable) = access.and_then(|access| access.executable.as_ref()) {
        return executable
            .iter()
            .map(|(name, key)| GrantedTool {
                name: name.clone(),
                key: key.clone(),
                handler: None,
            })
            .collect();
    }
    links_in_order(turn, children, adverts)
        .into_iter()
        .filter_map(|Advert(tool)| bound.get(*tool).ok().map(|bound| (*tool, bound)))
        .filter_map(|(tool, bound)| match &bound.descriptor.family {
            FamilyDescriptor::Tool { name, .. } => Some(GrantedTool {
                name: name.clone(),
                key: bound.key.clone(),
                handler: Some(tool),
            }),
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Memory { .. }
            | FamilyDescriptor::Retrieve { .. }
            | FamilyDescriptor::Custom { .. } => None,
        })
        .collect()
}

/// Count a landed completion's usage once before deferred invalid-call decisions
/// consume the turn, including responses completed after an early stream decision.
pub fn record_usage(
    mut commands: Commands,
    mut turns: Query<(Entity, &ChildOf, &mut Outputs), Unread>,
    effects: Query<LandedEffect, NotRetrieval>,
    runs: Query<(&Usage, &RunPhase)>,
) {
    for (turn, turn_of, mut outs) in &mut turns {
        if outs.usage_recorded {
            continue;
        }
        let run = turn_of.parent();
        let Ok((usage, &RunPhase::AwaitingModel)) = runs.get(run) else {
            continue;
        };
        let Some(EffectOutcome(Ok(Outcome::Completion(response)))) =
            landed_effect(turn, &effects).map(|effect| effect.outcome)
        else {
            continue;
        };
        commands.entity(run).insert(Usage(usage.0 + response.usage));
        outs.usage_recorded = true;
    }
}

/// The pending invalid calls of `turn`, each with its resolution, a
/// streamed call's id resolved to the block's final identity once the
/// stream landed.
fn pending_invalid_calls(
    turn: Entity,
    invalid_calls: &Query<(Entity, &ChildOf, &InvalidCall, &Resolution)>,
    stream: Option<&BusStreamed>,
) -> Vec<(Entity, InvalidCall, Resolution)> {
    invalid_calls
        .iter()
        .filter(|(_, child_of, _, _)| child_of.parent() == turn)
        .map(|(entity, _, call, resolution)| {
            let mut call = call.clone();
            if let Some(offset) = call.stream_offset
                && let Some(stream) = stream
                && let Some(id) = stream_invalid::completed_call_id(&stream.events, offset)
            {
                call.id = id;
            }
            (entity, call, resolution.clone())
        })
        .collect()
}

/// The turn's content after its invalid calls' edits: repairs rename their
/// call; ignores drop theirs. What is left is the turn.
fn edited_content(
    content: &[AssistantContent],
    pending: &[(Entity, InvalidCall, Resolution)],
) -> Vec<AssistantContent> {
    let mut content = content.to_vec();
    for (_, call, resolution) in pending {
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
                    AssistantContent::ToolCall(tool_call) => tool_call.id != call.id,
                    AssistantContent::Text(_)
                    | AssistantContent::Reasoning(_)
                    | AssistantContent::Image(_) => true,
                });
            }
            Resolution::Fail | Resolution::Retry { .. } | Resolution::Skip { .. } => {}
        }
    }
    content
}

/// An invalid call's `Fail`: the delivered prefix (if any) becomes history,
/// the turn is read, and the run fails `UnknownToolCall`.
fn fail_unknown_call(
    commands: &mut Commands,
    assets: &mut BinaryAssets,
    run: Entity,
    turn: Entity,
    outs: &Outputs,
    call: InvalidCall,
) -> Result<(), ContentError> {
    commands.entity(turn).insert(Materialised);
    if !call.prefix.is_empty() {
        let assistant = MessageParts::Assistant {
            id: outs.message_id.clone(),
            content: call.prefix.clone(),
        };
        spawn_deferred(commands, assets, run, assistant)?;
    }
    commands
        .entity(run)
        .end(Failed(Failure::UnknownToolCall { name: call.name }));
    Ok(())
}

/// An invalid call's `Retry` or `Skip`: the turn up to the call becomes
/// history (a streamed turn is abandoned where the call surfaced), then
/// one user utterance answering the call and its peers with the feedback,
/// every result `Skipped`; the turn is read and the run wants another
/// (a retry spends one of `InvalidRetries`).
fn abandon_turn(
    commands: &mut Commands,
    assets: &mut BinaryAssets,
    run: Entity,
    turn: Entity,
    outs: &Outputs,
    events: Option<&[rig_core::streaming::StreamEvent]>,
    allowed_names: &[String],
    call: &InvalidCall,
    feedback: &str,
    retries: Option<InvalidRetries>,
) -> Result<(), ContentError> {
    commands.entity(turn).insert(Materialised);
    let (content, diagnostic_id) =
        if let Some(AssistantContent::ToolCall(diagnostic)) = call.prefix.last() {
            (call.prefix.clone(), diagnostic.id.clone())
        } else {
            policy::partial_turn_at(&outs.content, events, &call.id, allowed_names)
        };
    let assistant = MessageParts::Assistant {
        id: outs.message_id.clone(),
        content: content.clone(),
    };
    spawn_deferred(commands, assets, run, assistant)?;
    let results = policy::invalid_peer_results(&content, &diagnostic_id, feedback);
    let skipped = match &results {
        MessageParts::User { content } => vec![ToolResultStatus::Skipped; content.len()],
        MessageParts::Assistant { .. } => Vec::new(),
    };
    spawn_deferred_with(commands, assets, run, results, skipped)?;
    let mut run_commands = commands.entity(run);
    run_commands.insert(RunPhase::Assembling);
    if let Some(retries) = retries {
        run_commands.insert(retries);
    }
    Ok(())
}

/// Apply invalid-call resolutions after usage accounting, in fail/retry/skip/edit
/// precedence. Failure, exhausted retries, and skips under `ToolChoice::None`
/// fail immediately, including mid-stream. Other decisions wait for completion;
/// edits also wait for all delivered names to be validated. Retries and skips
/// preserve the rejected prefix as history. Consumed call entities are despawned.
pub fn judge_invalid_calls(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    mut turns: Query<(Entity, &ChildOf, &mut Outputs), Unread>,
    effects: Query<LandedEffect, NotRetrieval>,
    runs: Query<(&RunOf, &RunSeq, &InvalidRetries, &OutputToolName, &RunPhase)>,
    invalid_calls: Query<(Entity, &ChildOf, &InvalidCall, &Resolution)>,
    children: Query<&Children>,
    adverts: Query<&Advert>,
    bound: Query<&Bound>,
    access: Query<&ToolAccess>,
    choices: Query<&ToolChoiceSpec>,
    policies: Query<&InvalidCalls>,
) {
    let mut turns: Vec<_> = turns
        .iter_mut()
        .filter_map(|(turn, turn_of, outs)| {
            let run = turn_of.parent();
            runs.get(run)
                .ok()
                .map(|(_, seq, ..)| (*seq, turn, run, outs))
        })
        .collect();
    turns.sort_by_key(|(seq, ..)| *seq);
    for (_, turn, run, mut outs) in turns {
        let Ok((RunOf(agent), _, invalid_retries, minted, &RunPhase::AwaitingModel)) =
            runs.get(run)
        else {
            continue;
        };
        let agent = *agent;
        let stream = landed_effect(turn, &effects).and_then(|effect| effect.streamed);
        let pending = pending_invalid_calls(turn, &invalid_calls, stream);
        if pending.is_empty() {
            continue;
        }
        let budget = setting(run, agent, &policies).map_or(0, |p| p.retries);
        let tool_choice = setting(run, agent, &choices).and_then(|c| c.0.as_ref());
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
            continue;
        }
        // EOF can arrive with more name events than this pass judged.
        // Retain earlier edits while discovery visits that delivered tail,
        // so the next decision sees the repaired/ignored prefix in order.
        if matches!(verdict, InvalidVerdict::Edit)
            && stream.is_some_and(|stream| {
                outs.stream_validated < stream_invalid::validation_len(stream)
            })
        {
            continue;
        }
        for (entity, _, _) in &pending {
            commands.entity(*entity).despawn();
        }
        let judged = match verdict {
            InvalidVerdict::Fail(call) => {
                fail_unknown_call(&mut commands, &mut assets, run, turn, &outs, call)
            }
            InvalidVerdict::Retry(call, feedback) | InvalidVerdict::Skip(call, feedback) => {
                let retries = matches!(invalid_verdict(&pending), InvalidVerdict::Retry(..))
                    .then(|| InvalidRetries(invalid_retries.0 + 1));
                let granted =
                    granted_tools(turn, &children, &adverts, &bound, access.get(turn).ok());
                let allowed_names: Vec<String> = granted
                    .into_iter()
                    .map(|tool| tool.name)
                    .chain(minted.0.clone())
                    .collect();
                abandon_turn(
                    &mut commands,
                    &mut assets,
                    run,
                    turn,
                    &outs,
                    stream.map(|stream| stream.events.as_slice()),
                    &allowed_names,
                    &call,
                    &feedback,
                    retries,
                )
            }
            InvalidVerdict::Edit => {
                outs.content = edited_content(&outs.content, &pending);
                Ok(())
            }
        };
        if let Err(error) = judged {
            fail_content(&mut commands, run, error);
        }
    }
}

/// Schedule and observe a provider retry when permitted, preserving history and
/// the model-call budget. Otherwise fail the run with provider or cancellation status.
fn provider_failed(
    commands: &mut Commands,
    run: Entity,
    report: &rig_core::error::ErrorReport,
    budget: usize,
    retried: usize,
    witness: Option<(&crate::bus::Witnessing, &crate::bus::Subjects)>,
) {
    if report.kind != ErrorKind::Cancelled && report.retryable && retried < budget {
        let attempt = retried + 1;
        commands.entity(run).insert((
            ProviderRetried(attempt),
            ProviderRetrying,
            RunPhase::Assembling,
        ));
        if let Some((witness, subjects)) = witness {
            witness::observe_provider_retry(
                witness,
                subjects.of_scope(run),
                attempt,
                budget,
                report,
            );
        }
        return;
    }
    let failure = if report.kind == ErrorKind::Cancelled {
        Failure::Cancelled(report.clone())
    } else {
        Failure::Provider(report.clone())
    };
    commands.entity(run).end(Failed(failure));
}

/// Read completed turns without pending invalid calls into [`TurnRead`].
/// Unsupported outcomes and truncated, answerless responses fail the run;
/// provider errors retry within budget or fail. Unpermitted ordinary tool calls
/// create invalid-call entities and leave the turn unread until resolved.
pub fn read_turn(
    mut commands: Commands,
    turns: Query<(Entity, &ChildOf, &Outputs), Unread>,
    effects: Query<LandedEffect, NotRetrieval>,
    runs: Query<(
        &RunOf,
        &RunSeq,
        &ProviderRetried,
        &OutputToolName,
        &RunPhase,
    )>,
    pending: Query<&ChildOf, With<InvalidCall>>,
    children: Query<&Children>,
    adverts: Query<&Advert>,
    bound: Query<&Bound>,
    access: Query<&ToolAccess>,
    provider_retries: Query<&ProviderRetries>,
    subjects: crate::bus::Subjects,
    witness: Option<Res<crate::bus::Witnessing>>,
) {
    let mut turns: Vec<_> = turns
        .iter()
        .filter_map(|(turn, turn_of, outs)| {
            let run = turn_of.parent();
            runs.get(run)
                .ok()
                .map(|(_, seq, ..)| (*seq, turn, run, outs))
        })
        .collect();
    turns.sort_by_key(|(seq, ..)| *seq);
    for (_, turn, run, outs) in turns {
        let Ok((RunOf(agent), _, provider_retried, minted, &RunPhase::AwaitingModel)) =
            runs.get(run)
        else {
            continue;
        };
        if !outs.done || pending.iter().any(|child_of| child_of.parent() == turn) {
            continue;
        }
        let Some(effect) = landed_effect(turn, &effects) else {
            continue;
        };
        commands.entity(turn).insert(Materialised);
        let response = match &effect.outcome.0 {
            Ok(Outcome::Completion(response)) => response,
            Ok(other) => {
                commands
                    .entity(run)
                    .end(Failed(Failure::Unsupported(format!(
                        "a {} answer to a completion",
                        other.family()
                    ))));
                continue;
            }
            Err(report) => {
                let budget = setting(run, *agent, &provider_retries)
                    .map_or(DEFAULT_PROVIDER_RETRIES, |retries| retries.0);
                let witness = witness.as_deref().map(|witness| (witness, &subjects));
                provider_failed(
                    &mut commands,
                    run,
                    report,
                    budget,
                    provider_retried.0,
                    witness,
                );
                continue;
            }
        };
        if turn_delivered_no_answer(&outs.content)
            && let Some(reason) = response
                .finish_reason()
                .filter(|reason| reason.truncated_output())
        {
            let report = rig_core::error::ErrorReport::from(
                &rig_core::error::ProviderError::Response(reason.no_answer_message()),
            );
            commands.entity(run).end(Failed(Failure::Provider(report)));
            continue;
        }
        let access = access.get(turn).ok();
        let granted = granted_tools(turn, &children, &adverts, &bound, access);
        let allowed = access.and_then(|access| access.allowed.as_ref());
        let read = TurnRead {
            message_id: response.message_id.clone(),
            content: outs.content.clone(),
            granted,
            assistant: None,
        };
        let invalid: Vec<_> = read
            .calls()
            .filter(|call| {
                let name = call.function.name.as_str();
                (!read.granted.iter().any(|tool| tool.name == name)
                    || allowed.is_some_and(|allowed| !allowed.contains(name)))
                    && minted.0.as_deref() != Some(name)
            })
            .collect();
        if !invalid.is_empty() {
            commands.entity(turn).remove::<Materialised>();
            for call in invalid {
                commands.spawn((
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
            continue;
        }
        commands.entity(turn).insert(read);
    }
}

/// Materialise assistant history or retry a tool-free turn. Empty turns settle
/// empty unless retried; retries retain history only when feedback is supplied.
/// Tool-bearing retries fail as unsupported. Returns content conversion errors.
fn say_assistant(
    commands: &mut Commands,
    assets: &mut BinaryAssets,
    run: Entity,
    turn: Entity,
    read: &mut TurnRead,
    retry: Option<&Retry>,
) -> Result<(), ContentError> {
    let feedback_utterance = |commands: &mut Commands, assets: &mut BinaryAssets, feedback| {
        let user = MessageParts::User {
            content: vec![UserContent::text(feedback)],
        };
        spawn_deferred(commands, assets, run, user).map(drop)
    };
    if policy::turn_is_empty(&read.content) {
        commands.entity(turn).remove::<(Retry, TurnRead)>();
        match retry {
            Some(Retry { feedback }) => {
                if let Some(feedback) = feedback {
                    feedback_utterance(commands, assets, feedback)?;
                }
                commands.entity(run).phase(RunPhase::Assembling);
            }
            None => {
                commands
                    .entity(run)
                    .end((RunResult(String::new()), Settled));
            }
        }
        return Ok(());
    }
    let assistant = MessageParts::Assistant {
        id: read.message_id.clone(),
        content: read.content.clone(),
    };
    if let Some(Retry { feedback }) = retry {
        commands.entity(turn).remove::<(Retry, TurnRead)>();
        if read.calls().next().is_some() {
            commands.entity(run).end(Failed(Failure::Unsupported(
                "a retry of a tool-bearing turn: steer the tool calls instead".to_owned(),
            )));
            return Ok(());
        }
        if let Some(feedback) = feedback {
            spawn_deferred(commands, assets, run, assistant)?;
            feedback_utterance(commands, assets, feedback)?;
        }
        commands.entity(run).phase(RunPhase::Assembling);
        return Ok(());
    }
    read.assistant = Some(spawn_deferred(commands, assets, run, assistant)?);
    Ok(())
}

/// Materialise assistant history in run order, processing retries and empty
/// answers. Content errors fail only the affected run and clear its transient read.
pub fn materialise_assistant(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    mut turns: Query<(Entity, &ChildOf, &mut TurnRead, Option<&Retry>)>,
    runs: Query<&RunSeq>,
) {
    let mut turns: Vec<_> = turns
        .iter_mut()
        .filter_map(|(turn, turn_of, read, retry)| {
            let run = turn_of.parent();
            runs.get(run).ok().map(|seq| (*seq, turn, run, read, retry))
        })
        .collect();
    turns.sort_by_key(|(seq, ..)| *seq);
    for (_, turn, run, mut read, retry) in turns {
        if let Err(error) = say_assistant(&mut commands, &mut assets, run, turn, &mut read, retry) {
            fail_content(&mut commands, run, error);
            commands.entity(turn).remove::<TurnRead>();
        }
    }
}

/// Spawn granted tool calls in call order with resolved context and handlers.
/// Calls beyond the concurrency limit receive batch-owned holds. Consumes the
/// turn read, links its assistant, and moves the run to `ResolvingTools`.
pub fn materialise_batch(
    mut commands: Commands,
    turns: Query<(Entity, &ChildOf, &TurnRead)>,
    runs: Query<(&RunOf, &RunSeq)>,
    tool_policies: Query<&ToolPolicy>,
    contexts: Query<&ToolContextSpec>,
) {
    let mut turns: Vec<_> = turns
        .iter()
        .filter_map(|(turn, turn_of, read)| {
            let run = turn_of.parent();
            runs.get(run)
                .ok()
                .map(|(RunOf(agent), seq)| (*seq, turn, run, *agent, read))
        })
        .collect();
    turns.sort_by_key(|(seq, ..)| *seq);
    for (_, turn, run, agent, read) in turns {
        let batch: Vec<_> = read
            .calls()
            .filter_map(|call| {
                read.granted
                    .iter()
                    .find(|tool| tool.name == call.function.name)
                    .map(|tool| (call, tool))
            })
            .collect();
        let (Some(assistant), false) = (read.assistant, batch.is_empty()) else {
            continue;
        };
        let concurrency = setting(run, agent, &tool_policies)
            .map_or(1, |policy| policy.concurrency)
            .max(1);
        let inputs = setting(run, agent, &contexts)
            .map(|spec| spec.0.for_dispatch())
            .unwrap_or_default();
        let count = batch.len();
        for (index, (call, tool)) in batch.into_iter().enumerate() {
            let mut effect = commands.spawn((
                PendingEffect::new(
                    tool.key.clone(),
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
            if let Some(handler) = tool.handler {
                effect.insert(ServedBy(handler));
            }
            if index >= concurrency {
                effect.insert(BatchHeld);
                let entity = effect.id();
                commands.queue(move |world: &mut World| {
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
        commands
            .entity(turn)
            .remove::<TurnRead>()
            .insert((Batch { calls: count }, TurnAssistant(assistant)));
        commands.entity(run).phase(RunPhase::ResolvingTools);
    }
}

/// The reprompt a read turn in `Tool` mode earns while the budget lasts
/// (one `OutputRetries`, under `MaxTurns`): an output-tool call missing
/// required fields is answered with a `Skipped` result naming them; a text
/// where the tool was due, unless it already is the structured output
/// (CONTRACT §4), is asked for the tool. `None` when the turn earns no
/// reprompt: `materialise_answer` settles it.
fn reprompt_for(
    read: &TurnRead,
    name: &str,
    schema: Option<&serde_json::Value>,
) -> Option<(MessageParts, Vec<ToolResultStatus>)> {
    match read.calls().find(|call| call.function.name == name) {
        Some(call) => {
            let missing = schema
                .map(|schema| policy::missing_required_fields(schema, &call.function.arguments))
                .unwrap_or_default();
            if missing.is_empty() {
                return None;
            }
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
            Some((reprompt, vec![ToolResultStatus::Skipped]))
        }
        None => {
            if policy::text_satisfies_schema(schema, &policy::answer_text(&read.content)) {
                return None;
            }
            let reprompt = MessageParts::User {
                content: vec![UserContent::text(policy::text::reprompt_text_answer(name))],
            };
            Some((reprompt, Vec::new()))
        }
    }
}

/// Reprompt tool-output turns at most once and within the model-call budget.
/// Commits feedback to history, consumes the turn read, and resumes assembly;
/// content errors fail the run. Exhausted budgets leave the answer for settlement.
pub fn materialise_reprompt(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    turns: Query<(Entity, &ChildOf, &TurnRead, &Folded)>,
    runs: Query<(&RunOf, &RunSeq, &Cursor, &OutputRetries, &OutputToolName)>,
    outputs: Query<&Output>,
    max_turns: Query<&MaxTurns>,
) {
    let mut turns: Vec<_> = turns
        .iter()
        .filter(|(_, _, _, Folded(mode))| *mode == OutputKind::Tool)
        .filter_map(|(turn, turn_of, read, _)| {
            let run = turn_of.parent();
            runs.get(run)
                .ok()
                .map(|(_, seq, ..)| (*seq, turn, run, read))
        })
        .collect();
    turns.sort_by_key(|(seq, ..)| *seq);
    for (_, turn, run, read) in turns {
        let Ok((RunOf(agent), _, cursor, retries, OutputToolName(Some(name)))) = runs.get(run)
        else {
            continue;
        };
        let agent = *agent;
        let limit = setting(run, agent, &max_turns).map_or(1, |limit| limit.0);
        if retries.0 >= 1 || cursor.turn >= limit {
            continue;
        }
        let schema = setting(run, agent, &outputs).and_then(|output| output.schema.as_ref());
        let Some((reprompt, statuses)) = reprompt_for(read, name, schema) else {
            continue;
        };
        commands
            .entity(turn)
            .remove::<TurnRead>()
            .insert(Reprompt(reprompt.to_message()));
        match spawn_deferred_with(&mut commands, &mut assets, run, reprompt, statuses) {
            Ok(_) => {
                commands
                    .entity(run)
                    .end((OutputRetries(retries.0 + 1), RunPhase::Assembling));
            }
            Err(error) => fail_content(&mut commands, run, error),
        }
    }
}

/// Settle remaining reads with output-tool arguments or assistant text.
/// Output-tool history retains non-call content and appends the JSON answer;
/// the effect record retains the original call. Clears transient reads and fails
/// the run if history conversion fails.
pub fn materialise_answer(
    mut commands: Commands,
    mut assets: ResMut<BinaryAssets>,
    turns: Query<(Entity, &ChildOf, &TurnRead, &Folded)>,
    runs: Query<(&RunSeq, &OutputToolName)>,
) {
    let mut turns: Vec<_> = turns
        .iter()
        .filter_map(|(turn, turn_of, read, folded)| {
            let run = turn_of.parent();
            runs.get(run)
                .ok()
                .map(|(seq, minted)| (*seq, turn, run, read, folded, minted))
        })
        .collect();
    turns.sort_by_key(|(seq, ..)| *seq);
    for (_, turn, run, read, Folded(mode), minted) in turns {
        commands.entity(turn).remove::<TurnRead>();
        let output_call = match (mode, minted.0.as_deref(), read.assistant) {
            (OutputKind::Tool, Some(name), Some(assistant)) => read
                .calls()
                .find(|call| call.function.name == name)
                .map(|call| (assistant, call)),
            _ => None,
        };
        let Some((assistant, call)) = output_call else {
            commands
                .entity(run)
                .end((RunResult(policy::answer_text(&read.content)), Settled));
            continue;
        };
        let output = call.function.arguments.to_string();
        let mut final_content: Vec<_> = read
            .content
            .iter()
            .filter(|part| !matches!(part, AssistantContent::ToolCall(_)))
            .cloned()
            .collect();
        final_content.push(AssistantContent::text(output.clone()));
        let restated = MessageParts::Assistant {
            id: read.message_id.clone(),
            content: final_content,
        };
        match replace_deferred(&mut commands, &mut assets, assistant, restated) {
            Ok(()) => {
                commands.entity(run).end((RunResult(output), Settled));
            }
            Err(error) => fail_content(&mut commands, run, error),
        }
    }
}

/// Cancel an active run when its pending completion or batch tool effect is
/// removed. Existing failures remain unchanged.
pub fn effect_cancelled(
    removed: On<bevy_ecs::lifecycle::Remove, PendingEffect>,
    effects: Query<(&ChildOf, Has<ToolCallSlot>), With<PendingEffect>>,
    turns: Query<TurnState, With<Turn>>,
    runs: Query<RunState, With<Run>>,
    mut commands: Commands,
) {
    let effect = removed.event().entity;
    let Ok((turn_of, is_tool_call)) = effects.get(effect) else {
        return;
    };
    let turn = turn_of.parent();
    let Ok(TurnStateItem {
        run_of,
        materialised,
        batched,
    }) = turns.get(turn)
    else {
        return;
    };
    let run = run_of.parent();
    let Ok(RunStateItem { phase, failed }) = runs.get(run) else {
        return;
    };
    // A run already ended keeps its ending: `run_cancelled` writes the
    // reason before it despawns what was pending.
    if failed {
        return;
    }
    let cancelled = Failed(Failure::Cancelled(rig_core::serve::cancelled()));
    if is_tool_call && batched && phase == Some(&RunPhase::ResolvingTools) {
        commands.entity(turn).remove::<Batch>();
        commands.entity(run).end(cancelled);
    } else if !is_tool_call && !materialised && phase == Some(&RunPhase::AwaitingModel) {
        commands.entity(turn).insert(Materialised);
        commands.entity(run).end(cancelled);
    }
}

/// Apply a run cancellation unless it already ended. Marks turns materialised
/// and despawns unissued effects without records. In-flight effects remain owned
/// by their handlers, which complete their records.
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
    for child in children
        .get(run)
        .map(|c| c.iter().collect::<Vec<_>>())
        .unwrap_or_default()
    {
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
                .map(|c| c.iter().collect::<Vec<_>>())
                .unwrap_or_default()
            {
                if let Ok(false) = effects.get(effect) {
                    pending.push(effect);
                }
            }
        }
    }
    // The ending first, so the despawns' observer (`effect_cancelled`)
    // finds the run ended with this reason and leaves it.
    commands
        .entity(run)
        .remove::<RunPhase>()
        .insert(Failed(Failure::Cancelled(
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
