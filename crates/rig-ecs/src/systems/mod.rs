//! The agent's systems: one per named set of [`RigSet`], in the bus
//! module's `RigSchedule`, run to quiescence by its runner.
//!
//! | set | true before | written during |
//! |---|---|---|
//! | `Advance` | a run in `Assembling` has no fresh turn | a turn is spawned `ChildOf` the run with its adverts and attachments, or the run fails `MaxTurns` |
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
    parts::{ContentError, ContentGraph, replace_deferred, spawn_deferred, write_message},
};

use crate::agent::content::parts::{EditTarget, RequestPartEdit};
use bevy_ecs::prelude::*;
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
        AdditionalParams, Advert, Assembling, Attachment, AwaitingModel, Batch, Cancelled, Context,
        Conversation, Cursor, DEFAULT_PROVIDER_RETRIES, DocumentId, DocumentProps, DocumentText,
        Failed, Failure, Grant, InvalidCall, InvalidCalls, InvalidRetries, LoadingMemory,
        MaxTokens, MaxTurns, MemoryAppendScheduled, MessageParts, Order, OrderCounter, Output,
        OutputKind, OutputRetries, OutputToolConfig, OutputToolName, Outputs, Preamble, Prompt,
        ProviderRetried, ProviderRetries, ProviderRetrying, Remembered, Remembering, Remembers,
        Reprompt, RequestPatch, Resolution, ResolvingTools, Retrievable, Retrieval, RetrievalKind,
        Retrieves, Retrieving, Retry, Run, RunCounter, RunOf, RunResult, RunSeq, Settled,
        StreamRequested, Temperature, ToolAccess, ToolCallSlot, ToolChoiceSpec, ToolContextSpec,
        ToolPolicy, Turn, Unhandled, Usage, UsesModel, Utterance,
    },
    bus::{
        Bound, BusSet, EffectOutcome, Issued, PendingEffect, Progress, RigSchedule, Scope,
        Streamed as BusStreamed, ToolInputs,
    },
    policy::{self, RequestGraph},
};

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

/// A run that wants a turn: `Assembling`, not failed.
pub type Wanting = (With<Assembling>, Without<Failed>);
/// A run with no model of its own yet.
pub type Unselected = (With<Run>, Without<UsesModel>);
/// What `Fold` reads of an effect.
pub type EffectView = (
    &'static ChildOf,
    Option<&'static BusStreamed>,
    Option<&'static EffectOutcome>,
);
/// An invalid call nothing resolved.
pub type Unresolved = (With<InvalidCall>, Without<Resolution>);
/// A turn `Materialise` has not read.
pub type Unread = (With<Turn>, Without<Materialised>);
/// What `assemble` reads of a run.
pub type AssemblingView = (
    &'static RunOf,
    &'static RunSeq,
    &'static StreamRequested,
    Option<&'static UsesModel>,
    &'static OutputToolName,
);
/// The request settings `assemble` resolves, the run's over the agent's.
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
}
/// What `assemble` reads of a fresh turn: its run, its patch, whether it
/// is retrieving.
pub type FreshView = (
    Entity,
    &'static ChildOf,
    Option<&'static RequestPatch>,
    Has<Retrieving>,
);
/// A fresh turn whose retrievals are out.
pub type RetrievingTurn = (With<Fresh>, With<Retrieving>);
/// A remembering run whose persisted finalization has not scheduled an append.
pub type NeedsMemoryAppend = (
    With<Settled>,
    With<Remembering>,
    Without<MemoryAppendScheduled>,
);
/// A completion effect: any effect of a turn that is not a retrieval.
pub type NotRetrieval = (With<PendingEffect>, Without<Retrieval>);
/// What `materialise` reads besides the graph: the tool choice and access
/// settings, the provider-retry budget, and the witness a retry is told
/// to. Grouped: a system takes at most sixteen parameters.
#[derive(bevy_ecs::system::SystemParam)]
pub struct MaterialiseReads<'w, 's> {
    /// The tool choice, the run's over the agent's.
    pub choices: Query<'w, 's, &'static ToolChoiceSpec>,
    /// The tool access spec.
    pub access: Query<'w, 's, &'static ToolAccess>,
    /// The provider-retry budget, the run's over the agent's.
    pub provider_retries: Query<'w, 's, &'static ProviderRetries>,
    /// Subjects for the witness.
    pub subjects: crate::bus::Subjects<'w, 's>,
    /// The witness, if the world has one.
    pub witness: Option<Res<'w, crate::bus::Witnessing>>,
}

/// What `materialise` reads of a run awaiting its model.
pub type AwaitingView = (
    &'static RunOf,
    &'static Cursor,
    &'static OutputRetries,
    &'static InvalidRetries,
    &'static OutputToolName,
    &'static Usage,
    &'static ProviderRetried,
);
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

/// Install the agent runtime into `world`: the bus must be installed first
/// (it owns the schedule); this adds the sets, the counters, the observers
/// and the systems.
pub fn install_agent(world: &mut World) {
    assert!(
        world.contains_resource::<crate::bus::Policy>(),
        "install_agent needs the bus installed first: it runs in the bus's RigSchedule"
    );
    world.init_resource::<OrderCounter>();
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
        advance.in_set(RigSet::Advance),
        attach_retrieved
            .after(RigSet::Advance)
            .before(RigSet::Select),
        select.in_set(RigSet::Select),
        assemble.in_set(RigSet::Assemble),
        release_batch.in_set(RigSet::Release),
        (fold, discover_streamed_invalid_calls)
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

fn fail_content(commands: &mut Commands, run: Entity, error: ContentError) {
    commands
        .entity(run)
        .remove::<(
            Assembling,
            AwaitingModel,
            LoadingMemory,
            ResolvingTools,
            Settled,
        )>()
        .insert(Failed(Failure::Content(error)));
}

macro_rules! content_or_fail {
    ($result:expr, $commands:expr, $run:expr, $progress:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => {
                fail_content($commands, $run, error);
                $progress.mark();
                return;
            }
        }
    };
}

/// Why [`despawn_run`] left a run in the world.
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

/// Despawn an ended run and everything that is its: turns, utterances,
/// adverts, attachments and the settled effects under them (`ChildOf` is
/// linked, so the despawn is deep), a `Streamed` fold included. The world
/// keeps nothing of a run by itself — a host that runs for long must
/// despawn the runs it is done reading, or their graphs and folds
/// accumulate for the life of the world. Refused while the run has not
/// ended or an effect of it is still in flight; nothing is despawned then.
pub fn despawn_run(world: &mut World, run: Entity) -> Result<(), RunBusy> {
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

/// Spawn a run of `agent` with `prompt` as its first utterance, after
/// `history`: the host's one entry point. The prompt is a user message's
/// parts (`&str` text, or text and images kept in their given order,
/// [`Prompt`]). Returns the run entity.
pub fn spawn_run(
    world: &mut World,
    agent: Entity,
    history: &[MessageParts],
    prompt: impl Into<Prompt>,
    streamed: bool,
    max_turns: Option<usize>,
) -> Entity {
    let prompt = MessageParts::User {
        content: prompt.into().0,
    };
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
    // An agent that remembers, and a run given no history: the conversation
    // is loaded before the first turn (CONTRACT §11).
    let memory = history
        .is_empty()
        .then(|| {
            let handler = world.get::<Remembers>(agent).map(|remembers| remembers.0)?;
            let conversation = world
                .get::<Conversation>(agent)
                .map(|conversation| conversation.0.clone())?;
            let key = world.get::<Bound>(handler).map(|bound| bound.key.clone())?;
            Some((key, conversation))
        })
        .flatten();
    let mut run = world.spawn((
        Run,
        RunOf(agent),
        RunSeq(seq),
        StreamRequested(streamed),
        Cursor::default(),
        OutputRetries::default(),
        crate::agent::InvalidRetries::default(),
        ProviderRetried::default(),
        OutputToolName::default(),
        Usage::default(),
        Scope(format!("{owner}/run#{seq}")),
    ));
    if let Some(limit) = max_turns {
        run.insert(MaxTurns(limit));
    }
    match &memory {
        Some((_, conversation)) => {
            run.insert((
                LoadingMemory,
                Remembering,
                Conversation(conversation.clone()),
            ));
        }
        None => {
            run.insert(Assembling);
        }
    }
    let run = run.id();
    for parts in history.iter().cloned().chain(std::iter::once(prompt)) {
        if let Err(error) = spawn_utterance(world, run, parts) {
            world
                .entity_mut(run)
                .remove::<(Assembling, LoadingMemory)>()
                .insert(Failed(Failure::Content(error)));
            return run;
        }
    }
    if let Some((key, conversation)) = memory {
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

    run
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
) -> Vec<&'a L> {
    let mut found: Vec<(&Order, &L)> = children
        .get(owner)
        .map(|children| {
            children
                .iter()
                .filter_map(|child| links.get(child).ok().map(|(link, order)| (order, link)))
                .collect()
        })
        .unwrap_or_default();
    found.sort_by_key(|(order, _)| **order);
    found.into_iter().map(|(_, link)| link).collect()
}

/// `RigSet::Advance`: a run in `Assembling` with no fresh turn gets one —
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
                .remove::<Assembling>()
                .insert(Failed(Failure::MaxTurns { limit }));
            progress.mark();
            continue;
        }
        if holds.get(run).is_ok_and(|holds| {
            commits
                .iter()
                .any(|(parent, commit)| parent.parent() == run && holds.blocks(commit.turn))
        }) {
            continue;
        }
        let turn = commands
            .spawn((Turn, Fresh, next_order_in(&mut orders), ChildOf(run)))
            .id();
        let retrieves = children
            .get(*agent)
            .map(|children| children.iter().any(|child| retrievals.get(child).is_ok()))
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
        let mut links: Vec<(&Grant, &Order, bool)> = children
            .get(*agent)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| grants.get(child).ok())
                    .collect()
            })
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
            let document = documents
                .iter()
                .find(|(_, existing)| existing.0 == id)
                .map(|(entity, _)| entity)
                .unwrap_or_else(|| commands.spawn((DocumentId(id), DocumentText(text))).id());
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
                for message in messages {
                    if let Some(parts) = MessageParts::from_message(message) {
                        let utterance = content_or_fail!(
                            spawn_deferred(
                                &mut commands,
                                &mut assets,
                                run,
                                parts,
                                next_order_in(&mut orders)
                            ),
                            &mut commands,
                            run,
                            progress
                        );
                        commands.entity(utterance).insert(Remembered);
                    }
                }
                // The prompt (and any history given) comes after what was
                // loaded: its order is re-stamped past the loaded ones.
                let mut existing: Vec<(Entity, Order)> = children
                    .get(run)
                    .map(|children| {
                        children
                            .iter()
                            .filter_map(|child| utterances.get(child).ok())
                            .map(|(entity, order)| (entity, *order))
                            .collect()
                    })
                    .unwrap_or_default();
                existing.sort_by_key(|(_, order)| *order);
                for (entity, _) in existing {
                    commands.entity(entity).insert(next_order_in(&mut orders));
                }
                commands
                    .entity(run)
                    .remove::<LoadingMemory>()
                    .insert(Assembling);
            }
            Ok(other) => {
                commands
                    .entity(run)
                    .remove::<LoadingMemory>()
                    .insert(Failed(Failure::Memory(rig_core::error::ErrorReport::new(
                        ErrorKind::Internal,
                        format!(
                            "the memory handler answered a load with a {} outcome",
                            other.family()
                        ),
                    ))));
            }
            Err(report) => {
                commands
                    .entity(run)
                    .remove::<LoadingMemory>()
                    .insert(Failed(Failure::Memory(report.clone())));
            }
        }
        progress.mark();
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
        let Some(key) = memories
            .get(*agent)
            .ok()
            .and_then(|Remembers(memory)| bound.get(*memory).ok())
            .map(|bound| bound.key.clone())
        else {
            continue;
        };
        let said: Result<Vec<_>, ContentError> = children
            .get(run)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| utterances.get(child).ok())
                    .filter(|(_, _, remembered)| !*remembered)
                    .map(|(entity, order, _)| {
                        content.message(entity).map(|message| (*order, message))
                    })
                    .collect()
            })
            .unwrap_or_else(|_| Ok(Vec::new()));
        let mut said = content_or_fail!(said, &mut commands, run, progress);
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

/// Ordered request edit links, including malformed links for explicit validation.
pub type PartEditView = (
    Entity,
    &'static ChildOf,
    Option<&'static Order>,
    Option<&'static EditTarget>,
    &'static RequestPartEdit,
);

/// `RigSet::Assemble`: for every fresh turn, in run order, gather the
/// graph — the run's settings over the agent's, the utterances in order,
/// the attachments in order, the adverts in order, the model's descriptor
/// — resolve the output mode, mint the output tool's name once per run,
/// fold, and spawn the effect `ChildOf` the turn. The run is then
/// `AwaitingModel`.
/// A missing selected model or non-completion binding instead terminates the
/// run with a provider `HandlerUnavailable` report; it never silently waits.
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn assemble(
    mut commands: Commands,
    fresh: Query<FreshView, With<Fresh>>,
    runs: Query<AssemblingView, (With<Run>, Without<Failed>)>,
    children: Query<&Children>,
    utterances: Query<(Entity, &Order), With<Utterance>>,
    content: ContentGraph,
    part_edits: Query<PartEditView>,
    retrievals: Query<(&Retrieves, &Order, &Retrieval)>,
    retrieving: Query<(), With<Retrieval>>,
    adverts: Query<(&Advert, &Order)>,
    attachments: Query<(&Attachment, &Order)>,
    documents: Query<(&DocumentId, &DocumentText, Option<&DocumentProps>)>,
    bound: Query<&Bound>,
    settings: Settings,
    mut progress: ResMut<Progress>,
) {
    let Settings {
        preambles,
        temperatures,
        max_tokens,
        params,
        choices,
        outputs,
        output_tools,
        tool_access,
    } = settings;
    let mut turns: Vec<(Entity, Entity, RunSeq, Option<&RequestPatch>, bool)> = fresh
        .iter()
        .filter_map(|(turn, child_of, patch, retrieving)| {
            let run = child_of.parent();
            runs.get(run)
                .ok()
                .map(|(_, seq, _, _, _)| (turn, run, *seq, patch, retrieving))
        })
        .collect();
    turns.sort_by_key(|(_, _, seq, _, _)| *seq);

    for (turn, run, _, patch, is_retrieving) in turns {
        let Ok((RunOf(agent), _, StreamRequested(stream), model, minted)) = runs.get(run) else {
            continue;
        };
        let agent = *agent;
        let model_bound = model.and_then(|UsesModel(model)| bound.get(*model).ok());
        let Some(model_bound) = model_bound else {
            commands.entity(run).remove::<Assembling>().insert(Failed(Failure::Provider(
                rig_core::error::ErrorReport::new(rig_core::error::ErrorKind::HandlerUnavailable,
                    "the run has no bound completion model; its selected model or its agent's binding was removed"),
            )));
            commands.entity(turn).remove::<Fresh>();
            progress.mark();
            continue;
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
                commands
                    .entity(run)
                    .remove::<Assembling>()
                    .insert(Failed(Failure::Provider(
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
                continue;
            }
        };

        let requested: Result<_, ContentError> = (|| {
            let mut links: Vec<_> = part_edits
                .iter()
                .filter(|(_, parent, _, _, _)| parent.parent() == turn)
                .collect();
            if links
                .iter()
                .any(|(_, _, order, target, _)| order.is_none() || target.is_none())
            {
                return Err(ContentError::Missing);
            }
            links.sort_by_key(|(_, _, order, _, _)| order.copied());
            if links
                .windows(2)
                .any(|pair| pair.first().map(|x| x.2) == pair.get(1).map(|x| x.2))
            {
                return Err(ContentError::DuplicateOrder);
            }
            let mut edits = std::collections::BTreeMap::new();
            let mut consumed = Vec::new();
            for (link, _, _, target, edit) in links {
                let target = target.ok_or(ContentError::Missing)?;
                let utterance = content.target_utterance(target.0)?;
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
                edits.insert(target.0, edit.clone());
                consumed.push(link);
            }
            Ok((edits, consumed))
        })();
        let (edits, consumed_edits) = content_or_fail!(requested, &mut commands, run, progress);

        let history: Result<Vec<_>, ContentError> = children
            .get(run)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| utterances.get(child).ok())
                    .map(|(entity, order)| {
                        content
                            .message_with(entity, &edits)
                            .map(|message| (*order, message))
                    })
                    .collect()
            })
            .unwrap_or_else(|_| Ok(Vec::new()));
        let mut history = content_or_fail!(history, &mut commands, run, progress);
        history.sort_by_key(|(order, _)| *order);

        if is_retrieving {
            // The first pass over a retrieving turn (CONTRACT §12): one
            // `Retrieve` effect per index, in link order, `ChildOf` the
            // turn; the fold waits for `attach_retrieved`.
            let spawned = children
                .get(turn)
                .map(|children| children.iter().any(|child| retrieving.get(child).is_ok()))
                .unwrap_or(false);
            if spawned {
                continue;
            }
            let query = policy::retrieval_query(
                &history
                    .iter()
                    .map(|(_, parts)| parts.clone())
                    .collect::<Vec<_>>(),
            );
            let mut indexes: Vec<(&Retrieves, &Order, &Retrieval)> = children
                .get(agent)
                .map(|children| {
                    children
                        .iter()
                        .filter_map(|child| retrievals.get(child).ok())
                        .collect()
                })
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
                    RetrievalKind::Documents => {
                        rig_core::effect::RetrieveQuery::TopN { req: request }
                    }
                    RetrievalKind::Tools => {
                        rig_core::effect::RetrieveQuery::TopNIds { req: request }
                    }
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
            continue;
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
            links_in_order(turn, &children, &attachments)
                .into_iter()
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
                    .chain(history.last().map(|(_, parts)| parts.clone()))
                    .collect()
            });
        let merged_params: Option<serde_json::Value> = match (
            setting(run, agent, &params).and_then(|p| p.0.clone()),
            patch.and_then(|p| p.additional_params.clone()),
        ) {
            (Some(base), Some(patched)) if base.is_object() && patched.is_object() => {
                Some(rig_core::json_utils::merge(base, patched))
            }
            (base, patched) => patched.or(base),
        };

        let preamble = patch
            .and_then(|p| p.preamble.as_deref())
            .or_else(|| setting(run, agent, &preambles).and_then(|preamble| preamble.0.as_deref()));
        let temperature = patch
            .and_then(|p| p.temperature)
            .or_else(|| setting(run, agent, &temperatures).and_then(|t| t.0));
        let max_tokens = patch
            .and_then(|p| p.max_tokens)
            .or_else(|| setting(run, agent, &max_tokens).and_then(|m| m.0));
        let additional_params = merged_params.as_ref();
        let tool_choice = patch
            .and_then(|p| p.tool_choice.as_ref())
            .or_else(|| setting(run, agent, &choices).and_then(|c| c.0.as_ref()));
        let output = setting(run, agent, &outputs).cloned().unwrap_or_default();
        let output_tool_config = setting(run, agent, &output_tools);
        let reserved_name = output_tool_config.and_then(|config| config.name.as_deref());

        let mut access = setting(run, agent, &tool_access)
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
            commands.entity(run).remove::<Assembling>().insert(Failed(
                Failure::OutputToolCollision {
                    name: output_tool.clone(),
                },
            ));
            progress.mark();
            continue;
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
                None => history.iter().map(|(_, parts)| parts).collect(),
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
        for link in consumed_edits {
            commands.entity(link).despawn();
        }
        commands.entity(turn).insert(access);
        commands.spawn((
            PendingEffect::new(
                model_bound.key.clone(),
                EffectKind::Completion {
                    request,
                    stream: *stream,
                },
            ),
            ChildOf(turn),
        ));
        commands
            .entity(turn)
            .remove::<(Fresh, RequestPatch)>()
            .insert((Folded(resolved), Outputs::default()));
        commands
            .entity(run)
            .remove::<Assembling>()
            .insert(AwaitingModel);
        progress.mark();
    }
}

/// `RigSet::Fold`: the turn's outputs from its effect — the text so far
/// while it streams (`Changed<Outputs>` is the delta signal), the folded
/// answer when it lands.
pub fn fold(
    effects: Query<EffectView, NotRetrieval>,
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
            Some(EffectOutcome(Ok(Outcome::Completion(response)))) => {
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
            Some(EffectOutcome(Ok(_))) | Some(EffectOutcome(Err(_))) => {
                outputs.done = true;
                progress.mark();
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
            children
                .iter()
                .filter_map(|child| tools.get(child).ok())
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
        let Ok((minted, _, cursor)) = runs.get(run) else {
            continue;
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
                continue;
            }
            // The ending first, so the despawns' observer finds the run
            // ended with this failure and leaves it.
            commands.entity(turn).remove::<Batch>();
            commands
                .entity(run)
                .remove::<ResolvingTools>()
                .insert(Failed(failure));
            for (entity, _, issued, outcome, _) in &calls {
                if !*issued && outcome.is_none() {
                    commands.entity(*entity).despawn();
                }
            }
            progress.mark();
            continue;
        }
        if calls.len() < batch.calls || calls.iter().any(|(_, _, _, outcome, _)| outcome.is_none())
        {
            continue;
        }
        let mut parts = Vec::with_capacity(calls.len());
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
                Ok(part) => parts.push(part),
                Err(failure) => {
                    failed = Some(failure);
                    break;
                }
            }
        }
        commands.entity(turn).remove::<Batch>();
        if let Some(failure) = failed {
            commands
                .entity(run)
                .remove::<ResolvingTools>()
                .insert(Failed(failure));
            progress.mark();
            continue;
        }
        let results = MessageParts::User { content: parts };
        let results_entity = content_or_fail!(
            spawn_deferred(
                &mut commands,
                &mut assets,
                run,
                results,
                next_order_in(&mut orders)
            ),
            &mut commands,
            run,
            progress
        );
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
                    .remove::<ResolvingTools>()
                    .insert((RunResult(arguments), Settled));
            }
            None => {
                commands
                    .entity(run)
                    .remove::<ResolvingTools>()
                    .insert(Assembling);
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
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn materialise(
    mut commands: Commands,
    mut turns: Query<(Entity, &ChildOf, &mut Outputs, &Folded, Option<&Retry>), Unread>,
    effects: Query<(&ChildOf, &EffectOutcome, Option<&BusStreamed>), NotRetrieval>,
    runs: Query<(AwaitingView, &RunSeq), With<AwaitingModel>>,
    children: Query<&Children>,
    adverts: Query<(&Advert, &Order)>,
    bound: Query<&Bound>,
    outputs: Query<&Output>,
    max_turns: Query<&MaxTurns>,
    policies: Query<&InvalidCalls>,
    reads: MaterialiseReads,
    tool_policies: Query<&ToolPolicy>,
    contexts: Query<&ToolContextSpec>,
    invalid_calls: Query<(Entity, &ChildOf, &InvalidCall, &Resolution)>,
    (mut assets, mut orders): (ResMut<BinaryAssets>, ResMut<OrderCounter>),
    mut progress: ResMut<Progress>,
) {
    let MaterialiseReads {
        choices,
        access,
        provider_retries,
        subjects,
        witness,
    } = reads;
    let mut turns: Vec<_> = turns.iter_mut().collect();
    turns.sort_by_key(|(_, turn_of, _, _, _)| runs.get(turn_of.parent()).map(|(_, seq)| *seq).ok());
    for (turn, turn_of, mut outs, Folded(mode), retry) in turns {
        let run = turn_of.parent();
        let Ok((
            (RunOf(agent), cursor, retries, invalid_retries, minted, usage, provider_retried),
            _,
        )) = runs.get(run)
        else {
            continue;
        };
        let agent = *agent;
        let tool_choice = setting(run, agent, &choices).and_then(|c| c.0.clone());

        // The tools this turn advertised, by name, with their keys.
        let mut granted: Vec<(String, rig_core::effect::HandlerKey)> =
            links_in_order(turn, &children, &adverts)
                .into_iter()
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
            && let Some((_, EffectOutcome(Ok(Outcome::Completion(response))), _)) = effects
                .iter()
                .find(|(parent, _, _)| parent.parent() == turn)
        {
            commands.entity(run).insert(Usage(usage.0 + response.usage));
            outs.usage_recorded = true;
        }

        // Pending invalid calls of this turn: consumed first.
        let pending: Vec<(Entity, InvalidCall, Resolution)> = invalid_calls
            .iter()
            .filter(|(_, child_of, _, _)| child_of.parent() == turn)
            .map(|(entity, _, call, resolution)| {
                let mut call = call.clone();
                if let Some(offset) = call.stream_offset
                    && let Some((_, _, Some(stream))) = effects
                        .iter()
                        .find(|(parent, _, _)| parent.parent() == turn)
                    && let Some(id) = stream_invalid::completed_call_id(&stream.events, offset)
                {
                    call.id = id;
                }
                (entity, call, resolution.clone())
            })
            .collect();
        if !pending.is_empty() {
            let budget = setting(run, agent, &policies).map_or(0, |p| p.retries);
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
                && effects.iter().any(|(parent, _, stream)| {
                    parent.parent() == turn
                        && stream.is_some_and(|stream| {
                            outs.stream_validated < stream_invalid::validation_len(stream)
                        })
                })
            {
                continue;
            }
            for (entity, _, _) in &pending {
                commands.entity(*entity).despawn();
            }
            match verdict {
                InvalidVerdict::Fail(call) => {
                    if !call.prefix.is_empty() {
                        let assistant = MessageParts::Assistant {
                            id: outs.message_id.clone(),
                            content: call.prefix.clone(),
                        };
                        content_or_fail!(
                            spawn_deferred(
                                &mut commands,
                                &mut assets,
                                run,
                                assistant,
                                next_order_in(&mut orders)
                            ),
                            &mut commands,
                            run,
                            progress
                        );
                    }
                    commands.entity(turn).insert(Materialised);
                    commands
                        .entity(run)
                        .remove::<AwaitingModel>()
                        .insert(Failed(Failure::UnknownToolCall { name: call.name }));
                    progress.mark();
                    continue;
                }
                InvalidVerdict::Retry(call, feedback) | InvalidVerdict::Skip(call, feedback) => {
                    let retried = matches!(invalid_verdict(&pending), InvalidVerdict::Retry(..));
                    // A streamed turn is abandoned where the call surfaced.
                    let events = effects
                        .iter()
                        .find(|(child_of, _, _)| child_of.parent() == turn)
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
                    content_or_fail!(
                        spawn_deferred(
                            &mut commands,
                            &mut assets,
                            run,
                            assistant,
                            next_order_in(&mut orders)
                        ),
                        &mut commands,
                        run,
                        progress
                    );
                    let results = policy::invalid_peer_results(&content, &diagnostic_id, &feedback);
                    content_or_fail!(
                        spawn_deferred(
                            &mut commands,
                            &mut assets,
                            run,
                            results,
                            next_order_in(&mut orders)
                        ),
                        &mut commands,
                        run,
                        progress
                    );
                    commands.entity(turn).insert(Materialised);
                    let mut run_commands = commands.entity(run);
                    run_commands.remove::<AwaitingModel>().insert(Assembling);
                    if retried {
                        run_commands.insert(InvalidRetries(invalid_retries.0 + 1));
                    }
                    progress.mark();
                    continue;
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
                    outs.content = content;
                }
            }
        }

        if !outs.done {
            continue;
        }
        let Some((_, EffectOutcome(outcome), _)) = effects
            .iter()
            .find(|(child_of, _, _)| child_of.parent() == turn)
        else {
            continue;
        };
        commands.entity(turn).insert(Materialised);

        let response = match outcome {
            Ok(Outcome::Completion(response)) => response,
            Ok(other) => {
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert(Failed(Failure::Unsupported(format!(
                        "a {} answer to a completion",
                        other.family()
                    ))));
                progress.mark();
                continue;
            }
            Err(report) => {
                // A retryable provider failure with budget left re-issues
                // the same request over the same history (CONTRACT §5):
                // the lost turn is read and leaves nothing; the run wants
                // a turn again, marked so `Advance` does not count it.
                let budget = setting(run, agent, &provider_retries)
                    .map_or(DEFAULT_PROVIDER_RETRIES, |retries| retries.0);
                if report.kind != ErrorKind::Cancelled
                    && report.retryable
                    && provider_retried.0 < budget
                {
                    let attempt = provider_retried.0 + 1;
                    commands.entity(turn).insert(Materialised);
                    commands.entity(run).remove::<AwaitingModel>().insert((
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
                    progress.mark();
                    continue;
                }
                let failure = if report.kind == ErrorKind::Cancelled {
                    Failure::Cancelled(report.clone())
                } else {
                    Failure::Provider(report.clone())
                };
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert(Failed(failure));
                progress.mark();
                continue;
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
            let report = rig_core::error::ErrorReport::from(
                &rig_core::completion::CompletionError::ResponseError(reason.no_answer_message()),
            );
            commands
                .entity(run)
                .remove::<AwaitingModel>()
                .insert(Failed(Failure::Provider(report)));
            progress.mark();
            continue;
        }

        // An empty turn is not history, and answers nothing. A retry
        // written on it (CONTRACT §9.4) still asks again: the feedback
        // becomes history, the empty turn does not, and another turn
        // begins; without one, the run settles on the empty answer.
        if policy::turn_is_empty(&content) {
            if let Some(Retry { feedback }) = retry {
                commands.entity(turn).remove::<Retry>();
                if let Some(feedback) = feedback {
                    let user = MessageParts::User {
                        content: vec![UserContent::text(feedback)],
                    };
                    content_or_fail!(
                        spawn_deferred(
                            &mut commands,
                            &mut assets,
                            run,
                            user,
                            next_order_in(&mut orders)
                        ),
                        &mut commands,
                        run,
                        progress
                    );
                }
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert(Assembling);
                progress.mark();
                continue;
            }
            commands
                .entity(run)
                .remove::<AwaitingModel>()
                .insert((RunResult(String::new()), Settled));
            progress.mark();
            continue;
        }

        let schema = setting(run, agent, &outputs).and_then(|output| output.schema.clone());
        let limit = setting(run, agent, &max_turns).map_or(1, |limit| limit.0);

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
            progress.mark();
            continue;
        }

        // A retry written on the turn (CONTRACT §9.4): tool-free turns only.
        if let Some(Retry { feedback }) = retry {
            commands.entity(turn).remove::<Retry>();
            if !calls.is_empty() {
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert(Failed(Failure::Unsupported(
                        "a retry of a tool-bearing turn: steer the tool calls instead".to_owned(),
                    )));
                progress.mark();
                continue;
            }
            if let Some(feedback) = feedback {
                let assistant = MessageParts::Assistant {
                    id: response.message_id.clone(),
                    content: content.clone(),
                };
                content_or_fail!(
                    spawn_deferred(
                        &mut commands,
                        &mut assets,
                        run,
                        assistant,
                        next_order_in(&mut orders)
                    ),
                    &mut commands,
                    run,
                    progress
                );
                let user = MessageParts::User {
                    content: vec![UserContent::text(feedback)],
                };
                content_or_fail!(
                    spawn_deferred(
                        &mut commands,
                        &mut assets,
                        run,
                        user,
                        next_order_in(&mut orders)
                    ),
                    &mut commands,
                    run,
                    progress
                );
            }
            commands
                .entity(run)
                .remove::<AwaitingModel>()
                .insert(Assembling);
            progress.mark();
            continue;
        }

        // The assistant turn is history.
        let assistant = MessageParts::Assistant {
            id: response.message_id.clone(),
            content: content.clone(),
        };
        let assistant_entity = content_or_fail!(
            spawn_deferred(
                &mut commands,
                &mut assets,
                run,
                assistant,
                next_order_in(&mut orders)
            ),
            &mut commands,
            run,
            progress
        );

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
            let concurrency = setting(run, agent, &tool_policies)
                .map_or(1, |policy| policy.concurrency)
                .max(1);
            let inputs = setting(run, agent, &contexts)
                .map(|spec| spec.0.for_dispatch())
                .unwrap_or_default();
            let count = batch.len();
            for (index, call, key) in batch {
                let mut effect = commands.spawn((
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
                .insert((Batch { calls: count }, TurnAssistant(assistant_entity)));
            commands
                .entity(run)
                .remove::<AwaitingModel>()
                .insert(ResolvingTools);
            progress.mark();
            continue;
        }

        match (*mode, output_tool) {
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
                            content_or_fail!(
                                replace_deferred(
                                    &mut commands,
                                    &mut assets,
                                    assistant_entity,
                                    MessageParts::Assistant {
                                        id: response.message_id.clone(),
                                        content: final_content
                                    }
                                ),
                                &mut commands,
                                run,
                                progress
                            );
                            commands
                                .entity(run)
                                .remove::<AwaitingModel>()
                                .insert((RunResult(output), Settled));
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
                            commands
                                .entity(turn)
                                .insert(Reprompt(reprompt.to_message()));
                            content_or_fail!(
                                spawn_deferred(
                                    &mut commands,
                                    &mut assets,
                                    run,
                                    reprompt,
                                    next_order_in(&mut orders)
                                ),
                                &mut commands,
                                run,
                                progress
                            );
                            commands
                                .entity(run)
                                .remove::<AwaitingModel>()
                                .insert((OutputRetries(retries.0 + 1), Assembling));
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
                            content: vec![UserContent::text(policy::text::reprompt_text_answer(
                                name,
                            ))],
                        };
                        commands
                            .entity(turn)
                            .insert(Reprompt(reprompt.to_message()));
                        content_or_fail!(
                            spawn_deferred(
                                &mut commands,
                                &mut assets,
                                run,
                                reprompt,
                                next_order_in(&mut orders)
                            ),
                            &mut commands,
                            run,
                            progress
                        );
                        commands
                            .entity(run)
                            .remove::<AwaitingModel>()
                            .insert((OutputRetries(retries.0 + 1), Assembling));
                    }
                    None => {
                        commands
                            .entity(run)
                            .remove::<AwaitingModel>()
                            .insert((RunResult(policy::answer_text(&content)), Settled));
                    }
                }
            }
            (OutputKind::Tool, None)
            | (OutputKind::Auto | OutputKind::Native | OutputKind::Prompted, _) => {
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert((RunResult(policy::answer_text(&content)), Settled));
            }
        }
        progress.mark();
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
        commands
            .entity(run)
            .remove::<ResolvingTools>()
            .insert(cancelled);
    } else if !is_tool_call && !materialised && awaiting {
        commands.entity(turn).insert(Materialised);
        commands
            .entity(run)
            .remove::<AwaitingModel>()
            .insert(cancelled);
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
        .remove::<(Assembling, AwaitingModel, ResolvingTools, LoadingMemory)>()
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
