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
//! | `Materialise` | a complete turn is unread, or its batch has landed | `land_batch`: the results as one user utterance, or a failure; `materialise`: the assistant utterance, the answer, a reprompt, an invalid call, the tool batch, or a failure |
//! | `Settle` | a run settled or failed this pass | nothing yet (observers fire on `Settled`/`Failed`) |
//!
//! The first steering slot is any system before `Assemble`: it edits the
//! graph (utterances, documents, grants, settings).

use bevy_app::{App, Plugin};
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    completion::message::{AssistantContent, ToolChoice, ToolResultContent, UserContent},
    effect::{EffectKind, FamilyDescriptor, Outcome},
    error::ErrorKind,
};

use crate::{
    agent::{
        AdditionalParams, Advert, Assembling, Attachment, AwaitingModel, Batch, Cancelled, Context,
        Cursor, DocumentId, DocumentProps, DocumentText, Failed, Failure, Grant, InvalidCall,
        InvalidCalls, InvalidRetries, MaxTokens, MaxTurns, MessageParts, Order, OrderCounter,
        Output, OutputKind, OutputRetries, OutputToolName, Outputs, Parts, Preamble, Reprompt,
        RequestPatch, Resolution, ResolvingTools, Retry, Run, RunCounter, RunOf, RunResult, RunSeq,
        Settled, Streamed, Temperature, ToolCallSlot, ToolChoiceSpec, ToolContextSpec, ToolPolicy,
        Turn, Unhandled, Usage, UsesModel, Utterance,
    },
    bus::{
        Bound, BusPlugin, BusSet, EffectOutcome, Held, Issued, PendingEffect, Progress,
        RigSchedule, Scope, Streamed as BusStreamed, ToolInputs,
    },
    policy::{self, RequestGraph},
};

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
    &'static Streamed,
    &'static UsesModel,
    &'static OutputToolName,
);
/// What `materialise` reads of a run awaiting its model.
pub type AwaitingView = (
    &'static RunOf,
    &'static Cursor,
    &'static OutputRetries,
    &'static InvalidRetries,
    &'static OutputToolName,
    &'static Usage,
);
/// What the cancel observer reads of a turn: its run, whether it was
/// read, whether its batch is out.
pub type TurnState = (&'static ChildOf, Has<Materialised>, Has<Batch>);

/// A fresh turn: spawned by `Advance`, not yet folded by `Assemble`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Fresh;

/// The output mode the turn was folded under, pinned.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Folded(pub OutputKind);

/// A turn `Materialise` has read.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Materialised;

/// The agent runtime: [`BusPlugin`] must be added first (it owns the
/// schedule); this adds the sets, the counters and the systems.
#[derive(Debug, Clone, Default)]
pub struct AgentPlugin {
    /// Ambiguity detection for the agent's systems' ordering.
    pub ambiguity: Option<LogLevel>,
}

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<BusPlugin>(),
            "AgentPlugin needs BusPlugin added first: it runs in the bus's RigSchedule"
        );
        app.init_resource::<OrderCounter>()
            .init_resource::<RunCounter>();
        app.add_observer(effect_cancelled);
        app.add_observer(run_cancelled);
        let mut schedules = app.world_mut().resource_mut::<Schedules>();
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
                RigSet::Settle,
            )
                .chain()
                .after(BusSet::Judge),
        );
        schedule.add_systems((
            advance.in_set(RigSet::Advance),
            select.in_set(RigSet::Select),
            assemble.in_set(RigSet::Assemble),
            release_batch.in_set(RigSet::Release),
            fold.in_set(RigSet::Fold),
            (resolve_invalid_defaults, land_batch, materialise)
                .chain()
                .in_set(RigSet::Materialise),
        ));
    }
}

/// Spawn a run of `agent` with `prompt` as its first utterance, after
/// `history`: the host's one entry point. Returns the run entity.
pub fn spawn_run(
    world: &mut World,
    agent: Entity,
    history: &[MessageParts],
    prompt: &str,
    streamed: bool,
    max_turns: Option<usize>,
) -> Entity {
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
    let mut run = world.spawn((
        Run,
        RunOf(agent),
        RunSeq(seq),
        Streamed(streamed),
        Cursor::default(),
        Assembling,
        OutputRetries::default(),
        crate::agent::InvalidRetries::default(),
        OutputToolName::default(),
        Usage::default(),
        Scope(format!("{owner}/run#{seq}")),
    ));
    if let Some(limit) = max_turns {
        run.insert(MaxTurns(limit));
    }
    let run = run.id();
    for parts in history {
        spawn_utterance(world, run, parts.clone());
    }
    spawn_utterance(
        world,
        run,
        MessageParts::User {
            content: vec![UserContent::text(prompt)],
        },
    );
    run
}

/// Spawn one utterance `ChildOf` `run`, next in order.
pub fn spawn_utterance(world: &mut World, run: Entity, parts: MessageParts) -> Entity {
    let order = next_order(world);
    world
        .spawn((Utterance, parts.role(), Parts(parts), order, ChildOf(run)))
        .id()
}

/// The next [`Order`].
pub fn next_order(world: &mut World) -> Order {
    let mut counter = world.resource_mut::<OrderCounter>();
    let order = Order(counter.0);
    counter.0 += 1;
    order
}

fn next_order_in(counter: &mut OrderCounter) -> Order {
    let order = Order(counter.0);
    counter.0 += 1;
    order
}

/// A run's effective setting: its own component, else its agent's.
fn setting<'a, C: Component>(run: Entity, agent: Entity, query: &'a Query<&C>) -> Option<&'a C> {
    query.get(run).ok().or_else(|| query.get(agent).ok())
}

/// The links of one kind under `owner`, in order.
fn links_in_order<'a, L: Component>(
    owner: Entity,
    children: &Query<&Children>,
    links: &'a Query<(&L, &Order)>,
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
/// context link, in the agent's order — or, at its budget, fails
/// `MaxTurns`.
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn advance(
    mut commands: Commands,
    runs: Query<(Entity, &RunOf, &Cursor, &RunSeq), Wanting>,
    fresh: Query<&ChildOf, With<Fresh>>,
    children: Query<&Children>,
    grants: Query<(&Grant, &Order)>,
    contexts: Query<(&Context, &Order)>,
    max_turns: Query<&MaxTurns>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    let mut runs: Vec<_> = runs.iter().collect();
    runs.sort_by_key(|(_, _, _, seq)| **seq);
    for (run, RunOf(agent), cursor, _) in runs {
        if fresh.iter().any(|child_of| child_of.parent() == run) {
            continue;
        }
        let limit = setting(run, *agent, &max_turns).map_or(1, |limit| limit.0);
        if cursor.turn >= limit {
            commands
                .entity(run)
                .remove::<Assembling>()
                .insert(Failed(Failure::MaxTurns { limit }));
            progress.mark();
            continue;
        }
        let turn = commands
            .spawn((Turn, Fresh, next_order_in(&mut orders), ChildOf(run)))
            .id();
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
        commands.entity(run).insert(Cursor {
            turn: cursor.turn + 1,
        });
        progress.mark();
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

/// `RigSet::Assemble`: for every fresh turn, in run order, gather the
/// graph — the run's settings over the agent's, the utterances in order,
/// the attachments in order, the adverts in order, the model's descriptor
/// — resolve the output mode, mint the output tool's name once per run,
/// fold, and spawn the effect `ChildOf` the turn. The run is then
/// `AwaitingModel`.
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn assemble(
    mut commands: Commands,
    fresh: Query<(Entity, &ChildOf, Option<&RequestPatch>), With<Fresh>>,
    runs: Query<AssemblingView, (With<Run>, Without<Failed>)>,
    children: Query<&Children>,
    utterances: Query<(&Parts, &Order), With<Utterance>>,
    adverts: Query<(&Advert, &Order)>,
    attachments: Query<(&Attachment, &Order)>,
    documents: Query<(&DocumentId, &DocumentText, Option<&DocumentProps>)>,
    bound: Query<&Bound>,
    preambles: Query<&Preamble>,
    temperatures: Query<&Temperature>,
    max_tokens: Query<&MaxTokens>,
    params: Query<&AdditionalParams>,
    choices: Query<&ToolChoiceSpec>,
    outputs: Query<&Output>,
    mut progress: ResMut<Progress>,
) {
    let mut turns: Vec<(Entity, Entity, RunSeq, Option<&RequestPatch>)> = fresh
        .iter()
        .filter_map(|(turn, child_of, patch)| {
            let run = child_of.parent();
            runs.get(run)
                .ok()
                .map(|(_, seq, _, _, _)| (turn, run, *seq, patch))
        })
        .collect();
    turns.sort_by_key(|(_, _, seq, _)| *seq);

    for (turn, run, _, patch) in turns {
        let Ok((RunOf(agent), _, Streamed(stream), UsesModel(model), minted)) = runs.get(run)
        else {
            continue;
        };
        let agent = *agent;
        let Ok(model_bound) = bound.get(*model) else {
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
            | FamilyDescriptor::Custom { .. } => false,
        };

        let mut history: Vec<(&Order, &Parts)> = children
            .get(run)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| {
                        utterances
                            .get(child)
                            .ok()
                            .map(|(parts, order)| (order, parts))
                    })
                    .collect()
            })
            .unwrap_or_default();
        history.sort_by_key(|(order, _)| **order);

        // The turn's patch (CONTRACT §9.3), folded in as `prepare_request`
        // folded a completion-call hook's.
        let tools: Vec<&Bound> = links_in_order(turn, &children, &adverts)
            .into_iter()
            .filter_map(|Advert(tool)| bound.get(*tool).ok())
            .filter(|bound| {
                match (
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
                    .chain(history.last().map(|(_, parts)| parts.0.clone()))
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
        let output_tool = minted
            .0
            .clone()
            .unwrap_or_else(|| policy::output_tool_name(&granted_names));
        let callable = policy::output_tool_callable(tool_choice, &output_tool);
        // A committed output tool (minted on an earlier turn) stays the
        // mode whatever this turn's choice says (CONTRACT §9.3).
        let resolved = if minted.0.is_some() {
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
        if let Some(name) = &minted.0
            && granted_names.contains(&name.as_str())
        {
            // A tool granted after the mint took the output tool's name: a
            // request that advertised both would be ambiguous, so the run
            // fails here, named, as rig-agent's docs say it must.
            commands.entity(turn).remove::<Fresh>();
            commands
                .entity(run)
                .remove::<Assembling>()
                .insert(Failed(Failure::OutputToolCollision { name: name.clone() }));
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
                None => history.iter().map(|(_, parts)| &parts.0).collect(),
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
        };
        let request = policy::fold_request(&graph);
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
    effects: Query<EffectView, With<PendingEffect>>,
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
                outputs.content = response.choice.clone();
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
/// `InvalidCalls.unhandled`, written last in `Materialise`'s chain so a
/// user system before it wins.
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
/// it was issued, its outcome, whether it is held.
pub type ToolChildView = (
    Entity,
    &'static ToolCallSlot,
    Option<&'static Issued>,
    Option<&'static EffectOutcome>,
    Has<Held>,
);

/// The tool children of `turn`, by call index.
fn batch_children<'a>(
    turn: Entity,
    children: &Query<&Children>,
    tools: &'a Query<ToolChildView>,
) -> Vec<(
    Entity,
    &'a ToolCallSlot,
    bool,
    Option<&'a EffectOutcome>,
    bool,
)> {
    let mut found: Vec<_> = children
        .get(turn)
        .map(|children| {
            children
                .iter()
                .filter_map(|child| tools.get(child).ok())
                .map(|(entity, slot, issued, outcome, held)| {
                    (entity, slot, issued.is_some(), outcome, held)
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort_by_key(|(_, slot, _, _, _)| slot.index);
    found
}

/// `RigSet::Release`: a turn's batch is let through up to the run's
/// `ToolPolicy.concurrency` — every call beyond it was spawned `Held`, and
/// is released in call order as earlier ones land. Once a landed outcome
/// is one the run fails on, nothing more is released (fail-fast: in-flight
/// calls drain, unstarted ones never start).
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
        // Released and not landed — taken by `Dispatch` or about to be.
        let active = batch
            .iter()
            .filter(|(_, _, _, outcome, held)| !*held && outcome.is_none())
            .count();
        let mut free = concurrency.saturating_sub(active);
        for (entity, _, issued, _, held) in &batch {
            if free == 0 {
                break;
            }
            if *held && !issued {
                commands.entity(*entity).remove::<Held>();
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
    turns: Query<(Entity, &ChildOf, &Batch, &Outputs)>,
    runs: Query<&OutputToolName, With<ResolvingTools>>,
    children: Query<&Children>,
    tools: Query<ToolChildView>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    for (turn, turn_of, batch, outs) in &turns {
        let run = turn_of.parent();
        let Ok(minted) = runs.get(run) else {
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
            for (entity, _, issued, outcome, _) in &calls {
                if !*issued && outcome.is_none() {
                    commands.entity(*entity).despawn();
                }
            }
            commands.entity(turn).remove::<Batch>();
            commands
                .entity(run)
                .remove::<ResolvingTools>()
                .insert(Failed(failure));
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
        commands.spawn((
            Utterance,
            results.role(),
            Parts(results),
            next_order_in(&mut orders),
            ChildOf(run),
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
        progress.mark();
    }
}

/// What the pending invalid calls of a turn amount to, in precedence:
/// a `Fail` fails the run; else a `Retry` retries the turn; else a `Skip`
/// answers the call and skips the turn; else repairs and ignores edit the
/// turn's content and the turn goes on.
enum InvalidVerdict {
    Fail(String),
    Retry(InvalidCall, String),
    Skip(InvalidCall, String),
    Edit,
}

fn invalid_verdict(pending: &[(Entity, InvalidCall, Resolution)]) -> InvalidVerdict {
    if let Some((_, call, _)) = pending
        .iter()
        .find(|(_, _, resolution)| matches!(resolution, Resolution::Fail))
    {
        return InvalidVerdict::Fail(call.name.clone());
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
    effects: Query<(&ChildOf, &EffectOutcome, Option<&BusStreamed>), With<PendingEffect>>,
    runs: Query<AwaitingView, With<AwaitingModel>>,
    children: Query<&Children>,
    adverts: Query<(&Advert, &Order)>,
    bound: Query<&Bound>,
    outputs: Query<&Output>,
    max_turns: Query<&MaxTurns>,
    policies: Query<&InvalidCalls>,
    choices: Query<&ToolChoiceSpec>,
    tool_policies: Query<&ToolPolicy>,
    contexts: Query<&ToolContextSpec>,
    invalid_calls: Query<(Entity, &ChildOf, &InvalidCall, &Resolution)>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    for (turn, turn_of, mut outs, Folded(mode), retry) in &mut turns {
        let run = turn_of.parent();
        let Ok((RunOf(agent), cursor, retries, invalid_retries, minted, usage)) = runs.get(run)
        else {
            continue;
        };
        let agent = *agent;
        let tool_choice = setting(run, agent, &choices).and_then(|c| c.0.clone());

        // Pending invalid calls of this turn: consumed first.
        let pending: Vec<(Entity, InvalidCall, Resolution)> = invalid_calls
            .iter()
            .filter(|(_, child_of, _, _)| child_of.parent() == turn)
            .map(|(entity, _, call, resolution)| (entity, call.clone(), resolution.clone()))
            .collect();
        if !pending.is_empty() {
            for (entity, _, _) in &pending {
                commands.entity(*entity).despawn();
            }
            let budget = setting(run, agent, &policies).map_or(0, |p| p.retries);
            let verdict = match invalid_verdict(&pending) {
                InvalidVerdict::Retry(call, _) if invalid_retries.0 >= budget => {
                    InvalidVerdict::Fail(call.name)
                }
                InvalidVerdict::Skip(call, _) if matches!(tool_choice, Some(ToolChoice::None)) => {
                    InvalidVerdict::Fail(call.name)
                }
                verdict => verdict,
            };
            match verdict {
                InvalidVerdict::Fail(name) => {
                    commands.entity(turn).insert(Materialised);
                    commands
                        .entity(run)
                        .remove::<AwaitingModel>()
                        .insert(Failed(Failure::UnknownToolCall { name }));
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
                    let content = policy::partial_turn_at(&outs.content, events, &call.id);
                    let assistant = MessageParts::Assistant {
                        id: outs.message_id.clone(),
                        content: content.clone(),
                    };
                    commands.spawn((
                        Utterance,
                        assistant.role(),
                        Parts(assistant),
                        next_order_in(&mut orders),
                        ChildOf(run),
                    ));
                    let results = policy::invalid_peer_results(&content, &call.id, &feedback);
                    commands.spawn((
                        Utterance,
                        results.role(),
                        Parts(results),
                        next_order_in(&mut orders),
                        ChildOf(run),
                    ));
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
                                        && tool_call.id.as_str() == call.id
                                    {
                                        tool_call.function.name = to.clone();
                                    }
                                }
                            }
                            Resolution::Ignore => {
                                content.retain(|part| match part {
                                    AssistantContent::ToolCall(tool_call) => {
                                        tool_call.id.as_str() != call.id
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
        if !pending.is_empty() {
            // The usage was counted when the turn was first read.
        } else {
            commands.entity(run).insert(Usage(usage.0 + response.usage));
        }
        let content = outs.content.clone();

        // An empty turn is not history, and answers nothing.
        if policy::turn_is_empty(&content) {
            commands
                .entity(run)
                .remove::<AwaitingModel>()
                .insert((RunResult(String::new()), Settled));
            progress.mark();
            continue;
        }

        // The tools this turn advertised, by name, with their keys.
        let granted: Vec<(String, rig_core::effect::HandlerKey)> =
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
        let output_tool = minted.0.as_deref();
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
                !granted.iter().any(|(name, _)| *name == call.function.name)
                    && output_tool != Some(call.function.name.as_str())
            })
            .collect();
        if !invalid.is_empty() {
            commands.entity(turn).remove::<Materialised>();
            for call in invalid {
                commands.spawn((
                    InvalidCall {
                        id: call.id.to_string(),
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone(),
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
                commands.spawn((
                    Utterance,
                    assistant.role(),
                    Parts(assistant),
                    next_order_in(&mut orders),
                    ChildOf(run),
                ));
                let user = MessageParts::User {
                    content: vec![UserContent::text(feedback)],
                };
                commands.spawn((
                    Utterance,
                    user.role(),
                    Parts(user),
                    next_order_in(&mut orders),
                    ChildOf(run),
                ));
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
        commands.spawn((
            Utterance,
            assistant.role(),
            Parts(assistant),
            next_order_in(&mut orders),
            ChildOf(run),
        ));

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
                    effect.insert(Held);
                }
            }
            commands.entity(turn).insert(Batch { calls: count });
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
                            commands
                                .entity(run)
                                .remove::<AwaitingModel>()
                                .insert((RunResult(call.function.arguments.to_string()), Settled));
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
                            commands.spawn((
                                Utterance,
                                reprompt.role(),
                                Parts(reprompt),
                                next_order_in(&mut orders),
                                ChildOf(run),
                            ));
                            commands
                                .entity(run)
                                .remove::<AwaitingModel>()
                                .insert((OutputRetries(retries.0 + 1), Assembling));
                        }
                    }
                    None if can_reprompt => {
                        let reprompt = MessageParts::User {
                            content: vec![UserContent::text(policy::text::reprompt_text_answer(
                                name,
                            ))],
                        };
                        commands
                            .entity(turn)
                            .insert(Reprompt(reprompt.to_message()));
                        commands.spawn((
                            Utterance,
                            reprompt.role(),
                            Parts(reprompt),
                            next_order_in(&mut orders),
                            ChildOf(run),
                        ));
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
    runs: Query<(Has<AwaitingModel>, Has<ResolvingTools>), With<Run>>,
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
    let Ok((awaiting, resolving)) = runs.get(run) else {
        return;
    };
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
    reasons: Query<&Cancelled>,
    children: Query<&Children>,
    turns: Query<(), With<Turn>>,
    effects: Query<Has<Issued>, With<PendingEffect>>,
    mut commands: Commands,
) {
    let run = added.event().entity;
    let Ok(Cancelled(reason)) = reasons.get(run) else {
        return;
    };
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
    for effect in pending {
        commands.entity(effect).despawn();
    }
    for turn in unread {
        commands
            .entity(turn)
            .insert(Materialised)
            .remove::<(Batch, Fresh, RequestPatch)>();
    }
    commands
        .entity(run)
        .remove::<(Assembling, AwaitingModel, ResolvingTools)>()
        .insert(Failed(Failure::Cancelled(
            rig_core::error::ErrorReport::new(ErrorKind::Cancelled, reason.clone()),
        )));
}
