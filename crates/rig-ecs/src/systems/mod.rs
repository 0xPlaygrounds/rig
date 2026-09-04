//! The agent's systems: one per named set of [`RigSet`], in the bus
//! module's `RigSchedule`, run to quiescence by its runner.
//!
//! | set | true before | written during |
//! |---|---|---|
//! | `Advance` | a run in `Assembling` has no fresh turn | a turn is spawned `ChildOf` the run with its adverts and attachments, or the run fails `MaxTurns` |
//! | `Select` | a run may lack a model of its own | the agent's `UsesModel` is copied to the run |
//! | `Assemble` | a fresh turn's graph is complete | the fold spawns the turn's effect; the run is `AwaitingModel` |
//! | `Patch` | the folded effect is a `PendingEffect` | a user system may rewrite it (the second steering slot) |
//! | *the bus's `Gate`, `Dispatch`, `Collect`, `Judge`* | | |
//! | `Fold` | the effect may have streamed or landed | `Outputs` on the turn, per tick |
//! | `Judge` | the turn's outputs are complete | a user system may rewrite them |
//! | `Materialise` | a complete turn is unread | the assistant utterance, the answer, a reprompt, an invalid call, or a failure |
//! | `Settle` | a run settled or failed this pass | nothing yet (observers fire on `Settled`/`Failed`) |
//!
//! The first steering slot is any system before `Assemble`: it edits the
//! graph (utterances, documents, grants, settings).

use bevy_app::{App, Plugin};
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    completion::message::{AssistantContent, ToolResultContent, UserContent},
    effect::{EffectKind, FamilyDescriptor, Outcome},
    error::ErrorKind,
};

use crate::{
    agent::{
        AdditionalParams, Advert, Assembling, Attachment, AwaitingModel, Context, Cursor,
        DocumentId, DocumentProps, DocumentText, Failed, Failure, Grant, InvalidCall, InvalidCalls,
        MaxTokens, MaxTurns, MessageParts, Order, OrderCounter, Output, OutputKind, OutputRetries,
        OutputToolName, Outputs, Parts, Preamble, Reprompt, Resolution, Run, RunCounter, RunOf,
        RunResult, RunSeq, Settled, Streamed, Temperature, ToolChoiceSpec, Turn, Unhandled, Usage,
        UsesModel, Utterance,
    },
    bus::{
        Bound, BusPlugin, BusSet, EffectOutcome, PendingEffect, Progress, RigSchedule, Scope,
        Streamed as BusStreamed,
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
            fold.in_set(RigSet::Fold),
            (resolve_invalid_defaults, materialise)
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
    fresh: Query<(Entity, &ChildOf), With<Fresh>>,
    runs: Query<(&RunOf, &RunSeq, &Streamed, &UsesModel, &OutputToolName), With<Run>>,
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
    let mut turns: Vec<(Entity, Entity, RunSeq)> = fresh
        .iter()
        .filter_map(|(turn, child_of)| {
            let run = child_of.parent();
            runs.get(run)
                .ok()
                .map(|(_, seq, _, _, _)| (turn, run, *seq))
        })
        .collect();
    turns.sort_by_key(|(_, _, seq)| *seq);

    for (turn, run, _) in turns {
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

        let tools: Vec<&Bound> = links_in_order(turn, &children, &adverts)
            .into_iter()
            .filter_map(|Advert(tool)| bound.get(*tool).ok())
            .collect();
        let attached: Vec<rig_core::completion::Document> =
            links_in_order(turn, &children, &attachments)
                .into_iter()
                .filter_map(|Attachment(document)| documents.get(*document).ok())
                .map(|(id, text, props)| rig_core::completion::Document {
                    id: id.0.clone(),
                    text: text.0.clone(),
                    additional_props: props.map(|props| props.0.clone()).unwrap_or_default(),
                })
                .collect();

        let preamble = setting(run, agent, &preambles).and_then(|preamble| preamble.0.as_deref());
        let temperature = setting(run, agent, &temperatures).and_then(|t| t.0);
        let max_tokens = setting(run, agent, &max_tokens).and_then(|m| m.0);
        let additional_params = setting(run, agent, &params).and_then(|p| p.0.as_ref());
        let tool_choice = setting(run, agent, &choices).and_then(|c| c.0.as_ref());
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
        let resolved = policy::resolve_output(
            output.mode,
            output.schema.is_some(),
            granted_names.len(),
            callable,
            composes,
        );
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
            utterances: history.iter().map(|(_, parts)| &parts.0).collect(),
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
            .remove::<Fresh>()
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

/// `RigSet::Materialise`: a complete, unread turn becomes graph — the
/// assistant utterance (unless the turn is empty), the answer and
/// `Settled`, a reprompt and another turn, an invalid call awaiting its
/// resolution, or `Failed`.
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn materialise(
    mut commands: Commands,
    turns: Query<(Entity, &ChildOf, &Outputs, &Folded), Unread>,
    effects: Query<(&ChildOf, &EffectOutcome), With<PendingEffect>>,
    runs: Query<(&RunOf, &Cursor, &OutputRetries, &OutputToolName, &Usage), With<AwaitingModel>>,
    children: Query<&Children>,
    adverts: Query<(&Advert, &Order)>,
    bound: Query<&Bound>,
    outputs: Query<&Output>,
    max_turns: Query<&MaxTurns>,
    invalid_calls: Query<(Entity, &ChildOf, &InvalidCall, &Resolution)>,
    mut orders: ResMut<OrderCounter>,
    mut progress: ResMut<Progress>,
) {
    for (turn, turn_of, outs, Folded(mode)) in &turns {
        let run = turn_of.parent();
        let Ok((RunOf(agent), cursor, retries, minted, usage)) = runs.get(run) else {
            continue;
        };
        let agent = *agent;

        // Pending invalid calls of this turn: consumed first.
        let pending: Vec<(Entity, &InvalidCall, Resolution)> = invalid_calls
            .iter()
            .filter(|(_, child_of, _, _)| child_of.parent() == turn)
            .map(|(entity, _, call, resolution)| (entity, call, *resolution))
            .collect();
        if !pending.is_empty() {
            let mut failed = None;
            let mut ignored = false;
            for (entity, call, resolution) in &pending {
                match resolution {
                    Resolution::Fail => {
                        failed = Some(Failure::UnknownToolCall {
                            name: call.name.clone(),
                        });
                    }
                    Resolution::Ignore => {
                        ignored = true;
                    }
                    Resolution::Retry | Resolution::Repair | Resolution::Skip => {
                        failed = Some(Failure::Unsupported(format!(
                            "resolution {resolution:?} for `{}`: a later stage's",
                            call.name
                        )));
                    }
                }
                commands.entity(*entity).despawn();
            }
            if let Some(failure) = failed {
                commands.entity(turn).insert(Materialised);
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert(Failed(failure));
                progress.mark();
                continue;
            }
            if ignored {
                // The invalid calls are dropped from the turn; what is left
                // is read as the answer, and an empty turn is an empty one.
                let kept: Vec<AssistantContent> = outs
                    .content
                    .iter()
                    .filter(|part| match part {
                        AssistantContent::ToolCall(call) => !pending
                            .iter()
                            .any(|(_, invalid, _)| invalid.id == call.id.to_string()),
                        AssistantContent::Text(_)
                        | AssistantContent::Reasoning(_)
                        | AssistantContent::Image(_) => true,
                    })
                    .cloned()
                    .collect();
                commands.entity(turn).insert(Materialised);
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert((RunResult(policy::answer_text(&kept)), Settled));
                progress.mark();
                continue;
            }
        }

        if !outs.done {
            continue;
        }
        let Some((_, EffectOutcome(outcome))) = effects
            .iter()
            .find(|(child_of, _)| child_of.parent() == turn)
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
        commands.entity(run).insert(Usage(usage.0 + response.usage));
        let content = &response.choice;

        // An empty turn is not history, and answers nothing.
        if policy::turn_is_empty(content) {
            commands
                .entity(run)
                .remove::<AwaitingModel>()
                .insert((RunResult(String::new()), Settled));
            progress.mark();
            continue;
        }

        let granted: Vec<String> = links_in_order(turn, &children, &adverts)
            .into_iter()
            .filter_map(|Advert(tool)| bound.get(*tool).ok())
            .filter_map(|bound| match &bound.descriptor.family {
                FamilyDescriptor::Tool { name, .. } => Some(name.clone()),
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
                !granted.contains(&call.function.name)
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

        // A call to a granted tool: dispatched as a tool effect in a later
        // stage; here the run cannot go on.
        if let Some(call) = calls
            .iter()
            .find(|call| granted.contains(&call.function.name))
        {
            commands
                .entity(run)
                .remove::<AwaitingModel>()
                .insert(Failed(Failure::Unsupported(format!(
                    "the tool call `{}`: tool dispatch is stage 3's",
                    call.function.name
                ))));
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
                            .insert((RunResult(policy::answer_text(content)), Settled));
                    }
                }
            }
            (OutputKind::Tool, None)
            | (OutputKind::Auto | OutputKind::Native | OutputKind::Prompted, _) => {
                commands
                    .entity(run)
                    .remove::<AwaitingModel>()
                    .insert((RunResult(policy::answer_text(content)), Settled));
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
    effects: Query<&ChildOf, With<PendingEffect>>,
    turns: Query<&ChildOf, (With<Turn>, Without<Materialised>)>,
    runs: Query<(), With<AwaitingModel>>,
    mut commands: Commands,
) {
    let effect = removed.event().entity;
    let Ok(turn_of) = effects.get(effect) else {
        return;
    };
    let turn = turn_of.parent();
    let Ok(run_of) = turns.get(turn) else {
        return;
    };
    let run = run_of.parent();
    if runs.get(run).is_err() {
        return;
    }
    commands.entity(turn).insert(Materialised);
    commands
        .entity(run)
        .remove::<AwaitingModel>()
        .insert(Failed(Failure::Cancelled(rig_core::serve::cancelled())));
}
