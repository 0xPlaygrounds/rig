//! SN-22: race two models, keep the faster.
//!
//! `RIG_HOME/race.toml` names virtual models, each raced over two catalog
//! models: `race-fast = ["provider-a/model-x", "provider-b/model-y"]`. Each
//! is a catalog model of the vendor `race` (`race/race-fast`), so `/model`
//! lists it and an agent, a subagent or a restored session picks it like
//! any model; `/model race-fast` picks it too.
//!
//! A turn's model call to a race model goes to its first member, and the
//! plugin sends a sibling turn call of the same turn to the other one, both
//! built by the kernel from the agent and each fitted to its own model.
//! The first to stream content (text, reasoning or a tool call) wins: it
//! [`Supersedes`] the other, which the kernel cancels and never puts in
//! the conversation. A member that fails before it streamed drops out the
//! same way; if both fail, the last failure lists every member's error. A
//! winner that fails mid-stream is superseded by a turn call to the other
//! member alone ("restarting on …"), so the stored reply is that member's
//! whole reply. What a cancelled member was sent is counted in the
//! spending at its model's price, its input tokens estimated, and `/race`
//! shows each race model's wins, median time to first content and
//! cancelled spend.

use std::collections::BTreeMap;
use std::path::Path;

use rig_core::catalog::{Catalog, ModelSpec, Transport};
use rig_core::completion::{ModelRef, Usage};
use rig_core::effect::{FamilyDescriptor, HandlerDescriptor, model_key};
use rig_core::error::ErrorKind;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_harness::prelude::*;
use rig_harness::rig_ecs::effects::Handler;
use rig_memory::{HeuristicTokenCounter, TokenCounter};
use rig_models::ModelItem;
use rig_telemetry::{SessionSpending, Spending, TurnSpending};
use serde::{Deserialize, Serialize};

/// The file under `RIG_HOME`.
pub const FILE: &str = "race.toml";

/// The vendor the race models are filed under: `race/<name>`.
pub const VENDOR: &str = "race";

/// How many times to first content are kept per member.
const SAMPLES: usize = 1000;

/// The race models of `race.toml` (or inserted before the plugin), by
/// name, each with its two members' catalog references.
#[derive(Resource, Clone, Debug, Default)]
pub struct RaceModels {
    /// Each race model's members, by its name.
    pub models: BTreeMap<String, Vec<String>>,
    /// What in the file could not be used.
    pub problems: Vec<String>,
}

impl RaceModels {
    /// The race models of the file at `path`; none when there is no file.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text),
            Err(why) if why.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(why) => Self {
                problems: vec![format!("Could not read {}: {why}.", path.display())],
                ..Self::default()
            },
        }
    }

    /// The race models `text`, in the file's shape, names: each with
    /// exactly two `vendor/model` members.
    pub fn parse(text: &str) -> Self {
        let file: BTreeMap<String, Vec<String>> = match toml::from_str(text) {
            Ok(file) => file,
            Err(why) => {
                return Self {
                    problems: vec![format!("{FILE}: {why}")],
                    ..Self::default()
                };
            }
        };
        let mut races = Self::default();
        for (name, members) in file {
            let named = members.iter().all(|member| member.contains('/'));
            if members.len() == 2 && named && !name.contains('/') {
                races.models.insert(name, members);
            } else {
                races.problems.push(format!(
                    "{FILE}: `{name}` skipped: it needs two members, each `vendor/model`."
                ));
            }
        }
        races
    }
}

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaceSystems {
    /// In `PrepareSystems::Edit`: each call to a race model routed to a
    /// member, a turn's sent to both.
    Route,
    /// In `ReviewSystems::Edit`: the failure of a race every member failed
    /// lists each member's error.
    ListErrors,
    /// In `ReviewSystems::Act`: a member that failed drops out, or, if it
    /// had won, the other member is asked alone.
    DropOut,
}

/// What each race model did, kept with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RaceStats(pub BTreeMap<String, RaceTally>);

/// One race model's tally.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RaceTally {
    /// Each member's, by catalog reference.
    pub members: BTreeMap<String, MemberTally>,
    /// Requests cancelled because the other member answered first.
    pub cancelled: u64,
    /// The input tokens they were sent, estimated.
    pub cancelled_tokens: u64,
    /// What they cost in USD, at their models' list prices.
    pub cancelled_cost: f64,
}

/// One member's tally.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemberTally {
    /// Replies of its that went into the conversation.
    pub wins: u64,
    /// Its times to first content as the winner, newest last, in ms.
    pub first_content_ms: Vec<u64>,
}

/// On a turn whose current model call is raced.
#[derive(Component, Debug)]
struct Racing {
    /// The race model.
    model: String,
    /// Its members.
    members: Vec<String>,
    /// The member that streamed first.
    winner: Option<usize>,
    /// The members that failed before streaming, with their errors.
    failures: Vec<(usize, String)>,
}

/// On a turn call of a race: the member it goes to, and whether it is
/// sent to it alone, after the winner failed.
#[derive(Component, Clone, Copy, Debug)]
struct Racer {
    member: usize,
    solo: bool,
}

/// On a racer whose failure was handled.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct DroppedOut;

/// On the racer that carries the failure of a race every member failed.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct ListsErrors;

/// Races the models of `race.toml`; adds `/race`.
#[derive(Default)]
pub struct RaceModelsPlugin;

impl Plugin for RaceModelsPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<RaceModels>() {
            let path = RigHome::of(app.world()).root().join(FILE);
            app.insert_resource(RaceModels::load(&path));
        }
        let names: Vec<(String, String)> = app
            .world()
            .resource::<RaceModels>()
            .models
            .iter()
            .map(|(name, members)| (name.clone(), members.join(" | ")))
            .collect();
        for (name, members) in names {
            let label = format!("{name}  race of {members}");
            let command = format!("model {VENDOR}/{name}");
            app.world_mut().spawn(ModelItem::new(name, label, command));
        }
        app.init_resource::<RaceStats>()
            .configure_sets(
                AgentPass,
                (
                    RaceSystems::Route.in_set(PrepareSystems::Edit),
                    RaceSystems::ListErrors.in_set(ReviewSystems::Edit),
                    RaceSystems::DropOut.in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    route.in_set(RaceSystems::Route),
                    list_errors.in_set(RaceSystems::ListErrors),
                    drop_out.in_set(RaceSystems::DropOut),
                ),
            )
            .add_systems(Startup, report_problems)
            .add_systems(Update, claim.before(AgentLoop))
            .add_observer(claim_at_once)
            .add_observer(tally)
            .add_command(
                "race",
                "Show each race model's wins, time to first content and cancelled spend",
                show,
            );
    }

    /// Files each race model in the catalog, once every plugin added its
    /// models, so its window and tools are its members'.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let races = world.resource::<RaceModels>().models.clone();
        let mut models = world.get_resource_or_init::<Models>();
        let Ok(vendor) = ProviderId::declare(VENDOR) else {
            return;
        };
        for (name, members) in &races {
            let specs: Vec<ModelSpec> = members
                .iter()
                .filter_map(|member| models.0.catalog().resolve(member).ok())
                .map(|found| found.spec.clone())
                .collect();
            let mut spec = ModelSpec::new(vendor, name)
                .with_display_name(format!("{name} (race of {})", members.join(", ")))
                .with_tools(specs.iter().all(|spec| spec.tools));
            let windows = specs.iter().map(|spec| spec.context_window);
            if let Some(Some(window)) = windows.clone().min()
                && windows.len() == members.len()
            {
                spec = spec.with_context_window(window);
            }
            if let Some(first) = specs.first() {
                spec = spec.with_reasoning(first.reasoning.clone());
            }
            models.0.catalog_mut().insert(spec);
        }
        models.0.provide(vendor, serve_nothing);
    }
}

/// A race model's own handler: it is never sent anything, since each call
/// goes to a member.
fn serve_nothing(
    spec: &ModelSpec,
    _: &Catalog,
    _: &Transport,
) -> Result<ErasedHandler, ConnectError> {
    Ok(unavailable(
        &spec.reference(),
        "a race model's calls go to its members",
    ))
}

/// A handler named `reference` that answers every call with `why`.
fn unavailable(reference: &str, why: impl Into<String>) -> ErasedHandler {
    let descriptor = HandlerDescriptor {
        key: model_key(reference),
        family: FamilyDescriptor::Completion {
            model: ModelRef::from(reference),
            capabilities: Default::default(),
        },
        layers: Vec::new(),
    };
    ErasedHandler::unavailable(descriptor, why)
}

/// Says what in `race.toml` was skipped.
fn report_problems(races: Res<RaceModels>, mut notices: MessageWriter<Notice>) {
    for problem in &races.problems {
        notices.write(Notice::warn(None, problem.clone()));
    }
}

/// A model call as [`route`] finds it.
type Routed<'a> = (
    Entity,
    &'a mut ModelRequest,
    &'a Effect,
    Option<&'a ForAgent>,
    Option<&'a CallOf>,
    Option<&'a Racer>,
    Has<TurnCall>,
);

/// Routes each call admitted for a race model to a member: a racer to its
/// own, a new turn call to the first member while a sibling call of the
/// same turn goes to the second, and any other call, such as a summary, to
/// the first. Each gets its member's own cache setting; Start fits the
/// rest of the request to it.
fn route(
    mut calls: Query<Routed, (Added<Admitted>, ModelCalls)>,
    agents: Query<(&Connection, &Effort)>,
    races: Res<RaceModels>,
    models: Res<Models>,
    mut commands: Commands,
) {
    for (call, mut request, effect, agent, of, racer, turn_call) in &mut calls {
        let own = agent.and_then(|agent| agents.get(agent.0).ok());
        let target = request.connection.as_ref().or(own.map(|(to, _)| to));
        let Some(race) = target
            .filter(|to| to.spec.provider.vendor() == VENDOR)
            .map(|to| to.spec.clone())
        else {
            continue;
        };
        let name = race.id.clone();
        let Some(members) = races.models.get(&name) else {
            continue;
        };
        let member = match (racer, of, agent) {
            (Some(racer), ..) => racer.member,
            (None, Some(&CallOf(turn)), Some(&ForAgent(agent))) if turn_call => {
                commands.entity(turn).insert(Racing {
                    model: name.clone(),
                    members: members.clone(),
                    winner: None,
                    failures: Vec::new(),
                });
                commands.entity(call).insert(Racer {
                    member: 0,
                    solo: false,
                });
                commands.spawn((
                    TurnCall,
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::child_of(effect.parent),
                    Racer {
                        member: 1,
                        solo: false,
                    },
                ));
                0
            }
            _ => 0,
        };
        let Some(reference) = members.get(member) else {
            continue;
        };
        // A member that cannot be reached fails at once, saying why, and
        // drops out like any member that fails.
        let connection = models.connect(reference).unwrap_or_else(|why| Connection {
            spec: race,
            handler: Handler(unavailable(reference, why)),
        });
        let effort = own.and_then(|(_, effort)| effort.0);
        request.request.options.cache = connection.spec.default_options(effort).cache;
        request.connection = Some(connection);
    }
}

/// Whether `event` is content: text, reasoning, or a tool call.
fn is_content(event: &StreamEvent) -> bool {
    match event {
        StreamEvent::Text { text, .. } | StreamEvent::Reasoning { text, .. } => !text.is_empty(),
        StreamEvent::Start { kind, .. } => matches!(kind, PartKind::ToolCall | PartKind::Image),
        StreamEvent::Arguments { .. } | StreamEvent::End { .. } => true,
    }
}

/// Claims the race for the first racer that streams content, as its
/// stream's first item comes: a delta moves no stage, so this does not
/// wait for the next frame's [`claim`].
fn claim_at_once(_: On<Add<Streamed>>, mut commands: Commands) {
    commands.run_system_cached(claim);
}

/// The first racer of a race to stream content wins: it supersedes the
/// other one, which the kernel then cancels and never commits, and its
/// time to first content is kept.
fn claim(
    mut deltas: MessageReader<StreamDelta>,
    racers: Query<
        (
            &Racer,
            &CallOf,
            Option<&Streamed>,
            Option<&Started>,
            Has<Supersedes>,
        ),
        (With<TurnCall>, Without<Settled>, Without<SupersededBy>),
    >,
    mut turns: Query<(&mut Racing, &Calls)>,
    mut stats: ResMut<RaceStats>,
    mut commands: Commands,
) {
    for delta in deltas.read() {
        if !is_content(&delta.event) {
            continue;
        }
        let Ok((racer, &CallOf(turn), streamed, started, superseding)) = racers.get(delta.effect)
        else {
            continue;
        };
        let Ok((mut racing, calls)) = turns.get_mut(turn) else {
            continue;
        };
        if racer.solo || racing.winner.is_some() {
            continue;
        }
        racing.winner = Some(racer.member);
        let loser = calls.iter().find(|&call| {
            call != delta.effect && racers.get(call).is_ok_and(|(other, ..)| !other.solo)
        });
        if let (Some(loser), false) = (loser, superseding) {
            commands.entity(delta.effect).insert(Supersedes(loser));
        }
        if let (Some(streamed), Some(started), Some(member)) =
            (streamed, started, racing.members.get(racer.member))
        {
            let waited = streamed
                .first
                .duration_since(started.at)
                .unwrap_or_default();
            let tally = stats.0.entry(racing.model.clone()).or_default();
            let samples = &mut tally
                .members
                .entry(member.clone())
                .or_default()
                .first_content_ms;
            samples.push(u64::try_from(waited.as_millis()).unwrap_or(u64::MAX));
            if samples.len() > SAMPLES {
                samples.remove(0);
            }
        }
    }
}

/// The racers of a race still running.
fn running(
    calls: &Calls,
    racers: &Query<&Racer, (Without<Outcome>, Without<SupersededBy>, Without<Settled>)>,
) -> bool {
    calls
        .iter()
        .any(|call| racers.get(call).is_ok_and(|racer| !racer.solo))
}

/// When every member of a race failed before streaming, the last failure
/// lists each member's error, so the request fails once, saying why each
/// failed.
fn list_errors(
    mut failed: Query<
        (Entity, &Racer, &CallOf, &mut Outcome),
        (
            Unreviewed,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<DroppedOut>,
        ),
    >,
    alive: Query<&Racer, (Without<Outcome>, Without<SupersededBy>, Without<Settled>)>,
    turns: Query<(&Racing, &Calls)>,
    mut commands: Commands,
) {
    let mut listed: Vec<Entity> = Vec::new();
    let mut lists: Vec<(Entity, Entity, String)> = Vec::new();
    for (call, racer, &CallOf(turn), outcome) in &failed {
        let Ok((racing, calls)) = turns.get(turn) else {
            continue;
        };
        let Err(report) = &outcome.0 else {
            continue;
        };
        if racer.solo || racing.winner.is_some() || running(calls, &alive) {
            continue;
        }
        let name = racing
            .members
            .get(racer.member)
            .cloned()
            .unwrap_or_default();
        match lists.iter_mut().find(|(of, ..)| *of == turn) {
            Some((.., text)) => text.push_str(&format!("; {name}: {}", report.message)),
            None => {
                let mut text: Vec<String> = racing
                    .failures
                    .iter()
                    .map(|(member, why)| {
                        let other = racing.members.get(*member).cloned().unwrap_or_default();
                        format!("{other}: {why}")
                    })
                    .collect();
                text.push(format!("{name}: {}", report.message));
                lists.push((turn, call, text.join("; ")));
            }
        }
    }
    for (turn, keeper, text) in lists {
        let Ok((racing, _)) = turns.get(turn) else {
            continue;
        };
        if let Ok((.., mut outcome)) = failed.get_mut(keeper)
            && let Err(report) = &outcome.0
        {
            let why = format!("every member of {} failed: {text}", racing.model);
            let mut listing = report.clone();
            listing.message = why;
            *outcome = Outcome(Err(listing));
            listed.push(keeper);
        }
    }
    for keeper in listed {
        commands.entity(keeper).insert(ListsErrors);
    }
}

/// A racer that failed drops out. Before any member streamed, the member
/// still running, else the one that lists every error, supersedes it, so
/// it never reaches the conversation and shows no error. A winner that
/// failed mid-stream is superseded by a call to the other member alone,
/// which takes its partial reply off the screen.
fn drop_out(
    failed: Query<
        (Entity, &Racer, &CallOf, &Outcome, &Effect, &ForAgent),
        (
            Unreviewed,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<DroppedOut>,
        ),
    >,
    alive: Query<&Racer, (Without<Outcome>, Without<SupersededBy>, Without<Settled>)>,
    keepers: Query<(), With<ListsErrors>>,
    superseding: Query<(), With<Supersedes>>,
    mut turns: Query<(&mut Racing, &Calls)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, racer, of, outcome, effect, &ForAgent(agent)) in &failed {
        let Err(report) = &outcome.0 else {
            continue;
        };
        if racer.solo || report.kind == ErrorKind::Cancelled {
            continue;
        }
        let Ok((mut racing, calls)) = turns.get_mut(of.0) else {
            continue;
        };
        commands.entity(call).insert(DroppedOut);
        if racing.winner == Some(racer.member) {
            let dropped = |member: usize| racing.failures.iter().any(|(it, _)| *it == member);
            let next = (0..racing.members.len()).find(|&it| it != racer.member && !dropped(it));
            let Some(reference) = next.and_then(|next| racing.members.get(next)) else {
                continue;
            };
            notices.write(Notice::info(agent, format!("restarting on {reference}")));
            commands.spawn((
                TurnCall,
                Supersedes(call),
                CallOf(of.0),
                ForAgent(agent),
                Effect::child_of(effect.parent),
                Racer {
                    member: next.unwrap_or_default(),
                    solo: true,
                },
            ));
            continue;
        }
        racing.failures.push((racer.member, report.message.clone()));
        let free = |other: Entity| other != call && !superseding.contains(other);
        let successor = calls
            .iter()
            .find(|&other| free(other) && alive.get(other).is_ok_and(|other| !other.solo))
            .or_else(|| {
                calls
                    .iter()
                    .find(|&other| free(other) && keepers.contains(other))
            });
        if let Some(successor) = successor {
            commands.entity(successor).insert(Supersedes(call));
        }
    }
}

/// Counts each racer that settled: a reply in the conversation is a win
/// for its member; a racer cancelled because the other answered first is
/// counted as spending what it was sent, its input tokens estimated and
/// priced at its model's list price.
fn tally(
    settled: On<Add<Settled>>,
    calls: Query<(
        &Racer,
        &CallOf,
        Option<&Outcome>,
        Option<&SentTo>,
        &ModelRequest,
        &Effect,
        Option<&ForAgent>,
        Has<SupersededBy>,
    )>,
    turns: Query<&Racing>,
    mut spent: Query<&mut Spending>,
    mut turn_spent: Query<&mut TurnSpending>,
    session: Option<ResMut<SessionSpending>>,
    mut stats: ResMut<RaceStats>,
) {
    let Ok((racer, &CallOf(turn), outcome, sent, request, effect, agent, superseded)) =
        calls.get(settled.entity)
    else {
        return;
    };
    let Ok(racing) = turns.get(turn) else {
        return;
    };
    let Some(member) = racing.members.get(racer.member) else {
        return;
    };
    let tally = stats.0.entry(racing.model.clone()).or_default();
    if outcome.and_then(Outcome::completion).is_some() && !superseded {
        tally.members.entry(member.clone()).or_default().wins += 1;
        return;
    }
    let (Some(SentTo(connection)), None, true) = (sent, outcome, superseded) else {
        return;
    };
    let counter = HeuristicTokenCounter::default();
    let messages = request.request.chat_history.iter();
    let tokens = messages
        .map(|message| counter.count(message))
        .sum::<usize>();
    let tokens = u64::try_from(tokens).unwrap_or(u64::MAX);
    let mut usage = Usage::new();
    usage.input_tokens = Some(tokens);
    usage.output_tokens = Some(0);
    usage.total_tokens = Some(tokens);
    usage.cost = connection
        .spec
        .pricing
        .as_ref()
        .and_then(|pricing| pricing.cost(&usage));
    tally.cancelled += 1;
    tally.cancelled_tokens += tokens;
    tally.cancelled_cost += usage.cost.map_or(0.0, |cost| cost.total);
    let model = connection.spec.reference();
    let purpose = effect.purpose.as_deref();
    if let Some(mut spending) = agent.and_then(|agent| spent.get_mut(agent.0).ok()) {
        spending.record(Some(&model), purpose, &usage);
    }
    if let Ok(mut turn_spending) = turn_spent.get_mut(turn) {
        turn_spending.0.record(&usage);
    }
    if let Some(mut session) = session {
        session.0.record(Some(&model), purpose, &usage);
    }
}

/// The median of `samples`, in ms.
fn median(samples: &[u64]) -> Option<u64> {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted.get(sorted.len() / 2).copied()
}

/// `/race`: each race model's wins per member, median time to first
/// content per member, and cancelled spend.
fn show(
    In(args): In<CommandArgs>,
    races: Res<RaceModels>,
    stats: Res<RaceStats>,
    mut notices: MessageWriter<Notice>,
) {
    if races.models.is_empty() {
        let text = format!("No race models: name some in RIG_HOME/{FILE}.");
        notices.write(Notice::info(args.agent, text));
        return;
    }
    let mut lines = Vec::new();
    for (name, members) in &races.models {
        let tally = stats.0.get(name).cloned().unwrap_or_default();
        let total: u64 = tally.members.values().map(|member| member.wins).sum();
        lines.push(format!("{name}: {total} replies"));
        for member in members {
            let own = tally.members.get(member).cloned().unwrap_or_default();
            let share = (own.wins * 100).checked_div(total).unwrap_or(0);
            let first = median(&own.first_content_ms).map_or_else(
                || "no first content yet".to_owned(),
                |ms| format!("median first content {ms} ms"),
            );
            lines.push(format!("  {member}: won {} ({share}%), {first}", own.wins));
        }
        lines.push(format!(
            "  cancelled: {} requests, ~{} input tokens, ${:.4}",
            tally.cancelled, tally.cancelled_tokens, tally.cancelled_cost
        ));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

#[cfg(test)]
mod tests;
