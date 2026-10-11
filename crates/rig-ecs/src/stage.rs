//! The stages every effect moves through within a frame. An effect is an
//! entity whose components are its request and its outcome, such as an
//! [`EffectRequest`] and, once answered, an [`Outcome`]; generic components
//! say how far it got. Each run of the [`AgentPass`] schedule goes through
//! the [`Stage`] sets in order, and the loop in `Update` ([`AgentLoop`])
//! runs it again until nothing moved, so an effect spawned, answered or
//! released goes as far as it can in that frame.
//!
//! A plugin steps in with an ordinary system in a stage that queries stage
//! data, ordered against other plugins with `.before` and `.after`. Every
//! effect is [`Admitted`] first, so every Prepare and Gate system sees it,
//! in their order, before it starts, wherever it was spawned. The first
//! answer wins (`insert_if_new`); a reviewer replaces an outcome with
//! `insert`, and the record keeps both. Any entity [`HoldOf`] an effect
//! keeps it where it is, and holds add up. A system that fails in Prepare,
//! Gate or Review closes its stage for that pass, whatever the app's error
//! handler: what Prepare admitted is answered with an error, Start starts
//! nothing, Commit settles nothing. From the second pass of a frame on,
//! `Time`, `Time<Real>` and `Time<Virtual>` advance by zero, so timers count
//! each frame once.
//!
//! ```
//! use rig_ecs::prelude::*;
//!
//! // Refuse every effect of the kind `deploy` when it is about to start.
//! fn no_deploys(effects: Query<(Entity, &EffectRequest), Undecided>, mut commands: Commands) {
//!     for (effect, request) in &effects {
//!         if request.kind.name() == "deploy" {
//!             commands.entity(effect).insert_if_new(Outcome::refused("no deploys today"));
//!         }
//!     }
//! }
//!
//! let mut app = App::new();
//! app.add_plugins(AgentPlugin)
//!     .add_systems(AgentPass, no_deploys.in_set(Stage::Gate));
//! ```

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use bevy_app::prelude::*;
use bevy_ecs::error::{BevyError, ErrorContext, ErrorHandler, FallbackErrorHandler};
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{LogLevel, ScheduleBuildSettings, ScheduleLabel, SingleThreadedExecutor};
use bevy_ecs::world::DeferredWorld;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use bevy_time::{Real, Time, Virtual};
use rig_core::effect::{
    self, Disposition, EffectId, EffectKind, EffectRecord, HandlerKey, tool_key,
};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::{Origin, Recorder, cancelled};
use web_time::{SystemTime, UNIX_EPOCH};

use super::agent::{AgentId, Notice};
use super::calls::{Running, Wake, arrive};
use super::effects::{Effects, Handler};

/// One pass over the stages. Plugins add their systems to it in a
/// [`Stage`]: `app.add_systems(AgentPass, gate.in_set(Stage::Gate))`.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AgentPass;

/// The system in `Update` that runs [`AgentPass`] until nothing moves. Work
/// done after it in a frame, such as an effect spawned in `PostUpdate`, waits
/// for the next frame: a system that spawns, answers or releases effects
/// runs `.before(AgentLoop)`.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AgentLoop;

/// The stages of a pass, run in this order.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// The kernel: finished tasks become their effects' [`Outcome`]s.
    Arrive,
    /// Plugins read, replace (`insert`) or hold an [`Outcome`]
    /// ([`Unreviewed`]).
    Review,
    /// The kernel settles the answered effects nothing holds, and the
    /// superseded ones, and writes their [`Recorded`].
    Commit,
    /// The kernel admits the effects spawned since the last pass.
    Admit,
    /// Plugins edit the requests admitted this pass (`Added<Admitted>`).
    Prepare,
    /// Plugins hold ([`HoldOf`]) or answer ([`Outcome`]) what is about to
    /// start ([`Undecided`]).
    Gate,
    /// The kernel starts what is admitted, unheld and unanswered. An
    /// effect it does not run stays open, [`Started`], until answered.
    Start,
}

impl Stage {
    /// Every stage, in pass order.
    const ALL: [Self; 7] = [
        Self::Arrive,
        Self::Review,
        Self::Commit,
        Self::Admit,
        Self::Prepare,
        Self::Gate,
        Self::Start,
    ];
}

/// The kernel's systems that open a stage, before any of its own.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Enter(Stage);

/// Every effect. Spawning it takes the next effect id, and sets `parent`
/// from the effect it is a `ChildOf`, when not given. Despawned before it
/// settles, it is recorded as cancelled. Insert it once: inserting it again
/// loses its id.
#[derive(Component, Debug)]
#[component(on_add = take_id, on_remove = record_unsettled)]
pub struct Effect {
    id: EffectId,
    /// The effect that asked for this one, in the record.
    pub parent: Option<EffectId>,
    /// What it is for, such as `"summary"`, in the record.
    pub purpose: Option<Cow<'static, str>>,
    /// How its first answer came.
    answered: Option<Disposition>,
    /// Whether a later answer replaced the first.
    replaced: bool,
    /// The first answer, once replaced.
    raw: Option<Result<effect::Outcome, ErrorReport>>,
}

impl Default for Effect {
    fn default() -> Self {
        Self::child_of(None)
    }
}

impl Effect {
    /// An effect asked for by `parent`, such as a successor keeping the
    /// parent of the effect it supersedes.
    pub fn child_of(parent: Option<EffectId>) -> Self {
        Self {
            id: EffectId::from_raw(0),
            parent,
            purpose: None,
            answered: None,
            replaced: false,
            raw: None,
        }
    }

    /// This effect, made for `purpose`.
    #[must_use]
    pub fn with_purpose(mut self, purpose: impl Into<Cow<'static, str>>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    /// Its id; 0 until it is spawned.
    pub fn id(&self) -> EffectId {
        self.id
    }
}

/// A plugin's rig-core effect: the kernel dispatches `kind` to `handler` at
/// Start, recorded, and its answer becomes the [`Outcome`]. Without a
/// handler it is open: Start marks it [`Started`], and it waits for any
/// system to answer it.
#[derive(Component, Clone)]
#[require(Effect)]
pub struct EffectRequest {
    /// The effect.
    pub kind: EffectKind,
    /// What answers it, if the kernel runs it.
    pub handler: Option<Handler>,
}

/// The agent a stage entity works for. Without one, the effect is the
/// app's. Despawning the agent despawns its work.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = AgentWork)]
pub struct ForAgent(pub Entity);

/// The stage entities working for this agent.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = ForAgent, linked_spawn)]
pub struct AgentWork(Vec<Entity>);

/// Inserted by the kernel in [`Stage::Admit`] on every effect spawned since
/// the last pass. Start starts only admitted effects.
#[derive(Component, Reflect, Debug, Default)]
#[component(storage = "SparseSet", on_add = moved::<Admitted>)]
#[reflect(Component, Default)]
pub struct Admitted;

/// When the kernel started the effect.
#[derive(Component, Clone, Copy, Debug)]
#[component(on_add = moved::<Started>)]
pub struct Started {
    /// The wall-clock time.
    pub at: SystemTime,
}

/// The effect's answer. The first one wins: a gate or an answering system
/// inserts it with `insert_if_new`. A reviewer replaces it with `insert`,
/// and the record keeps the first as its outcome and the last as the
/// reviewed one. Inserted on a running effect, it cancels the handler.
#[derive(Component, Clone, Debug)]
#[component(on_insert = answered, on_discard = keep_raw)]
pub struct Outcome(pub Result<effect::Outcome, ErrorReport>);

impl Outcome {
    /// A refusal for `why`, decided before the effect ran.
    pub fn refused(why: impl Into<String>) -> Self {
        Self(Err(ErrorReport::new(ErrorKind::Denied, why)))
    }
}

/// Keeps the effect it names where it is: not started, or not settled. Any
/// entity can be a hold; despawn it, or remove this, to release it. Holds
/// add up: releasing one never releases another's.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = Holds)]
pub struct HoldOf(pub Entity);

/// The holds on an effect, removed with the last one. Despawning the
/// effect despawns its holds.
#[derive(Component, Reflect, Debug)]
#[component(on_remove = moved::<Holds>)]
#[reflect(Component)]
#[relationship_target(relationship = HoldOf, linked_spawn)]
pub struct Holds(Vec<Entity>);

/// This effect replaces the one it names, such as a retry on another
/// model: the old one never starts, its task is cancelled if it runs, and
/// its outcome is never applied. It settles at the next Commit, recorded
/// as superseded.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = SupersededBy)]
pub struct Supersedes(pub Entity);

/// The effect that replaces this one.
#[derive(Component, Reflect, Debug)]
#[component(on_add = moved::<SupersededBy>)]
#[reflect(Component)]
#[relationship_target(relationship = Supersedes)]
pub struct SupersededBy(Entity);

impl SupersededBy {
    /// The successor.
    pub fn successor(&self) -> Entity {
        self.0
    }
}

/// The effect was committed. It stays until the next frame's loop starts,
/// for `Added<Settled>` readers, then it is despawned.
#[derive(Component, Reflect, Debug, Default)]
#[component(storage = "SparseSet", on_add = moved::<Settled>)]
#[reflect(Component, Default)]
pub struct Settled;

/// The record of every effect that settled or was cancelled, in settle
/// order, and of the dispatches nested in their handlers. Only effects of a
/// kind the kernel knows, such as an [`EffectRequest`], have a record.
#[derive(Message, Clone, Debug)]
pub struct Recorded(pub EffectRecord);

/// Effects about to start: admitted, not started, unanswered, unheld and
/// not superseded. A gate's filter.
pub type Undecided = (
    With<Admitted>,
    Without<Started>,
    Without<Outcome>,
    Without<Holds>,
    Without<Settled>,
    Without<SupersededBy>,
);

/// Answered effects nothing holds, not settled yet. A reviewer's filter.
pub type Unreviewed = (With<Outcome>, Without<Holds>, Without<Settled>);

/// The record `effect` leaves when it settles, for an effect in flight:
/// as it stands, or as cancelled when it has no answer yet. `None` when it
/// is not an effect, or the kernel does not know its kind.
pub fn record_of(world: &World, effect: Entity) -> Option<EffectRecord> {
    assemble(world, effect, None, false)
}

/// How many passes a frame runs at most.
const MAX_PASSES: usize = 16;

/// The loop's bookkeeping.
#[derive(Resource, Default)]
struct Passes {
    /// Bumped by the hooks whenever a stage entity moves.
    moved: u64,
    /// `moved` when the loop last ended quiet.
    quiet_at: u64,
    /// What moved last, for the warning when the loop does not settle.
    last: Option<(&'static str, Entity)>,
    /// The effects admitted this pass.
    admitted: Vec<Entity>,
    /// The failed systems already named in a notice.
    noticed: HashSet<String>,
}

thread_local! {
    /// The stage running on this thread, for the error handler.
    static STAGE: Cell<Option<Stage>> = const { Cell::new(None) };
    /// The systems that failed this pass.
    static FAILED: RefCell<Vec<Failure>> = const { RefCell::new(Vec::new()) };
    /// The app's error handler, which the kernel's forwards to.
    static APP_HANDLER: Cell<Option<ErrorHandler>> = const { Cell::new(None) };
}

/// Adds the stages, the loop and their bookkeeping.
pub(crate) fn build(app: &mut App) {
    app.init_resource::<Passes>()
        .add_message::<Recorded>()
        .edit_schedule(AgentPass, |pass| {
            pass.set_executor(SingleThreadedExecutor::new());
            if cfg!(debug_assertions) {
                // Unordered systems writing the same data in one stage are
                // reported, as their order is meaningless.
                pass.set_build_settings(ScheduleBuildSettings {
                    ambiguity_detection: LogLevel::Warn,
                    ..Default::default()
                });
            }
        })
        .configure_sets(
            AgentPass,
            (
                Enter(Stage::Arrive),
                Stage::Arrive,
                Enter(Stage::Review),
                Stage::Review,
                Enter(Stage::Commit),
                Stage::Commit,
                Enter(Stage::Admit),
                Stage::Admit,
                Enter(Stage::Prepare),
                Stage::Prepare,
                Enter(Stage::Gate),
                Stage::Gate,
                Enter(Stage::Start),
                Stage::Start,
            )
                .chain(),
        )
        .configure_sets(
            AgentPass,
            (
                Stage::Review,
                Stage::Commit,
                Stage::Admit,
                Stage::Prepare,
                Stage::Gate,
                Stage::Start,
            )
                .distributive_run_if(moved_since_quiet),
        )
        .add_systems(
            AgentPass,
            (
                arrive.in_set(Stage::Arrive),
                settle.in_set(Stage::Commit),
                admit.in_set(Stage::Admit),
                answer_unprepared.in_set(Enter(Stage::Gate)),
                start.in_set(Stage::Start),
            ),
        )
        .add_systems(
            Update,
            run_passes
                .in_set(AgentLoop)
                .run_if(any_with_component::<Effect>),
        );
    for stage in Stage::ALL {
        app.add_systems(AgentPass, entering(stage).in_set(Enter(stage)));
    }
}

/// The marker system that opens `stage`.
fn entering(stage: Stage) -> impl Fn() + Clone + Send + Sync + 'static {
    move || STAGE.set(Some(stage))
}

/// Whether anything moved since the loop last ended quiet: the stages after
/// Arrive have nothing to do otherwise.
fn moved_since_quiet(passes: Res<Passes>) -> bool {
    passes.moved != passes.quiet_at
}

/// Despawns what settled last frame, then runs passes until one moves
/// nothing, with zero time steps after the first, and the kernel's error
/// handler in place of the app's.
fn run_passes(world: &mut World, settled: &mut QueryState<Entity, With<Settled>>) {
    let done: Vec<Entity> = settled.iter(world).collect();
    for entity in done {
        // A settled effect may go with its parent.
        world.try_despawn(entity).ok();
    }
    let clocks = Clocks::save(world);
    let mut zeroed = false;
    let app_handler = world.get_resource::<FallbackErrorHandler>().copied();
    APP_HANDLER.set(Some(world.fallback_error_handler()));
    world.insert_resource(FallbackErrorHandler(fail_closed));
    let quiet = world.try_schedule_scope(AgentPass, |world, pass| {
        (0..MAX_PASSES).any(|n| {
            if n == 1 {
                Clocks::zero(world);
                zeroed = true;
            }
            FAILED.with_borrow_mut(Vec::clear);
            let before = moves(world);
            pass.run(world);
            moves(world) == before
        })
    });
    STAGE.set(None);
    FAILED.with_borrow_mut(Vec::clear);
    APP_HANDLER.set(None);
    match app_handler {
        Some(handler) => world.insert_resource(handler),
        None => {
            world.remove_resource::<FallbackErrorHandler>();
        }
    }
    if zeroed {
        clocks.restore(world);
    }
    let Some(mut passes) = world.get_resource_mut::<Passes>() else {
        return;
    };
    match quiet {
        Ok(true) => passes.quiet_at = passes.moved,
        Ok(false) => match passes.last {
            Some((what, entity)) => warn!(
                "the agent loop still moved after {MAX_PASSES} passes; the last change: {what} \
                 on {entity}"
            ),
            None => warn!("the agent loop still moved after {MAX_PASSES} passes"),
        },
        Err(_) => {}
    }
}

/// The counter of moves.
fn moves(world: &World) -> u64 {
    world
        .get_resource::<Passes>()
        .map_or(0, |passes| passes.moved)
}

/// The clocks as the frame left them, put back after passes with a zero
/// time step. A change a stage system makes to them, such as a pause, is
/// lost then.
struct Clocks {
    time: Option<Time>,
    real: Option<Time<Real>>,
    virtual_time: Option<Time<Virtual>>,
}

impl Clocks {
    fn save(world: &World) -> Self {
        Self {
            time: world.get_resource::<Time>().copied(),
            real: world.get_resource::<Time<Real>>().copied(),
            virtual_time: world.get_resource::<Time<Virtual>>().copied(),
        }
    }

    /// Steps every clock by zero: the same elapsed time, a zero delta.
    fn zero(world: &mut World) {
        if let Some(mut time) = world.get_resource_mut::<Time>() {
            time.advance_by(Duration::ZERO);
        }
        if let Some(mut time) = world.get_resource_mut::<Time<Real>>() {
            time.advance_by(Duration::ZERO);
        }
        if let Some(mut time) = world.get_resource_mut::<Time<Virtual>>() {
            time.advance_by(Duration::ZERO);
        }
    }

    fn restore(self, world: &mut World) {
        if let (Some(saved), Some(mut time)) = (self.time, world.get_resource_mut::<Time>()) {
            *time = saved;
        }
        if let (Some(saved), Some(mut time)) = (self.real, world.get_resource_mut::<Time<Real>>()) {
            *time = saved;
        }
        if let (Some(saved), Some(mut time)) =
            (self.virtual_time, world.get_resource_mut::<Time<Virtual>>())
        {
            *time = saved;
        }
    }
}

/// The error handler during the passes: notes a system that failed in a
/// stage, then hands the error to the app's handler.
fn fail_closed(error: BevyError, context: ErrorContext) {
    let in_system = matches!(
        context,
        ErrorContext::System { .. } | ErrorContext::RunCondition { .. }
    );
    if in_system && let Some(stage) = STAGE.get() {
        let message = error.to_string();
        let failure = Failure {
            stage,
            system: context.to_string(),
            error: message.lines().next().unwrap_or_default().to_owned(),
        };
        FAILED.with_borrow_mut(|failed| failed.push(failure));
    }
    let handler = APP_HANDLER
        .get()
        .unwrap_or(FallbackErrorHandler::default().0);
    handler(error, context);
}

/// A system that failed in a stage.
#[derive(Clone)]
struct Failure {
    stage: Stage,
    /// Which system, as Bevy names it: "System `name` failed".
    system: String,
    /// The first line of its error.
    error: String,
}

/// The first system that failed in `stage` this pass.
fn failed_in(stage: Stage) -> Option<Failure> {
    FAILED.with_borrow(|failed| failed.iter().find(|failed| failed.stage == stage).cloned())
}

/// The notice that a system failed and what that closed, once per system
/// and stage.
fn closed(passes: &mut Passes, failure: Failure, what: &str) -> Option<Notice> {
    let Failure {
        stage,
        system,
        error,
    } = failure;
    let text = format!("{system} in {stage:?} ({error}): {what}.");
    passes
        .noticed
        .insert(format!("{stage:?} {system}"))
        .then(|| Notice::error(None, text))
}

/// Counts a move of a stage entity, by the component `C`.
fn moved<C: Component>(mut world: DeferredWorld, context: HookContext) {
    if let Some(mut passes) = world.get_resource_mut::<Passes>() {
        passes.moved = passes.moved.wrapping_add(1);
        passes.last = Some((std::any::type_name::<C>(), context.entity));
    }
}

/// Gives a new effect its id, and its parent from the effect it is a
/// `ChildOf`.
fn take_id(mut world: DeferredWorld, context: HookContext) {
    let entity = context.entity;
    let id = world.get_resource::<Effects>().map(Effects::next_id);
    let parent = world
        .get::<ChildOf>(entity)
        .and_then(|of| world.get::<Effect>(of.parent()))
        .map(Effect::id);
    if let Some(mut effect) = world.get_mut::<Effect>(entity) {
        if let Some(id) = id {
            effect.id = id;
        }
        if effect.parent.is_none() {
            effect.parent = parent;
        }
    }
    moved::<Effect>(world, context);
}

/// Records an effect despawned before it settled as cancelled.
fn record_unsettled(mut world: DeferredWorld, context: HookContext) {
    if world.get::<Settled>(context.entity).is_some() {
        return;
    }
    if let Some(record) = assemble(&world, context.entity, Some(Disposition::Cancelled), true) {
        world.write_message(Recorded(record));
    }
}

/// Notes how an effect's first answer came, or that a later one replaced
/// it.
fn answered(mut world: DeferredWorld, context: HookContext) {
    let entity = context.entity;
    let started = world.get::<Started>(entity).is_some();
    let running = world.get::<Running>(entity).is_some();
    if let Some(mut effect) = world.get_mut::<Effect>(entity) {
        match effect.answered {
            None => {
                effect.answered = Some(match (started, running) {
                    (false, _) => Disposition::AnsweredBeforeStart,
                    (true, true) => Disposition::AnsweredWhileRunning,
                    (true, false) => Disposition::Ran,
                });
            }
            Some(_) => effect.replaced = true,
        }
    }
    moved::<Outcome>(world, context);
}

/// Keeps an unsettled effect's first answer before it is replaced.
fn keep_raw(mut world: DeferredWorld, context: HookContext) {
    let entity = context.entity;
    let keeps = world.get::<Settled>(entity).is_none()
        && world
            .get::<Effect>(entity)
            .is_some_and(|effect| effect.answered.is_some() && effect.raw.is_none());
    let Some(first) = world
        .get::<Outcome>(entity)
        .filter(|_| keeps)
        .map(|outcome| outcome.0.clone())
    else {
        return;
    };
    if let Some(mut effect) = world.get_mut::<Effect>(entity) {
        effect.raw = Some(first);
    }
}

/// Admits every effect spawned since the last pass.
fn admit(
    new: Query<Entity, (With<Effect>, Without<Admitted>, Without<Settled>)>,
    mut passes: ResMut<Passes>,
    mut commands: Commands,
) {
    passes.admitted.clear();
    for entity in &new {
        passes.admitted.push(entity);
        commands.entity(entity).try_insert(Admitted);
    }
}

/// After a Prepare system failed, answers what was admitted this pass with
/// an error: its preparation is unknown, and edits do not run twice.
fn answer_unprepared(
    mut passes: ResMut<Passes>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(failure) = failed_in(Stage::Prepare) else {
        return;
    };
    let why = format!("{} in Prepare: {}", failure.system, failure.error);
    for &entity in &passes.admitted {
        let failed = ErrorReport::new(ErrorKind::Internal, why.clone());
        commands
            .entity(entity)
            .try_insert_if_new(Outcome(Err(failed)));
    }
    let what = "what it admitted is answered with an error";
    notices.write_batch(closed(&mut passes, failure, what));
}

/// Starts every admitted, unheld, unanswered effect, unless a Gate system
/// failed this pass: an effect request with a handler is dispatched on the
/// kernel's recorded path, and one without stays open.
fn start(
    ready: Query<(Entity, &Effect, Option<&EffectRequest>, Option<&ForAgent>), Undecided>,
    agents: Query<&AgentId>,
    effects: Res<Effects>,
    wake: Res<Wake>,
    mut passes: ResMut<Passes>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if let Some(failure) = failed_in(Stage::Gate) {
        if !ready.is_empty() {
            let what = "nothing started this pass";
            notices.write_batch(closed(&mut passes, failure, what));
        }
        return;
    }
    let at = SystemTime::now();
    for (entity, effect, request, agent) in &ready {
        let mut started = commands.entity(entity);
        let Some(EffectRequest {
            kind,
            handler: Some(handler),
        }) = request
        else {
            started.try_insert(Started { at });
            continue;
        };
        let scope = agent
            .and_then(|agent| agents.get(agent.0).ok())
            .map(|id| Arc::from(id.0.as_str()));
        let origin = Origin {
            parent: effect.parent,
            scope,
            purpose: effect.purpose.clone(),
        };
        let work = effects.run(effect.id, origin, handler.0.clone(), kind.clone());
        let running = Running::answering(effect_pool(), &wake, work);
        started.try_insert((Started { at }, running));
    }
}

/// Plugin effects run on the async compute pool, as tool calls do.
fn effect_pool() -> &'static AsyncComputeTaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// The effects Commit settles: answered and unheld, or superseded.
type Due = (
    With<Effect>,
    Without<Settled>,
    Or<((With<Outcome>, Without<Holds>), With<SupersededBy>)>,
);

/// Settles the answered effects nothing holds, and the superseded ones, in
/// id order, unless a Review system failed this pass: cancels a handler
/// still running, and writes each one's record. Then writes the records of
/// dispatches nested in handlers.
fn settle(
    world: &mut World,
    due: &mut QueryState<(Entity, &Effect), Due>,
    live: &mut QueryState<&Effect, Without<Settled>>,
) {
    let mut settling: Vec<(EffectId, Entity)> = due
        .iter(world)
        .map(|(entity, effect)| (effect.id, entity))
        .collect();
    if let Some(failure) = failed_in(Stage::Review) {
        let what = "nothing settled this pass";
        let notice = world
            .get_resource_mut::<Passes>()
            .filter(|_| !settling.is_empty())
            .and_then(|mut passes| closed(&mut passes, failure, what));
        if let Some(notice) = notice {
            world.write_message(notice);
        }
        return;
    }
    settling.sort_unstable();
    for (_, entity) in settling {
        let superseded = world.get::<SupersededBy>(entity).is_some();
        let record = assemble(
            world,
            entity,
            superseded.then_some(Disposition::Superseded),
            true,
        );
        if let Ok(mut effect) = world.get_entity_mut(entity) {
            effect.remove::<Running>().insert(Settled);
        }
        if let Some(record) = record {
            world.write_message(Recorded(record));
        }
    }
    record_nested(world, live);
}

/// Writes the resolved records that belong to no unsettled effect: the
/// dispatches nested in handlers.
fn record_nested(world: &mut World, live: &mut QueryState<&Effect, Without<Settled>>) {
    let Some(buffer) = world
        .get_resource::<Effects>()
        .map(|effects| effects.buffer().clone())
    else {
        return;
    };
    if buffer.resolved() == 0 {
        return;
    }
    let live: HashSet<EffectId> = live.iter(world).map(Effect::id).collect();
    for record in buffer.records() {
        if !live.contains(&record.id) {
            buffer.remove(record.id);
            world.write_message(Recorded(record));
        }
    }
}

/// The record of `entity`: the handler's record from the kernel's buffer
/// when there is one (taken out of it with `take`), else one made from the
/// entity. `disposition` overrides how its answer came.
fn assemble(
    world: &World,
    entity: Entity,
    disposition: Option<Disposition>,
    take: bool,
) -> Option<EffectRecord> {
    let buffer = world.get_resource::<Effects>()?.buffer();
    let effect = world.get::<Effect>(entity)?;
    let current = world
        .get::<Outcome>(entity)
        .map(|outcome| outcome.0.clone());
    let (outcome, reviewed) = match (effect.replaced, &effect.raw, current) {
        (true, Some(raw), current) => (Some(raw.clone()), current),
        (_, _, current) => (current, None),
    };
    let disposition = disposition
        .or(effect.answered.filter(|_| outcome.is_some()))
        .unwrap_or(Disposition::Cancelled);
    let outcome = outcome.unwrap_or_else(|| match disposition {
        Disposition::Superseded => Err(ErrorReport::new(
            ErrorKind::Cancelled,
            "superseded before it was answered",
        )),
        _ => Err(cancelled()),
    });
    let id = effect.id;
    let handled = match take {
        true => buffer.take(id).or_else(|| {
            // Still in flight: its answer came from elsewhere, or none did.
            buffer.contains(id).then(|| {
                buffer.resolve(id, outcome.clone());
                buffer.take(id)
            })?
        }),
        false => buffer
            .records()
            .into_iter()
            .rev()
            .find(|record| record.id == id),
    };
    let mut record = match handled {
        Some(mut record) => {
            if disposition != Disposition::Ran {
                record.outcome = outcome;
            }
            record
        }
        None => {
            let (key, kind) = kind_of(world, entity)?;
            EffectRecord {
                tool_output: None,
                id,
                key,
                kind,
                outcome,
                events: None,
                stream_origin: None,
                parent: effect.parent,
                scope: scope_of(world, entity),
                purpose: effect.purpose.clone(),
                started_ms: world
                    .get::<Started>(entity)
                    .and_then(|started| millis(started.at)),
                finished_ms: millis(SystemTime::now()),
                reviewed: None,
                disposition: Disposition::Ran,
            }
        }
    };
    record.reviewed = reviewed;
    record.disposition = disposition;
    Some(record)
}

/// The handler key and kind of an effect the kernel knows.
fn kind_of(world: &World, entity: Entity) -> Option<(HandlerKey, EffectKind)> {
    let request = world.get::<EffectRequest>(entity)?;
    let key = match &request.handler {
        Some(handler) => handler.0.descriptor().key,
        None => match &request.kind {
            EffectKind::ToolCall { name, .. } => tool_key(name),
            other => HandlerKey::new(other.name()),
        },
    };
    Some((key, request.kind.clone()))
}

/// The stable id of the agent the effect works for.
fn scope_of(world: &World, entity: Entity) -> Option<Arc<str>> {
    let agent = world.get::<ForAgent>(entity)?.0;
    world
        .get::<AgentId>(agent)
        .map(|id| Arc::from(id.0.as_str()))
}

/// Milliseconds since the Unix epoch.
fn millis(at: SystemTime) -> Option<u64> {
    let since = at.duration_since(UNIX_EPOCH).ok()?;
    Some(u64::try_from(since.as_millis()).unwrap_or(u64::MAX))
}
