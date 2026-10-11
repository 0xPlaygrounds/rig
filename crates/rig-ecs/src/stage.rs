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
//! in their order, before it starts, wherever it was spawned. A gate
//! answers in the effect's [`Decision`], where the first decision wins; a
//! reviewer edits the [`Outcome`] in place, and the record keeps both. Bevy
//! sees both writes, so two unordered systems writing them are reported in
//! debug builds. Any entity [`HoldOf`] an effect
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
//! fn no_deploys(mut effects: Query<(&EffectRequest, &mut Decision), (Undecided, PluginEffects)>) {
//!     for (request, mut decision) in &mut effects {
//!         if request.kind.name() == "deploy" {
//!             decision.decide(Outcome::refused("no deploys today"));
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
use bevy_ecs::query::QueryData;
use bevy_ecs::schedule::{LogLevel, ScheduleBuildSettings, ScheduleLabel, SingleThreadedExecutor};
use bevy_ecs::world::DeferredWorld;
use bevy_log::tracing::Instrument;
use bevy_log::{info_span, warn};
use bevy_reflect::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use bevy_time::{Real, Time, Virtual};
use rig_core::completion::CompletionResponse;
use rig_core::effect::{
    self, Disposition, EffectId, EffectKind, EffectRecord, HandlerKey, tool_key,
};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::{Origin, Recorder, Reply, cancelled};
use rig_core::tool::{ToolExecutionError, ToolResult, ToolResultContext};
use web_time::SystemTime;

use super::agent::Notice;
use super::agent::{AgentId, CallOf, Turn, TurnOutcome};
use super::calls::{Running, Wake, arrive};
use super::effects::{Effects, Handler, Ran};
use super::inbox::{Incoming, accept, carry_on_accepted, message_kind};
use super::journal::{SessionLog, millis, now_ms};
use super::tools::{
    CallHandler, Footprint, ToolRequest, ToolStart, ToolStarts, complete_calls, compute_footprints,
};
use super::turn::{
    Adapters, ModelRequest, ModelStart, ModelStarts, SentTo, StreamDelta, TurnCall, back_off,
    build_requests, carry_on_called, commit_replies, commit_turn_calls, end_turns, feed,
    judge_failures, keeps_settled, model_pool, stop_calls, stream_partials, stream_reply,
    wait_in_order,
};

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

/// The stages of a pass, run in this order. The kernel's own work in each
/// is in the [module](self) doc and the plugin guide.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Finished tasks become their effects' [`Outcome`]s, streamed text is
    /// read ([`StreamDelta`]), and failed model calls get their
    /// [`RetryVerdict`](super::turn::RetryVerdict).
    Arrive,
    /// Plugins read, edit in place or hold an [`Outcome`]
    /// ([`Unreviewed`]), and a turn's proposed [`TurnOutcome`].
    Review,
    /// The kernel settles what is answered and unheld, writes its
    /// [`Recorded`], commits replies and their tool results, and ends the
    /// turns nothing holds.
    Commit,
    /// The kernel admits what was spawned since the last pass, builds the
    /// requests of [`TurnCall`]s and [`BuildFrom`](super::turn::BuildFrom)s,
    /// and marks the calls that wait for an earlier one [`Waiting`].
    Admit,
    /// Plugins edit the requests admitted this pass (`Added<Admitted>`).
    Prepare,
    /// Plugins hold ([`HoldOf`]) or decide ([`Decision`]) what is about to
    /// start ([`Undecided`]).
    Gate,
    /// The kernel applies the decisions made since the last pass as
    /// [`Outcome`]s, then starts what is admitted, unheld and unanswered: it
    /// accepts messages ([`Incoming`]), and runs handlers, a model call on
    /// its adapted, fitted request. What it does not run stays open,
    /// [`Started`], until answered.
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
#[require(Decision)]
pub struct Effect {
    id: EffectId,
    /// The effect that asked for this one, in the record.
    pub parent: Option<EffectId>,
    /// What it is for, such as `"summary"`, in the record.
    pub purpose: Option<Cow<'static, str>>,
    /// How its first answer came.
    answered: Option<Disposition>,
    /// The first answer, which a reviewer may edit.
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

    /// Whether its first answer came from its handler, which ran.
    pub(crate) fn ran(&self) -> bool {
        self.answered == Some(Disposition::Ran)
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

/// The effect's answer, applied by the kernel from its [`Decision`]. A
/// reviewer edits it in place (`Query<&mut Outcome, Unreviewed>`), and the
/// record keeps the first answer as its outcome and the edited one as the
/// reviewed one. Applied to a running effect, it cancels the handler. A
/// tool call's answer is its result: `Outcome::from(ToolResult)`.
#[derive(Component, Clone, Debug)]
#[component(on_insert = answered)]
pub struct Outcome(pub Result<effect::Outcome, ErrorReport>);

impl Outcome {
    /// A refusal for `why`, decided before the effect ran. A tool call's
    /// refusal reaches the model as the tool's refusal.
    pub fn refused(why: impl Into<String>) -> Self {
        Self(Err(ErrorReport::new(ErrorKind::Denied, why).refused()))
    }

    /// The response a model call answered with, if it did.
    pub fn completion(&self) -> Option<&CompletionResponse> {
        match &self.0 {
            Ok(effect::Outcome::Completion(response)) => Some(response),
            _ => None,
        }
    }

    /// What a tool call answered this way returns to the model: its
    /// result, or the failure as an error result.
    pub fn tool_result(&self) -> ToolResult {
        match &self.0 {
            Ok(effect::Outcome::ToolResult { result }) => result.clone(),
            Ok(other) => ToolResult::failed(ToolExecutionError::other(format!(
                "the tool answered with a {} outcome",
                other.family()
            ))),
            Err(report) => ToolResult::failed(report.clone().into()),
        }
    }

    /// An answer of a kind of its own, rig-core's custom outcome with
    /// `payload`, such as a plugin's taking over a message.
    pub fn custom(payload: impl Into<serde_json::Value>) -> Self {
        Self(Ok(effect::Outcome::Custom {
            payload: payload.into(),
        }))
    }
}

impl From<ToolResult> for Outcome {
    fn from(result: ToolResult) -> Self {
        Self(Ok(effect::Outcome::ToolResult { result }))
    }
}

impl From<Result<effect::Outcome, ErrorReport>> for Outcome {
    fn from(outcome: Result<effect::Outcome, ErrorReport>) -> Self {
        Self(outcome)
    }
}

/// Where an effect's answer is decided before the kernel applies it, on
/// every effect: the first decision wins. Gates decide in Gate, and the
/// owner of an open effect decides whenever its work is done, through
/// `Query<&mut Decision>`, so Bevy sees two unordered deciders. The kernel
/// applies the decision as the effect's [`Outcome`] when the pass reaches
/// Start, unless the effect was answered meanwhile; it stays decided
/// afterwards. Until then the effect is still [`Undecided`] for the gates
/// after the one that decided: a gate that holds or asks checks
/// [`is_decided`](Self::is_decided) first.
#[derive(Component, Debug, Default)]
pub struct Decision(Option<Outcome>);

impl Decision {
    /// Decides `outcome` unless a decision was made; whether this one won.
    pub fn decide(&mut self, outcome: impl Into<Outcome>) -> bool {
        if self.0.is_some() {
            return false;
        }
        self.0 = Some(outcome.into());
        true
    }

    /// The decision, if one was made.
    pub fn get(&self) -> Option<&Outcome> {
        self.0.as_ref()
    }

    /// Whether a decision was made.
    pub fn is_decided(&self) -> bool {
        self.0.is_some()
    }
}

/// What the effect's answer is if it is stopped before it answers, such as
/// a running command's output so far. Whoever runs the effect keeps it
/// current; the kernel answers with it, in place of "interrupted" or
/// "stopped", whenever it stops the effect: an
/// [`Interrupt`](super::agent::Interrupt), its turn failing or stopping,
/// its cancellation, and the app's exit, which logs it for the restart.
#[derive(Component, Clone, Debug)]
pub struct IfStopped(pub Outcome);

/// What a handler published beside its answer, on its effect once
/// answered, and recorded with it: a tool's facts, such as rig-core's
/// `FilesChanged` and `ExitStatus`, or a fact a layer of a model call's
/// handler published about the reply, such as an adapter's stream rewrite
/// ([`PublishedContext::insert`](rig_core::tool::PublishedContext::insert)).
/// Read a value with `published.0.get::<FilesChanged>()`.
#[derive(Component, Clone, Debug)]
pub struct Published(pub ToolResultContext);

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

/// On a call of a model's reply that waits for an earlier call of that
/// reply it must not run beside ([`Footprint`]) to settle. The kernel sets
/// and clears it when [`Stage::Gate`] opens, from the calls as Prepare left
/// them, so Gate sees each call when it is about to run.
#[derive(Component, Reflect, Debug, Default)]
#[component(storage = "SparseSet", on_remove = moved::<Waiting>)]
#[reflect(Component, Default)]
pub struct Waiting;

/// Effects about to start: admitted, not waiting, not started, unanswered,
/// unheld and not superseded. A gate's filter.
pub type Undecided = (
    With<Admitted>,
    Without<Waiting>,
    Without<Started>,
    Without<Outcome>,
    Without<Holds>,
    Without<Settled>,
    Without<SupersededBy>,
);

/// Answered effects nothing holds, not settled yet. A reviewer's filter;
/// it matches on every pass until Commit settles the effect, so a reviewer
/// edits once, keeping a marker on what it edited.
pub type Unreviewed = (With<Outcome>, Without<Holds>, Without<Settled>);

/// Tool calls ([`ToolRequest`]), so Bevy can see they are no other kind of
/// effect: two queries of different kinds never conflict. Two queries in
/// one system that write the same component must rule each other out like
/// this, or the system panics when it is initialized.
pub type ToolCalls = (
    With<ToolRequest>,
    Without<ModelRequest>,
    Without<Incoming>,
    Without<EffectRequest>,
);

/// Model calls whose request is built ([`ModelRequest`]).
pub type ModelCalls = (
    With<ModelRequest>,
    Without<ToolRequest>,
    Without<Incoming>,
    Without<EffectRequest>,
);

/// Messages on their way to an agent ([`Incoming`]).
pub type IncomingMessages = (
    With<Incoming>,
    Without<ToolRequest>,
    Without<ModelRequest>,
    Without<EffectRequest>,
);

/// Plugins' own effects ([`EffectRequest`]), such as a question. A system
/// over one of them names its marker too, such as `With<Ask>`.
pub type PluginEffects = (
    With<EffectRequest>,
    Without<ToolRequest>,
    Without<ModelRequest>,
    Without<Incoming>,
);

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
        .add_message::<StreamDelta>()
        .edit_schedule(AgentPass, |pass| {
            pass.set_executor(SingleThreadedExecutor::new());
            if cfg!(debug_assertions) {
                // Unordered systems writing the same data in one stage are
                // reported, as their order is meaningless: two deciders, two
                // reviewers. Holds and spawns go through Commands, which
                // Bevy does not see, and need no order.
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
                (arrive, stream_partials, judge_failures)
                    .chain()
                    .in_set(Stage::Arrive),
                (
                    stop_calls,
                    back_off,
                    settle,
                    commit_replies,
                    end_turns,
                    commit_turn_calls,
                )
                    .chain()
                    .in_set(Stage::Commit),
                (complete_calls, admit, carry_on_called, build_requests)
                    .chain()
                    .in_set(Stage::Admit),
                (
                    answer_unprepared,
                    compute_footprints.run_if(moved_since_quiet),
                    wait_in_order.run_if(moved_since_quiet),
                )
                    .chain()
                    .in_set(Enter(Stage::Gate)),
                apply_decisions.in_set(Enter(Stage::Start)),
                (accept, carry_on_accepted, start)
                    .chain()
                    .in_set(Stage::Start),
            ),
        )
        .add_systems(
            Update,
            run_passes
                .in_set(AgentLoop)
                .run_if(any_match_filter::<Or<(With<Effect>, With<Turn>)>>),
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

/// Despawns what settled last frame, but the calls of a reply still
/// running, then runs passes until one moves nothing, with zero time steps
/// after the first, and the kernel's error handler in place of the app's.
fn run_passes(world: &mut World, settled: &mut QueryState<Entity, With<Settled>>) {
    let done: Vec<Entity> = settled
        .iter(world)
        .filter(|&entity| !keeps_settled(world, entity))
        .collect();
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
pub(crate) struct Failure {
    stage: Stage,
    /// Which system, as Bevy names it: "System `name` failed".
    system: String,
    /// The first line of its error.
    error: String,
}

/// The first system that failed in `stage` this pass.
pub(crate) fn failed_in(stage: Stage) -> Option<Failure> {
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
pub(crate) fn moved<C: Component>(mut world: DeferredWorld, context: HookContext) {
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
    if let Some(record) = assemble(&world, context.entity, Some(Disposition::Cancelled)) {
        world.write_message(Recorded(record));
    }
}

/// Notes how an effect's first answer came, and keeps it for the record.
fn answered(mut world: DeferredWorld, context: HookContext) {
    let entity = context.entity;
    let started = world.get::<Started>(entity).is_some();
    let running = world.get::<Running>(entity).is_some();
    let first = world
        .get::<Effect>(entity)
        .filter(|effect| effect.answered.is_none())
        .and_then(|_| world.get::<Outcome>(entity))
        .map(|outcome| outcome.0.clone());
    if let (Some(first), Some(mut effect)) = (first, world.get_mut::<Effect>(entity)) {
        effect.answered = Some(match (started, running) {
            (false, _) => Disposition::AnsweredBeforeStart,
            (true, true) => Disposition::AnsweredWhileRunning,
            (true, false) => Disposition::Ran,
        });
        effect.raw = Some(first);
    }
    moved::<Outcome>(world, context);
}

/// Applies each new decision as its effect's [`Outcome`], in the order the
/// effects were spawned, unless it was answered meanwhile. The decision
/// stays, so it still reads as decided.
fn apply_decisions(
    decided: Query<
        (Entity, &Effect, &Decision),
        (Changed<Decision>, Without<Outcome>, Without<Settled>),
    >,
    mut commands: Commands,
) {
    let mut decided: Vec<_> = decided
        .iter()
        .filter_map(|(entity, effect, decision)| Some((effect.id(), entity, decision.get()?)))
        .collect();
    decided.sort_unstable_by_key(|(id, ..)| *id);
    for (_, effect, outcome) in decided {
        commands.entity(effect).try_insert(outcome.clone());
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
/// kernel's recorded path, a model call and a tool call as [`ModelStarts`]
/// and [`ToolStarts`] say, and any other effect stays open.
fn start(
    mut ready: Query<StartItem, Undecided>,
    agents: Query<&AgentId>,
    turns: Query<&TurnOutcome>,
    (tools, models): (ToolStarts, ModelStarts),
    effects: Res<Effects>,
    wake: Res<Wake>,
    log: Res<SessionLog>,
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
    // Before a call that may change something, the reply that asked for
    // it is on disk.
    let changes = ready.iter().any(|item| {
        item.tool.is_some()
            && !item
                .footprint
                .is_some_and(|footprint| footprint.0.reads_only())
    });
    if changes {
        log.flush();
    }
    // In the order they were spawned, such as a reply's calls in call order.
    let mut ready: Vec<_> = ready.iter_mut().collect();
    ready.sort_unstable_by_key(|item| item.effect.id());
    for item in ready {
        // Messages are accepted before.
        if item.message {
            continue;
        }
        let ending = item.of.and_then(|of| turns.get(of.0).ok());
        if ending.and_then(TurnOutcome::stops).is_some() {
            continue;
        }
        let (entity, effect) = (item.entity, item.effect);
        let agent = item.agent.map(|agent| agent.0);
        let caller = agent.and_then(|agent| agents.get(agent).ok());
        let scope = caller.map(|id| id.0.as_str()).unwrap_or_default();
        let origin = Origin {
            parent: effect.parent,
            scope: caller.map(|id| Arc::from(id.0.as_str())),
            purpose: effect.purpose.clone(),
        };
        let mut started = commands.entity(entity);
        if let Some(mut model) = item.model {
            let (connection, kind) = match models.start(&mut model, item.adapters, agent) {
                ModelStart::Run(connection, kind) => (connection, kind),
                ModelStart::Refused(report) => {
                    started.try_insert_if_new(Outcome(Err(report)));
                    continue;
                }
            };
            let reference = connection.spec.reference();
            let span =
                info_span!("model_call", agent = scope, effect = %effect.id, model = reference);
            let (sender, feed) = feed();
            let wake_up = wake.clone();
            let consume = move |reply| stream_reply(reply, sender, wake_up);
            let handler = connection.handler.0.clone();
            let work = effects.run(effect.id, origin, handler, kind, consume);
            let running = Running::finishing(model_pool(), &wake, work.instrument(span), answer);
            started.try_insert((Started { at }, running, feed, SentTo(connection)));
            continue;
        }
        let (handler, kind) = match (item.request, item.tool) {
            (
                Some(EffectRequest {
                    kind,
                    handler: Some(handler),
                }),
                _,
            ) => (handler.clone(), kind.clone()),
            (_, Some(ToolRequest { call })) => {
                match tools.start(call, item.handler, agent.zip(caller)) {
                    ToolStart::Run(handler, kind) => (handler, kind),
                    ToolStart::Open(open) => {
                        started.try_insert(Started { at });
                        let parent = effect.parent;
                        let (id, call) = (effect.id, call.clone());
                        commands.queue(move |world: &mut World| {
                            if world.get_entity(entity).is_ok() {
                                open(world, entity, id, parent, call);
                            }
                        });
                        continue;
                    }
                    ToolStart::Refused(report) => {
                        started.try_insert_if_new(Outcome(Err(report)));
                        continue;
                    }
                }
            }
            _ => {
                started.try_insert(Started { at });
                continue;
            }
        };
        let span = match &kind {
            EffectKind::ToolCall { name, .. } => {
                info_span!("tool_call", agent = scope, tool = name.as_str(), parent = ?effect.parent)
            }
            other => info_span!("effect", kind = other.name()),
        };
        let work = effects.run(effect.id, origin, handler.0, kind, Reply::into_outcome);
        let running = Running::finishing(effect_pool(), &wake, work.instrument(span), answer);
        started.try_insert((Started { at }, running));
    }
}

/// Puts what a handler's run left on its effect, unless it was answered
/// meanwhile: what a tool published, then the outcome.
fn answer(mut entity: EntityWorldMut, (outcome, published): Ran) {
    if entity.contains::<Outcome>() {
        return;
    }
    if let Some(published) = published {
        entity.insert(Published(published));
    }
    entity.insert(Outcome(outcome));
}

/// An effect as Start finds it.
#[derive(QueryData)]
#[query_data(mutable)]
struct StartItem {
    entity: Entity,
    effect: &'static Effect,
    request: Option<&'static EffectRequest>,
    tool: Option<&'static ToolRequest>,
    model: Option<&'static mut ModelRequest>,
    adapters: Option<&'static Adapters>,
    turn_call: Has<TurnCall>,
    message: Has<Incoming>,
    handler: Option<&'static CallHandler>,
    footprint: Option<&'static Footprint>,
    agent: Option<&'static ForAgent>,
    of: Option<&'static CallOf>,
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
        settle_now(world, entity, superseded.then_some(Disposition::Superseded));
    }
    record_nested(world, live);
}

/// Settles `entity` now, its record's disposition overridden by
/// `disposition`: a task still running is dropped, which cancels it, it is
/// [`Settled`], and its record is written.
fn settle_now(world: &mut World, entity: Entity, disposition: Option<Disposition>) {
    let record = assemble(world, entity, disposition);
    if let Ok(mut effect) = world.get_entity_mut(entity) {
        effect.remove::<Running>().insert(Settled);
    }
    if let Some(record) = record {
        world.write_message(Recorded(record));
    }
}

/// Despawns whatever holds `effect`.
pub(crate) fn despawn_holds(world: &mut World, effect: Entity) {
    let holds: Vec<Entity> = world
        .get::<Holds>(effect)
        .map(|holds| holds.iter().collect())
        .unwrap_or_default();
    for hold in holds {
        world.try_despawn(hold).ok();
    }
}

/// Despawns `holds`, the holds on an effect or a turn's outcome.
pub(crate) fn release(commands: &mut Commands, holds: Option<&Holds>) {
    for hold in holds.into_iter().flat_map(Holds::iter) {
        commands.entity(hold).try_despawn();
    }
}

/// Settles `entity` at once, cancelled, if it is an unsettled effect, such
/// as a call of a turn being interrupted: whatever holds it is despawned,
/// it is answered with its [`IfStopped`], else `why`, unless it has an
/// answer, its task is cancelled, and its record is written.
pub(crate) fn cancel(world: &mut World, entity: Entity, why: &str) {
    let unsettled = world
        .get_entity(entity)
        .is_ok_and(|effect| effect.contains::<Effect>() && !effect.contains::<Settled>());
    if !unsettled {
        return;
    }
    despawn_holds(world, entity);
    let Ok(mut effect) = world.get_entity_mut(entity) else {
        return;
    };
    if !effect.contains::<Outcome>() {
        let stopped = match effect.get::<IfStopped>() {
            Some(IfStopped(outcome)) => outcome.clone(),
            None => Outcome(Err(ErrorReport::new(ErrorKind::Cancelled, why))),
        };
        effect.insert(stopped);
    }
    settle_now(world, entity, Some(Disposition::Cancelled));
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

/// The record of `entity`: the handler's record, taken out of the kernel's
/// buffer, when there is one, else one made from the entity.
/// `disposition` overrides how its answer came.
fn assemble(
    world: &World,
    entity: Entity,
    disposition: Option<Disposition>,
) -> Option<EffectRecord> {
    let buffer = world.get_resource::<Effects>()?.buffer();
    let effect = world.get::<Effect>(entity)?;
    let current = world
        .get::<Outcome>(entity)
        .map(|outcome| outcome.0.clone());
    // Applied, an outcome's ticks are equal; a review moves only `changed`.
    let reviewed = world
        .get_entity(entity)
        .ok()
        .and_then(|entity| entity.get_change_ticks::<Outcome>())
        .is_some_and(|ticks| ticks.changed != ticks.added);
    let (outcome, reviewed) = match (reviewed, &effect.raw, current) {
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
    let handled = buffer.take(id).or_else(|| {
        // Still in flight: its answer came from elsewhere, or none did.
        buffer.contains(id).then(|| {
            buffer.resolve(id, outcome.clone());
            buffer.take(id)
        })?
    });
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
                tool_output: world
                    .get::<Published>(entity)
                    .map(|published| published.0.clone()),
                id,
                key,
                kind,
                outcome,
                events: None,
                stream_origin: None,
                parent: effect.parent,
                scope: scope_of(world, entity),
                purpose: effect.purpose.clone(),
                turn: None,
                started_ms: world
                    .get::<Started>(entity)
                    .map(|started| millis(started.at)),
                finished_ms: Some(now_ms()),
                reviewed: None,
                disposition: Disposition::Ran,
            }
        }
    };
    record.reviewed = reviewed;
    record.disposition = disposition;
    record.turn = world
        .get::<CallOf>(entity)
        .and_then(|call| world.get::<Turn>(call.0))
        .map(|turn| turn.index);
    Some(record)
}

/// The handler key and kind of an effect the kernel knows.
fn kind_of(world: &World, entity: Entity) -> Option<(HandlerKey, EffectKind)> {
    if let Some(ToolRequest { call }) = world.get::<ToolRequest>(entity) {
        let name = call.function.name.as_str();
        let key = world
            .get::<CallHandler>(entity)
            .map_or_else(|| tool_key(name), |handler| handler.0.0.descriptor().key);
        return Some((key, EffectKind::from(call)));
    }
    if let Some(model) = world.get::<ModelRequest>(entity) {
        let key = model.connection.as_ref().map_or_else(
            || HandlerKey::new("completion"),
            |connection| connection.handler.0.descriptor().key,
        );
        let kind = EffectKind::Completion {
            request: model.request.clone(),
            stream: true,
        };
        return Some((key, kind));
    }
    if let Some(message) = world.get::<Incoming>(entity) {
        return Some(message_kind(message));
    }
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
