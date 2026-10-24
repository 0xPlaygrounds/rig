//! The stages every effect moves through within a frame. An effect is an
//! entity whose components are its request and its outcome, such as an
//! [`EffectRequest`] and, once answered, an [`Outcome`]; generic components
//! say how far it got. Each run of the [`AgentPass`] schedule goes through
//! the [`Stage`] sets in order, and the loop in `Update` ([`AgentLoop`])
//! runs it again until nothing moved, so an effect spawned, answered or
//! released goes as far as it can in that frame.
//!
//! A plugin steps in with an ordinary system in a stage that queries stage
//! data, ordered by its phase ([`PrepareSystems`], [`ReviewSystems`],
//! [`GateSystems`]) and against other plugins' public sets. Every effect is
//! [`Admitted`] first, so every Prepare and Gate system sees it before it
//! starts. A gate decides in the effect's [`Decision`] (the first decision
//! wins), a reviewer edits the [`Outcome`] in place, and any entity
//! [`HoldOf`] an effect keeps it where it is. A failing Prepare, Gate or
//! Review system closes its stage for that pass. From a frame's second pass
//! on, Bevy's clocks advance by zero, so timers count each frame once.
//!
//! ```
//! use rig_ecs::prelude::*;
//!
//! // Refuse every effect of the kind `deploy` when it is about to start.
//! fn no_deploys(mut effects: Query<(&EffectRequest, &mut Decision), (ReadyToStart, PluginEffects)>) {
//!     for (request, mut decision) in &mut effects {
//!         if request.kind.name() == "deploy" {
//!             decision.decide(Decider::plugin("no-deploys"), Outcome::refused("no deploys today"));
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
use bevy_ecs::schedule::{
    InternedSystemSet, LogLevel, ScheduleBuildSettings, ScheduleLabel, SingleThreadedExecutor,
};
use bevy_ecs::world::DeferredWorld;
use bevy_log::tracing::Instrument;
use bevy_log::{info_span, warn};
use bevy_reflect::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use bevy_time::{Real, Time, Virtual};
use rig_core::completion::CompletionResponse;
use rig_core::effect::{
    self, Decider, Disposition, EffectId, EffectKind, EffectRecord, HandlerKey, tool_key,
};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::{DispatchContext, Recorder, Reply, cancelled};
use rig_core::tool::{ToolExecutionError, ToolResult, ToolResultContext};
use rig_core::{millis, now_ms};
use web_time::SystemTime;

use super::agent::{AgentId, CallOf, Conversation, Notice, Turn, TurnOutcome};
use super::effects::{Effects, Handler, Ran};
use super::inbox::{Incoming, accept, carry_on_accepted, message_kind, settle_unsent};
use super::restore::{Interrupted, answer_interrupted, mark_started};
use super::session_log::{Drain, SessionLog, save_now};
use super::tasks::{Running, Wake, arrive};
use super::tools::{
    CallHandler, Footprint, ToolRequest, ToolStart, ToolStarts, complete_calls, compute_footprints,
    redact_results, refuse_calls,
};
use super::turn::{
    Adapters, DroppedOptions, ModelRequest, ModelStarts, SentTo, StreamDelta, TurnCall, back_off,
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
    /// ([`ReadyToSettle`]), and a turn's proposed [`TurnOutcome`], in the
    /// [`ReviewSystems`] phases. Before them the kernel redacts every
    /// secret in a tool call's answer ([`rig_core::redact`]).
    Review,
    /// The kernel settles what is answered and unheld, writes its
    /// [`EffectRecorded`], commits replies and their tool results, and ends the
    /// turns nothing holds.
    Commit,
    /// The kernel admits what was spawned since the last pass, builds the
    /// requests of [`TurnCall`]s and [`SideCall`](super::turn::SideCall)s,
    /// and marks the calls that wait for an earlier one [`Waiting`].
    Admit,
    /// Plugins edit the requests admitted this pass (`Added<Admitted>`), in
    /// the [`PrepareSystems`] phases.
    Prepare,
    /// Plugins hold ([`HoldOf`]) or decide ([`Decision`]) what is about to
    /// start ([`ReadyToStart`]), once the kernel answered the tool calls it
    /// refuses ([`ToolRequest`]).
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

/// The phases of [`Stage::Prepare`], run in this order. A plugin puts a
/// Prepare system in the phase of what it does, and so is ordered against
/// every other plugin's, without naming them.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrepareSystems {
    /// What came in is rewritten before anything is read from it: a
    /// message's text, a tool call's arguments.
    Input,
    /// The parts that come from what a request names and from its agent go
    /// in: the files a message names, a default system prompt, the
    /// directory a tool call runs in.
    Compose,
    /// The composed request is edited as it will be sent: its prompt,
    /// messages, tools and options, the model it goes to, its [`Adapters`],
    /// a tool call's [`CallHandler`].
    Edit,
    /// The request as it will be sent ([`ModelStarts::sent`]) is measured
    /// against the window of the model it goes to and fitted, such as by
    /// compaction.
    Fit,
}

/// The phases of [`Stage::Review`], run in this order.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReviewSystems {
    /// An [`Outcome`], a failed model call's
    /// [`RetryVerdict`](super::turn::RetryVerdict) or a turn's proposed
    /// [`TurnOutcome`] is edited in place.
    Edit,
    /// What was reviewed is acted on: a failed call taken over by a
    /// successor, a turn held or carried on, a result followed up.
    Act,
}

/// The positions around [`Stage::Gate`], in this order: `FromRecord`,
/// every gate (`in_set(Stage::Gate)`), `After`, then `Fallback`. A system
/// that fails in `FromRecord` or `After` closes Gate for the pass, as a
/// gate does.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GateSystems {
    /// Decides effects from a record of an earlier run of them, such as a
    /// session replay, before every gate, so each gate sees the effect
    /// decided. The record holds what every gate decided in that run; a
    /// gate deciding again would judge today's settings and files instead.
    FromRecord,
    /// After every gate had its say: an answer that stands in for running
    /// the effect (a cached result, a dry run, an editor's buffer), which
    /// must not answer what a gate refused, and a coordinator of a group
    /// that starts together or not at all. A decision made in Gate is
    /// applied at Start, so here a decided effect still matches
    /// [`ReadyToStart`]: read its [`Decision`]. A tool call the kernel refuses
    /// is answered before every gate: read its [`Outcome`]. No gate sees an
    /// effect held from its spawn, so a coordinator spawns its members
    /// unheld and queries them without [`ReadyToStart`]. While any member is
    /// held by another or [`Waiting`], it holds the free ones with holds of
    /// its own; once none is, it refuses them all if one was refused or
    /// answered with an error, else releases its holds.
    After,
    /// The kernel's own answers once every plugin had its say, such as an
    /// [`Interrupted`] call's or an [`Unsent`](super::inbox::Unsent)
    /// message's. A default plugin's fallback goes between
    /// (`.after(GateSystems::After).before(GateSystems::Fallback)`), so it
    /// sees what every plugin decided or held and the kernel has not
    /// answered yet.
    Fallback,
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
    raw: Option<Result<effect::Answered, ErrorReport>>,
    /// Who decided the answer, when the kernel applied a decision.
    decided_by: Option<Decider>,
}

impl Default for Effect {
    fn default() -> Self {
        Self::with_parent_id(None)
    }
}

impl Effect {
    /// An effect asked for by `parent`, such as a successor keeping the
    /// parent of the effect it supersedes.
    pub fn with_parent_id(parent: Option<EffectId>) -> Self {
        Self {
            id: EffectId::from_raw(0),
            parent,
            purpose: None,
            answered: None,
            raw: None,
            decided_by: None,
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
/// reviewer edits it in place (`Query<&mut Outcome, ReadyToSettle>`), and the
/// record keeps the first answer as its outcome and the edited one as the
/// reviewed one. Applied to a running effect, it cancels the handler. A
/// tool call's answer is its result: `Outcome::from(ToolResult)`.
#[derive(Component, Clone, Debug)]
#[component(on_insert = answered)]
pub struct Outcome(pub Result<effect::Answered, ErrorReport>);

impl Outcome {
    /// A refusal for `why`, decided before the effect ran. A tool call's
    /// refusal reaches the model as the tool's refusal.
    pub fn refused(why: impl Into<String>) -> Self {
        Self(Err(ErrorReport::new(ErrorKind::Denied, why).refused()))
    }

    /// The response a model call answered with, if it did.
    pub fn completion(&self) -> Option<&CompletionResponse> {
        match &self.0 {
            Ok(effect::Answered::Completion(response)) => Some(response),
            _ => None,
        }
    }

    /// What a tool call answered this way returns to the model: its
    /// result, or the failure as an error result.
    pub fn tool_result(&self) -> ToolResult {
        match &self.0 {
            Ok(effect::Answered::ToolResult { result }) => result.clone(),
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
        Self(Ok(effect::Answered::Custom {
            payload: payload.into(),
        }))
    }
}

impl From<ToolResult> for Outcome {
    fn from(result: ToolResult) -> Self {
        Self(Ok(effect::Answered::ToolResult { result }))
    }
}

impl From<Result<effect::Answered, ErrorReport>> for Outcome {
    fn from(outcome: Result<effect::Answered, ErrorReport>) -> Self {
        Self(outcome)
    }
}

/// Where an effect's answer is decided before the kernel applies it, on
/// every effect: the first decision wins. Gates decide in Gate, and the
/// owner of an open effect decides whenever its work is done, through
/// `Query<&mut Decision>`, so Bevy sees two unordered deciders. The kernel
/// applies the decision as the effect's [`Outcome`] when the pass reaches
/// Start, unless the effect was answered meanwhile, and its record names
/// who decided (`EffectRecord::decided_by`); it stays decided afterwards.
/// An effect no one decided, such as a call every gate let run, records
/// no decider. Until then the effect is still [`ReadyToStart`] for the gates
/// after the one that decided: a gate that holds or asks checks
/// [`is_decided`](Self::is_decided) first.
#[derive(Component, Debug, Default)]
pub struct Decision(Option<(Outcome, Decider)>);

impl Decision {
    /// Decides `outcome` for `by`, such as `Decider::plugin("protected-paths")`,
    /// unless a decision was made; whether this one won.
    pub fn decide(&mut self, by: Decider, outcome: impl Into<Outcome>) -> bool {
        if self.0.is_some() {
            return false;
        }
        self.0 = Some((outcome.into(), by));
        true
    }

    /// The decision, if one was made.
    pub fn get(&self) -> Option<&Outcome> {
        self.0.as_ref().map(|(outcome, _)| outcome)
    }

    /// Who made the decision, if one was made.
    pub fn by(&self) -> Option<&Decider> {
        self.0.as_ref().map(|(_, by)| by)
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
/// [`StopTurn`](super::agent::StopTurn), its turn failing or stopping,
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
/// add up: releasing one never releases another's, and every release wakes
/// the loop.
#[derive(Component, Reflect, Debug)]
#[component(on_remove = moved::<HoldOf>)]
#[reflect(Component)]
#[relationship(relationship_target = Holds)]
pub struct HoldOf(pub Entity);

/// The holds on an effect, removed with the last one. Despawning the
/// effect despawns its holds.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = HoldOf, linked_spawn)]
pub struct Holds(Vec<Entity>);

/// This effect is replaced by the one it names, such as a call retried on
/// another model, or each losing member of a race: it never starts, its
/// task is cancelled if it runs, and its outcome is never applied. It
/// settles at the next Commit, recorded as superseded. Any number of
/// effects may name one successor: a retry is
/// `commands.spawn(retry).add_related::<SupersededBy>(&[call])`, and each
/// loser of a race gets `insert(SupersededBy(winner))`.
#[derive(Component, Reflect, Debug)]
#[component(on_add = moved::<SupersededBy>)]
#[reflect(Component)]
#[relationship(relationship_target = Supersedes)]
pub struct SupersededBy(pub Entity);

impl SupersededBy {
    /// The successor.
    pub fn successor(&self) -> Entity {
        self.0
    }
}

/// The effects this one replaces ([`SupersededBy`]).
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = SupersededBy)]
pub struct Supersedes(Vec<Entity>);

/// The effect was committed. It stays until the next frame's loop starts,
/// for `Added<Settled>` readers, then it is despawned.
#[derive(Component, Reflect, Debug, Default)]
#[component(storage = "SparseSet", on_add = moved::<Settled>)]
#[reflect(Component, Default)]
pub struct Settled;

/// When an effect carrying `M` is superseded, moves `M`, and the hold the
/// effect keeps ([`HoldOf`]), to the successor, so the plugin that owns the
/// effect by `M` takes the successor's outcome. The successor's hold is
/// inserted before the old one is removed, so the held effect is never
/// unheld in between: `app.add_observer(follow_successor::<Summarizing>)`.
pub fn follow_successor<M: Component>(
    superseded: On<Add<SupersededBy>>,
    owned: Query<(&SupersededBy, Option<&HoldOf>), With<M>>,
    mut commands: Commands,
) {
    let old = superseded.entity;
    let Ok((by, hold)) = owned.get(old) else {
        return;
    };
    let new = by.successor();
    let held = hold.map(|&HoldOf(held)| held);
    commands.queue(move |world: &mut World| {
        let Ok(mut to) = world.get_entity_mut(new) else {
            return;
        };
        if let Some(held) = held {
            to.insert(HoldOf(held));
        }
        let Ok(mut from) = world.get_entity_mut(old) else {
            return;
        };
        let marker = from.take::<M>();
        if held.is_some() {
            from.remove::<HoldOf>();
        }
        if let (Some(marker), Ok(mut to)) = (marker, world.get_entity_mut(new)) {
            to.insert(marker);
        }
    });
}

/// When an effect settles, despawns its `ChildOf` children carrying `M`
/// that are still open, such as a request a gate holds when a stop ends
/// the call that made it: each is recorded as cancelled, which ends
/// whatever held it. `app.add_observer(despawn_open_children::<TaskCall>)`.
pub fn despawn_open_children<M: Component>(
    settled: On<Add<Settled>>,
    children: Query<&Children>,
    open: Query<(), (With<M>, Without<Settled>)>,
    mut commands: Commands,
) {
    let children = children.get(settled.entity).into_iter().flatten();
    for &child in children.filter(|&&child| open.contains(child)) {
        commands.entity(child).despawn();
    }
}

/// The record of every effect that settled or was cancelled, in settle
/// order, and of the dispatches nested in their handlers. Only effects of a
/// kind the kernel knows, such as an [`EffectRequest`], have a record.
#[derive(Message, Clone, Debug)]
pub struct EffectRecorded(pub EffectRecord);

/// On a call of a model's reply that waits for an earlier call of that
/// reply it must not run beside ([`Footprint`]) to settle. The kernel sets
/// and clears it when [`Stage::Gate`] opens, from the calls as they stand
/// then, so Gate sees each call when it is about to run.
#[derive(Component, Reflect, Debug, Default)]
#[component(storage = "SparseSet", on_remove = moved::<Waiting>)]
#[reflect(Component, Default)]
pub struct Waiting;

/// Effects about to start: admitted, not waiting, not started, unanswered,
/// unheld, not superseded and not a restored call the kernel answers
/// ([`Interrupted`]). A gate's filter.
pub type ReadyToStart = (
    With<Admitted>,
    Without<Waiting>,
    Without<Started>,
    Without<Outcome>,
    Without<Holds>,
    Without<Settled>,
    Without<SupersededBy>,
    Without<Interrupted>,
);

/// Answered effects nothing holds, not settled yet. A reviewer's filter;
/// it matches on every pass until Commit settles the effect, so a reviewer
/// edits once, keeping a marker on what it edited.
pub type ReadyToSettle = (With<Outcome>, Without<Holds>, Without<Settled>);

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
        .add_message::<EffectRecorded>()
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
                GateSystems::FromRecord,
                Stage::Gate,
                GateSystems::After,
                GateSystems::Fallback,
                Enter(Stage::Start),
                Stage::Start,
            )
                .chain(),
        )
        .configure_sets(
            AgentPass,
            (
                (
                    PrepareSystems::Input,
                    PrepareSystems::Compose,
                    PrepareSystems::Edit,
                    PrepareSystems::Fit,
                )
                    .chain()
                    .in_set(Stage::Prepare),
                (ReviewSystems::Edit, ReviewSystems::Act)
                    .chain()
                    .in_set(Stage::Review),
            ),
        )
        .add_systems(
            AgentPass,
            // After every reviewer and before anything is committed: no
            // key reaches the model, the conversation or a log, and no
            // plugin has to order itself against it. A plugin that swaps
            // secrets for placeholders in Review leaves nothing to redact.
            redact_results.in_set(Enter(Stage::Commit)),
        )
        .configure_sets(
            AgentPass,
            (
                Stage::Review,
                Stage::Commit,
                Stage::Admit,
                Stage::Prepare,
                GateSystems::FromRecord,
                Stage::Gate,
                GateSystems::After,
                GateSystems::Fallback,
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
                    back_off.run_if(passed(Stage::Review)),
                    settle,
                    commit_replies,
                    end_turns.run_if(passed(Stage::Review)),
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
                refuse_calls
                    .after(GateSystems::FromRecord)
                    .before(Stage::Gate)
                    .run_if(moved_since_quiet),
                (answer_interrupted, settle_unsent)
                    .chain()
                    .in_set(GateSystems::Fallback),
                apply_decisions.in_set(Enter(Stage::Start)),
                (accept.run_if(passed(Stage::Gate)), carry_on_accepted, start)
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
    // The conversations changed are saved between the stages, so a
    // `ConversationChanged` reader in a later stage of the pass sees the change, and
    // at the end of the pass; each save keeps followed components with
    // their lines (`SaveConversations`).
    let mut after: Option<InternedSystemSet> = None;
    for stage in Stage::ALL {
        let save = save_in_pass.before(Enter(stage));
        match after {
            Some(after) => app.add_systems(AgentPass, save.after(after)),
            None => app.add_systems(AgentPass, save),
        };
        after = Some(match stage {
            Stage::Gate => GateSystems::Fallback.intern(),
            stage => stage.intern(),
        });
    }
    app.add_systems(AgentPass, save_in_pass.after(Stage::Start));
}

/// Saves the agent conversations changed since it last ran, so a change
/// made in a stage is announced to the later stages of the pass; one that
/// saved any counts as a move, so the loop runs another pass, where the
/// earlier stages see it too.
fn save_in_pass(world: &mut World) {
    if let Some(agent) = save_now(world)
        && let Some(mut passes) = world.get_resource_mut::<Passes>()
    {
        saved_some(&mut passes, agent);
    }
}

/// Counts the save of `agent`'s conversation as a move.
fn saved_some(passes: &mut Passes, agent: Entity) {
    passes.moved = passes.moved.wrapping_add(1);
    passes.last = Some((std::any::type_name::<Conversation>(), agent));
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
    let mut clocks = None;
    let app_handler = world.get_resource::<FallbackErrorHandler>().copied();
    APP_HANDLER.set(Some(world.fallback_error_handler()));
    world.insert_resource(FallbackErrorHandler(fail_closed));
    let quiet = world.try_schedule_scope(AgentPass, |world, pass| {
        (0..MAX_PASSES).any(|n| {
            if n == 1 {
                clocks = Some((
                    zero_step::<()>(world),
                    zero_step::<Real>(world),
                    zero_step::<Virtual>(world),
                ));
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
    if let Some((time, real, virtual_time)) = clocks {
        put_back(world, time);
        put_back(world, real);
        put_back(world, virtual_time);
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

/// Steps the clock `Time<T>` by zero (the same elapsed time, a zero
/// delta), and returns it as it was, to put back after the passes. A change
/// a stage system makes to a clock after the first pass, such as a pause,
/// is lost then.
fn zero_step<T: Default + Copy + Send + Sync + 'static>(world: &mut World) -> Option<Time<T>> {
    let mut time = world.get_resource_mut::<Time<T>>()?;
    let was = *time;
    time.advance_by(Duration::ZERO);
    Some(was)
}

/// Puts back a clock [`zero_step`] took.
fn put_back<T: Default + Copy + Send + Sync + 'static>(world: &mut World, was: Option<Time<T>>) {
    if let (Some(was), Some(mut time)) = (was, world.get_resource_mut::<Time<T>>()) {
        *time = was;
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

/// Whether no system failed in `stage` this pass: a run condition, read
/// right before its system because the pass runs single-threaded.
fn passed(stage: Stage) -> impl Fn() -> bool + Clone {
    move || failed_in(stage).is_none()
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
        world.write_message(EffectRecorded(record));
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
    mut decided: Query<
        (Entity, &mut Effect, &Decision),
        (Changed<Decision>, Without<Outcome>, Without<Settled>),
    >,
    mut commands: Commands,
) {
    let mut decided: Vec<_> = decided
        .iter_mut()
        .filter_map(|(entity, effect, decision)| Some((entity, effect, decision.0.as_ref()?)))
        .collect();
    decided.sort_unstable_by_key(|(_, effect, _)| effect.id());
    for (entity, mut effect, (outcome, by)) in decided {
        effect.decided_by = Some(by.clone());
        commands.entity(entity).try_insert(outcome.clone());
    }
}

/// Admits every effect spawned since the last pass.
fn admit(
    new: Query<Entity, (With<Effect>, Without<Admitted>, Without<Settled>)>,
    mut commands: Commands,
) {
    for entity in &new {
        commands.entity(entity).try_insert(Admitted);
    }
}

/// After a Prepare system failed, answers what was admitted this pass with
/// an error: its preparation is unknown, and edits do not run twice.
fn answer_unprepared(
    admitted: Query<Entity, Added<Admitted>>,
    mut passes: ResMut<Passes>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(failure) = failed_in(Stage::Prepare) else {
        return;
    };
    let why = format!("{} in Prepare: {}", failure.system, failure.error);
    for entity in &admitted {
        let failed = ErrorReport::new(ErrorKind::Internal, why.clone());
        commands
            .entity(entity)
            .try_insert_if_new(Outcome(Err(failed)));
    }
    let what = "what it admitted is answered with an error";
    notices.write_batch(closed(&mut passes, failure, what));
}

/// The agents' conversations as Start marks calls on them, and saves them.
type Saving<'w, 's> = ParamSet<
    'w,
    's,
    (
        Drain<'static, 'static>,
        Query<'static, 'static, &'static mut Conversation>,
    ),
>;

/// Starts every admitted, unheld, unanswered, undecided effect, unless a
/// Gate system failed this pass: an effect request with a handler is
/// dispatched on the kernel's recorded path, a model call and a tool call
/// as [`ModelStarts`] and [`ToolStarts`] say, and any other effect stays
/// open.
fn start(
    mut ready: Query<StartItem, ReadyToStart>,
    agents: Query<&AgentId>,
    turns: Query<&TurnOutcome>,
    (tools, models): (ToolStarts, ModelStarts),
    effects: Res<Effects>,
    wake: Res<Wake>,
    (log, mut saving): (Res<SessionLog>, Saving),
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
    let stopping = |item: &StartItemReadOnlyItem| {
        let ending = item.of.and_then(|of| turns.get(of.0).ok());
        ending.and_then(TurnOutcome::stops).is_some()
    };
    // Before a call that may change something starts, the reply that asked
    // for it and the call's mark are written to the store (not yet synced
    // to disk). When they cannot be, no such call starts: a restart could
    // not tell that it may have run.
    let changing: Vec<_> = ready
        .iter()
        .filter(|item| !item.message && !item.decision.is_decided() && !stopping(item))
        .filter(|item| !item.footprint.is_some_and(|it| it.0.reads_only()))
        .filter_map(|item| Some((item.agent?.0, item.tool?.call.id.clone())))
        .collect();
    let mut unsaved = None;
    if !changing.is_empty() {
        let mut conversations = saving.p1();
        for (agent, call) in &changing {
            if let Ok(mut conversation) = conversations.get_mut(*agent) {
                mark_started(&mut conversation, call);
            }
        }
        if let Some(agent) = saving.p0().save() {
            saved_some(&mut passes, agent);
        }
        unsaved = log.flush().err().map(|failure| {
            let why = format!(
                "Not run: the session can no longer be saved ({failure}). Nothing changed."
            );
            ErrorReport::new(ErrorKind::Internal, why)
        });
    }
    // In the order they were spawned, such as a reply's calls in call order.
    let mut ready: Vec<_> = ready.iter_mut().collect();
    ready.sort_unstable_by_key(|item| item.effect.id());
    for item in ready {
        // Messages are accepted before. An effect decided since the
        // decisions were applied, such as by an observer of an answer
        // they applied, is answered by the next pass's.
        if item.message || item.decision.is_decided() {
            continue;
        }
        let ending = item.of.and_then(|of| turns.get(of.0).ok());
        if ending.and_then(TurnOutcome::stops).is_some() {
            continue;
        }
        let (entity, effect) = (item.entity, item.effect);
        let changes = item.tool.is_some() && !item.footprint.is_some_and(|it| it.0.reads_only());
        if let Some(unsaved) = unsaved.as_ref().filter(|_| changes) {
            commands
                .entity(entity)
                .try_insert_if_new(Outcome(Err(unsaved.clone())));
            continue;
        }
        let agent = item.agent.map(|agent| agent.0);
        let caller = agent.and_then(|agent| agents.get(agent).ok());
        let scope = caller.map(|id| id.0.as_str()).unwrap_or_default();
        let origin = DispatchContext {
            parent: effect.parent,
            scope: caller.map(|id| Arc::from(id.0.as_str())),
            purpose: effect.purpose.clone(),
        };
        let mut started = commands.entity(entity);
        if let Some(mut model) = item.model {
            let (connection, mut dropped) = match models.start(&mut model, item.adapters, agent) {
                Ok(started) => started,
                Err(report) => {
                    started.try_insert_if_new(Outcome(Err(report)));
                    continue;
                }
            };
            let kind = EffectKind::Completion {
                request: model.request.clone(),
                stream: true,
            };
            dropped.extend(connection.handler.0.check(&kind));
            let reference = connection.spec.reference();
            let span =
                info_span!("model_call", agent = scope, effect = %effect.id, model = reference);
            let (sender, feed) = feed();
            let wake_up = wake.clone();
            let consume = move |reply| stream_reply(reply, sender, wake_up);
            let handler = connection.handler.0.clone();
            let work = effects.run(effect.id, origin, handler, kind, consume);
            let running = Running::finishing(model_pool(), &wake, work.instrument(span), answer);
            started.try_insert((
                Started { at },
                running,
                feed,
                SentTo(connection),
                DroppedOptions(dropped),
            ));
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
            (_, Some(ToolRequest { call })) => match tools.start(call, item.handler, agent) {
                ToolStart::Run(handler, kind) => (handler, kind),
                ToolStart::Open(open, tool) => {
                    commands.queue(move |world: &mut World| {
                        if let Err(report) = open(world, tool, entity, Started { at })
                            && let Ok(mut call) = world.get_entity_mut(entity)
                        {
                            call.insert_if_new(Outcome(Err(report)));
                        }
                    });
                    continue;
                }
                ToolStart::Refused(report) => {
                    started.try_insert_if_new(Outcome(Err(report)));
                    continue;
                }
            },
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
    decision: &'static Decision,
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
        world.write_message(EffectRecorded(record));
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
    if let Ok(mut effect) = world.get_entity_mut(entity) {
        effect.despawn_related::<Holds>();
    }
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
    for record in buffer.take_resolved_where(|id| !live.contains(&id)) {
        world.write_message(EffectRecorded(record));
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
                published: world
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
                first_item_ms: None,
                finished_ms: Some(now_ms()),
                reviewed: None,
                disposition: Disposition::Ran,
                decided_by: None,
            }
        }
    };
    // A stream fed from outside the buffer's dispatch still timed its first item.
    record.first_item_ms = record.first_item_ms.or_else(|| {
        let timing = world.get::<super::turn::StreamTiming>(entity)?;
        Some(millis(timing.first))
    });
    record.reviewed = reviewed;
    record.disposition = disposition;
    record.decided_by.clone_from(&effect.decided_by);
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
            .map_or_else(|| tool_key(name), |handler| handler.0.0.key().clone());
        return Some((key, EffectKind::from(call)));
    }
    if let Some(model) = world.get::<ModelRequest>(entity) {
        let key = model.connection.as_ref().map_or_else(
            || HandlerKey::new("completion"),
            |connection| connection.handler.0.key().clone(),
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
        Some(handler) => handler.0.key().clone(),
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
