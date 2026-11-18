//! What the agents are doing, kept for views: every agent's [`Activity`]
//! (its status and its open tool calls), and with the default `tui` feature
//! its status in the terminal view's status line. The status comes from all of the agent's work
//! (`AgentWork`), not only its turn: a question waiting for the user's
//! answer, and a plugin's named effect for the agent, such as a review
//! after the turn, show too. A view reads them instead of
//! deriving them from turns and calls; the agent tree is the agents'
//! [`SpawnedBy`] and [`Spawned`] relationship.
//!
//! [`Activity`] is updated in `PostUpdate`, in [`ActivitySystems`], and
//! changes only when what it says changes, so `Changed<Activity>` tells a
//! view to redraw.
//!
//! While the shown agent works, its status says what it does and, once it
//! has lasted a few seconds, for how long. The bars before it are the
//! [`animation`]'s.

use std::fmt;

use rig_harness::prelude::*;

pub mod animation;

/// Keeps every agent's [`Activity`].
#[derive(Default)]
pub struct ActivityPlugin;

impl Plugin for ActivityPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, Activity>()
            .init_resource::<rig_ecs::retry::RetryPolicy>()
            .add_systems(PostUpdate, update_activity.in_set(ActivitySystems));
        #[cfg(feature = "tui")]
        app.register_required_components::<Agent, StatusClock>()
            .init_resource::<Time<Real>>()
            .configure_sets(
                PostUpdate,
                ActivitySystems.before(rig_tui::StatusSystems::Show),
            )
            .add_systems(PostUpdate, show_status.in_set(rig_tui::StatusSystems::Show));
        #[cfg(feature = "tui")]
        rig_tui::AppStatusPartsExt::add_status_part(app, &STATUS, "What it is doing");
    }
}

/// The system in `PostUpdate` that updates each agent's [`Activity`],
/// before the terminal view's status line. A view that reads it in
/// `PostUpdate` runs after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivitySystems;

/// What an agent is doing; on every agent.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component)]
pub struct Activity {
    /// Its status.
    pub status: ActivityStatus,
    /// Its turn's tool calls without an output yet, in order.
    pub tools: Vec<ToolActivity>,
}

impl Activity {
    /// Whether a turn runs.
    pub fn is_busy(&self) -> bool {
        self.status != ActivityStatus::Idle
    }
}

/// An agent's status.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub enum ActivityStatus {
    /// No turn runs.
    #[default]
    Idle,
    /// Waiting for its model.
    Thinking,
    /// Its model's answer streams: text or a tool call.
    Writing,
    /// Its tools run.
    RunningTools,
    /// A plugin's effect for the agent runs, in its turn or after it, such
    /// as a summary of the older conversation; the effect's [`Name`], such
    /// as `compacting`.
    Busy(String),
    /// A question for the user about the agent waits for its answer; the
    /// [`Ask`]'s title. An inline one shows only while no turn runs.
    Asking(String),
    /// Waiting `seconds` before retry `attempt` of a failed model call.
    Retrying {
        /// The retry's number, from 1.
        attempt: u32,
        /// Whole seconds left, rounded up.
        seconds: u64,
    },
}

impl fmt::Display for ActivityStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => f.write_str("idle"),
            Self::Thinking => f.write_str("thinking"),
            Self::Writing => f.write_str("writing"),
            Self::RunningTools => f.write_str("running tools"),
            Self::Busy(what) => f.write_str(what),
            Self::Asking(title) => write!(f, "asking: {title}"),
            Self::Retrying { attempt, seconds } => write!(f, "retry {attempt} in {seconds}s"),
        }
    }
}

/// A tool call of the turn without an outcome yet.
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
pub struct ToolActivity {
    /// The tool's name.
    pub name: String,
    /// Whether it waits for earlier calls of its reply.
    pub waiting: bool,
    /// The names of the holds on it, which say why, such as `approval`;
    /// empty when nothing holds it. A hold without a [`Name`] is
    /// [`UNNAMED_HOLD`].
    pub held_by: Vec<String>,
    /// Its [`ToolProgress`] line, which its owner keeps current while it
    /// runs, such as `12 of 40 files`.
    pub progress: Option<String>,
}

/// What [`ToolActivity::held_by`] calls a hold without a [`Name`].
pub const UNNAMED_HOLD: &str = "unnamed hold";

/// A plugin's unsettled effect for an agent, which shows by its name.
type Named = (
    With<Effect>,
    Without<TurnCall>,
    Without<ToolRequest>,
    Without<Incoming>,
    Without<Ask>,
    Without<Settled>,
);

/// Each agent's activity, from its questions, its turn's calls and its
/// other work.
fn update_activity(
    mut agents: Query<(&mut Activity, Option<&ActiveTurn>, Option<&AgentWork>), With<Agent>>,
    turns: Query<&Calls>,
    tools: Query<
        (
            &ToolRequest,
            Has<Waiting>,
            Option<&Holds>,
            Option<&ToolProgress>,
            Has<Outcome>,
        ),
        Without<Settled>,
    >,
    (holds, names): (Query<&Holds>, Query<&Name>),
    waits: Query<&Backoff>,
    others: Query<&Name, Named>,
    asks: Query<&Ask, (Without<Outcome>, Without<Settled>)>,
    partials: Query<&Partial, (Without<Settled>, Without<SupersededBy>)>,
) {
    for (mut activity, turn, work) in &mut agents {
        let work: Vec<Entity> = work.map(|work| work.iter().collect()).unwrap_or_default();
        // An inline question leaves a running turn its status: the turn
        // does not wait for it.
        let asks = work.iter().filter_map(|effect| asks.get(*effect).ok());
        let mut asks = asks.filter(|ask| turn.is_none() || !ask.inline);
        let asking = asks
            .next()
            .map(|ask| ActivityStatus::Asking(ask.title.clone()));
        let Some(turn) = turn else {
            let named = work.iter().find_map(|effect| others.get(*effect).ok());
            let status = asking
                .or_else(|| named.map(|name| ActivityStatus::Busy(name.as_str().to_owned())))
                .unwrap_or_default();
            activity.set_if_neq(Activity {
                status,
                tools: Vec::new(),
            });
            continue;
        };
        let calls: Vec<Entity> = turns
            .get(turn.turn())
            .map(|calls| calls.iter().collect())
            .unwrap_or_default();
        // A failed model call waits for its retry, held by its backoff.
        let backoff = calls
            .iter()
            .filter_map(|call| holds.get(*call).ok())
            .find_map(|held| held.iter().find_map(|hold| waits.get(hold).ok()));
        let status = if let Some(asking) = asking {
            asking
        } else if let Some(wait) = backoff {
            ActivityStatus::Retrying {
                attempt: wait.attempt,
                seconds: wait.seconds_left(),
            }
        } else if let Some(name) = calls.iter().find_map(|call| others.get(*call).ok()) {
            ActivityStatus::Busy(name.as_str().to_owned())
        } else if calls.iter().any(|call| tools.contains(*call)) {
            ActivityStatus::RunningTools
        } else if calls
            .iter()
            .filter_map(|call| partials.get(*call).ok())
            .any(|partial| !partial.text.trim().is_empty() || !partial.calls.is_empty())
        {
            ActivityStatus::Writing
        } else {
            ActivityStatus::Thinking
        };
        let open_tools = calls
            .iter()
            .filter_map(|call| tools.get(*call).ok())
            .filter(|(.., done)| !done)
            .map(|(request, waiting, held, progress, _)| ToolActivity {
                name: request.call.function.name.as_str().to_owned(),
                waiting,
                held_by: held
                    .into_iter()
                    .flat_map(Holds::iter)
                    .map(|hold| {
                        names
                            .get(hold)
                            .map_or(UNNAMED_HOLD, Name::as_str)
                            .to_owned()
                    })
                    .collect(),
                progress: progress.map(|ToolProgress(line)| line.clone()),
            })
            .collect();
        activity.set_if_neq(Activity {
            status,
            tools: open_tools,
        });
    }
}

/// Where an agent's status is in the status line.
#[cfg(feature = "tui")]
const STATUS: rig_tui::StatusItem =
    rig_tui::StatusItem::new("status", rig_tui::Side::Left, 50, u8::MAX);

/// When an agent's status last changed, on Bevy's real clock, and the
/// whole seconds its status line shows since then, if any.
#[cfg(feature = "tui")]
#[derive(Component, Default)]
struct StatusClock {
    status: ActivityStatus,
    since: Duration,
    shown: Option<u64>,
    step: Option<usize>,
}

/// The seconds a status lasts before its clock shows, so short steps do
/// not flicker.
#[cfg(feature = "tui")]
const CLOCK_AFTER: u64 = 5;

/// Each agent's status, in its status line: green when idle, yellow at
/// work, after the [`animation`]'s bars when it runs, red waiting to retry. Work that has lasted
/// [`CLOCK_AFTER`] seconds shows how long, such as `thinking… 1m12s`, so a
/// silent model does not look frozen.
#[cfg(feature = "tui")]
fn show_status(
    mut agents: Query<(Ref<Activity>, &mut StatusClock, &mut rig_tui::StatusItems)>,
    time: Res<Time<Real>>,
    policy: Res<rig_ecs::retry::RetryPolicy>,
    pulse: Option<Res<animation::Pulse>>,
) {
    use rig_tui::Tone;
    let now = time.elapsed();
    for (activity, mut clock, mut items) in &mut agents {
        if clock.status != activity.status {
            clock.status = activity.status.clone();
            clock.since = now;
        }
        let seconds = now.saturating_sub(clock.since).as_secs();
        let busy = works(&activity.status);
        let shown = (busy && seconds >= CLOCK_AFTER).then_some(seconds);
        let step = pulse
            .as_ref()
            .and_then(|pulse| pulse.step())
            .filter(|_| busy);
        if !activity.is_changed() && shown == clock.shown && step == clock.step {
            continue;
        }
        clock.shown = shown;
        clock.step = step;
        // No bars without the animation; still ones while it stands.
        let wave = pulse
            .as_ref()
            .map(|_| format!("{} ", animation::wave(step)));
        let wave = wave.unwrap_or_default();
        let shown = match (&activity.status, shown) {
            (ActivityStatus::Idle, _) => STATUS.says("idle", Tone::Green),
            (ActivityStatus::Asking(title), _) => {
                STATUS.says(format!("asking: {title}"), Tone::Cyan)
            }
            (ActivityStatus::Retrying { attempt, seconds }, _) => STATUS.says(
                format!("retry {attempt}/{} in {seconds}s…", policy.max_retries),
                Tone::Red,
            ),
            (status, Some(seconds)) => {
                STATUS.says(format!("{wave}{status}… {}", lasted(seconds)), Tone::Yellow)
            }
            (status, None) => STATUS.says(format!("{wave}{status}…"), Tone::Yellow),
        };
        items.show(shown);
    }
}

/// `seconds` as `12s`, `4m03s` or `1h02m`.
#[cfg(feature = "tui")]
fn lasted(seconds: u64) -> String {
    let (hours, minutes) = (seconds / 3600, seconds / 60 % 60);
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{:02}s", seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

/// Whether `status` is work the agent does itself, not a wait.
#[cfg(feature = "tui")]
pub(crate) fn works(status: &ActivityStatus) -> bool {
    matches!(
        status,
        ActivityStatus::Thinking
            | ActivityStatus::Writing
            | ActivityStatus::RunningTools
            | ActivityStatus::Busy(_)
    )
}

#[cfg(test)]
mod tests;
