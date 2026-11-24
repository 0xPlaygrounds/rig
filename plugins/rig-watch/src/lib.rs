//! Wakes an idle agent when something it waits on happens.
//!
//! An agent that directs other processes (workers in panes, reviewers, a
//! build) ends its turn and then hears nothing until the user types. With
//! the `watch` tool it says what it waits on, each a fact of a file
//! ([`Kind`]). Every few seconds the plugin looks. A watch whose fact holds
//! fires once: the agent gets a message that names it, which starts a turn
//! if it is idle. An agent in a turn is looked at once the turn is over.
//! While the agent has not said it is done, a message also comes every
//! `every_minutes` with nothing fired, for a worker that hangs. `done` in a
//! call, `/watch off` or the cap on wakes ends it; `/watch` lists what is
//! watched. [`Watching`] is saved with the session.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_watch::WatchPlugin;
//!
//! let mut app = App::new();
//! app.add_plugins(WatchPlugin);
//! ```

use std::fmt;
use std::fs;
use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::clock::ClockPlugin;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// How often the facts are looked at.
const LOOK: Duration = Duration::from_secs(5);
/// How much of a file `has` and `lacks` read.
const READ_MAX: u64 = 4096;
/// The wakes one arming may make before it ends by itself.
const WAKES_MAX: u32 = 60;
/// The minutes between wakes with nothing fired, when a call names none.
const EVERY_MINUTES: u32 = 10;

/// The tool and the command.
const NAME: &str = "watch";
/// The name the plugin decides its calls and writes its wakes as
/// ([`Decider::plugin`], [`Author::plugin`]).
const ORIGIN: &str = "rig-watch";

const HELP: &str = "What the agent waits on: /watch lists it, /watch off ends it";

const DESCRIPTION: &str = "Wait for facts of files without the user: after your turn ends you are \
    woken by a message when one holds, and every `every_minutes` (default 10) while nothing has, \
    until you call this with `done: true`. Use it whenever you would otherwise end a turn waiting \
    on other processes: a worker's report file (`exists`), a pane's title or state file changing \
    (`has`/`lacks` TEXT in its first 4 KiB), a folder removed (`gone`). Each watch fires once. \
    Calling again adds watches; `done: true` ends all of it, and you MUST call that when the \
    whole task is finished or needs the user. End your turn after arming; do not poll in a loop.";

/// The `watch` tool, `/watch`, and the wakes.
#[derive(Default)]
pub struct WatchPlugin;

impl Plugin for WatchPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        // The wall clock the looks are timed by, which the agent has already.
        if !app.is_plugin_added::<ClockPlugin>() {
            app.add_plugins(ClockPlugin);
        }
        app.world_mut()
            .spawn((Name::new("watch timer"), WatchTimer));
        app.register_required_components::<Agent, Watching>()
            .add_command(NAME, HELP, command)
            .add_open_tool(NAME, DESCRIPTION, options, on_watch)
            .add_observer(look)
            .add_systems(PostUpdate, arm);
    }
}

/// What an agent waits on, saved with it.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Watching {
    /// Whether the agent is woken at all: from its first watch until `done`.
    pub active: bool,
    /// What has not fired yet.
    pub watches: Vec<Watch>,
    /// The minutes between wakes with nothing fired.
    pub every_minutes: u32,
    /// When the agent was last woken or armed, in seconds since the epoch.
    pub last: u64,
    /// The wakes left before the arming ends by itself.
    pub wakes_left: u32,
}

/// The fact of a file a [`Watch`] waits for.
#[derive(Reflect, Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Default, Clone, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// The file or folder is there.
    #[default]
    Exists,
    /// It is not.
    Gone,
    /// The file's first 4 KiB hold the text.
    Has,
    /// They do not; a file that is not there lacks every text.
    Lacks,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Exists => "exists",
            Self::Gone => "gone",
            Self::Has => "has",
            Self::Lacks => "lacks",
        })
    }
}

/// One fact waited for, and one watch of a `watch` call.
#[derive(Reflect, Deserialize, JsonSchema, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Default, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Watch {
    /// A few words you will recognise it by, such as `task 7c report`.
    pub label: String,
    /// `exists`, `gone`, `has` or `lacks`.
    pub kind: Kind,
    /// The file or folder, by its whole path.
    pub path: String,
    /// For `has` and `lacks`: the text looked for in the file's first 4 KiB.
    #[serde(default)]
    pub text: String,
}

impl fmt::Display for Watch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} {}", self.label, self.kind, self.path)?;
        if self.text.is_empty() {
            Ok(())
        } else {
            write!(f, " {:?}", self.text)
        }
    }
}

impl Watch {
    /// Whether the fact holds now.
    fn holds(&self) -> bool {
        let has = || {
            let mut start = Vec::new();
            let file = fs::File::open(&self.path).ok()?;
            file.take(READ_MAX).read_to_end(&mut start).ok()?;
            Some(String::from_utf8_lossy(&start).contains(&self.text))
        };
        match self.kind {
            Kind::Exists => fs::symlink_metadata(&self.path).is_ok(),
            Kind::Gone => fs::symlink_metadata(&self.path).is_err(),
            Kind::Has => has() == Some(true),
            Kind::Lacks => has() != Some(true),
        }
    }

    /// Why the watch cannot be taken, if it cannot.
    fn refused(&self) -> Option<String> {
        if !self.path.starts_with('/') {
            return Some(format!("`{}`: give the whole path.", self.path));
        }
        let texted = matches!(self.kind, Kind::Has | Kind::Lacks);
        (texted && self.text.is_empty()).then(|| format!("`{}` needs a text.", self.kind))
    }
}

/// The entity whose [`WakeAt`] is the next look.
#[derive(Component)]
struct WatchTimer;

/// Now, in seconds since the epoch.
fn now() -> u64 {
    let now = SystemTime::now().duration_since(UNIX_EPOCH);
    now.map_or(0, |now| now.as_secs())
}

/// The arguments of a `watch` call.
#[derive(Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
struct Args {
    /// Watches to add.
    #[serde(default)]
    add: Vec<Watch>,
    /// Minutes between wakes while nothing fires; kept from the last call if not given.
    #[serde(default)]
    every_minutes: Option<u32>,
    /// End all watching: the task is finished, or waits on the user.
    #[serde(default)]
    done: bool,
}

/// Does what a call asks on the agent's [`Watching`], at the time `at`, and
/// says what the tool answers. A call with a watch that cannot be taken
/// (`Err`, with why) changes nothing.
fn apply(watching: &mut Watching, args: Args, at: u64) -> Result<String, String> {
    if args.done {
        *watching = Watching::default();
        return Ok("Watching ended: you are not woken again.".to_owned());
    }
    if let Some(why) = args.add.iter().find_map(Watch::refused) {
        return Err(why);
    }
    watching.watches.extend(args.add);
    if let Some(every) = args.every_minutes {
        watching.every_minutes = every.max(1);
    } else if watching.every_minutes == 0 {
        watching.every_minutes = EVERY_MINUTES;
    }
    if !watching.active {
        watching.wakes_left = WAKES_MAX;
    }
    watching.active = true;
    watching.last = at;
    Ok(format!(
        "Watching {} thing(s); woken when one holds, and every {} min otherwise, at most {} \
         more times. End your turn now; call with `done: true` when the task is finished.",
        watching.watches.len(),
        watching.every_minutes,
        watching.wakes_left
    ))
}

/// A `watch` call.
fn on_watch(
    mut called: On<ToolCalled<Args>>,
    mut agents: Query<&mut Watching>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let args = std::mem::take(&mut called.args);
    let result = match agents.get_mut(called.agent) {
        Ok(mut watching) => apply(&mut watching, args, now()),
        Err(_) => Err("Not an agent.".to_owned()),
    };
    let result = match result {
        Ok(said) => ToolResult::success(said.into()),
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin(ORIGIN), result);
    }
}

/// `/watch`, `/watch off`.
fn command(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Watching>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(mut watching) = agents.get_mut(args.agent) else {
        return;
    };
    let text = match args.args.trim() {
        "off" => {
            *watching = Watching::default();
            "Watching ended.".to_owned()
        }
        "" if !watching.active => "Nothing is watched.".to_owned(),
        "" => {
            let mut lines = vec![format!(
                "Woken every {} min with nothing fired; {} wakes left. /watch off ends it.",
                watching.every_minutes, watching.wakes_left
            )];
            lines.extend(watching.watches.iter().map(|watch| format!("  {watch}")));
            lines.join("\n")
        }
        _ => {
            notices.write(Notice::error(args.agent, HELP));
            return;
        }
    };
    notices.write(Notice::info(args.agent, text));
}

/// The timer runs while any agent watches, and not otherwise.
fn arm(
    agents: Query<&Watching>,
    timer: Single<(Entity, Has<WakeAt>), With<WatchTimer>>,
    mut commands: Commands,
) {
    let (timer, armed) = *timer;
    let wanted = agents.iter().any(|watching| watching.active);
    if wanted && !armed {
        commands.entity(timer).insert(WakeAt::after(LOOK));
    } else if !wanted && armed {
        commands.entity(timer).remove::<(WakeAt, Due)>();
    }
}

/// `at`, seconds since the epoch, as the local date and time of day.
fn local(at: u64) -> String {
    let at = i64::try_from(at).ok();
    let at = at.and_then(|at| chrono::DateTime::from_timestamp(at, 0));
    at.map_or_else(String::new, |at| {
        let at = at.with_timezone(&chrono::Local);
        at.format("%Y-%m-%d %H:%M:%S %:z").to_string()
    })
}

/// The message a wake at the time `at` is, or `None` when there is nothing
/// to say yet. The watches that held are taken off.
fn wake(watching: &mut Watching, at: u64) -> Option<String> {
    let waited = at.saturating_sub(watching.last);
    let late = waited >= u64::from(watching.every_minutes) * 60;
    if !late && !watching.watches.iter().any(Watch::holds) {
        return None;
    }
    let (fired, kept): (Vec<_>, Vec<_>) = watching.watches.drain(..).partition(Watch::holds);
    watching.watches = kept;
    watching.last = at;
    watching.wakes_left = watching.wakes_left.saturating_sub(1);
    let mut lines = vec![format!("[watch] {}", local(at))];
    if fired.is_empty() {
        let minutes = waited / 60;
        lines.push(format!(
            "Nothing fired in {minutes} min; look at what you wait on."
        ));
    } else {
        lines.push("Fired:".to_owned());
        lines.extend(fired.iter().map(|watch| format!("  {watch}")));
    }
    if !watching.watches.is_empty() {
        lines.push("Still watched:".to_owned());
        lines.extend(watching.watches.iter().map(|watch| format!("  {watch}")));
    }
    lines.push(if watching.wakes_left == 0 {
        *watching = Watching::default();
        "That was the last wake: watching has ended. Arm it again if needed.".to_owned()
    } else {
        "Act on it, add watches for what you start, and end your turn; `done: true` when the \
         task is finished or needs the user."
            .to_owned()
    });
    Some(lines.join("\n"))
}

/// The timer went off: every watching agent that is in no turn is looked
/// at, and woken if there is something to say.
fn look(
    due: On<Add<Due>>,
    timers: Query<(), With<WatchTimer>>,
    mut agents: Query<(Entity, &mut Watching), Without<ActiveTurn>>,
    mut commands: Commands,
) {
    if !timers.contains(due.entity) {
        return;
    }
    let at = now();
    for (agent, mut watching) in &mut agents {
        if !watching.active {
            continue;
        }
        // A look that finds nothing changes nothing, so nothing is saved.
        if let Some(text) = wake(watching.bypass_change_detection(), at) {
            watching.set_changed();
            let message = Deliver::new(agent, text, DeliveryMode::Queue);
            commands.trigger(message.with_author(Author::plugin(ORIGIN)));
        }
    }
    commands.entity(due.entity).insert(WakeAt::after(LOOK));
}
