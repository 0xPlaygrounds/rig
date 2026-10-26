//! What the command line chooses beyond the launcher's own options: the
//! options plugins take ([`Flag`]), the front of the run ([`Front`]) and
//! the commands that choose one ([`Front::on_command`]), and the secrets
//! plugins ask for ([`Secret`]).
//!
//! A plugin declares its options in `build` with `app.add_flag(..)` and
//! reads them from [`Invoked`]: `rig --json` reaches the agent as the flag
//! `json`, `rig --serve 127.0.0.1:80` as `serve` with its value. The agent
//! refuses an option no plugin of its build declared. Each declaration is
//! an entity (`front:json`, `flag:serve`, `secret:crypt`), so it is marked
//! [`ProvidedBy`](crate::ProvidedBy) its plugin, and another plugin can
//! turn it off with `Disabled`. The launcher reads the command line
//! against what the build declared, which the build describes when asked
//! ([`describe`](crate::describe)).
//!
//! A front shows the agents to someone and takes what they type, such as the
//! terminal view or `--print`. One front takes each run: the one whose
//! command or flag was given, else the first offered (in `plugins.toml`
//! order) for the kind of run. Each sets itself up in `finish` only if
//! `app.runs_front(name)`. The plugin guide's
//! [fronts section](crate::plugin_guide#fronts-options-and-commands) covers
//! sessionless fronts and fronts with a store of their own.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! #[derive(Default)]
//! struct JsonFront;
//!
//! impl Plugin for JsonFront {
//!     fn build(&self, app: &mut App) {
//!         app.add_front(Front::new("json").on_flag("json", "Stream the session as JSON lines"))
//!             // `rig stats [--all]`: not `on_terminal`, so it runs headless
//!             // and its output and exit code pass through the launcher.
//!             .add_front(Front::new("stats").sessionless().on_command(
//!                 "stats",
//!                 "[--all]",
//!                 "Count the saved sessions",
//!             ));
//!     }
//!
//!     fn finish(&self, app: &mut App) {
//!         if app.runs_front("json") {
//!             // Own stdout and stdin from here.
//!         }
//!         if app.runs_front("stats") {
//!             let words = app.world().get_resource::<Invoked>().and_then(Invoked::command);
//!             let all = words.is_some_and(|(_, words)| words.iter().any(|word| word == "--all"));
//!             // Add a `Startup` system that prints, then triggers `Exit`.
//!             # drop(all);
//!         }
//!     }
//! }
//! ```

use std::fmt::Write as _;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::entity_disabling::Disabled;
use bevy_ecs::prelude::*;
use bevy_ecs::query::Allow;
use rig_harness_protocol::{RESERVED_COMMANDS, SessionDir, StartReason};
use rig_memory::{SessionError, Store};

use super::launcher::Invoked;
use super::session::{AppStart, StoreOpener};
use crate::describe::Describing;
use crate::secrets::Secret;

/// An option a plugin takes on the command line: `--name`, or with a
/// value, `--name VALUE` (also written `--name=VALUE`).
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct Flag {
    /// Its name: lowercase letters, digits and dashes.
    pub name: &'static str,
    /// What its value is, such as `ADDR`, when it takes one.
    pub value: Option<&'static str>,
    /// What it does, for `rig help`.
    pub help: &'static str,
}

impl Flag {
    /// `--name`, which takes no value.
    pub fn new(name: &'static str, help: &'static str) -> Self {
        Self {
            name,
            value: None,
            help,
        }
    }

    /// The same option, taking a value described as `value`.
    pub fn value(mut self, value: &'static str) -> Self {
        self.value = Some(value);
        self
    }

    /// How a refusal lists it.
    fn usage(&self) -> String {
        let option = match self.value {
            Some(value) => format!("--{} {value}", self.name),
            None => format!("--{}", self.name),
        };
        format!("  {option:<23}{}", self.help)
    }
}

/// A top-level command, `rig <name> <usage>`, that gives its front the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    /// Its name: lowercase letters, digits and dashes, and none of the
    /// launcher's own (`build`, `plugin`, `help`).
    pub name: &'static str,
    /// What its arguments are, such as `<tool> [path]`; may be empty.
    pub usage: &'static str,
    /// What it does, for `rig help`.
    pub help: &'static str,
}

/// A front a plugin offers, and which runs it can take.
#[derive(Component, Clone, Debug)]
pub struct Front {
    /// Its name, which [`AppCliExt::runs_front`] asks about.
    pub name: &'static str,
    /// Whether it takes a run on an interactive terminal. A command's
    /// front that does not runs headless: the launcher passes its output
    /// and exit code through and never restarts it.
    pub terminal: bool,
    /// Whether it takes a run without one (`--print`, stdin piped in).
    /// This and [`terminal`](Self::terminal) only pick the front of a run
    /// that names no command or option: a front with either takes only
    /// the runs that name it.
    pub headless: bool,
    /// The option that gives it the run, whatever the terminal.
    pub flag: Option<Flag>,
    /// The command that gives it the run, whatever the terminal.
    pub command: Option<Command>,
    /// Whether its runs keep no session: a run it takes removes its
    /// session's directory when it ends, and a resumed session's run is
    /// refused.
    pub sessionless: bool,
    /// Where a run it takes keeps its session, in place of
    /// [`StoreOpener`] ([`Self::session_store`]).
    pub session_store: Option<StoreOpener>,
}

impl Front {
    /// A front that takes no run until told which.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            terminal: false,
            headless: false,
            flag: None,
            command: None,
            sessionless: false,
            session_store: None,
        }
    }

    /// It takes a run on an interactive terminal.
    pub fn on_terminal(mut self) -> Self {
        self.terminal = true;
        self
    }

    /// It takes a run without an interactive terminal.
    pub fn headless(mut self) -> Self {
        self.headless = true;
        self
    }

    /// Its runs keep no session.
    pub fn sessionless(mut self) -> Self {
        self.sessionless = true;
        self
    }

    /// A run it takes keeps its session in the store `open` opens for the
    /// session's directory, such as one of a chat thread's, instead of the
    /// one [`StoreOpener`] opens. Only the running session's store comes
    /// from it: imports and listings still open theirs with
    /// [`StoreOpener`]. The front is chosen, and the store opened, once
    /// every plugin is built.
    pub fn session_store(
        mut self,
        open: impl Fn(&SessionDir) -> Result<Arc<dyn Store>, SessionError> + Send + Sync + 'static,
    ) -> Self {
        self.session_store = Some(StoreOpener::new(open));
        self
    }

    /// It takes any run with `--flag`, an option it declares with `help`.
    pub fn on_flag(mut self, flag: &'static str, help: &'static str) -> Self {
        self.flag = Some(Flag::new(flag, help));
        self
    }

    /// It takes any run of `rig <name> …`, a command whose arguments are
    /// `usage` and which does what `help` says; the words after the name
    /// are read with [`Invoked::command`]. Two plugins with one command,
    /// or a command of the launcher's own name, fail the build.
    pub fn on_command(
        mut self,
        name: &'static str,
        usage: &'static str,
        help: &'static str,
    ) -> Self {
        self.command = Some(Command { name, usage, help });
        self
    }
}

/// The front that takes this run, once chosen.
#[derive(Resource, Default, Debug)]
pub struct Cli {
    /// The front of this run, once chosen; why none could be, if so.
    chosen: Option<Result<Option<Front>, String>>,
}

impl Cli {
    /// The front of this run, once chosen.
    pub fn front(&self) -> Option<&'static str> {
        self.chosen().map(|front| front.name)
    }

    /// Whether the front of this run keeps no session.
    pub fn sessionless(&self) -> bool {
        self.chosen().is_some_and(|front| front.sessionless)
    }

    /// The front of this run, once chosen.
    pub(crate) fn chosen(&self) -> Option<&Front> {
        self.chosen.as_ref()?.as_ref().ok()?.as_ref()
    }
}

/// On the entity of each declaration of an option, front or secret: its
/// place in the order they were declared, which is the order fronts are
/// offered in and options described.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Declared(pub usize);

/// Every declaration of a `T` not turned off, in the order made, with its
/// entity.
pub(crate) fn declared<T: Component + Clone>(world: &mut World) -> Vec<(Entity, T)> {
    let mut declared: Vec<(Declared, Entity, T)> = world
        .query::<(&Declared, Entity, &T)>()
        .iter(world)
        .map(|(&order, entity, it)| (order, entity, it.clone()))
        .collect();
    declared.sort_unstable_by_key(|(order, ..)| *order);
    declared
        .into_iter()
        .map(|(_, entity, it)| (entity, it))
        .collect()
}

/// The declared options, the fronts' included, in order.
pub(crate) fn flags(world: &mut World) -> Vec<Flag> {
    let fronts = declared::<Front>(world);
    declared::<Flag>(world)
        .into_iter()
        .map(|(_, flag)| flag)
        .chain(fronts.into_iter().filter_map(|(_, front)| front.flag))
        .collect()
}

/// Why the command line does not fit this build: an option no plugin
/// declared, a value where none goes or none where one does, an unknown
/// command, two fronts given the run, or a front that keeps no session
/// given a resumed one.
fn refusal_of(world: &mut World, invoked: &Invoked) -> Option<String> {
    let flags = flags(world);
    for (name, value) in &invoked.args.flags {
        let Some(flag) = flags.iter().find(|flag| flag.name == name) else {
            let known: String = flags.iter().fold(String::new(), |mut text, flag| {
                let _ = writeln!(text, "{}", flag.usage());
                text
            });
            return Some(if known.is_empty() {
                format!("unknown option `--{name}`: no plugin of this build takes options")
            } else {
                format!("unknown option `--{name}`; this build's plugins take:\n{known}")
            });
        };
        match (flag.value, value) {
            (Some(value_name), None) => {
                return Some(format!("`--{name}` needs a value: --{name} {value_name}"));
            }
            (None, Some(_)) => return Some(format!("`--{name}` takes no value")),
            _ => {}
        }
    }
    let resumed = world
        .get_resource::<AppStart>()
        .is_some_and(|start| start.0 == StartReason::Resumed);
    match world
        .get_resource::<Cli>()
        .and_then(|cli| cli.chosen.as_ref())
    {
        Some(Err(why)) => Some(why.clone()),
        Some(Ok(Some(front))) if front.sessionless && resumed => Some(format!(
            "the front `{}` keeps no session, so it cannot resume one",
            front.name
        )),
        _ => None,
    }
}

/// Chooses the front of the run, once: none while the build describes
/// itself.
pub(crate) fn choose(world: &mut World) -> Option<&'static str> {
    if world.get_resource_or_init::<Cli>().chosen.is_none() {
        let chosen = if world.contains_resource::<Describing>() {
            Ok(None)
        } else {
            let invoked = world.get_resource::<Invoked>().cloned().unwrap_or_default();
            let fronts: Vec<Front> = declared::<Front>(world)
                .into_iter()
                .map(|(_, front)| front)
                .collect();
            front_of(&fronts, &invoked)
        };
        world.resource_mut::<Cli>().chosen = Some(chosen);
    }
    world.resource::<Cli>().front()
}

/// The front of `fronts` that takes the run `invoked` describes.
fn front_of(fronts: &[Front], invoked: &Invoked) -> Result<Option<Front>, String> {
    let command = invoked.command().map(|(name, _)| name);
    let given: Vec<&Front> = fronts
        .iter()
        .filter(|front| {
            let flagged = front
                .flag
                .as_ref()
                .is_some_and(|flag| invoked.flag(flag.name));
            let commanded = front
                .command
                .as_ref()
                .is_some_and(|given| Some(given.name) == command);
            flagged || commanded
        })
        .collect();
    let interactive = invoked.interactive();
    match given.as_slice() {
        [front] => Ok(Some((*front).clone())),
        [] => match command {
            Some(command) => Err(format!(
                "unknown command `{command}`: no plugin of this build adds it"
            )),
            None => Ok(fronts
                .iter()
                .filter(|front| front.command.is_none() && front.flag.is_none())
                .find(|front| {
                    if interactive {
                        front.terminal
                    } else {
                        front.headless
                    }
                })
                .cloned()),
        },
        several => {
            let names: Vec<String> = several
                .iter()
                .filter_map(|front| match (&front.command, &front.flag) {
                    (Some(given), _) if Some(given.name) == command => {
                        Some(format!("the command `{}`", given.name))
                    }
                    (_, Some(flag)) => Some(format!("--{}", flag.name)),
                    _ => None,
                })
                .collect();
            Err(format!(
                "{} each choose the front; give one",
                names.join(" and ")
            ))
        }
    }
}

/// Declaring options, fronts and secrets while the app is built.
pub trait AppCliExt {
    /// Declares an option of this plugin.
    fn add_flag(&mut self, flag: Flag) -> &mut Self;

    /// Offers a front, and declares its option and command.
    fn add_front(&mut self, front: Front) -> &mut Self;

    /// Asks for a secret, which the launcher gets before the agent starts
    /// (see [`Secrets`](crate::secrets::Secrets)).
    fn add_secret(&mut self, secret: Secret) -> &mut Self;

    /// Whether the front `name` takes this run. Ask in `finish` or
    /// `cleanup`, once every plugin offered its fronts: the first answer
    /// chooses the run's front.
    fn runs_front(&mut self, name: &str) -> bool;
}

/// Spawns the declaration `what` as an entity named `name`, after every
/// declaration before it.
fn declare(app: &mut App, name: String, what: impl Bundle) {
    let world = app.world_mut();
    // A declaration turned off still holds its place.
    let mut made = world.query_filtered::<&Declared, Allow<Disabled>>();
    let order = made.iter(world).map(|it| it.0 + 1).max().unwrap_or(0);
    world.spawn((Name::new(name), Declared(order), what));
}

impl AppCliExt for App {
    fn add_flag(&mut self, flag: Flag) -> &mut Self {
        declare(self, format!("flag:{}", flag.name), flag);
        self
    }

    fn add_front(&mut self, front: Front) -> &mut Self {
        declare(self, format!("front:{}", front.name), front);
        self
    }

    fn add_secret(&mut self, secret: Secret) -> &mut Self {
        declare(self, format!("secret:{}", secret.name), secret);
        self
    }

    fn runs_front(&mut self, name: &str) -> bool {
        choose(self.world_mut()) == Some(name)
    }
}

impl Invoked {
    /// Whether the command line has the plugin option `--name`.
    pub fn flag(&self, name: &str) -> bool {
        self.args.flags.iter().any(|(flag, _)| flag == name)
    }

    /// The value of the plugin option `--name VALUE`, if given.
    pub fn flag_value(&self, name: &str) -> Option<&str> {
        self.args
            .flags
            .iter()
            .find(|(flag, _)| flag == name)
            .and_then(|(_, value)| value.as_deref())
    }

    /// The command `rig <command> …` runs and the words after it, if the
    /// run is a command's.
    pub fn command(&self) -> Option<(&str, &[String])> {
        let command = self.args.command.as_deref()?;
        Some((command, &self.args.arguments))
    }
}

/// Why `command`, a front's, cannot be a command: a launcher's word, or a
/// name the launcher cannot read.
pub(crate) fn command_refusal(command: &Command) -> Option<String> {
    if RESERVED_COMMANDS.contains(&command.name) {
        return Some(format!(
            "the command `{}` is the launcher's own",
            command.name
        ));
    }
    (!rig_harness_protocol::is_name(command.name)).then(|| {
        format!(
            "the command `{}` is not lowercase letters, digits and dashes",
            command.name
        )
    })
}

/// Chooses the front of the run, if no front asked yet, and why the run
/// cannot go on: a command line the build does not take, or an
/// interactive terminal no front takes. Nothing while the build describes
/// itself.
pub(crate) fn refusal(app: &mut App) -> Option<String> {
    let world = app.world_mut();
    if world.contains_resource::<Describing>() {
        return None;
    }
    let invoked = world.get_resource::<Invoked>().cloned().unwrap_or_default();
    let front = choose(world);
    if let Some(why) = refusal_of(world, &invoked) {
        return Some(why);
    }
    (invoked.interactive() && front.is_none()).then(|| {
        "this build has no front for an interactive terminal: enable an \
         interactive front plugin, or run with -p or with stdin piped in"
            .to_owned()
    })
}

#[cfg(test)]
mod tests;
