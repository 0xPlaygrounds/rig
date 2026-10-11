//! What the command line chooses beyond the launcher's own options: the
//! options plugins take ([`Flag`]), and the front of the run ([`Front`]).
//!
//! A plugin declares its options in `build` with `app.add_flag(..)` and
//! reads them from [`Invoked`]: `rig --json` reaches the agent as the flag
//! `json`. The agent refuses an option no plugin of its build declared,
//! and lists the declared ones for the launcher's `rig help`.
//!
//! A front shows the agents to someone and takes what they type: the
//! terminal view, `--print`, a window, a web server. Each offers itself in
//! `build` with `app.add_front(..)`, saying which runs it can take: an
//! interactive terminal, a run without one, or a run whose command line
//! has its flag. One front takes each run: the one whose flag was given,
//! else the first offered (in `plugins.toml` order) for the kind of run.
//! Each front asks `app.runs_front(name)` in its `finish`, once every
//! plugin offered what it has, and sets itself up only then. An
//! interactive run that no front takes says so and exits. A front whose
//! runs leave nothing to resume, such as one replaying a recording, is
//! [`Front::sessionless`]: a run it takes removes its session's directory
//! when it ends.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! #[derive(Default)]
//! struct JsonFront;
//!
//! impl Plugin for JsonFront {
//!     fn build(&self, app: &mut App) {
//!         app.add_front(Front::new("json").on_flag("json", "Stream the session as JSON lines"));
//!     }
//!
//!     fn finish(&self, app: &mut App) {
//!         if app.runs_front("json") {
//!             // Own stdout and stdin from here.
//!         }
//!     }
//! }
//! ```

use std::fmt::Write as _;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::launcher::Invoked;

/// An option a plugin takes on the command line: `--name`, or with a
/// value, `--name=VALUE`.
#[derive(Clone, Debug, PartialEq, Eq)]
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

    /// How `rig help` shows it.
    fn usage(&self) -> String {
        let option = match self.value {
            Some(value) => format!("--{}={value}", self.name),
            None => format!("--{}", self.name),
        };
        format!("  {option:<23}{}", self.help)
    }
}

/// A front a plugin offers, and which runs it can take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Front {
    /// Its name, which [`AppCliExt::runs_front`] asks about.
    pub name: &'static str,
    /// Whether it takes a run on an interactive terminal.
    pub terminal: bool,
    /// Whether it takes a run without one (`--print`, stdin piped in).
    pub headless: bool,
    /// The option that gives it the run, whatever the terminal.
    pub flag: Option<Flag>,
    /// Whether its runs keep no session: a run it takes removes its
    /// session's directory when it ends.
    pub sessionless: bool,
}

impl Front {
    /// A front that takes no run until told which.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            terminal: false,
            headless: false,
            flag: None,
            sessionless: false,
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

    /// It takes any run with `--flag`, an option it declares with `help`.
    pub fn on_flag(mut self, flag: &'static str, help: &'static str) -> Self {
        self.flag = Some(Flag::new(flag, help));
        self
    }
}

/// The options the plugins declared, and the fronts they offered with the
/// one that takes this run.
#[derive(Resource, Default, Debug)]
pub struct Cli {
    flags: Vec<Flag>,
    fronts: Vec<Front>,
    /// The front of this run, once chosen; why none could be, if so.
    chosen: Option<Result<Option<&'static str>, String>>,
}

impl Cli {
    /// The declared options, the fronts' included, in order.
    pub fn flags(&self) -> impl Iterator<Item = &Flag> {
        self.flags
            .iter()
            .chain(self.fronts.iter().filter_map(|front| front.flag.as_ref()))
    }

    /// The front of this run, once chosen.
    pub fn front(&self) -> Option<&'static str> {
        self.chosen.clone().and_then(Result::ok).flatten()
    }

    /// Whether the front of this run keeps no session.
    pub fn sessionless(&self) -> bool {
        let front = self.front();
        let chosen = self
            .fronts
            .iter()
            .find(|offered| Some(offered.name) == front);
        chosen.is_some_and(|front| front.sessionless)
    }

    /// The options as `rig help` lists them, one per line.
    pub fn usage(&self) -> String {
        self.flags().fold(String::new(), |mut text, flag| {
            let _ = writeln!(text, "{}", flag.usage());
            text
        })
    }

    /// Why the command line does not fit this build: an option no plugin
    /// declared, a value where none goes or none where one does, or two
    /// fronts given the run.
    pub fn refusal(&self, invoked: &Invoked) -> Option<String> {
        for (name, value) in &invoked.args.flags {
            let Some(flag) = self.flags().find(|flag| flag.name == name) else {
                let known = self.usage();
                return Some(if known.is_empty() {
                    format!("unknown option `--{name}`: no plugin of this build takes options")
                } else {
                    format!("unknown option `--{name}`; this build's plugins take:\n{known}")
                });
            };
            match (flag.value, value) {
                (Some(value_name), None) => {
                    return Some(format!("`--{name}` needs a value: --{name}={value_name}"));
                }
                (None, Some(_)) => return Some(format!("`--{name}` takes no value")),
                _ => {}
            }
        }
        match &self.chosen {
            Some(Err(why)) => Some(why.clone()),
            _ => None,
        }
    }

    /// Chooses the front of the run `invoked` describes, once.
    fn choose(&mut self, invoked: &Invoked) -> Option<&'static str> {
        if self.chosen.is_none() {
            let given: Vec<&Front> = self
                .fronts
                .iter()
                .filter(|front| {
                    front
                        .flag
                        .as_ref()
                        .is_some_and(|flag| invoked.flag(flag.name))
                })
                .collect();
            let interactive = invoked.interactive();
            self.chosen = Some(match given.as_slice() {
                [front] => Ok(Some(front.name)),
                [] => Ok(self
                    .fronts
                    .iter()
                    .find(|front| {
                        if interactive {
                            front.terminal
                        } else {
                            front.headless
                        }
                    })
                    .map(|front| front.name)),
                several => {
                    let names: Vec<String> = several
                        .iter()
                        .filter_map(|front| front.flag.as_ref())
                        .map(|flag| format!("--{}", flag.name))
                        .collect();
                    Err(format!(
                        "{} each choose the front; give one",
                        names.join(" and ")
                    ))
                }
            });
        }
        self.front()
    }
}

/// Declaring options and fronts while the app is built.
pub trait AppCliExt {
    /// Declares an option of this plugin.
    fn add_flag(&mut self, flag: Flag) -> &mut Self;

    /// Offers a front, and declares its option.
    fn add_front(&mut self, front: Front) -> &mut Self;

    /// Whether the front `name` takes this run. Ask in `finish` or
    /// `cleanup`, once every plugin offered its fronts: the first answer
    /// chooses the run's front.
    fn runs_front(&mut self, name: &str) -> bool;
}

impl AppCliExt for App {
    fn add_flag(&mut self, flag: Flag) -> &mut Self {
        self.world_mut()
            .get_resource_or_init::<Cli>()
            .flags
            .push(flag);
        self
    }

    fn add_front(&mut self, front: Front) -> &mut Self {
        self.world_mut()
            .get_resource_or_init::<Cli>()
            .fronts
            .push(front);
        self
    }

    fn runs_front(&mut self, name: &str) -> bool {
        let invoked = self
            .world()
            .get_resource::<Invoked>()
            .cloned()
            .unwrap_or_default();
        let mut cli = self.world_mut().get_resource_or_init::<Cli>();
        cli.choose(&invoked) == Some(name)
    }
}

impl Invoked {
    /// Whether the command line has the plugin option `--name`.
    pub fn flag(&self, name: &str) -> bool {
        self.args.flags.iter().any(|(flag, _)| flag == name)
    }

    /// The value of the plugin option `--name=VALUE`, if given.
    pub fn flag_value(&self, name: &str) -> Option<&str> {
        self.args
            .flags
            .iter()
            .find(|(flag, _)| flag == name)
            .and_then(|(_, value)| value.as_deref())
    }
}

/// Chooses the front of the run, if no front asked yet, and why the run
/// cannot go on: a command line the build does not take, or an
/// interactive terminal no front takes.
pub(crate) fn refusal(app: &mut App) -> Option<String> {
    let invoked = app
        .world()
        .get_resource::<Invoked>()
        .cloned()
        .unwrap_or_default();
    let mut cli = app.world_mut().get_resource_or_init::<Cli>();
    let front = cli.choose(&invoked);
    if let Some(why) = cli.refusal(&invoked) {
        return Some(why);
    }
    (invoked.interactive() && front.is_none()).then(|| {
        "this build has no front for an interactive terminal: enable rig-tui \
         (`rig plugin add rig_tui::TuiPlugin --crate rig-tui`), or run with -p or with stdin \
         piped in"
            .to_owned()
    })
}

#[cfg(test)]
mod tests;
