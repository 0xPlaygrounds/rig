//! Plugin corpus (held-out): H24: Command palette (spec `H24-command-palette.md`).
//!
//! One fuzzy list over everything the user can do, built afresh from what
//! exists each time the filter changes: the slash commands (with their
//! help), the catalog models that can be reached (`/model <id>`), the
//! agents (shown), the tools (with their description, and whether they are
//! off), and the [`PaletteEntry`]s any plugin spawns. A filter matches a
//! subsequence of an entry's label, scored by the word starts it hits
//! ([`rank`]); with an empty filter the recent picks come first, kept in
//! `RIG_HOME` ([`RecentPicks`]).
//!
//! Under the default `tui` feature, Ctrl-P opens it over the terminal view
//! (see the `tui` module).

use std::cmp::{Reverse, max};
use std::path::PathBuf;

use bevy_ecs::system::{SystemParam, SystemState};
use bevy_log::warn;
use rig_harness::prelude::*;
use rig_tools::fs::{read_json, write_json};

#[cfg(feature = "tui")]
mod tui;

#[cfg(feature = "tui")]
pub use tui::{Activate, PaletteSystems};

/// How many recent picks are kept.
const RECENT: usize = 20;

/// What a letter at the start of a word scores.
const WORD_START: i32 = 8;
/// What a letter right after the previous one matched scores besides.
const ADJACENT: i32 = 4;

/// An entry of the palette a plugin adds: spawn one, and despawn it (or
/// insert Bevy's `Disabled`) to take it out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct PaletteEntry {
    /// What is listed and matched, such as `Open the changelog`.
    pub label: String,
    /// What is shown beside it.
    pub detail: String,
    /// The command line it runs, without its `/`, such as `files`.
    pub command: String,
}

impl PaletteEntry {
    /// `label`, shown with `detail`, runs `command`.
    pub fn new(
        label: impl Into<String>,
        detail: impl Into<String>,
        command: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            command: command.into(),
        }
    }
}

/// Where an entry comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A slash command.
    Command,
    /// A catalog model.
    Model,
    /// An agent.
    Agent,
    /// A tool.
    Tool,
    /// A plugin's [`PaletteEntry`].
    Plugin,
}

impl Kind {
    /// Its name, as listed.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::Model => "model",
            Self::Agent => "agent",
            Self::Tool => "tool",
            Self::Plugin => "action",
        }
    }
}

/// What picking an entry does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Runs this command line (without its `/`) at once.
    Run(String),
    /// Puts `/name ` in the input, for a command that takes arguments.
    Prefill(String),
    /// Shows this agent.
    Focus(Entity),
    /// Says what the tool does.
    Describe,
}

/// One line of the palette.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Where it comes from.
    pub kind: Kind,
    /// What is matched and listed first, such as `/model`.
    pub label: String,
    /// What is shown beside it, such as a command's help.
    pub detail: String,
    /// A tool that is off: disabled, or not one the agent shown may call.
    pub off: bool,
    /// What picking it does.
    pub action: Action,
}

impl Entry {
    /// How [`RecentPicks`] names it.
    pub fn key(&self) -> String {
        format!("{}:{}", self.kind.tag(), self.label)
    }
}

/// The entries picked last, newest first, by [`Entry::key`], kept in
/// `RIG_HOME/command-palette.json`.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct RecentPicks(pub Vec<String>);

impl RecentPicks {
    /// The file in `home`.
    pub fn path(home: &Home) -> PathBuf {
        home.root().join("command-palette.json")
    }

    /// The picks kept in `home`; none when there is no file.
    pub fn load(home: &Home) -> Self {
        Self(read_json(&Self::path(home)).unwrap_or_default())
    }

    /// Puts `entry` first and keeps the list in `home`.
    pub fn record(&mut self, entry: &Entry, home: &Home) {
        let key = entry.key();
        self.0.retain(|picked| *picked != key);
        self.0.insert(0, key);
        self.0.truncate(RECENT);
        let path = Self::path(home);
        if let Err(failure) = write_json(&path, &self.0) {
            warn!(
                "could not keep the palette's recent picks in {}: {failure}",
                path.display()
            );
        }
    }

    /// How recently `key` was picked: 0 is the last pick.
    fn position(&self, key: &str) -> Option<usize> {
        self.0.iter().position(|picked| picked == key)
    }
}

/// Everything the palette lists, read live.
#[derive(SystemParam)]
pub struct PaletteSources<'w, 's> {
    commands: Query<'w, 's, (&'static Name, &'static SlashCommand)>,
    models: Option<Res<'w, Models>>,
    agents: Query<'w, 's, (Entity, &'static Name), With<Agent>>,
    tools: Query<'w, 's, (&'static ToolDef, Has<Disabled>)>,
    access: Query<'w, 's, &'static ToolAccess>,
    shown: Option<Res<'w, ShownAgent>>,
    entries: Query<'w, 's, &'static PaletteEntry>,
    recent: Res<'w, RecentPicks>,
}

impl PaletteSources<'_, '_> {
    /// Every entry, in listing order: commands by name, models as the
    /// catalog orders those that can be reached, agents, tools by name,
    /// then the plugins' entries.
    pub fn entries(&self) -> Vec<Entry> {
        let mut commands: Vec<(&Name, &SlashCommand)> = self.commands.iter().collect();
        commands.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        let commands = commands.into_iter().map(|(name, command)| {
            let name = name.as_str();
            let takes_args = command
                .help
                .split_once('<')
                .is_some_and(|(_, rest)| rest.contains('>'));
            Entry {
                kind: Kind::Command,
                label: name.to_owned(),
                detail: command.help.clone(),
                off: false,
                action: if takes_args {
                    Action::Prefill(name.to_owned())
                } else {
                    Action::Run(name.trim_start_matches('/').to_owned())
                },
            }
        });
        let models = self.models.iter().flat_map(|models| {
            models.0.reachable().into_iter().map(|spec| {
                let reference = spec.reference();
                Entry {
                    kind: Kind::Model,
                    detail: format!("switch to {}", spec.display_name),
                    off: false,
                    action: Action::Run(format!("model {reference}")),
                    label: reference,
                }
            })
        });
        let agents = self.agents.iter().map(|(agent, name)| Entry {
            kind: Kind::Agent,
            label: name.as_str().to_owned(),
            detail: "show this agent".to_owned(),
            off: false,
            action: Action::Focus(agent),
        });
        let allowed = self
            .shown
            .as_ref()
            .and_then(|shown| shown.0)
            .and_then(|agent| self.access.get(agent).ok());
        let mut tools: Vec<Entry> = self
            .tools
            .iter()
            .map(|(def, disabled)| {
                let name = def.0.name.as_str();
                let denied = allowed.is_some_and(|access| !access.allows(name));
                let description = def.0.description.lines().next().unwrap_or_default();
                Entry {
                    kind: Kind::Tool,
                    label: name.to_owned(),
                    detail: description.to_owned(),
                    off: disabled || denied,
                    action: Action::Describe,
                }
            })
            .collect();
        tools.sort_by(|a, b| a.label.cmp(&b.label));
        let plugins = self.entries.iter().map(|entry| Entry {
            kind: Kind::Plugin,
            label: entry.label.clone(),
            detail: entry.detail.clone(),
            off: false,
            action: Action::Run(entry.command.clone()),
        });
        commands
            .chain(models)
            .chain(agents)
            .chain(tools)
            .chain(plugins)
            .filter(|entry| !entry.label.is_empty())
            .collect()
    }

    /// The entries `filter` matches, best first ([`rank`]).
    pub fn rank(&self, filter: &str) -> Vec<Entry> {
        rank(self.entries(), filter, &self.recent)
    }
}

/// The palette's list for `filter`, as the palette would show it now.
pub fn ranked(world: &mut World, filter: &str) -> Vec<Entry> {
    let mut sources = SystemState::<PaletteSources>::new(world);
    sources
        .get(world)
        .map(|sources| sources.rank(filter))
        .unwrap_or_default()
}

/// The `entries` whose label holds `filter`'s letters in order (spaces
/// ignored, case too), best first: the highest [`score`], then the most
/// recently picked, then the shortest label. With an empty filter, every
/// entry: the recent picks first, newest first, then the others in order.
pub fn rank(entries: Vec<Entry>, filter: &str, recent: &RecentPicks) -> Vec<Entry> {
    let filtering = !filter.trim().is_empty();
    let mut scored: Vec<(i32, usize, Entry)> = entries
        .into_iter()
        .enumerate()
        .filter_map(|(order, entry)| Some((score(filter, &entry.label)?, order, entry)))
        .collect();
    scored.sort_by_cached_key(|(score, order, entry)| {
        let length = if filtering {
            entry.label.chars().count()
        } else {
            0
        };
        let picked = recent.position(&entry.key()).unwrap_or(usize::MAX);
        (Reverse(*score), picked, length, *order)
    });
    scored.into_iter().map(|(.., entry)| entry).collect()
}

/// How well `pattern` matches `text` as a subsequence, ignoring case and
/// the pattern's spaces, or `None` when it does not: the best sum over the
/// letters matched of `WORD_START` for a letter that starts a word (after
/// a character that is no letter or digit, or a capital after a small
/// letter), else 1, plus `ADJACENT` for each letter right after the one
/// before. An empty pattern scores 0.
pub fn score(pattern: &str, text: &str) -> Option<i32> {
    let text: Vec<char> = text.chars().collect();
    let before = std::iter::once(None).chain(text.iter().copied().map(Some));
    let bonuses: Vec<i32> = before
        .zip(&text)
        .map(|(before, &letter)| match before {
            None => WORD_START,
            Some(before) if !before.is_alphanumeric() => WORD_START,
            Some(before) if before.is_lowercase() && letter.is_uppercase() => WORD_START,
            Some(_) => 1,
        })
        .collect();
    let lower: Vec<char> = text.iter().map(|&letter| lowercase(letter)).collect();
    // The best score of the letters so far with the last one at each
    // position of `text`, if they match there.
    let mut best: Option<Vec<Option<i32>>> = None;
    let wanted = pattern.chars().filter(|letter| !letter.is_whitespace());
    for wanted in wanted.map(lowercase) {
        let mut row = Vec::with_capacity(lower.len());
        // The previous letter's best before the position before this one,
        // and at the position before.
        let (mut earlier, mut adjacent): (Option<i32>, Option<i32>) = (None, None);
        for (position, (&letter, &bonus)) in lower.iter().zip(&bonuses).enumerate() {
            let here = match &best {
                _ if letter != wanted => None,
                None => Some(bonus),
                Some(_) => max(
                    earlier.map(|score| score + bonus),
                    adjacent.map(|score| score + bonus + ADJACENT),
                ),
            };
            row.push(here);
            earlier = max(earlier, adjacent);
            adjacent = best
                .as_ref()
                .and_then(|best| best.get(position).copied().flatten());
        }
        best = Some(row);
    }
    match best {
        None => Some(0),
        Some(row) => row.into_iter().flatten().max(),
    }
}

/// `letter` in lower case, when that is one letter.
fn lowercase(letter: char) -> char {
    letter.to_lowercase().next().unwrap_or(letter)
}

/// The command palette: [`PaletteEntry`], the recent picks, and with the
/// `tui` feature, the overlay on Ctrl-P.
#[derive(Default)]
pub struct CommandPalettePlugin;

impl Plugin for CommandPalettePlugin {
    fn build(&self, app: &mut App) {
        let recent = RecentPicks::load(&RigHome::of(app.world()));
        app.init_resource::<RigHome>()
            .insert_resource(recent)
            .register_type::<PaletteEntry>();
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

#[cfg(all(test, feature = "tui"))]
mod tests;
