//! FH-04: thinking keywords. A user's message that says `think`, `think
//! hard` or `ultrathink` (whole words, any case, the longest match, outside
//! code) raises the reasoning setting of every model request of the turn it
//! goes with, then the next turn is back to the agent's own setting. The
//! message itself is left as typed, and the agent's [`ReasoningSetting`] is never
//! changed. Keywords, budgets and whether subagents inherit the raised
//! level are read from `RIG_HOME/think.toml`.

use std::collections::{BTreeMap, HashSet};

use rig_basics::subagents::SpawnSubagent;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::completion::{Effort as Level, Reasoning};
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

#[cfg(test)]
mod tests;

/// After the agent's reasoning setting, gone just after it.
const THINK_ITEM: StatusItem = StatusItem::new("fh-thinking-keywords", Side::Left, 45, 12);

/// The keywords and budgets of `RIG_HOME/think.toml`:
///
/// ```toml
/// inherit = false          # subagents started in the turn get the level
/// [keywords]
/// "think" = "low"
/// "think hard" = "medium"
/// [budgets]                # tokens, for models that take a budget
/// low = 4096
/// ```
///
/// A `[keywords]` table replaces the default keywords; a `[budgets]` table
/// replaces the default budget of each level it names.
#[derive(Resource, Clone, Debug)]
pub struct ThinkConfig {
    /// Each keyword, lower case, and the level it asks for.
    pub keywords: Vec<(String, Level)>,
    /// The reasoning budget each level stands for, by the level's word.
    pub budgets: BTreeMap<String, u32>,
    /// Whether a subagent started during a raised turn gets its level.
    pub inherit: bool,
    /// What was wrong with the file, shown at startup.
    pub problems: Vec<String>,
}

impl Default for ThinkConfig {
    fn default() -> Self {
        let keywords = [
            ("think", Level::Low),
            ("think hard", Level::Medium),
            ("think more", Level::Medium),
            ("think harder", Level::High),
            ("ultrathink", Level::High),
        ];
        let budgets = [("low", 4096), ("medium", 16_384), ("high", 32_768)];
        Self {
            keywords: keywords
                .map(|(word, level)| (word.to_owned(), level))
                .into(),
            budgets: budgets
                .map(|(name, tokens)| (name.to_owned(), tokens))
                .into(),
            inherit: false,
            problems: Vec::new(),
        }
    }
}

impl ThinkConfig {
    /// The config in the TOML file at `path`: the defaults for what it
    /// leaves out, or when there is none.
    pub fn read(path: &std::path::Path) -> Self {
        let file = Settings::read(path);
        let mut config = Self {
            problems: file.problems().to_vec(),
            ..Self::default()
        };
        match file.get::<BTreeMap<String, Level>>("keywords") {
            Ok(keywords) if !keywords.is_empty() => {
                let keywords = keywords.into_iter();
                config.keywords = keywords
                    .map(|(word, level)| (word.to_lowercase(), level))
                    .collect();
            }
            Ok(_) => {}
            Err(why) => config.problems.push(why),
        }
        match file.get::<BTreeMap<String, u32>>("budgets") {
            Ok(budgets) => config.budgets.extend(budgets),
            Err(why) => config.problems.push(why),
        }
        match file.get::<bool>("inherit") {
            Ok(inherit) => config.inherit = inherit,
            Err(why) => config.problems.push(why),
        }
        config
    }

    /// The keyword `text` asks for and its level: the longest one found
    /// as whole words, any case, outside fenced blocks and inline code.
    pub fn find(&self, text: &str) -> Option<(&str, Level)> {
        let words = words(&prose(text));
        self.keywords
            .iter()
            .filter(|(keyword, _)| {
                let wanted: Vec<&str> = keyword.split_whitespace().collect();
                !wanted.is_empty() && words.windows(wanted.len()).any(|seen| seen == wanted)
            })
            .max_by_key(|(keyword, _)| (keyword.split_whitespace().count(), keyword.len()))
            .map(|(keyword, level)| (keyword.as_str(), *level))
    }

    /// The setting `level` is on `spec`: a budget, below the request's
    /// output limit `limit`, on a model that takes budgets and no levels;
    /// the level otherwise. `None` on a model that does not reason.
    pub fn reasoning(
        &self,
        level: Level,
        spec: &ModelSpec,
        limit: Option<u64>,
    ) -> Option<Reasoning> {
        let support = &spec.reasoning;
        if !support.supported() {
            return None;
        }
        let budget = support
            .budget()
            .filter(|_| support.levels().is_some_and(<[_]>::is_empty));
        let Some(range) = budget else {
            return Some(Reasoning::Effort(level));
        };
        let limit = limit.or(spec.max_output_tokens.map(u64::from));
        let below = limit.map_or(u32::MAX, |limit| {
            u32::try_from(limit.saturating_sub(1)).unwrap_or(u32::MAX)
        });
        let wanted = self
            .budgets
            .get(level.as_str())
            .copied()
            .unwrap_or(*range.end());
        let tokens = wanted.min(below).min(*range.end()).max(*range.start());
        Some(Reasoning::Budget { tokens })
    }
}

/// `text` without its fenced code blocks and inline code.
fn prose(text: &str) -> String {
    text.split("```")
        .step_by(2)
        .flat_map(|outside| outside.split('`').step_by(2))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The words of `text`, lower case.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// On a user's message that asked for more thinking, and on the turn it
/// went with: the keyword and its level.
#[derive(Component, Clone, Debug)]
pub struct ThinkKeyword {
    /// The keyword as configured.
    pub keyword: String,
    /// The level it asks for.
    pub level: Level,
}

/// On a turn whose requests are raised: the level they carry.
#[derive(Component, Clone, Copy, Debug)]
pub struct Raised(pub Level);

/// The plugin's stage systems, which edit model requests and subagent
/// requests in [`PrepareSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThinkSystems;

/// Raises a turn's reasoning setting when the user's message names a
/// thinking keyword.
#[derive(Default)]
pub struct ThinkingKeywordsPlugin;

impl Plugin for ThinkingKeywordsPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<ThinkConfig>() {
            let path = RigHome::of(app.world()).root().join("think.toml");
            app.insert_resource(ThinkConfig::read(&path));
        }
        app.configure_sets(AgentPass, ThinkSystems.in_set(PrepareSystems::Edit))
            .add_systems(Startup, show_problems)
            .add_systems(
                AgentPass,
                (
                    mark_messages.in_set(PrepareSystems::Input),
                    (raise_requests, raise_subagents).in_set(ThinkSystems),
                ),
            )
            .add_systems(PostUpdate, show_raised.in_set(StatusSystems::Show))
            .add_observer(carry_to_turn);
    }
}

/// Says what was wrong with `think.toml`.
fn show_problems(config: Res<ThinkConfig>, mut notices: MessageWriter<Notice>) {
    for problem in &config.problems {
        notices.write(Notice::warn(None, format!("think.toml: {problem}")));
    }
}

/// Marks each user's message that names a keyword with it; its text stays.
fn mark_messages(
    messages: Query<(Entity, &Incoming), (Added<Admitted>, IncomingMessages)>,
    config: Res<ThinkConfig>,
    mut commands: Commands,
) {
    for (message, incoming) in &messages {
        if incoming.author.kind != AuthorKind::User {
            continue;
        }
        if let Some((keyword, level)) = config.find(incoming.text()) {
            let keyword = keyword.to_owned();
            commands
                .entity(message)
                .insert(ThinkKeyword { keyword, level });
        }
    }
}

/// A keyword message's keyword goes on the turn it is delivered into, which
/// outlives the message's settled effect.
fn carry_to_turn(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&ThinkKeyword, &DeliveredIn)>,
    mut commands: Commands,
) {
    if let Ok((keyword, &DeliveredIn(turn))) = messages.get(delivered.entity) {
        commands.entity(turn).try_insert(keyword.clone());
    }
}

/// Each model call of a turn a keyword message went with carries the
/// keyword's level, unless the model has no reasoning setting, which is
/// said once per model.
fn raise_requests(
    mut calls: Query<(&mut ModelRequest, &CallOf, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    keywords: Query<&ThinkKeyword, With<TurnOf>>,
    agents: Query<Option<&Connection>>,
    config: Res<ThinkConfig>,
    mut warned: Local<HashSet<String>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (mut call, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok(keyword) = keywords.get(turn) else {
            continue;
        };
        let own = agents.get(agent).ok().flatten();
        let Some(spec) = call.connection.as_ref().or(own).map(|to| to.spec.clone()) else {
            continue;
        };
        let limit = call.request.max_tokens;
        match config.reasoning(keyword.level, &spec, limit) {
            Some(reasoning) => {
                call.request_mut().options.reasoning = Some(reasoning);
                commands.entity(turn).insert(Raised(keyword.level));
            }
            None => {
                call.request_mut().options.reasoning = None;
                if warned.insert(spec.reference()) {
                    let text = format!(
                        "{} has no reasoning setting; ignoring '{}'",
                        spec.display_name, keyword.keyword
                    );
                    notices.write(Notice::info(agent, text));
                }
            }
        }
    }
}

/// With `inherit`, a subagent started in a raised turn, at no reasoning
/// setting of its own, gets the turn's level.
fn raise_subagents(
    mut requests: Query<(&mut SpawnSubagent, &ForAgent), Added<Admitted>>,
    parents: Query<&ActiveTurn>,
    raised: Query<&Raised>,
    config: Res<ThinkConfig>,
) {
    if !config.inherit {
        return;
    }
    for (mut request, &ForAgent(parent)) in &mut requests {
        let level = parents.get(parent).and_then(|turn| raised.get(turn.turn()));
        if let (Ok(Raised(level)), None) = (level, &request.effort) {
            request.effort = Some(level.as_str().to_owned());
        }
    }
}

/// Shows a raised turn's level on the status line while it runs.
fn show_raised(mut turns: Query<(&Raised, &mut StatusItems), Changed<Raised>>) {
    for (Raised(level), mut items) in &mut turns {
        items.show(THINK_ITEM.says(format!("think: {}", level.as_str()), Tone::Yellow));
    }
}
