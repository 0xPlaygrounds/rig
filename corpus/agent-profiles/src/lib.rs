//! Agent profiles: named settings for an agent, defined in Markdown files
//! in `RIG_HOME/agents/*.md` and `<project>/.rig/agents/*.md` (a project's
//! profile replaces a home profile of the same name). A file's frontmatter
//! says the profile's `description`, `model`, `effort` and `tools` (an
//! allowlist); its body is extra system prompt for that agent alone.
//!
//! `/profile <name>` applies one to the agent shown: its prompt as a
//! [`PromptSection`] of that agent's, its tools as the agent's
//! [`ToolAccess`] (never more than the agent had), its model and its
//! reasoning setting. `/profile none` takes it off and restores the
//! settings from before, and `/profile` alone lists the profiles. The
//! applied profile is an [`AppliedProfile`], saved with the agent, so a
//! restart or `/resume` brings it back. Agents offered the `task` tool see
//! the profiles listed in their system prompt.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rig_harness::prelude::*;
use serde::Deserialize;

/// The tool that starts a subagent (rig-subagents' `task`).
const TASK: &str = "task";

/// Loads the profiles, adds `/profile`, keeps each applied profile's prompt
/// section, and lists the profiles to agents that can start subagents.
#[derive(Default)]
pub struct AgentProfilesPlugin;

impl Plugin for AgentProfilesPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<ProfileDirs>() {
            app.insert_resource(ProfileDirs::from_env());
        }
        let profiles = Profiles::read(app.world().resource::<ProfileDirs>());
        app.insert_resource(profiles)
            .add_observer(reread_on_turn)
            .add_observer(show_prompt)
            .add_observer(hide_prompt)
            .add_observer(list_for_spawners)
            .add_command(
                "profile",
                "Apply an agent profile: /profile <name>, /profile none, or /profile to list them",
                apply_profile,
            );
    }
}

/// The directories profiles are read from, the later ones overriding the
/// earlier ones by name.
#[derive(Resource, Clone, Debug, Default)]
pub struct ProfileDirs(pub Vec<PathBuf>);

impl ProfileDirs {
    /// `RIG_HOME/agents`, then the working directory's `.rig/agents`.
    pub fn from_env() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self(vec![
            Home::from_env().root().join("agents"),
            cwd.join(".rig").join("agents"),
        ])
    }
}

/// One profile, as its file says.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Profile {
    /// One line on what it is for: the frontmatter's `description`, else
    /// the body's first line.
    pub description: String,
    /// A catalog model, as `vendor/model`.
    pub model: Option<String>,
    /// A reasoning setting the model takes, such as `low` or `high`.
    pub effort: Option<String>,
    /// The only tools the agent may use, when given.
    pub tools: Option<Vec<String>>,
    /// Extra system prompt for the agent.
    pub prompt: String,
}

/// A profile file's frontmatter.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Frontmatter {
    description: Option<String>,
    model: Option<String>,
    effort: Option<String>,
    tools: Option<Vec<String>>,
}

impl Profile {
    /// The profile a file's `text` defines: optional YAML frontmatter
    /// between `---` lines, then the prompt.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim_start_matches('\u{feff}');
        let (front, body) = match text.strip_prefix("---") {
            Some(rest) => {
                let (front, after) = rest
                    .split_once("\n---")
                    .ok_or("its frontmatter has no closing `---`")?;
                (front, after.split_once('\n').map_or("", |(_, body)| body))
            }
            None => ("", text),
        };
        let front: Frontmatter = match front.trim() {
            "" => Frontmatter::default(),
            yaml => serde_yaml::from_str(yaml).map_err(|why| why.to_string())?,
        };
        let prompt = body.trim().to_owned();
        let first_line = prompt.lines().next().unwrap_or_default().trim().to_owned();
        Ok(Self {
            description: front.description.unwrap_or(first_line),
            model: front.model,
            effort: front.effort,
            tools: front.tools,
            prompt,
        })
    }
}

/// The profiles by name, and the files that could not be read.
#[derive(Resource, Clone, Debug, Default)]
pub struct Profiles {
    /// The profiles by name.
    pub by_name: BTreeMap<String, Profile>,
    /// Why each file that was left out could not be read.
    pub problems: Vec<String>,
}

impl Profiles {
    /// Reads every `*.md` file of `dirs`; a missing directory has none.
    pub fn read(dirs: &ProfileDirs) -> Self {
        let mut profiles = Self::default();
        for dir in &dirs.0 {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
            paths.sort();
            for path in paths {
                let named = path.file_stem().and_then(|stem| stem.to_str());
                let (Some(name), true) = (named, path.extension().is_some_and(|ext| ext == "md"))
                else {
                    continue;
                };
                let read = std::fs::read_to_string(&path).map_err(|why| why.to_string());
                match read.and_then(|text| Profile::parse(&text)) {
                    Ok(profile) => {
                        profiles.by_name.insert(name.to_owned(), profile);
                    }
                    Err(why) => profiles.problems.push(format!("{}: {why}", path.display())),
                }
            }
        }
        profiles
    }

    /// The names, as `a, b`, or `none`.
    fn names(&self) -> String {
        let names: Vec<&str> = self.by_name.keys().map(String::as_str).collect();
        match names.is_empty() {
            true => "none".to_owned(),
            false => names.join(", "),
        }
    }
}

/// An agent's settings before a profile was applied.
#[derive(Reflect, Clone)]
pub struct Before {
    /// Its tool access.
    pub tools: ToolAccess,
    /// Its catalog model.
    pub model: Option<String>,
    /// Its reasoning setting.
    pub effort: Effort,
}

/// On an agent: the profile applied to it, with its prompt as applied and
/// the settings it replaced. Saved with the session.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Saved)]
pub struct AppliedProfile {
    /// The profile's name.
    pub name: String,
    /// Its prompt, in the agent's prompt section.
    pub prompt: String,
    /// What `/profile none` restores.
    pub before: Before,
}

/// On the prompt section an [`AppliedProfile`] adds to its agent.
#[derive(Component)]
struct ProfilePrompt;

/// Re-reads the profiles when a turn starts, so an edited file counts from
/// the next message.
fn reread_on_turn(
    _started: On<Add<TurnOf>>,
    dirs: Res<ProfileDirs>,
    mut profiles: ResMut<Profiles>,
) {
    *profiles = Profiles::read(&dirs);
}

/// The profile prompt sections, by agent.
type ProfilePrompts<'w, 's> =
    Query<'w, 's, (Entity, &'static SectionOf, &'static mut PromptSection), With<ProfilePrompt>>;

/// Gives an agent the prompt section of the profile inserted on it, as
/// `/profile` or a restore inserts it.
fn show_prompt(
    inserted: On<Insert<AppliedProfile>>,
    applied: Query<&AppliedProfile>,
    mut sections: ProfilePrompts,
    mut commands: Commands,
) {
    let agent = inserted.entity;
    let Ok(profile) = applied.get(agent) else {
        return;
    };
    let text = PromptSection::new(
        PromptSection::ORDER_ROLE,
        "agent_profile",
        profile.prompt.clone(),
    );
    match sections.iter_mut().find(|(_, of, _)| of.0 == agent) {
        Some((.., mut section)) => {
            section.set_if_neq(text);
        }
        None => {
            commands.spawn((
                Name::new("prompt:agent_profile"),
                ProfilePrompt,
                text,
                SectionOf(agent),
            ));
        }
    }
}

/// Takes the profile's prompt section off its agent with the profile.
fn hide_prompt(
    removed: On<Remove<AppliedProfile>>,
    sections: ProfilePrompts,
    mut commands: Commands,
) {
    for (section, ..) in sections.iter().filter(|(_, of, _)| of.0 == removed.entity) {
        commands.entity(section).try_despawn();
    }
}

/// Lists the profiles in the system prompt of an agent offered `task`.
fn list_for_spawners(mut prepare: On<PrepareRequest>, profiles: Res<Profiles>) {
    let spawns = prepare.tools.iter().any(|tool| tool.name.as_str() == TASK);
    if !spawns || profiles.by_name.is_empty() {
        return;
    }
    let lines: Vec<String> = profiles
        .by_name
        .iter()
        .map(|(name, profile)| format!("- {name}: {}", profile.description))
        .collect();
    prepare.preamble.push_str(&format!(
        "\n\n<agent_profiles>\nNamed agent profiles the user can apply with /profile:\n{}\n\
         </agent_profiles>",
        lines.join("\n")
    ));
}

/// What `/profile` reads and changes of the agent.
type AgentSettings<'a> = (
    Option<&'a AppliedProfile>,
    &'a ToolAccess,
    Option<&'a ModelChoice>,
    &'a Effort,
    Has<ActiveTurn>,
);

/// `/profile <name>`, `/profile none`, or `/profile` to list them.
fn apply_profile(
    In(args): In<CommandArgs>,
    agents: Query<AgentSettings, With<Agent>>,
    dirs: Res<ProfileDirs>,
    mut profiles: ResMut<Profiles>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((applied, tools, model, &effort, busy)) = agents.get(agent) else {
        return;
    };
    *profiles = Profiles::read(&dirs);
    for problem in &profiles.problems {
        notices.write(Notice::info(agent, format!("Profile left out: {problem}.")));
    }
    let name = args.args.as_str();
    if name.is_empty() {
        let shown = applied.map_or("none".to_owned(), |applied| applied.name.clone());
        let text = format!("Profiles: {}. Applied: {shown}.", profiles.names());
        notices.write(Notice::info(agent, text));
        return;
    }
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let before = applied.map_or_else(
        || Before {
            tools: tools.clone(),
            model: model.map(|model| model.0.clone()),
            effort,
        },
        |applied| applied.before.clone(),
    );
    let mut agent_commands = commands.entity(agent);
    if name == "none" {
        let Some(applied) = applied else {
            notices.write(Notice::info(agent, "No profile is applied."));
            return;
        };
        if let Some(model) = before
            .model
            .filter(|before| model.is_none_or(|now| now.0 != *before))
        {
            agent_commands.insert(ModelChoice(model));
        }
        agent_commands
            .insert((before.tools, before.effort))
            .remove::<AppliedProfile>();
        let text = format!("Profile `{}` taken off.", applied.name);
        notices.write(Notice::info(agent, text));
        return;
    }
    let Some(profile) = profiles.by_name.get(name) else {
        let known = profiles.names();
        notices.write(Notice::error(
            agent,
            format!("Unknown profile `{name}`. Known profiles: {known}."),
        ));
        return;
    };
    let (new_model, new_effort) = match (&profile.model, &profile.effort) {
        (None, None) => (before.model.clone().map(ModelChoice), before.effort),
        (asked_model, asked_effort) => {
            let parent = before.model.clone().map(ModelChoice);
            match ModelChoice::inherit(
                &models.0,
                (parent.as_ref(), before.effort),
                asked_model.as_deref(),
                asked_effort.as_deref(),
            ) {
                Ok(settled) => settled,
                Err(why) => {
                    let why = format!("Profile `{name}` not applied: {why}.");
                    notices.write(Notice::error(agent, why));
                    return;
                }
            }
        }
    };
    let access = match &profile.tools {
        Some(names) => ToolAccess::Only(
            names
                .iter()
                .filter(|name| before.tools.allows(name))
                .cloned()
                .collect(),
        ),
        None => before.tools.clone(),
    };
    // The model first: its connection is in place when the reasoning
    // setting is checked against it.
    if let Some(choice) = new_model.filter(|choice| model != Some(choice)) {
        agent_commands.insert(choice);
    }
    agent_commands.insert(access).insert(new_effort);
    agent_commands.insert(AppliedProfile {
        name: name.to_owned(),
        prompt: profile.prompt.clone(),
        before,
    });
    let text = format!("Profile `{name}` applied: {}.", profile.description);
    notices.write(Notice::info(agent, text));
}

#[cfg(test)]
mod tests;
