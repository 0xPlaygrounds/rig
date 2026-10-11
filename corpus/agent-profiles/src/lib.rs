//! Agent profiles: named settings for an agent, defined in Markdown files
//! in `RIG_HOME/agents/*.md` and `<project>/.rig/agents/*.md` (a project's
//! profile replaces a home profile of the same name), watched for changes.
//! A file's frontmatter says the profile's `description`, `model`, `effort`
//! and `tools` (an allowlist); its body is extra system prompt for that
//! agent alone.
//!
//! `/profile <name>` applies one to the agent shown: its prompt as a
//! [`PromptSection`] of that agent's, its tools as the agent's
//! [`ToolAccess`] (never more than the agent had), its model and its
//! reasoning setting. `/profile none` takes it off and restores the
//! settings from before, and `/profile` alone lists the profiles. The
//! applied profile is an [`AppliedProfile`], saved with the agent, so a
//! restart or `/resume` brings it back.
//!
//! rig-subagents' `task` tool gets a `profile` argument, and the profiles
//! are listed in its rules, which are in the system prompt of every agent
//! offered `task`. A Prepare system takes the argument off the call before
//! `task` reads it, a Prepare system on the [`SpawnSubagent`] it asks for
//! gives it the profile's model, reasoning setting and tools, and a Review
//! system gives the agent spawned the profile's prompt.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rig_harness::prelude::*;
use rig_subagents::{SpawnSubagent, SpawnedAgent, TASK};
use serde::Deserialize;

/// The `task` argument naming a profile.
const PROFILE: &str = "profile";

/// The start of the `task` rule that lists the profiles.
const LISTED: &str = "Agent profiles, for `task`'s `profile`:";

/// Reads and watches the profiles, adds `/profile`, keeps each applied
/// profile's prompt section, and lets `task` start a subagent as a profile.
#[derive(Default)]
pub struct AgentProfilesPlugin;

impl Plugin for AgentProfilesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Profiles>()
            .add_message::<FileChanges>()
            .add_observer(show_prompt)
            .add_observer(hide_prompt)
            .add_systems(Startup, watch_dirs)
            .add_systems(
                Update,
                (
                    read_profiles,
                    offer_to_task.run_if(resource_changed::<Profiles>),
                )
                    .chain()
                    .before(AgentLoop),
            )
            .add_systems(
                AgentPass,
                (
                    (take_task_profile, profile_spawn).in_set(Stage::Prepare),
                    refuse_unknown.in_set(Stage::Gate),
                    prompt_spawned.in_set(Stage::Review),
                ),
            )
            .add_command(
                "profile",
                "Apply an agent profile: /profile <name>, /profile none, or /profile to list them",
                apply_profile,
            );
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

    /// `tools` narrowed to this profile's allowlist: never more than it was.
    fn narrow(&self, tools: &ToolAccess) -> ToolAccess {
        match &self.tools {
            Some(names) => {
                let allowed = names.iter().filter(|name| tools.allows(name));
                ToolAccess::Only(allowed.cloned().collect())
            }
            None => tools.clone(),
        }
    }
}

/// The profiles by name.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct Profiles(pub BTreeMap<String, Profile>);

impl Profiles {
    /// The profile `name`, or a refusal naming the known ones.
    fn get(&self, name: &str) -> Result<&Profile, String> {
        self.0.get(name).ok_or_else(|| {
            let names: Vec<&str> = self.0.keys().map(String::as_str).collect();
            let known = if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(", ")
            };
            format!("Unknown profile `{name}`. Known profiles: {known}.")
        })
    }
}

/// On a [`FileWatch`] of a directory profiles are read from; later ones
/// override earlier ones by name.
#[derive(Component)]
struct ProfileDir(usize);

/// Watches `RIG_HOME/agents`, then the project's `.rig/agents`.
fn watch_dirs(home: Res<RigHome>, root: Res<ProjectRoot>, mut commands: Commands) {
    let dirs = [
        home.0.root().join("agents"),
        root.0.join(".rig").join("agents"),
    ];
    for (rank, dir) in dirs.into_iter().enumerate() {
        commands.spawn((
            Name::new("agent profiles"),
            ProfileDir(rank),
            FileWatch::new(dir),
        ));
    }
}

/// Reads every `*.md` file of the watched directories when they are
/// watched and whenever their files change; a missing directory has none.
fn read_profiles(
    dirs: Query<(Entity, &ProfileDir, Ref<FileWatch>)>,
    mut changes: MessageReader<FileChanges>,
    mut profiles: ResMut<Profiles>,
    mut notices: MessageWriter<Notice>,
) {
    let changed = changes.read().any(|change| dirs.contains(change.watch));
    if !changed && !dirs.iter().any(|(.., watch)| watch.is_added()) {
        return;
    }
    let mut ranked: Vec<(usize, PathBuf)> = dirs
        .iter()
        .map(|(_, rank, watch)| (rank.0, watch.path.clone()))
        .collect();
    ranked.sort();
    let mut read = BTreeMap::new();
    for (_, dir) in ranked {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
        paths.sort();
        for path in paths
            .iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        {
            let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let text = std::fs::read_to_string(path).map_err(|why| why.to_string());
            match text.and_then(|text| Profile::parse(&text)) {
                Ok(profile) => {
                    read.insert(name.to_owned(), profile);
                }
                Err(why) => {
                    let why = format!("Profile left out: {}: {why}.", path.display());
                    notices.write(Notice::info(None, why));
                }
            }
        }
    }
    profiles.set_if_neq(Profiles(read));
}

/// Declares `task`'s `profile` argument and lists the profiles in its
/// rules, so every agent offered `task` sees them in its system prompt.
fn offer_to_task(profiles: Res<Profiles>, mut tools: Query<(&mut ToolDef, &mut ToolRules)>) {
    let Some((mut def, mut rules)) = tools
        .iter_mut()
        .find(|(def, _)| def.0.name.as_str() == TASK)
    else {
        return;
    };
    rules.0.retain(|rule| !rule.starts_with(LISTED));
    let properties = def
        .0
        .parameters
        .get_mut("properties")
        .and_then(serde_json::Value::as_object_mut);
    let Some(properties) = properties else {
        return;
    };
    if profiles.0.is_empty() {
        properties.shift_remove(PROFILE);
        return;
    }
    let listed: Vec<String> = profiles
        .0
        .iter()
        .map(|(name, profile)| format!("- {name}: {}", profile.description))
        .collect();
    rules.0.push(format!("{LISTED}\n{}", listed.join("\n")));
    let argument = serde_json::json!({
        "type": "string",
        "description": "An agent profile the subagent starts as, by name: its prompt, tools, \
            model and reasoning setting. Your settings when absent."
    });
    properties.insert(PROFILE.to_owned(), argument);
}

/// On a `task` call that named a profile, which its arguments no longer
/// carry.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct TaskProfile(String);

/// Takes the `profile` argument off each `task` call as it is admitted,
/// before `task` reads its arguments, and keeps it on the call.
fn take_task_profile(
    mut calls: Query<(Entity, &mut ToolRequest), Added<Admitted>>,
    mut commands: Commands,
) {
    for (call, mut request) in &mut calls {
        let function = &mut request.call.function;
        if function.name.as_str() != TASK {
            continue;
        }
        if let Some(profile) = function.arguments.shift_remove(PROFILE) {
            let name = profile
                .as_str()
                .map_or_else(|| profile.to_string(), str::to_owned);
            commands.entity(call).insert(TaskProfile(name));
        }
    }
}

/// The profile of the `task` call a spawn request is for, if it named one.
fn named_profile<'a>(of: Option<&ChildOf>, calls: &'a Query<&TaskProfile>) -> Option<&'a str> {
    let profile = calls.get(of?.parent()).ok()?;
    Some(profile.0.as_str())
}

/// Gives a subagent asked for with a profile the profile's model and
/// reasoning setting where the call named none, and its tools narrowed to
/// the profile's.
fn profile_spawn(
    mut requests: Query<(&mut SpawnSubagent, &ForAgent, Option<&ChildOf>), Added<Admitted>>,
    calls: Query<&TaskProfile>,
    parents: Query<&ToolAccess>,
    profiles: Res<Profiles>,
) {
    for (mut request, &ForAgent(parent), of) in &mut requests {
        let Some(Ok(profile)) = named_profile(of, &calls).map(|name| profiles.get(name)) else {
            continue;
        };
        let request = &mut *request;
        request.model = request.model.take().or_else(|| profile.model.clone());
        request.effort = request.effort.take().or_else(|| profile.effort.clone());
        let asked = match (&request.tools, parents.get(parent)) {
            (Some(names), _) => ToolAccess::Only(names.clone()),
            (None, Ok(access)) => access.clone(),
            (None, Err(_)) => ToolAccess::All,
        };
        if let ToolAccess::Only(names) = profile.narrow(&asked) {
            request.tools = Some(names);
        }
    }
}

/// Refuses a subagent asked for with a profile nobody defined.
fn refuse_unknown(
    requests: Query<(Entity, Option<&ChildOf>), (With<SpawnSubagent>, Undecided)>,
    calls: Query<&TaskProfile>,
    profiles: Res<Profiles>,
    mut commands: Commands,
) {
    for (request, of) in &requests {
        if let Some(Err(why)) = named_profile(of, &calls).map(|name| profiles.get(name)) {
            commands
                .entity(request)
                .insert_if_new(Outcome::refused(why));
        }
    }
}

/// Gives a subagent spawned with a profile the profile's prompt, saved with
/// it as its [`AppliedProfile`].
fn prompt_spawned(
    spawned: Query<(&SpawnedAgent, Option<&ChildOf>), (With<SpawnSubagent>, Unreviewed)>,
    calls: Query<&TaskProfile>,
    agents: Query<Settings, Without<AppliedProfile>>,
    profiles: Res<Profiles>,
    mut commands: Commands,
) {
    for (&SpawnedAgent(agent), of) in &spawned {
        let Some(name) = named_profile(of, &calls) else {
            continue;
        };
        let (Ok(profile), Ok((tools, model, &effort))) = (profiles.get(name), agents.get(agent))
        else {
            continue;
        };
        commands.entity(agent).insert(AppliedProfile {
            name: name.to_owned(),
            prompt: profile.prompt.clone(),
            before: Before {
                tools: tools.clone(),
                model: model.map(|model| model.0.clone()),
                effort,
            },
        });
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
/// the settings it replaced (for a subagent started as the profile, its
/// settings as it started). Saved with the session.
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

/// The profile prompt sections, by agent.
type ProfilePrompts<'w, 's> =
    Query<'w, 's, (Entity, &'static SectionOf, &'static mut PromptSection), With<ProfilePrompt>>;

/// Gives an agent the prompt section of the profile inserted on it, as
/// `/profile`, `task` or a restore inserts it.
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

/// What a profile changes of an agent.
type Settings<'a> = (&'a ToolAccess, Option<&'a ModelChoice>, &'a Effort);

/// Gives `agent` the model `choice` and the reasoning setting `effort`
/// together, connected here, so the setting is checked against the new
/// model; `now` is its model.
fn switch_model(
    agent: &mut EntityCommands,
    models: &Models,
    (now, choice): (Option<&ModelChoice>, Option<ModelChoice>),
    effort: Effort,
) -> Result<(), String> {
    match choice.filter(|choice| now != Some(choice)) {
        Some(choice) => {
            let connection = models.connect(&choice.0)?;
            agent.insert((connection, choice, effort));
        }
        None => {
            agent.insert(effort);
        }
    }
    Ok(())
}

/// `/profile <name>`, `/profile none`, or `/profile` to list them.
fn apply_profile(
    In(args): In<CommandArgs>,
    agents: Query<(Settings, Option<&AppliedProfile>, Has<ActiveTurn>), With<Agent>>,
    profiles: Res<Profiles>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(((tools, model, &effort), applied, busy)) = agents.get(agent) else {
        return;
    };
    let name = args.args.as_str();
    if name.is_empty() {
        let shown = applied.map_or("none", |applied| applied.name.as_str());
        let names: Vec<&str> = profiles.0.keys().map(String::as_str).collect();
        let text = format!("Profiles: {}. Applied: {shown}.", names.join(", "));
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
    let mut entity = commands.entity(agent);
    if name == "none" {
        let Some(applied) = applied else {
            notices.write(Notice::info(agent, "No profile is applied."));
            return;
        };
        let restored = (model, before.model.map(ModelChoice));
        if let Err(why) = switch_model(&mut entity, &models, restored, before.effort) {
            notices.write(Notice::error(agent, why));
            return;
        }
        entity.insert(before.tools).remove::<AppliedProfile>();
        let text = format!("Profile `{}` taken off.", applied.name);
        notices.write(Notice::info(agent, text));
        return;
    }
    let settled = profiles.get(name).and_then(|profile| {
        let parent = before.model.clone().map(ModelChoice);
        let (choice, new_effort) = ModelChoice::inherit(
            &models.0,
            (parent.as_ref(), before.effort),
            profile.model.as_deref(),
            profile.effort.as_deref(),
        )?;
        switch_model(&mut entity, &models, (model, choice), new_effort)?;
        Ok(profile)
    });
    let profile = match settled {
        Ok(profile) => profile,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    entity.insert((
        profile.narrow(&before.tools),
        AppliedProfile {
            name: name.to_owned(),
            prompt: profile.prompt.clone(),
            before,
        },
    ));
    let text = format!("Profile `{name}` applied: {}.", profile.description);
    notices.write(Notice::info(agent, text));
}

#[cfg(test)]
mod tests;
