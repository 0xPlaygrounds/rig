//! Skills loaded on demand (spec `28-skills.md`). A skill is a directory
//! `RIG_HOME/skills/<name>/` or `<project>/.rig/skills/<name>/` with a
//! `SKILL.md` whose frontmatter has a `name` and a `description`, and maybe
//! more files its body names. The system prompt lists only each skill's
//! name and description; the model loads a body with `load_skill`, and the
//! user with `/skill <name> [text]`, which sends it with their message.
//! `/skills` lists them and which were loaded this session.
//!
//! The skill directories are read again when a turn starts (as the project
//! instructions are), by `/skills`, and by each load, so skills appear and
//! disappear as their directories change; a `SKILL.md` whose frontmatter
//! is broken is skipped and reported once.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool that loads a skill.
pub const LOAD_SKILL: &str = "load_skill";

/// The heading of the skill list in the system prompt.
pub const HEADING: &str = "Skills you can load";

/// Adds the skill list to the system prompt, `load_skill`, `/skill` and
/// `/skills`.
#[derive(Default)]
pub struct SkillsPlugin;

impl Plugin for SkillsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkillRoots>()
            .init_resource::<Reported>()
            .register_required_components::<Agent, LoadedSkills>()
            .add_open_tool(
                LOAD_SKILL,
                "Load a skill listed under \"Skills you can load\": returns its instructions, \
                 with the paths of the files it names.",
                ToolOptions {
                    rules: &[
                        "When a task fits a skill under \"Skills you can load\", call \
                         `load_skill` with its name before starting, and read the files it \
                         names with `read`.",
                    ],
                    footprint: Footprint::ReadOnly,
                },
                load_skill,
            )
            .add_command(
                "skill",
                "Send a skill with your message: /skill <name> [text]",
                send_skill,
            )
            .add_command(
                "skills",
                "List the skills and which were loaded this session",
                list_skills,
            )
            .add_observer(refresh_on_turn)
            .add_observer(forget_turn)
            .add_observer(forget_condensed);
        let section = PromptSection::new(PromptSection::ORDER_PROJECT + 10, "skills", "");
        app.world_mut()
            .spawn((Name::new("prompt:skills"), SkillsSection, section));
    }
}

/// The directories whose subdirectories are skills, most general first: a
/// skill of a later one replaces a skill of the same name.
#[derive(Resource, Clone, Debug)]
pub struct SkillRoots(pub Vec<PathBuf>);

impl Default for SkillRoots {
    /// `RIG_HOME/skills`, then the working directory's `.rig/skills`.
    fn default() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let home = Home::from_env().root().join("skills");
        Self(vec![home, cwd.join(".rig").join("skills")])
    }
}

/// The skills an agent loaded, kept with the session.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved, Default)]
pub struct LoadedSkills {
    /// Every skill loaded this session, in order.
    pub session: Vec<String>,
    /// The skills loaded since the agent's last turn ended.
    pub turn: Vec<String>,
}

impl LoadedSkills {
    /// Marks `name` loaded.
    fn add(&mut self, name: &str) {
        for list in [&mut self.session, &mut self.turn] {
            if !list.iter().any(|loaded| loaded == name) {
                list.push(name.to_owned());
            }
        }
    }
}

/// The prompt section listing the skills.
#[derive(Component)]
struct SkillsSection;

/// The broken `SKILL.md` files already reported.
#[derive(Resource, Default)]
struct Reported(HashSet<PathBuf>);

/// A skill found on disk.
#[derive(Debug)]
struct Skill {
    description: String,
    /// Its `SKILL.md`.
    file: PathBuf,
}

/// The skills of every root, by name, and the `SKILL.md` files skipped,
/// with why.
#[derive(Default)]
struct Found {
    skills: BTreeMap<String, Skill>,
    broken: Vec<(PathBuf, String)>,
}

/// A `SKILL.md`'s frontmatter.
#[derive(Deserialize)]
struct Frontmatter {
    name: String,
    description: String,
}

impl Found {
    /// Reads the skill directories of `roots` now.
    fn scan(roots: &SkillRoots) -> Self {
        let mut found = Self::default();
        for root in &roots.0 {
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            let mut files: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path().join("SKILL.md"))
                .filter(|file| file.is_file())
                .collect();
            files.sort();
            for file in files {
                match read(&file) {
                    Ok((front, _)) => {
                        let skill = Skill {
                            description: front.description,
                            file,
                        };
                        found.skills.insert(front.name, skill);
                    }
                    Err(why) => found.broken.push((file, why)),
                }
            }
        }
        found
    }

    /// The prompt section's text: each skill's name and description.
    fn listing(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut lines = vec![format!("{HEADING} (with `{LOAD_SKILL}`):")];
        for (name, skill) in &self.skills {
            lines.push(format!("- {name}: {}", skill.description));
        }
        lines.join("\n")
    }

    /// The skill `name`'s body for the model, its relative paths made
    /// absolute, or why it cannot be loaded.
    fn load(&self, name: &str) -> Result<String, String> {
        let Some(skill) = self.skills.get(name) else {
            let names: Vec<&str> = self.skills.keys().map(String::as_str).collect();
            return Err(format!(
                "No skill `{name}`. The skills: {}.",
                names.join(", ")
            ));
        };
        let (_, body) = read(&skill.file)?;
        let dir = skill.file.parent().unwrap_or(Path::new("."));
        Ok(format!(
            "<skill name=\"{name}\" dir=\"{}\">\n{}\n</skill>",
            dir.display(),
            resolve(body.trim(), dir)
        ))
    }
}

/// `file`'s frontmatter and body, or why it is broken.
fn read(file: &Path) -> Result<(Frontmatter, String), String> {
    let text = std::fs::read_to_string(file).map_err(|error| error.to_string())?;
    let rest = text
        .strip_prefix("---")
        .and_then(|rest| rest.split_once('\n'))
        .filter(|(after, _)| after.trim().is_empty())
        .map(|(_, rest)| rest)
        .ok_or("no frontmatter: the file must start with a `---` line")?;
    let (yaml, after) = format!("\n{rest}")
        .split_once("\n---")
        .map(|(yaml, after)| (yaml.to_owned(), after.to_owned()))
        .ok_or("the frontmatter has no closing `---` line")?;
    let body = after.split_once('\n').map_or("", |(_, body)| body);
    let front: Frontmatter =
        serde_yaml::from_str(&yaml).map_err(|error| format!("broken frontmatter: {error}"))?;
    let name = front.name.trim();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err(format!("the name `{name}` is empty or has spaces"));
    }
    Ok((front, body.to_owned()))
}

/// `body` with each relative path of a file in `dir`, as a Markdown link's
/// target or a code span, made absolute.
fn resolve(body: &str, dir: &Path) -> String {
    let absolute = |text: &str| -> Option<String> {
        let relative = text.trim().trim_start_matches("./");
        let plain = !relative.is_empty()
            && !relative.contains("://")
            && !relative.starts_with(['/', '#', '~']);
        let path = dir.join(relative);
        (plain && path.exists()).then(|| path.display().to_string())
    };
    let mut spans = String::new();
    for (index, piece) in body.split('`').enumerate() {
        if index > 0 {
            spans.push('`');
        }
        let code = index % 2 == 1;
        spans.push_str(
            &code
                .then(|| absolute(piece))
                .flatten()
                .unwrap_or_else(|| piece.to_owned()),
        );
    }
    let mut pieces = spans.split("](");
    let mut out = pieces.next().unwrap_or_default().to_owned();
    for piece in pieces {
        out.push_str("](");
        match piece.split_once(')') {
            Some((target, rest)) => {
                out.push_str(&absolute(target).unwrap_or_else(|| target.to_owned()));
                out.push(')');
                out.push_str(rest);
            }
            None => out.push_str(piece),
        }
    }
    out
}

/// Reads the skills again, puts their list in the prompt section and
/// reports each newly broken `SKILL.md` once, about `agent`.
fn refresh(
    roots: &SkillRoots,
    sections: &mut Query<&mut PromptSection, With<SkillsSection>>,
    reported: &mut Reported,
    notices: &mut MessageWriter<Notice>,
    agent: Option<Entity>,
) -> Found {
    let found = Found::scan(roots);
    for mut section in sections.iter_mut() {
        let listing = found.listing();
        if section.text != listing {
            section.text = listing;
        }
    }
    reported
        .0
        .retain(|file| found.broken.iter().any(|(broken, _)| broken == file));
    for (file, why) in &found.broken {
        if reported.0.insert(file.clone()) {
            let text = format!("Skill {} skipped: {why}.", file.display());
            notices.write(Notice::info(agent, text));
        }
    }
    found
}

/// Reads the skills again when a turn starts, before its first request.
fn refresh_on_turn(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    roots: Res<SkillRoots>,
    mut sections: Query<&mut PromptSection, With<SkillsSection>>,
    mut reported: ResMut<Reported>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = turns.get(started.entity).ok().map(|of| of.0);
    refresh(&roots, &mut sections, &mut reported, &mut notices, agent);
}

/// A turn's end ends what was loaded in it.
fn forget_turn(ended: On<TurnEnded>, mut loaded: Query<&mut LoadedSkills>) {
    if let Ok(mut loaded) = loaded.get_mut(ended.entity) {
        loaded.turn.clear();
    }
}

/// A conversation condensed, such as by compaction, may have lost the
/// bodies loaded in this turn, so they may be loaded again.
fn forget_condensed(condensed: On<Insert<Condensed>>, mut loaded: Query<&mut LoadedSkills>) {
    if let Ok(mut loaded) = loaded.get_mut(condensed.entity) {
        loaded.turn.clear();
    }
}

/// `load_skill`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadSkill {
    /// The skill's name, as listed.
    pub name: String,
}

/// Answers a `load_skill` call with the skill's body, or says it is in the
/// turn already.
fn load_skill(
    called: On<ToolCalled<LoadSkill>>,
    roots: Res<SkillRoots>,
    mut loaded: Query<&mut LoadedSkills>,
    mut commands: Commands,
) {
    let name = called.args.name.trim();
    let mut loaded = loaded.get_mut(called.agent).ok();
    let already = loaded
        .as_ref()
        .is_some_and(|loaded| loaded.turn.iter().any(|done| done == name));
    let result = if already {
        Ok(format!(
            "The skill `{name}` is already loaded above in this turn; follow it from there."
        ))
    } else {
        Found::scan(&roots).load(name)
    };
    let output = match result {
        Ok(text) => {
            if let Some(loaded) = loaded.as_mut() {
                loaded.add(name);
            }
            ToolResult::success(text.into())
        }
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    commands
        .entity(called.call)
        .insert_if_new(ToolOutput(output));
}

/// `/skill <name> [text]`: sends the skill's body with the text, as the
/// user's message.
fn send_skill(
    In(args): In<CommandArgs>,
    roots: Res<SkillRoots>,
    mut loaded: Query<&mut LoadedSkills>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let line = args.args.as_str();
    let (name, text) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    if name.is_empty() {
        notices.write(Notice::error(args.agent, "/skill needs a skill's name."));
        return;
    }
    let body = match Found::scan(&roots).load(name) {
        Ok(body) => body,
        Err(why) => {
            notices.write(Notice::error(args.agent, why));
            return;
        }
    };
    if let Ok(mut loaded) = loaded.get_mut(args.agent) {
        loaded.add(name);
    }
    let message = format!("{body}\n\n{}", text.trim());
    let message = Deliver::new(args.agent, message.trim(), DeliveryMode::Steer);
    commands.trigger(message.literal());
}

/// `/skills`: the skills, which this agent loaded this session, and the
/// ones skipped.
fn list_skills(
    In(args): In<CommandArgs>,
    roots: Res<SkillRoots>,
    mut sections: Query<&mut PromptSection, With<SkillsSection>>,
    mut reported: ResMut<Reported>,
    loaded: Query<&LoadedSkills>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = Some(args.agent);
    let found = refresh(&roots, &mut sections, &mut reported, &mut notices, agent);
    let session = loaded
        .get(args.agent)
        .map(|loaded| loaded.session.as_slice());
    let session = session.unwrap_or_default();
    let mut lines = vec![format!("Skills ({} loaded this session):", session.len())];
    for (name, skill) in &found.skills {
        let mark = if session.contains(name) { "*" } else { " " };
        let file = skill.file.display();
        lines.push(format!("{mark} {name}: {} ({file})", skill.description));
    }
    if found.skills.is_empty() {
        let roots: Vec<String> = roots
            .0
            .iter()
            .map(|root| root.display().to_string())
            .collect();
        lines.push(format!("  none in {}", roots.join(", ")));
    }
    for (file, why) in &found.broken {
        lines.push(format!("  skipped {}: {why}", file.display()));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

#[cfg(test)]
mod tests;
