//! Skills loaded on demand (spec `28-skills.md`). A skill is a directory
//! `RIG_HOME/skills/<name>/` or `<project>/.rig/skills/<name>/` with a
//! `SKILL.md` whose frontmatter has a `name` and a `description`, and maybe
//! more files its body names. The system prompt lists only each skill's
//! name and description; the model loads a body with `load_skill`, and the
//! user with `/skill <name> [text]`, which sends it with their message.
//! `/skills` lists them and which were loaded this session.
//!
//! Both skill directories are watched, so skills appear and disappear as
//! they change; a `SKILL.md` whose frontmatter is broken is skipped and
//! reported once. A skill loaded in a turn is not loaded again in it.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_memory::summary_in;
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
        app.init_resource::<ProjectRoot>()
            .init_resource::<RigHome>()
            .init_resource::<Skills>()
            .add_message::<FileChanges>()
            .register_required_components::<Agent, LoadedSkills>()
            .register_required_components::<TurnOf, LoadedInTurn>()
            .add_open_tool(
                LOAD_SKILL,
                "Load a skill listed under \"Skills you can load\": returns its instructions, \
                 with the paths of the files it names. When a task fits a skill, load it before \
                 starting, and read the files it names with `read`.",
                ToolOptions {
                    footprint: Some(Touches::Reads(Scope::Anything)),
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
            .add_observer(delivered)
            .add_systems(Update, forget_compacted.after(AgentLoop))
            .add_systems(Startup, watch_roots)
            .add_systems(Update, refresh.before(AgentLoop));
        let section = PromptSection::new(PromptSection::ORDER_PROJECT + 10, "skills", "");
        app.world_mut()
            .spawn((Name::new("prompt:skills"), SkillsSection, section));
    }
}

/// A watched directory whose subdirectories are skills; a skill of a
/// higher `0` replaces one of the same name.
#[derive(Component)]
struct SkillRoot(u8);

/// The skills found, by name, and the `SKILL.md` files skipped, with why.
#[derive(Resource, Default)]
struct Skills {
    found: BTreeMap<String, Skill>,
    broken: Vec<(PathBuf, String)>,
    /// The broken files reported already.
    reported: HashSet<PathBuf>,
}

/// A skill found on disk.
#[derive(Debug)]
struct Skill {
    description: String,
    /// Its `SKILL.md`.
    file: PathBuf,
}

/// The skills an agent loaded this session, in order, kept with it.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved, Default)]
pub struct LoadedSkills(pub Vec<String>);

/// The skills loaded in a turn, on the turn.
#[derive(Component, Default, Debug)]
pub struct LoadedInTurn(pub Vec<String>);

/// On a message `/skill` sent: the skill it carries.
#[derive(Component)]
struct SkillMessage(String);

/// The prompt section listing the skills.
#[derive(Component)]
struct SkillsSection;

/// A `SKILL.md`'s frontmatter.
#[derive(Deserialize)]
struct Frontmatter {
    name: String,
    description: String,
}

/// Pushes `name` onto `list` unless it is there.
fn add(list: &mut Vec<String>, name: &str) {
    if !list.iter().any(|done| done == name) {
        list.push(name.to_owned());
    }
}

impl Skills {
    /// Reads the skill directories of `roots`, most general first.
    fn scan<'a>(&mut self, roots: impl IntoIterator<Item = &'a Path>) {
        self.found.clear();
        self.broken.clear();
        for root in roots {
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
                        let description = front.description;
                        self.found.insert(front.name, Skill { description, file });
                    }
                    Err(why) => self.broken.push((file, why)),
                }
            }
        }
    }

    /// The prompt section's text: each skill's name and description.
    fn listing(&self) -> String {
        if self.found.is_empty() {
            return String::new();
        }
        let mut lines = vec![format!("{HEADING} (with `{LOAD_SKILL}`):")];
        for (name, skill) in &self.found {
            lines.push(format!("- {name}: {}", skill.description));
        }
        lines.join("\n")
    }

    /// The skill `name`'s body for the model, its relative paths made
    /// absolute, or why it cannot be loaded.
    fn load(&self, name: &str) -> Result<String, String> {
        let Some(skill) = self.found.get(name) else {
            let names: Vec<&str> = self.found.keys().map(String::as_str).collect();
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
        let code = (index % 2 == 1).then(|| absolute(piece)).flatten();
        spans.push_str(code.as_deref().unwrap_or(piece));
    }
    let mut pieces = spans.split("](");
    let mut out = pieces.next().unwrap_or_default().to_owned();
    for piece in pieces {
        out.push_str("](");
        match piece.split_once(')') {
            Some((target, rest)) => {
                out.push_str(absolute(target).as_deref().unwrap_or(target));
                out.push(')');
                out.push_str(rest);
            }
            None => out.push_str(piece),
        }
    }
    out
}

/// Watches `RIG_HOME/skills` and the project's `.rig/skills`.
fn watch_roots(home: Res<RigHome>, project: Res<ProjectRoot>, mut commands: Commands) {
    let roots = [
        home.0.root().join("skills"),
        project.0.join(".rig").join("skills"),
    ];
    for (order, root) in (0..).zip(roots) {
        commands.spawn((SkillRoot(order), FileWatch::tree(root)));
    }
}

/// Reads the skills again when a root is watched or something under one
/// changed, keeps the prompt's list current, and reports each newly broken
/// `SKILL.md` once.
fn refresh(
    roots: Query<(&SkillRoot, Ref<FileWatch>)>,
    mut changes: MessageReader<FileChanges>,
    mut skills: ResMut<Skills>,
    mut sections: Query<&mut PromptSection, With<SkillsSection>>,
    mut notices: MessageWriter<Notice>,
) {
    let changed = changes.read().any(|change| roots.contains(change.watch));
    if !changed && !roots.iter().any(|(_, watch)| watch.is_added()) {
        return;
    }
    let mut ordered: Vec<_> = roots.iter().collect();
    ordered.sort_by_key(|(root, _)| root.0);
    skills.scan(ordered.iter().map(|(_, watch)| watch.path.as_path()));
    let listing = skills.listing();
    for mut section in &mut sections {
        if section.text != listing {
            section.text.clone_from(&listing);
        }
    }
    let Skills {
        broken, reported, ..
    } = &mut *skills;
    reported.retain(|file| broken.iter().any(|(gone, _)| gone == file));
    for (file, why) in broken.iter() {
        if reported.insert(file.clone()) {
            let text = format!("Skill {} skipped: {why}.", file.display());
            notices.write(Notice::error(None, text));
        }
    }
}

/// A `/skill` message delivered into a turn loads its skill there, and
/// for the session.
fn delivered(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&DeliveredIn, &SkillMessage, &ForAgent)>,
    mut turns: Query<&mut LoadedInTurn>,
    mut session: Query<&mut LoadedSkills>,
) {
    let Ok((&DeliveredIn(turn), SkillMessage(name), &ForAgent(agent))) =
        messages.get(delivered.entity)
    else {
        return;
    };
    if let Ok(mut turn) = turns.get_mut(turn) {
        add(&mut turn.0, name);
    }
    if let Ok(mut session) = session.get_mut(agent) {
        add(&mut session.0, name);
    }
}

/// A conversation compacted into a summary may have lost the bodies
/// loaded in the running turn, so they may be loaded again.
fn forget_compacted(
    mut committed: MessageReader<ConversationChanged>,
    agents: Query<&ActiveTurn>,
    mut loaded: Query<&mut LoadedInTurn>,
) {
    for ConversationChanged { agent, change } in committed.read() {
        if summary_in(&change.change).is_some()
            && let Ok(turn) = agents.get(*agent)
            && let Ok(mut loaded) = loaded.get_mut(turn.turn())
        {
            loaded.0.clear();
        }
    }
}

/// `load_skill`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadSkill {
    /// The skill's name, as listed.
    pub name: String,
}

/// Answers a `load_skill` call with the skill's body, or says its turn
/// has it already.
fn load_skill(
    called: On<ToolCalled<LoadSkill>>,
    skills: Res<Skills>,
    calls: Query<&CallOf>,
    mut turns: Query<&mut LoadedInTurn>,
    mut session: Query<&mut LoadedSkills>,
    mut decisions: Query<&mut Decision, ToolCalls>,
) {
    let name = called.args.name.trim();
    let mut turn = calls
        .get(called.call)
        .ok()
        .and_then(|&CallOf(turn)| turns.get_mut(turn).ok());
    let already = turn
        .as_ref()
        .is_some_and(|turn| turn.0.iter().any(|done| done == name));
    let result = if already {
        Ok(format!(
            "The skill `{name}` is already loaded above in this turn; follow it from there."
        ))
    } else {
        skills.load(name)
    };
    let output = match result {
        Ok(text) => {
            if let Some(turn) = turn.as_mut() {
                add(&mut turn.0, name);
            }
            if let Ok(mut session) = session.get_mut(called.agent) {
                add(&mut session.0, name);
            }
            ToolResult::success(text.into())
        }
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    if let Ok(mut decision) = decisions.get_mut(called.call) {
        decision.decide(Decider::plugin("skills"), output);
    }
}

/// `/skill <name> [text]`: sends the skill's body with the text, as the
/// user's message, titled with the skill, as the line was sent.
fn send_skill(
    In(args): In<CommandArgs>,
    skills: Res<Skills>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let line = args.args.as_str();
    let (name, text) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    if name.is_empty() {
        notices.write(Notice::error(args.agent, "/skill needs a skill's name."));
        return;
    }
    let body = match skills.load(name) {
        Ok(body) => body,
        Err(why) => {
            notices.write(Notice::error(args.agent, why));
            return;
        }
    };
    let text = format!("{body}\n\n{}", text.trim());
    let author = Author::user().titled(format!("skill {name}"));
    let message = Incoming::new(text.trim(), args.mode).with_author(author);
    let mark = SkillMessage(name.to_owned());
    commands.spawn((message, ForAgent(args.agent), mark));
}

/// `/skills`: the skills, which this agent loaded this session, and the
/// ones skipped.
fn list_skills(
    In(args): In<CommandArgs>,
    skills: Res<Skills>,
    roots: Query<&FileWatch, With<SkillRoot>>,
    loaded: Query<&LoadedSkills>,
    mut notices: MessageWriter<Notice>,
) {
    let session = loaded.get(args.agent).map(|loaded| loaded.0.as_slice());
    let session = session.unwrap_or_default();
    let mut lines = vec![format!("Skills ({} loaded this session):", session.len())];
    for (name, skill) in &skills.found {
        let mark = if session.contains(name) { "*" } else { " " };
        let file = skill.file.display();
        lines.push(format!("{mark} {name}: {} ({file})", skill.description));
    }
    if skills.found.is_empty() {
        let roots: Vec<String> = roots
            .iter()
            .map(|root| root.path.display().to_string())
            .collect();
        lines.push(format!("  none in {}", roots.join(", ")));
    }
    for (file, why) in &skills.broken {
        lines.push(format!("  skipped {}: {why}", file.display()));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

#[cfg(test)]
mod tests;
