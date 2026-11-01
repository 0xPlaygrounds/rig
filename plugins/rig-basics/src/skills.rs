//! Skills ([`SkillsPlugin`]): instructions for one kind of task, each a
//! directory with a `SKILL.md`, whose frontmatter gives its `name` and
//! `description`, and maybe more files the body names. The system prompt
//! lists each skill's name, description and path, and the model reads a
//! skill's file with its read tool when a task matches. `/skill:<name>
//! [text]` sends a skill with the user's message.
//!
//! Skills follow the Agent Skills format, so skills written for other
//! agents work unchanged. They are found once, when the agent starts, so the
//! prompt stays the same for the whole session and the provider's prompt
//! cache keeps it. A change takes effect on `/reload` or `/new`.
//!
//! ```markdown
//! ---
//! name: release-notes
//! description: Write release notes from the merged PRs. Use when asked for a changelog.
//! ---
//! Read `template.md`, then ...
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use bevy_ecs::message::Messages;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_ecs::agent::Notice;
use rig_ecs::inbox::Incoming;
use rig_ecs::prompt::PromptSection;
use rig_ecs::stage::ForAgent;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::rig_core::conversation::Author;
use rig_harness::services::{ProjectRoot, RigHome};
use serde::Deserialize;

/// The file that makes a directory a skill.
pub const SKILL_FILE: &str = "SKILL.md";

/// The longest description, in characters.
const MAX_DESCRIPTION: usize = 1024;

/// The longest name, in characters.
const MAX_NAME: usize = 64;

/// Lists the skills in the system prompt and adds a `/skill:<name>` command
/// for each.
#[derive(Default)]
pub struct SkillsPlugin;

impl Plugin for SkillsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>();
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let home = RigHome::of(world);
        let root = world.resource::<ProjectRoot>().0.clone();
        let (skills, problems) = discover(&roots(&root, home.root()));
        for problem in &problems {
            warn!("{problem}");
        }
        if world.contains_resource::<Messages<Notice>>() {
            for problem in problems {
                world.write_message(Notice::warn(None, problem));
            }
        }
        world.spawn((
            Name::new("prompt:skills"),
            PromptSection::new(
                PromptSection::ORDER_PROJECT + 10,
                "skills",
                listing(&skills),
            ),
        ));
        for skill in skills {
            let help = format!("Skill: {}", first_line(&skill.description));
            let name = format!("skill:{}", skill.name);
            app.add_command(
                &name,
                &help,
                move |In(args): In<CommandArgs>,
                      commands: Commands,
                      notices: MessageWriter<Notice>| {
                    send(&skill, args, commands, notices);
                },
            );
        }
    }
}

/// A skill found on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    /// Its name: the frontmatter's, else its directory's.
    pub name: String,
    /// When to use it.
    pub description: String,
    /// Its `SKILL.md`.
    pub file: PathBuf,
    /// Whether the system prompt leaves it out, so only `/skill:<name>`
    /// sends it (`disable-model-invocation: true`).
    pub manual: bool,
}

/// A `SKILL.md`'s frontmatter: the fields rig reads. Others, such as
/// `license` or `allowed-tools`, are allowed and ignored.
#[derive(Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    disable_model_invocation: bool,
}

/// The directories skills are found in, first first: `.rig/skills` and
/// `.agents/skills` of `dir` and of each directory above it up to the
/// repository root (the nearest holding `.git`; outside a repository, `dir`
/// alone), then `home/skills`, then `~/.agents/skills`.
pub fn roots(dir: &Path, home: &Path) -> Vec<PathBuf> {
    let repository = dir.ancestors().find(|place| place.join(".git").exists());
    let places: Vec<&Path> = match repository {
        Some(top) => dir
            .ancestors()
            .take_while(|place| *place != top)
            .chain([top])
            .collect(),
        None => vec![dir],
    };
    let mut roots: Vec<PathBuf> = places
        .into_iter()
        .flat_map(|place| [place.join(".rig/skills"), place.join(".agents/skills")])
        .collect();
    roots.push(home.join("skills"));
    if let Some(user) = std::env::home_dir() {
        roots.push(user.join(".agents/skills"));
    }
    roots
}

/// The skills under `roots`, by name, with what was wrong with the ones
/// skipped or doubtful. A directory holding a `SKILL.md` is a skill and is
/// not searched further; hidden directories and `node_modules` are skipped.
/// Of two skills with one name, the first root's wins.
pub fn discover(roots: &[PathBuf]) -> (Vec<Skill>, Vec<String>) {
    let mut skills: Vec<Skill> = Vec::new();
    let mut problems = Vec::new();
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut walk = Walk::default();
    for root in roots {
        walk.find(root);
        problems.append(&mut walk.problems);
        for file in std::mem::take(&mut walk.files) {
            let Ok(real) = file.canonicalize() else {
                continue;
            };
            if seen.contains(&real) {
                continue;
            }
            seen.push(real);
            match read(&file) {
                Ok((skill, mut doubts)) => {
                    problems.append(&mut doubts);
                    match skills.iter().find(|kept| kept.name == skill.name) {
                        Some(kept) => problems.push(format!(
                            "Skill {} skipped: {} has the name `{}` already.",
                            file.display(),
                            kept.file.display(),
                            skill.name
                        )),
                        None => skills.push(skill),
                    }
                }
                Err(why) => problems.push(format!("Skill {} skipped: {why}.", file.display())),
            }
        }
    }
    (skills, problems)
}

/// A search of skill directories that visits each directory once, by its
/// canonical path, however links lead back to it.
#[derive(Default)]
struct Walk {
    /// The `SKILL.md` files found, in name order.
    files: Vec<PathBuf>,
    /// Every directory searched.
    searched: HashSet<PathBuf>,
    /// The directories being searched, from the root down.
    path: Vec<PathBuf>,
    /// The directories a link led back to, each named once.
    looped: HashSet<PathBuf>,
    problems: Vec<String>,
}

impl Walk {
    /// Pushes the `SKILL.md` files under `dir` onto `files`, in name order.
    /// A directory searched already is skipped; one a link leads back to
    /// from below it is named once in `problems`.
    fn find(&mut self, dir: &Path) {
        let Ok(real) = dir.canonicalize() else {
            return;
        };
        if self.path.contains(&real) && self.looped.insert(real.clone()) {
            self.problems.push(format!(
                "Skills: {} leads back to {}, which is searched once.",
                dir.display(),
                real.display()
            ));
        }
        if !self.searched.insert(real.clone()) {
            return;
        }
        self.path.push(real);
        self.search(dir);
        self.path.pop();
    }

    fn search(&mut self, dir: &Path) {
        let file = dir.join(SKILL_FILE);
        if file.is_file() {
            self.files.push(file);
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut dirs: Vec<PathBuf> = entries
            .flatten()
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                !name.starts_with('.') && name != "node_modules"
            })
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        dirs.sort();
        for dir in dirs {
            self.find(&dir);
        }
    }
}

/// The skill `file` describes, with doubts about it, or why it is skipped.
fn read(file: &Path) -> Result<(Skill, Vec<String>), String> {
    let text = std::fs::read_to_string(file).map_err(|error| error.to_string())?;
    let (front, _) = split(&text)?;
    let front: Frontmatter = match front {
        Some(yaml) => serde_yaml::from_str::<Option<Frontmatter>>(yaml)
            .map_err(|error| format!("broken frontmatter: {error}"))?
            .unwrap_or_default(),
        None => Frontmatter::default(),
    };
    let description = front
        .description
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
        .ok_or("it has no description")?;
    let directory = file
        .parent()
        .and_then(Path::file_name)
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = front.name.map_or(directory, |name| name.trim().to_owned());
    let mut doubts = Vec::new();
    if description.chars().count() > MAX_DESCRIPTION {
        doubts.push(format!(
            "Skill {}: its description is longer than {MAX_DESCRIPTION} characters.",
            file.display()
        ));
    }
    if !valid_name(&name) {
        doubts.push(format!(
            "Skill {}: the name `{name}` should be lowercase letters, digits and single \
             hyphens, at most {MAX_NAME} characters.",
            file.display()
        ));
    }
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err(format!("the name `{name}` is empty or has spaces"));
    }
    let skill = Skill {
        name,
        description,
        file: file.to_owned(),
        manual: front.disable_model_invocation,
    };
    Ok((skill, doubts))
}

/// `text`'s YAML frontmatter, if it starts with a `---` line, and its body.
fn split(text: &str) -> Result<(Option<&str>, &str), String> {
    let Some(rest) = text
        .strip_prefix("---")
        .and_then(|rest| rest.split_once('\n'))
        .filter(|(after, _)| after.trim().is_empty())
        .map(|(_, rest)| rest)
    else {
        return Ok((None, text));
    };
    let end = rest
        .match_indices("---")
        .find(|(at, _)| *at == 0 || rest.get(..*at).is_some_and(|before| before.ends_with('\n')))
        .map(|(at, _)| at)
        .ok_or("the frontmatter has no closing `---` line")?;
    let yaml = rest.get(..end).unwrap_or_default();
    let body = rest
        .get(end..)
        .and_then(|after| after.split_once('\n'))
        .map_or("", |(_, body)| body);
    Ok((Some(yaml), body))
}

/// Whether `name` follows the Agent Skills format: lowercase letters,
/// digits and single hyphens inside, at most [`MAX_NAME`] characters.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--")
}

/// The prompt section's text: how to use skills and the ones the model may
/// pick. Empty, which leaves the section out, when there is none.
fn listing(skills: &[Skill]) -> String {
    let entries: Vec<String> = skills
        .iter()
        .filter(|skill| !skill.manual)
        .map(|skill| {
            format!(
                "  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    \
                 <location>{}</location>\n  </skill>",
                escape(&skill.name),
                escape(&skill.description),
                escape(&skill.file.display().to_string()),
            )
        })
        .collect();
    if entries.is_empty() {
        return String::new();
    }
    format!(
        "Skills give instructions for specific tasks. When a task matches a skill's \
         description, read the skill's file with the read tool before you start, and follow \
         it. A relative path in a skill is relative to the skill's directory, the directory \
         of its file.\n\n<available_skills>\n{}\n</available_skills>",
        entries.join("\n")
    )
}

/// `text` with XML's special characters escaped.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The first line of `text`, cut to a help line's length.
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    match line.char_indices().nth(80) {
        Some((at, _)) => format!("{}…", line.get(..at).unwrap_or(line)),
        None => line.to_owned(),
    }
}

/// The message `/skill:<name> [text]` sends: the skill's body, without its
/// frontmatter, with where it is, then the text.
pub fn message(skill: &Skill, text: &str) -> Result<String, String> {
    let file = std::fs::read_to_string(&skill.file).map_err(|error| error.to_string())?;
    let (_, body) = split(&file)?;
    let dir = skill.file.parent().unwrap_or(Path::new("."));
    let mut message = format!(
        "<skill name=\"{}\" location=\"{}\">\nRelative paths in this skill are relative to {}.\n\n{}\n</skill>",
        escape(&skill.name),
        escape(&skill.file.display().to_string()),
        dir.display(),
        body.trim()
    );
    if !text.trim().is_empty() {
        message.push_str("\n\n");
        message.push_str(text.trim());
    }
    Ok(message)
}

/// `/skill:<name> [text]`: sends the skill with the text, as the user's
/// message, as the line was sent.
fn send(
    skill: &Skill,
    args: CommandArgs,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    match message(skill, &args.args) {
        Ok(text) => {
            let author = Author::user().titled(format!("skill {}", skill.name));
            let message = Incoming::new(text, args.mode).with_author(author);
            commands.spawn((message, ForAgent(args.agent)));
        }
        Err(why) => {
            let why = format!("Cannot read {}: {why}.", skill.file.display());
            notices.write(Notice::error(args.agent, why));
        }
    }
}

#[cfg(test)]
mod tests;
