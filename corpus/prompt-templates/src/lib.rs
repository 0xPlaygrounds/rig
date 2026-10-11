//! Prompt templates: every `*.md` in `RIG_HOME/prompts/` and in the
//! project's `.rig/prompts/` is a slash command named after the file. Its
//! text, with `$1`, `$2`, `$@` and `${name}` replaced by what was typed
//! after the command, is sent as the user's message.
//!
//! A template may start with a YAML frontmatter:
//!
//! ```text
//! ---
//! description: Review a file      # the line `/help` shows
//! model: anthropic/claude-sonnet  # the turn it starts runs on this model
//! args: [file, focus]             # `${file}` is `$1`, `${focus}` is `$2`
//! ---
//! Review ${file} for bugs, mostly ${focus}.
//! ```
//!
//! Every named argument, and every `$N` the text uses, must be typed;
//! otherwise the command is refused with its usage line. The project's
//! template wins a name clash with the user's, and a template never
//! replaces another plugin's command: it is skipped with a warning. The
//! directories are read again before every slash command the user types,
//! so a template added, edited or removed counts without a restart.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use bevy_log::warn;
use rig_harness::prelude::*;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// Reads the template directories and makes each template a command.
#[derive(Default)]
pub struct PromptTemplatesPlugin;

impl Plugin for PromptTemplatesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TemplateDirs>()
            .init_resource::<Templates>()
            .add_systems(Startup, sync)
            .add_observer(sync_before_command)
            .add_observer(route_turn);
    }
}

/// The directories templates are read from, the most general first: a
/// later directory's template wins a name clash. By default
/// `RIG_HOME/prompts` and `.rig/prompts` of the working directory.
#[derive(Resource, Clone, Debug)]
pub struct TemplateDirs(pub Vec<PathBuf>);

impl Default for TemplateDirs {
    fn default() -> Self {
        let project = std::env::current_dir().unwrap_or_default();
        Self(vec![
            Home::from_env().root().join("prompts"),
            project.join(".rig").join("prompts"),
        ])
    }
}

/// The templates read last, by command name, and the warnings already
/// logged, so a skipped template is reported once.
#[derive(Resource, Default)]
struct Templates {
    by_name: BTreeMap<String, Template>,
    warned: BTreeSet<String>,
}

/// On the entity of a template's command.
#[derive(Component)]
struct TemplateCommand;

/// On an agent: the model of a template its next turn runs on.
#[derive(Component)]
struct NextTurnModel(Connection);

/// One template file.
#[derive(Clone, Debug)]
struct Template {
    path: PathBuf,
    /// `/help`'s line.
    help: String,
    model: Option<String>,
    /// The names of `$1`, `$2`, ….
    args: Vec<String>,
    body: String,
}

#[derive(Deserialize, Default)]
struct Frontmatter {
    description: Option<String>,
    model: Option<String>,
    #[serde(default)]
    args: Option<ArgNames>,
}

/// `args: [file, focus]`, or `args: file focus` (or `file, focus`).
#[derive(Deserialize)]
#[serde(untagged)]
enum ArgNames {
    List(Vec<String>),
    Line(String),
}

impl ArgNames {
    fn names(self) -> Vec<String> {
        match self {
            Self::List(names) => names,
            Self::Line(line) => line
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .collect(),
        }
    }
}

impl Template {
    /// The template in `text`, read from `path`.
    fn parse(path: &Path, text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let mut front = Vec::new();
        let body = if text.lines().next().map(str::trim_end) == Some("---") {
            lines.next();
            loop {
                match lines.next() {
                    Some(line) if line.trim_end() == "---" => break,
                    Some(line) => front.push(line),
                    None => return Err("its frontmatter has no closing `---`".to_owned()),
                }
            }
            lines.collect::<Vec<_>>().join("\n")
        } else {
            text.to_owned()
        };
        let front = front.join("\n");
        let front: Frontmatter = if front.trim().is_empty() {
            Frontmatter::default()
        } else {
            serde_yaml::from_str(&front).map_err(|why| format!("its frontmatter: {why}"))?
        };
        Ok(Self {
            path: path.to_owned(),
            help: front
                .description
                .unwrap_or_else(|| format!("Prompt template {}", path.display())),
            model: front.model,
            args: front.args.map(ArgNames::names).unwrap_or_default(),
            body: body.trim().to_owned(),
        })
    }

    /// The text with `typed` in place of its arguments, or the number of
    /// arguments it needs when fewer were typed.
    fn expand(&self, typed: &[String]) -> Result<String, usize> {
        let mut text = String::with_capacity(self.body.len());
        let mut needed = self.args.len();
        let mut rest = self.body.as_str();
        while let Some((before, after)) = rest.split_once('$') {
            text.push_str(before);
            let (position, after) = if let Some(after) = after.strip_prefix('$') {
                text.push('$');
                (None, after)
            } else if let Some(after) = after.strip_prefix('@') {
                text.push_str(&typed.join(" "));
                (None, after)
            } else if let Some((key, after)) = after
                .strip_prefix('{')
                .and_then(|braced| braced.split_once('}'))
                && let Some(position) = self.position(key)
            {
                (Some(position), after)
            } else {
                let digits = after.trim_start_matches(|c: char| c.is_ascii_digit());
                let number = after.len().saturating_sub(digits.len());
                match after.get(..number).and_then(|n| n.parse::<usize>().ok()) {
                    Some(position) if position > 0 => (Some(position), digits),
                    _ => {
                        text.push('$');
                        (None, after)
                    }
                }
            };
            if let Some(position) = position {
                needed = needed.max(position);
                if let Some(arg) = typed.get(position - 1) {
                    text.push_str(arg);
                }
            }
            rest = after;
        }
        text.push_str(rest);
        if typed.len() < needed {
            return Err(needed);
        }
        Ok(text)
    }

    /// The position, from 1, of the argument `key`: a number, or a name
    /// the frontmatter lists.
    fn position(&self, key: &str) -> Option<usize> {
        match key.parse::<usize>() {
            Ok(position) => (position > 0).then_some(position),
            Err(_) => self.args.iter().position(|name| name == key).map(|i| i + 1),
        }
    }

    /// `Usage: /name <file> <focus>` for `needed` arguments.
    fn usage(&self, name: &str, needed: usize) -> String {
        let args: Vec<String> = (1..=needed)
            .map(|position| match self.args.get(position - 1) {
                Some(arg) => format!("<{arg}>"),
                None => format!("<arg{position}>"),
            })
            .collect();
        format!("Usage: /{name} {}", args.join(" "))
    }
}

/// What was typed after a command, split at whitespace outside quotes.
fn split_args(line: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current: Option<String> = None;
    let mut quote = None;
    for c in line.chars() {
        match quote {
            Some(open) if c == open => quote = None,
            Some(_) => current.get_or_insert_default().push(c),
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                current.get_or_insert_default();
            }
            None if c.is_whitespace() => args.extend(current.take()),
            None => current.get_or_insert_default().push(c),
        }
    }
    args.extend(current);
    args
}

/// Every template of `dirs`, by name, and why any was skipped.
fn read_all(dirs: &[PathBuf]) -> (BTreeMap<String, Template>, Vec<String>) {
    let mut found = BTreeMap::new();
    let mut skipped = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for path in entries.flatten().map(|entry| entry.path()) {
            let name = path.file_stem().and_then(|stem| stem.to_str());
            let Some(name) = name.filter(|_| path.extension().is_some_and(|ext| ext == "md"))
            else {
                continue;
            };
            if name.is_empty() || name.contains(char::is_whitespace) {
                skipped.push(format!("{}: its name is not a command", path.display()));
                continue;
            }
            let read = std::fs::read_to_string(&path).map_err(|why| why.to_string());
            match read.and_then(|text| Template::parse(&path, &text)) {
                Ok(template) => {
                    found.insert(name.to_owned(), template);
                }
                Err(why) => skipped.push(format!("{}: {why}", path.display())),
            }
        }
    }
    (found, skipped)
}

/// Reads the templates again and makes the commands match: one per
/// template whose name no other command has, with its help.
fn sync(world: &mut World) {
    let Some(dirs) = world.get_resource::<TemplateDirs>().cloned() else {
        return;
    };
    let (read, mut skipped) = read_all(&dirs.0);
    let mut mine: BTreeMap<String, Entity> = BTreeMap::new();
    let mut taken = BTreeSet::new();
    let mut commands =
        world.query_filtered::<(Entity, &Name, Has<TemplateCommand>), With<SlashCommand>>();
    for (entity, name, template) in commands.iter(world) {
        let name = name.as_str().trim_start_matches('/').to_owned();
        if template {
            mine.insert(name, entity);
        } else {
            taken.insert(name);
        }
    }
    let mut kept = BTreeMap::new();
    for (name, template) in read {
        if taken.contains(&name) {
            let path = template.path.display();
            skipped.push(format!("{path}: /{name} is a command already"));
            continue;
        }
        match mine.remove(&name) {
            Some(entity) => {
                if let Some(mut command) = world.get_mut::<SlashCommand>(entity)
                    && command.help != template.help
                {
                    command.help.clone_from(&template.help);
                }
            }
            None => {
                let system = world.register_system(command_system(name.clone()));
                world.entity_mut(system.entity()).insert((
                    Name::new(format!("/{name}")),
                    SlashCommand {
                        help: template.help.clone(),
                    },
                    TemplateCommand,
                ));
            }
        }
        kept.insert(name, template);
    }
    // The commands of templates gone.
    for entity in mine.into_values() {
        world.despawn(entity);
    }
    let mut templates = world.get_resource_or_init::<Templates>();
    templates.by_name = kept;
    for why in skipped {
        if templates.warned.insert(why.clone()) {
            warn!("prompt template skipped: {why}");
        }
    }
}

/// Reads the templates again before a slash command the user typed runs:
/// this global observer runs before the kernel's, which runs the command.
fn sync_before_command(typed: On<Deliver>, mut commands: Commands) {
    if typed.command().is_some() {
        commands.queue(sync);
    }
}

/// The command of the template `name`.
fn command_system(
    name: String,
) -> impl FnMut(
    In<CommandArgs>,
    Res<Templates>,
    Option<Res<Models>>,
    Query<Has<ActiveTurn>>,
    Commands,
    MessageWriter<Notice>,
) {
    move |In(args), templates, models, busy, mut commands, mut notices| {
        let agent = args.agent;
        let busy = busy.get(agent).unwrap_or_default();
        match use_template(&name, &args, &templates, models.as_deref(), busy) {
            Ok((text, connection)) => {
                if let Some(connection) = connection {
                    commands.entity(agent).insert(NextTurnModel(connection));
                }
                if let Some(template) = templates.by_name.get(&name) {
                    let from = format!("/{name}: {}", template.path.display());
                    notices.write(Notice::info(agent, from));
                }
                // Sent as typed, even if it starts with `/`.
                commands.trigger(Deliver::new(agent, text, DeliveryMode::Steer).literal());
            }
            Err(why) => {
                notices.write(Notice::error(agent, why));
            }
        }
    }
}

/// The text the template `name` sends for `args`, and the model its turn
/// runs on; or why it cannot be used.
fn use_template(
    name: &str,
    args: &CommandArgs,
    templates: &Templates,
    models: Option<&Models>,
    busy: bool,
) -> Result<(String, Option<Connection>), String> {
    let template = templates
        .by_name
        .get(name)
        .ok_or_else(|| format!("The template /{name} is gone."))?;
    let text = template
        .expand(&split_args(&args.args))
        .map_err(|needed| template.usage(name, needed))?;
    if text.trim().is_empty() {
        return Err(format!("/{name} expands to nothing."));
    }
    let Some(model) = &template.model else {
        return Ok((text, None));
    };
    if busy {
        return Err(format!(
            "/{name} runs on {model}: wait until the running turn ends."
        ));
    }
    let models = models.ok_or_else(|| format!("/{name} runs on {model}: no models here."))?;
    Ok((text, Some(models.connect(model)?)))
}

/// Sends the turn a template started to the template's model.
fn route_turn(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    next: Query<&NextTurnModel>,
    mut commands: Commands,
) {
    let Ok(TurnOf(agent)) = turns.get(started.entity) else {
        return;
    };
    if let Ok(NextTurnModel(connection)) = next.get(*agent) {
        commands.entity(started.entity).insert(connection.clone());
        commands.entity(*agent).remove::<NextTurnModel>();
    }
}
