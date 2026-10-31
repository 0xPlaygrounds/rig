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
//! model: anthropic/claude-sonnet  # the turn its message goes to runs on this model
//! args: [file, focus]             # `${file}` is `$1`, `${focus}` is `$2`
//! ---
//! Review ${file} for bugs, mostly ${focus}.
//! ```
//!
//! Every named argument, and every `$N` the text uses, must be typed;
//! otherwise the command is refused with its usage line. The project's
//! template wins a name clash with the user's, and a template never
//! replaces another plugin's command: it is skipped with a warning. Both
//! directories are watched, from their first listing on, so a template
//! added, edited or removed counts without a restart.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use bevy_log::warn;
use rig_harness::prelude::*;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// Makes each template a command, and does again when one changes.
#[derive(Default)]
pub struct PromptTemplatesPlugin;

impl Plugin for PromptTemplatesPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<FileChanges>()
            .add_systems(Startup, watch)
            .add_systems(Update, reload)
            .configure_sets(
                AgentPass,
                PromptTemplatesSystems.in_set(PrepareSystems::Edit),
            )
            .add_systems(AgentPass, route.in_set(PromptTemplatesSystems));
    }
}

/// The plugin's Prepare system, which sends the turn of a template with a
/// `model` there: after the default prompt went in, before compaction
/// measures the request against the model it goes to.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptTemplatesSystems;

/// On the watches of the template directories.
#[derive(Component)]
struct TemplateWatch;

/// On the entity of a template's command.
#[derive(Component)]
struct TemplateCommand;

/// On a template's message whose turn runs on the template's model.
#[derive(Component)]
struct TemplateModel(Connection);

/// The directories templates are read from, the most general first: a
/// later directory's template wins a name clash.
fn dirs(world: &World) -> [PathBuf; 2] {
    let project = world
        .get_resource::<ProjectRoot>()
        .cloned()
        .unwrap_or_default();
    [
        RigHome::of(world).root().join("prompts"),
        project.0.join(".rig").join("prompts"),
    ]
}

/// Watches the template directories. Their first `FileChanges`, the
/// files there when armed, reads the templates.
fn watch(world: &mut World) {
    for dir in dirs(world) {
        world.spawn((TemplateWatch, FileWatch::new(dir)));
    }
}

/// One template file.
#[derive(Clone, Debug)]
struct Template {
    /// The command's name, without its `/`.
    name: String,
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
    args: Vec<String>,
}

impl Template {
    /// The template `name` in `text`, read from `path`.
    fn parse(name: &str, path: &Path, text: &str) -> Result<Self, String> {
        let (front, body) = match text.strip_prefix("---\n") {
            Some(rest) => match rest.strip_prefix("---") {
                Some(body) => ("", body),
                None => rest
                    .split_once("\n---")
                    .ok_or("its frontmatter has no closing `---`")?,
            },
            None => ("", text),
        };
        let front: Frontmatter = if front.trim().is_empty() {
            Frontmatter::default()
        } else {
            serde_yaml::from_str(front).map_err(|why| format!("its frontmatter: {why}"))?
        };
        Ok(Self {
            name: name.to_owned(),
            path: path.to_owned(),
            help: front
                .description
                .unwrap_or_else(|| format!("Prompt template {}", path.display())),
            model: front.model,
            args: front.args,
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
    fn usage(&self, needed: usize) -> String {
        let args: Vec<String> = (1..=needed)
            .map(|position| match self.args.get(position - 1) {
                Some(arg) => format!("<{arg}>"),
                None => format!("<arg{position}>"),
            })
            .collect();
        format!("Usage: /{} {}", self.name, args.join(" "))
    }

    /// The message this template sends for `args`, with the model its
    /// turn runs on; or why it cannot be used.
    fn message(&self, args: &CommandArgs, models: Option<&Models>) -> Result<Sent, String> {
        let text = self
            .expand(&split_args(&args.args))
            .map_err(|needed| self.usage(needed))?;
        if text.trim().is_empty() {
            return Err(format!("/{} expands to nothing.", self.name));
        }
        // The user's own words, sent as typed even if they start with `/`,
        // titled so the transcript names the template.
        let incoming = Incoming::new(text, args.mode)
            .with_author(Author::user().titled(self.name.clone()))
            .literal();
        let model = match &self.model {
            None => None,
            Some(model) => {
                let models = models.ok_or_else(|| format!("No models to run {model} on."))?;
                Some(TemplateModel(models.connect(model)?))
            }
        };
        Ok((incoming, model))
    }
}

/// A template's message, and the model its turn runs on.
type Sent = (Incoming, Option<TemplateModel>);

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
            match read.and_then(|text| Template::parse(name, &path, &text)) {
                Ok(template) => {
                    found.insert(name.to_owned(), template);
                }
                Err(why) => skipped.push(format!("{}: {why}", path.display())),
            }
        }
    }
    (found, skipped)
}

/// Reads the templates and replaces the template commands with one per
/// template whose name no other command has.
fn sync(world: &mut World) {
    let (read, mut skipped) = read_all(&dirs(world));
    let mut mine = world.query_filtered::<Entity, With<TemplateCommand>>();
    for command in mine.iter(world).collect::<Vec<_>>() {
        world.despawn(command);
    }
    let mut commands = world.query_filtered::<&Name, With<SlashCommand>>();
    let taken: BTreeSet<String> = commands
        .iter(world)
        .map(|name| name.as_str().to_owned())
        .collect();
    let plugin = world
        .get_resource::<PluginEntity<PromptTemplatesPlugin>>()
        .map(PluginEntity::entity);
    for (name, template) in read {
        let name = format!("/{name}");
        if taken.contains(&name) {
            skipped.push(format!(
                "{}: {name} is a command already",
                template.path.display()
            ));
            continue;
        }
        let help = template.help.clone();
        let system = world.register_system(command(template));
        let mut command = world.entity_mut(system.entity());
        command.insert((
            Name::new(name),
            SlashCommand {
                help,
                args: Args::Text,
            },
            TemplateCommand,
        ));
        if let Some(plugin) = plugin {
            command.insert(ProvidedBy(plugin));
        }
    }
    for why in skipped {
        warn!("prompt template skipped: {why}");
    }
}

/// Reads the templates when a watch is armed, and again when a file in
/// their directories changed.
fn reload(
    mut changes: MessageReader<FileChanges>,
    watches: Query<(), With<TemplateWatch>>,
    mut commands: Commands,
) {
    let mine = changes
        .read()
        .fold(false, |mine, change| mine | watches.contains(change.watch));
    if mine {
        commands.queue(sync);
    }
}

/// The command of `template`: sends its expansion as the user's message.
/// An error notice refuses it, with the line back in the input.
fn command(
    template: Template,
) -> impl FnMut(In<CommandArgs>, Option<Res<Models>>, Commands, MessageWriter<Notice>) {
    move |In(args), models, mut commands, mut notices| match template
        .message(&args, models.as_deref())
    {
        Ok((incoming, model)) => {
            let mut message = commands.spawn((incoming, ForAgent(args.agent)));
            if let Some(model) = model {
                message.insert(model);
            }
        }
        Err(why) => {
            notices.write(Notice::error(args.agent, why));
        }
    }
}

/// Sends each call of a turn a template's message went with to the
/// template's model.
fn route(
    mut calls: Query<(&mut ModelRequest, &CallOf), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    templates: Query<(&TemplateModel, &DeliveredIn)>,
) {
    for (mut request, &CallOf(turn)) in &mut calls {
        let model = templates.iter().find(|(_, into)| into.0 == turn);
        if let Some((TemplateModel(model), _)) = model {
            request.connection = Some(model.clone());
        }
    }
}
