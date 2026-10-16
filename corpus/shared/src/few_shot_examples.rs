//! Few-shot examples: example exchanges from `<project>/.rig/examples/*.md`
//! go with a model request when they fit it, and nowhere else.
//!
//! An example file has a frontmatter `when` (keywords, or globs of files
//! touched, separated by commas) and a body of alternating `## user` and
//! `## assistant` sections:
//!
//! ```text
//! ---
//! when: commit, commit message
//! ---
//! ## user
//! Commit the parser fix.
//! ## assistant
//! fix(parser): accept a trailing comma
//! ```
//!
//! A Prepare system ([`ExampleSystems`], before compaction measures the
//! request) puts the examples whose `when` matches the latest user text, or
//! a file the turn's tool calls said they read or changed ([`FilesRead`],
//! [`FilesChanged`]), right after the system prompt of each request built
//! from an agent, at most `max_examples` of them (the `[few_shot_examples]`
//! setting, 3 by default), in file-name order: the same examples make the
//! same prefix, so the provider's prompt cache keeps working. They are
//! never committed to the [`Conversation`], so the session log and
//! compaction's summaries never see them. The directory is watched, and
//! read on the watch's first listing and on each change. `/examples` says
//! which went with the agent's last request.

use std::path::{Path, PathBuf};

use glob::Pattern;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Message;
use rig_harness::rig_core::tool::{FilesChanged, FilesRead};
use serde::Deserialize;

/// Injects the project's few-shot examples into the requests they fit, and
/// adds `/examples`.
#[derive(Default)]
pub struct FewShotExamplesPlugin;

impl Plugin for FewShotExamplesPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<FewShotSettings>("few_shot_examples")
            .init_resource::<ProjectRoot>()
            .init_resource::<Library>()
            .add_message::<FileChanges>()
            .register_required_components::<TurnOf, Touched>()
            .register_required_components::<Agent, SentExamples>()
            .add_observer(track_files)
            .add_systems(Startup, watch)
            .add_systems(Update, reload.before(AgentLoop))
            .configure_sets(
                AgentPass,
                // Right after the whole system prompt, and measured by
                // compaction.
                ExampleSystems.in_set(PrepareSystems::Edit),
            )
            .add_systems(AgentPass, inject.in_set(ExampleSystems))
            .add_command(
                "examples",
                "List the few-shot examples that went with the last request",
                show,
            );
    }
}

/// The Prepare system that puts the examples in each request, in
/// [`PrepareSystems::Edit`], for other plugins that edit requests to order
/// against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExampleSystems;

/// The `[few_shot_examples]` settings table.
#[derive(Resource, Clone, Copy, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FewShotSettings {
    /// How many examples one request takes at most.
    pub max_examples: usize,
}

impl Default for FewShotSettings {
    fn default() -> Self {
        Self { max_examples: 3 }
    }
}

/// The names of the examples that went with the agent's last request.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct SentExamples(pub Vec<String>);

/// The files the turn's tool calls read or changed.
#[derive(Component, Clone, Debug, Default)]
struct Touched(Vec<String>);

/// The watch on the examples' directory.
#[derive(Component)]
struct ExamplesWatch;

/// The examples as last read, and the files skipped.
#[derive(Resource, Default)]
struct Library {
    examples: Vec<Example>,
    skipped: Vec<String>,
}

/// One example file.
struct Example {
    /// Its file name without `.md`.
    name: String,
    /// Keywords, each as its lowercase words.
    keywords: Vec<Vec<String>>,
    /// Globs of files touched.
    globs: Vec<Pattern>,
    /// The exchange, user first, assistant last.
    messages: Vec<Message>,
}

impl Example {
    /// Parses the example `name` from its file's `text`.
    fn parse(name: String, text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some("---") {
            return Err("no frontmatter".to_owned());
        }
        let mut when = None;
        for line in lines.by_ref() {
            if line.trim() == "---" {
                break;
            }
            if let Some(value) = line.trim().strip_prefix("when:") {
                when = Some(value.trim().trim_matches(['[', ']']).to_owned());
            }
        }
        let when = when.ok_or("no `when` in the frontmatter")?;
        let (mut keywords, mut globs) = (Vec::new(), Vec::new());
        for item in when.split(',') {
            let item = item.trim().trim_matches(['"', '\'']);
            if item.contains(['*', '?', '[', '/', '.']) {
                globs.push(Pattern::new(item).map_err(|error| format!("`{item}`: {error}"))?);
            } else if !item.is_empty() {
                keywords.push(words(item));
            }
        }
        Ok(Self {
            name,
            keywords,
            globs,
            messages: exchange(lines)?,
        })
    }

    /// Whether the example fits a request whose latest user text has
    /// `said` as words, in a turn that touched `files` in `project`.
    fn fits(&self, said: &[String], files: &[String], project: &Path) -> bool {
        let spoken = self.keywords.iter().any(|keyword| {
            !keyword.is_empty()
                && said
                    .windows(keyword.len())
                    .any(|run| run == keyword.as_slice())
        });
        spoken
            || files.iter().any(|file| {
                let path = Path::new(file);
                let relative = path.strip_prefix(project).unwrap_or(path);
                self.globs.iter().any(|glob| {
                    glob.matches_path(relative)
                        || !glob.as_str().contains('/')
                            && relative
                                .file_name()
                                .is_some_and(|name| glob.matches_path(Path::new(name)))
                })
            })
    }
}

/// The alternating `## user` / `## assistant` sections of `body`.
fn exchange<'a>(body: impl Iterator<Item = &'a str>) -> Result<Vec<Message>, String> {
    let mut sections: Vec<(bool, String)> = Vec::new();
    for line in body {
        match line.trim().to_lowercase().as_str() {
            "## user" => sections.push((true, String::new())),
            "## assistant" => sections.push((false, String::new())),
            _ => {
                if let Some((_, text)) = sections.last_mut() {
                    text.push_str(line);
                    text.push('\n');
                }
            }
        }
    }
    let alternates = sections
        .iter()
        .enumerate()
        .all(|(index, (user, text))| *user == (index % 2 == 0) && !text.trim().is_empty());
    if sections.is_empty() || !sections.len().is_multiple_of(2) || !alternates {
        return Err("the body is not alternating `## user` and `## assistant` sections".to_owned());
    }
    Ok(sections
        .into_iter()
        .map(|(user, text)| match user {
            true => Message::user(text.trim()),
            false => Message::assistant(text.trim()),
        })
        .collect())
}

/// The lowercase words of `text`.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The text of the latest user message that has any (a tool result has
/// none).
fn latest_user_text(messages: &[std::sync::Arc<Message>]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        let Message::User { content } = &**message else {
            return None;
        };
        let texts: Vec<&str> = content
            .iter()
            .filter_map(|part| match part {
                UserContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        (!texts.is_empty()).then(|| texts.join("\n"))
    })
}

/// The examples' directory in `root`.
fn examples_dir(root: &ProjectRoot) -> PathBuf {
    root.0.join(".rig").join("examples")
}

/// Reads the examples in `dir`, in file-name order.
fn read_library(dir: &Path) -> Library {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "md"))
        .collect();
    files.sort();
    let mut library = Library::default();
    for path in files {
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        let parsed = std::fs::read_to_string(&path)
            .map_err(|error| error.to_string())
            .and_then(|text| Example::parse(name.clone(), &text));
        match parsed {
            Ok(example) => library.examples.push(example),
            Err(why) => library.skipped.push(format!("{name}: {why}")),
        }
    }
    library
}

/// Watches the examples' directory.
fn watch(root: Res<ProjectRoot>, mut commands: Commands) {
    commands.spawn((ExamplesWatch, FileWatch::new(examples_dir(&root))));
}

/// Reads the examples on the watch's first listing and after each change.
fn reload(
    mut changes: MessageReader<FileChanges>,
    watches: Query<(), With<ExamplesWatch>>,
    root: Res<ProjectRoot>,
    mut library: ResMut<Library>,
) {
    if changes.read().any(|change| watches.contains(change.watch)) {
        *library = read_library(&examples_dir(&root));
    }
}

/// Notes the files any tool call says it read or changed on its turn.
fn track_files(
    published: On<Insert<Published>>,
    calls: Query<(&Published, &CallOf)>,
    mut turns: Query<&mut Touched>,
) {
    let Ok((published, &CallOf(turn))) = calls.get(published.entity) else {
        return;
    };
    let Ok(mut touched) = turns.get_mut(turn) else {
        return;
    };
    if let Ok(Some(FilesRead(read))) = published.0.get::<FilesRead>() {
        read.into_iter().for_each(|file| touched.note(file.path));
    }
    if let Ok(Some(changed)) = published.0.get::<FilesChanged>() {
        changed
            .paths()
            .for_each(|path| touched.note(path.to_owned()));
    }
}

impl Touched {
    /// Notes `path`, once.
    fn note(&mut self, path: String) {
        if !self.0.contains(&path) {
            self.0.push(path);
        }
    }
}

/// The requests built from an agent, as Prepare sees each once.
type NewRequests<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut ModelRequest,
        &'static ForAgent,
        Option<&'static CallOf>,
    ),
    (Added<Admitted>, Or<(With<TurnCall>, With<SideCall>)>),
>;

/// Puts the examples that fit each request right after its system prompt.
fn inject(
    mut calls: NewRequests,
    library: Res<Library>,
    settings: Res<FewShotSettings>,
    root: Res<ProjectRoot>,
    turns: Query<&Touched>,
    mut agents: Query<&mut SentExamples>,
) {
    for (mut call, &ForAgent(agent), of) in &mut calls {
        let files = of
            .and_then(|&CallOf(turn)| turns.get(turn).ok())
            .map(|touched| touched.0.as_slice())
            .unwrap_or_default();
        let Some(history) = &mut call.history else {
            continue;
        };
        let said = latest_user_text(history.messages())
            .map(|text| words(&text))
            .unwrap_or_default();
        let chosen: Vec<&Example> = library
            .examples
            .iter()
            .filter(|example| example.fits(&said, files, &root.0))
            .take(settings.max_examples)
            .collect();
        if let Ok(mut sent) = agents.get_mut(agent) {
            let names = chosen.iter().map(|example| example.name.clone()).collect();
            sent.set_if_neq(SentExamples(names));
        }
        let after_prompt = history
            .messages()
            .iter()
            .take_while(|message| matches!(&***message, Message::System { .. }))
            .count();
        let by = Author::plugin("few-shot-examples");
        let examples = chosen
            .iter()
            .flat_map(|example| example.messages.iter().cloned())
            .map(|message| Entry::new(message).by(by.clone()))
            .collect::<Vec<_>>();
        history.splice(after_prompt..after_prompt, examples, by);
    }
}

/// `/examples`.
fn show(
    In(args): In<CommandArgs>,
    agents: Query<&SentExamples>,
    library: Res<Library>,
    mut notices: MessageWriter<Notice>,
) {
    let sent = agents
        .get(args.agent)
        .map(|sent| sent.0.join(", "))
        .unwrap_or_default();
    let mut text = match sent.is_empty() {
        true => "No examples went with the last request.".to_owned(),
        false => format!("Examples with the last request: {sent}."),
    };
    for skipped in &library.skipped {
        text.push_str(&format!("\nSkipped {skipped}."));
    }
    notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;
