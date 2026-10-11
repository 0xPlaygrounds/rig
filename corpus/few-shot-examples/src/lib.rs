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
//! Before each request ([`PrepareRequest`]) the examples whose `when`
//! matches the latest user text or a file the turn's file tools touched go
//! first in its messages, right after the system prompt, at most
//! [`MaxExamples`] of them, in file-name order: the same examples make the
//! same prefix, so the provider's prompt cache keeps working. They are
//! never committed to the [`Conversation`], so the session log and
//! compaction's summaries never see them. `/examples` says which went with
//! the agent's last request.

use std::path::{Path, PathBuf};

use glob::Pattern;
use rig_core::completion::Message;
use rig_harness::prelude::*;
use serde_json::Value;

/// The tools whose `path` argument is a file the turn touched: the file
/// tools of rig-coding-tools, as rig-compaction tracks them.
const FILE_TOOLS: [&str; 3] = ["read", "edit", "write"];

/// Injects the project's few-shot examples into the requests they fit, and
/// adds `/examples`.
#[derive(Default)]
pub struct FewShotExamplesPlugin;

impl Plugin for FewShotExamplesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ExamplesProject>()
            .init_resource::<MaxExamples>()
            .init_resource::<Library>()
            .register_required_components::<TurnOf, Touched>()
            .register_required_components::<Agent, SentExamples>()
            .add_observer(load)
            .add_observer(track)
            .add_observer(inject)
            .add_command(
                "examples",
                "List the few-shot examples that went with the last request",
                show,
            );
    }
}

/// The project whose `.rig/examples` holds the examples: the working
/// directory unless set.
#[derive(Resource, Clone, Debug)]
pub struct ExamplesProject(pub PathBuf);

impl Default for ExamplesProject {
    fn default() -> Self {
        Self(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
}

impl ExamplesProject {
    /// The examples' directory.
    fn dir(&self) -> PathBuf {
        self.0.join(".rig").join("examples")
    }
}

/// How many examples one request takes at most.
#[derive(Resource, Clone, Copy, Debug)]
pub struct MaxExamples(pub usize);

impl Default for MaxExamples {
    fn default() -> Self {
        Self(3)
    }
}

/// The names of the examples that went with the agent's last request.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct SentExamples(pub Vec<String>);

/// The files the turn's file tools were called with.
#[derive(Component, Clone, Debug, Default)]
struct Touched(Vec<String>);

/// The examples read when the last turn started, and the files skipped.
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
fn latest_user_text(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        let Message::User { content } = message else {
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

/// Reads the examples when a turn starts, so an edited file counts from
/// the next message.
fn load(_started: On<Add<TurnOf>>, project: Res<ExamplesProject>, mut library: ResMut<Library>) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(project.dir())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "md"))
        .collect();
    files.sort();
    let mut read = Library::default();
    for path in files {
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        let parsed = std::fs::read_to_string(&path)
            .map_err(|error| error.to_string())
            .and_then(|text| Example::parse(name.clone(), &text));
        match parsed {
            Ok(example) => read.examples.push(example),
            Err(why) => read.skipped.push(format!("{name}: {why}")),
        }
    }
    *library = read;
}

/// Notes the file each of the turn's file tool calls names.
fn track(
    call: On<Add<CallOf>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    mut turns: Query<&mut Touched>,
) {
    let Ok((run, CallOf(turn))) = calls.get(call.entity) else {
        return;
    };
    let function = &run.call.function;
    if !FILE_TOOLS.contains(&function.name.as_str()) {
        return;
    }
    if let Some(path) = function.arguments.get("path").and_then(Value::as_str)
        && let Ok(mut touched) = turns.get_mut(*turn)
        && !touched.0.iter().any(|file| file == path)
    {
        touched.0.push(path.to_owned());
    }
}

/// Puts the examples that fit the request before its messages.
fn inject(
    mut prepare: On<PrepareRequest>,
    library: Res<Library>,
    max: Res<MaxExamples>,
    project: Res<ExamplesProject>,
    turns: Query<&Touched>,
    mut agents: Query<&mut SentExamples>,
) {
    let files = turns
        .get(prepare.entity)
        .map(|touched| touched.0.as_slice())
        .unwrap_or_default();
    let said = latest_user_text(&prepare.messages)
        .map(|text| words(&text))
        .unwrap_or_default();
    let chosen: Vec<&Example> = library
        .examples
        .iter()
        .filter(|example| example.fits(&said, files, &project.0))
        .take(max.0)
        .collect();
    if let Ok(mut sent) = agents.get_mut(prepare.agent) {
        let names = chosen.iter().map(|example| example.name.clone()).collect();
        sent.set_if_neq(SentExamples(names));
    }
    let examples = chosen
        .iter()
        .flat_map(|example| example.messages.iter().cloned());
    prepare.messages.splice(0..0, examples);
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
