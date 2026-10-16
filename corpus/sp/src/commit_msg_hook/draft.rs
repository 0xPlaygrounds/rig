//! What the drafting call reads and what is made of its answer: the staged
//! diff and recent subjects from git, the request, the message cleaned to
//! a subject of at most [`SUBJECT_CHARS`], and the message file written.

use std::path::Path;

use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_tools::process::Launcher;

/// The longest subject line written.
pub const SUBJECT_CHARS: usize = 72;

/// How many recent commit subjects the call reads.
const SUBJECTS: usize = 5;

/// What the drafting model is told.
const PREAMBLE: &str = "You write git commit messages. From the staged diff and the \
    repository's most recent commit subjects, write one commit message in the style of those \
    subjects: a subject line of at most 72 characters, then, only if the change needs \
    explaining, a blank line and a short body wrapped at 72 columns. Answer with the message \
    alone: no quotes, no code fences, no commentary.";

/// Whether git's `source` argument says the message is already there: from
/// `-m` or `-F`, a merge, a squash, or an existing commit.
pub fn has_message(source: &str) -> bool {
    matches!(source, "message" | "merge" | "squash" | "commit")
}

/// The staged changes and the latest commit subjects of a repository.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Staged {
    /// `git diff --cached`.
    pub diff: String,
    /// `git diff --cached --stat`: a line per file.
    pub summary: String,
    /// The last commits' subjects, newest first.
    pub subjects: Vec<String>,
}

impl Staged {
    /// Reads them in the repository at `root` with `git`, started by
    /// `launcher`. A repository with no commit yet has no subjects.
    ///
    /// # Errors
    ///
    /// When git cannot read the staged diff, such as outside a repository.
    pub fn read(root: &Path, launcher: &Launcher) -> Result<Self, String> {
        let diff = git(
            root,
            launcher,
            &["diff", "--cached", "--no-color", "--no-ext-diff"],
        )?;
        let summary = git(
            root,
            launcher,
            &["diff", "--cached", "--no-color", "--stat=120"],
        )?;
        let count = format!("-{SUBJECTS}");
        let log = git(root, launcher, &["log", &count, "--format=%s"]).unwrap_or_default();
        let subjects = log.lines().map(str::to_owned).collect();
        Ok(Self {
            diff,
            summary,
            subjects,
        })
    }
}

/// What `git args` printed in `root`.
///
/// # Errors
///
/// When git cannot start or fails, with the first line it said.
pub fn git(root: &Path, launcher: &Launcher, args: &[&str]) -> Result<String, String> {
    let output = launcher
        .command("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|why| format!("cannot run git: {why}"))?;
    if !output.status.success() {
        let said = String::from_utf8_lossy(&output.stderr);
        let line = said.lines().next().unwrap_or("git failed");
        return Err(format!("git {}: {line}", args.join(" ")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The drafting request for `staged`, its diff cut past `max_bytes` at a
/// line, the summary of every file kept, and the cut said.
pub fn request(staged: &Staged, max_bytes: usize) -> CompletionRequest {
    let mut prompt = String::from("Recent commit subjects, newest first:\n");
    if staged.subjects.is_empty() {
        prompt.push_str("(none: this is the first commit)\n");
    }
    for subject in &staged.subjects {
        prompt.push_str(&format!("- {subject}\n"));
    }
    let diff = &staged.diff;
    match cut(diff, max_bytes) {
        None => prompt.push_str(&format!("\nThe staged diff:\n{diff}")),
        Some(kept) => prompt.push_str(&format!(
            "\nThe staged diff is {} bytes, so it was cut after the first {} bytes. Every \
             changed file is in this summary:\n{}\nThe start of the staged diff (cut):\n{kept}",
            diff.len(),
            kept.len(),
            staged.summary,
        )),
    }
    CompletionRequest::new(Message::user(prompt)).preamble(PREAMBLE)
}

/// `text` cut to at most `max_bytes` at a line's end, when it is longer.
fn cut(text: &str, max_bytes: usize) -> Option<&str> {
    if text.len() <= max_bytes {
        return None;
    }
    let head = text.get(..text.floor_char_boundary(max_bytes))?;
    let end = head.rfind('\n').map_or(head.len(), |at| at + 1);
    head.get(..end)
}

/// The commit message in a model's `answer`: code fences dropped, the
/// first line the subject, at most [`SUBJECT_CHARS`] cut at a word, then
/// the body after a blank line; `None` when it holds no text.
pub fn message_of(answer: &str) -> Option<String> {
    let mut lines = answer
        .lines()
        .filter(|line| !line.trim_start().starts_with("```"))
        .map(str::trim_end)
        .skip_while(|line| line.trim().is_empty());
    let subject = subject(lines.next()?.trim());
    let body: Vec<&str> = lines.skip_while(|line| line.is_empty()).collect();
    let body = body.join("\n");
    let body = body.trim_end();
    Some(match body.is_empty() {
        true => subject,
        false => format!("{subject}\n\n{body}"),
    })
}

/// `line` in at most [`SUBJECT_CHARS`], cut at a word where it can be.
fn subject(line: &str) -> String {
    if line.chars().count() <= SUBJECT_CHARS {
        return line.to_owned();
    }
    let mut kept = String::new();
    for word in line.split_whitespace() {
        let longer = match kept.is_empty() {
            true => word.to_owned(),
            false => format!("{kept} {word}"),
        };
        if longer.chars().count() > SUBJECT_CHARS {
            break;
        }
        kept = longer;
    }
    match kept.is_empty() {
        true => line.chars().take(SUBJECT_CHARS).collect(),
        false => kept,
    }
}

/// Writes `message` at the top of the message file at `path`, above what
/// it holds (git's comment lines, a template), with a blank line between.
///
/// # Errors
///
/// When the file cannot be read or written; it is then as it was.
pub fn write_above(path: &Path, message: &str) -> std::io::Result<()> {
    let existing = std::fs::read_to_string(path)?;
    let rest = existing.trim_start_matches(['\n', '\r']);
    let text = match rest.is_empty() {
        true => format!("{message}\n"),
        false => format!("{message}\n\n{rest}"),
    };
    std::fs::write(path, text)
}
