//! The notes of one project and how they follow the code: pure data and
//! text, no Bevy.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Lines of context kept on each side of a note's lines, to find them
/// again.
const CONTEXT: usize = 3;

/// A note on lines of a file.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Note {
    /// Its number: `n3` is 3.
    pub id: u32,
    /// Its file, relative to the project's root when under it.
    pub path: String,
    /// The first line it is on, from 1; where it was, once stale.
    pub start: usize,
    /// The last line it is on.
    pub end: usize,
    /// The text of those lines when last found.
    pub anchor: Vec<String>,
    /// The lines just before them.
    pub before: Vec<String>,
    /// The lines just after them.
    pub after: Vec<String>,
    /// What it says.
    pub text: String,
    /// The day it was written, `YYYY-MM-DD`.
    pub date: String,
    /// Who wrote it: `agent` or `user`.
    pub by: String,
    /// Whether its lines were deleted or changed beyond recognition.
    pub stale: bool,
}

impl Note {
    /// Anchors it on `lines[from..to]` (0-based), with their context.
    pub fn anchor_at(&mut self, lines: &[&str], from: usize, to: usize) {
        let owned = |range: &[&str]| range.iter().map(|line| (*line).to_owned()).collect();
        self.start = from + 1;
        self.end = to;
        self.anchor = lines.get(from..to).map(owned).unwrap_or_default();
        self.before = lines
            .get(from.saturating_sub(CONTEXT)..from)
            .map(owned)
            .unwrap_or_default();
        self.after = lines
            .get(to..(to + CONTEXT).min(lines.len()))
            .map(owned)
            .unwrap_or_default();
        self.stale = false;
    }

    /// Where its lines are in `lines` now (0-based), and whether they are
    /// unchanged: the window that matches most of its anchored lines
    /// (ignoring indentation) and their context, nearest its old place. A
    /// window that matches only some of them needs at least half, and some
    /// context; none is recognizable otherwise.
    pub fn locate(&self, lines: &[&str]) -> Option<(usize, bool)> {
        let size = self.anchor.len();
        if size == 0 || lines.len() < size {
            return None;
        }
        let old = self.start.saturating_sub(1);
        let mut best: Option<(usize, usize, usize, bool)> = None;
        for at in 0..=lines.len() - size {
            let Some(window) = lines.get(at..at + size) else {
                continue;
            };
            let matched = window
                .iter()
                .zip(&self.anchor)
                .filter(|(line, anchored)| line.trim() == anchored.trim())
                .count();
            let exact = matched == size;
            if matched == 0 || matched * 2 < size {
                continue;
            }
            let context = self.context_at(lines, at);
            if !exact && context == 0 {
                continue;
            }
            let score = matched * 4 + context;
            let distance = at.abs_diff(old);
            let better = best.is_none_or(|(best_score, best_distance, ..)| {
                score > best_score || (score == best_score && distance < best_distance)
            });
            if better {
                best = Some((score, distance, at, exact));
            }
        }
        best.map(|(.., at, exact)| (at, exact))
    }

    /// How many of its context lines are around `lines[at..]` now.
    fn context_at(&self, lines: &[&str], at: usize) -> usize {
        let before = self.before.iter().rev().enumerate().filter(|(back, want)| {
            at.checked_sub(back + 1)
                .and_then(|index| lines.get(index))
                .is_some_and(|line| line.trim() == want.trim())
        });
        let after = self.after.iter().enumerate().filter(|(next, want)| {
            lines
                .get(at + self.anchor.len() + next)
                .is_some_and(|line| line.trim() == want.trim())
        });
        before.count() + after.count()
    }

    /// Finds its lines in `lines` again: re-anchored there, else stale.
    /// Whether anything about it changed.
    pub fn follow(&mut self, lines: &[&str]) -> bool {
        let was = self.clone();
        match self.locate(lines) {
            Some((at, _)) => self.anchor_at(lines, at, at + self.anchor.len()),
            None => self.stale = true,
        }
        *self != was
    }

    /// How a read shows it after its lines.
    pub fn line(&self) -> String {
        format!(
            "  ┆ note n{} ({}, {}): {}\n",
            self.id,
            self.date,
            self.by,
            self.text.replace('\n', "\n  ┆ ")
        )
    }

    /// How a read shows it, stale, at its end.
    pub fn stale_line(&self) -> String {
        format!(
            "  ┆ stale note n{} (was at lines {}–{}): {}\n",
            self.id,
            self.start,
            self.end,
            self.text.replace('\n', "\n  ┆ ")
        )
    }
}

/// A project's notes, as saved.
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub(crate) struct Book {
    /// The project's root, for whoever opens the file.
    pub root: String,
    /// The last note number given.
    pub last: u32,
    /// The notes.
    pub notes: Vec<Note>,
}

impl Book {
    /// `path`'s name in the book: relative to `root` when under it.
    pub fn key(root: &Path, path: &Path) -> String {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned()
    }

    /// The paths of the files with notes other than `path`'s.
    pub fn other_files(&self, root: &Path, path: &Path) -> Vec<PathBuf> {
        let key = Self::key(root, path);
        let files: HashSet<&str> = self
            .notes
            .iter()
            .map(|note| note.path.as_str())
            .filter(|file| *file != key)
            .collect();
        files.into_iter().map(|file| root.join(file)).collect()
    }

    /// Finds the notes on `path`, whose text is `text`, in it again, and
    /// takes over the notes of files that are `gone` whose lines are in it
    /// unchanged, as after a rename. Whether any note changed.
    pub fn follow(
        &mut self,
        root: &Path,
        path: &Path,
        text: &str,
        gone: &HashSet<PathBuf>,
    ) -> bool {
        let lines: Vec<&str> = text.lines().collect();
        let key = Self::key(root, path);
        let mut changed = false;
        for note in &mut self.notes {
            if note.path == key {
                changed |= note.follow(&lines);
            } else if gone.contains(&root.join(&note.path))
                && note.locate(&lines).is_some_and(|(_, exact)| exact)
            {
                note.path.clone_from(&key);
                note.follow(&lines);
                changed = true;
            }
        }
        changed
    }

    /// `output`, a read of `path` that showed `count` of its lines from
    /// line `first`, one per line, with each note on those lines
    /// after its last line shown, and each stale note of the file not in
    /// `shown` yet at the end, added to `shown`.
    pub fn annotate(
        &self,
        key: &str,
        output: &str,
        (first, count): (usize, usize),
        shown: &mut HashSet<u32>,
    ) -> String {
        let past = first + count;
        let live: Vec<&Note> = self
            .notes
            .iter()
            .filter(|note| note.path == key && !note.stale)
            .filter(|note| count > 0 && note.start < past && note.end >= first)
            .collect();
        let mut out = String::with_capacity(output.len());
        for (index, line) in output.split_inclusive('\n').enumerate() {
            out.push_str(line);
            if index >= count {
                continue;
            }
            let number = first + index;
            for note in live.iter().filter(|note| note.end.min(past - 1) == number) {
                end_line(&mut out);
                out.push_str(&note.line());
            }
        }
        let stale = self
            .notes
            .iter()
            .filter(|note| note.path == key && note.stale);
        for note in stale {
            if shown.insert(note.id) {
                end_line(&mut out);
                out.push_str(&note.stale_line());
            }
        }
        out
    }

    /// The notes listed by file, files with stale notes first, and in each
    /// file the stale ones first.
    pub fn listing(&self) -> String {
        let mut files: Vec<&str> = self.notes.iter().map(|note| note.path.as_str()).collect();
        files.sort_unstable();
        files.dedup();
        files.sort_by_key(|file| {
            !self
                .notes
                .iter()
                .any(|note| note.path == *file && note.stale)
        });
        let mut out = String::new();
        for file in files {
            out.push_str(file);
            out.push('\n');
            let mut notes: Vec<&Note> =
                self.notes.iter().filter(|note| note.path == file).collect();
            notes.sort_by_key(|note| (!note.stale, note.start));
            for note in notes {
                let (id, start, end) = (note.id, note.start, note.end);
                let at = if note.stale {
                    format!("stale n{id} (was at lines {start}–{end}")
                } else {
                    format!("n{id} (lines {start}–{end}")
                };
                let text = note.text.replace('\n', " ");
                out.push_str(&format!("  {at}, {}, {}): {text}\n", note.date, note.by));
            }
        }
        out
    }
}

/// Ends `out`'s last line, if it has an unfinished one.
fn end_line(out: &mut String) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}
