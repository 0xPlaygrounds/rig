//! `session-explode <session-dir> <out-dir> [--min-turns N] [--json]`:
//! turns every branch of a saved rig session into a standalone session of
//! its own, which the agent resumes, leaving the original untouched.
//!
//! A rig session is a directory (rig-memory's `rig-session/4` store: a log
//! per agent, blobs and files). For each leaf of the main agent's branch
//! tree (the active line and every kept branch that no other line
//! continues), the program writes one session directory into `out-dir`,
//! named by a new session id, holding exactly the path from the first
//! message to that leaf: its entries with their ids, the asides that sit
//! on it (plugins' own kinds and followed values among them, verbatim and
//! in order) but none from other branches, the agent's and the session's
//! saved values and the session's plugin files. Its listing
//! ([`SessionMeta`]) is titled `<original title> [branch k of n]`, its
//! [`ForkedFrom`] names the original session and the leaf message, and its
//! name is `<original name>-branch-<k>`, so with `out-dir` a rig home's
//! `sessions` directory `rig --resume <name>` resumes it.
//!
//! Each output is written in a hidden directory beside it and renamed
//! into place whole; no existing directory or name is overwritten (a taken
//! name gets a numeric suffix). An input that cannot be read whole is
//! refused before anything is written.

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use rig_core::conversation::{AsideId, Branch, Conversation, Entry, MessageId, shared_prefix};
use rig_harness_protocol::{SessionDir, SessionId};
use rig_memory::{DirStore, Files, ForkedFrom, SavedAgent, Session, SessionMeta};
use serde::Serialize;

/// How the program is called.
pub const USAGE: &str = "usage: session-explode <session-dir> <out-dir> [--min-turns N] [--json]";

/// Why the program stopped: printed after `session-explode: `.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    fn new(why: impl fmt::Display) -> Self {
        Self(why.to_string())
    }
}

/// What the command line asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// The session directory read.
    pub input: PathBuf,
    /// Where the outputs go.
    pub out: PathBuf,
    /// Leaves with fewer turns after their last branching point are
    /// skipped.
    pub min_turns: usize,
    /// Whether the outputs are printed as a JSON array.
    pub json: bool,
}

impl Options {
    /// Reads `args`, the arguments after the program's name.
    ///
    /// # Errors
    ///
    /// When they are not `<session-dir> <out-dir> [--min-turns N] [--json]`.
    pub fn parse(args: &[String]) -> Result<Self, Error> {
        let mut paths = Vec::new();
        let mut min_turns = 0;
        let mut json = false;
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => json = true,
                "--min-turns" => {
                    let n = args.next().and_then(|n| n.parse().ok());
                    min_turns = n.ok_or_else(|| Error::new(USAGE))?;
                }
                flag if flag.starts_with("--") => return Err(Error::new(USAGE)),
                path => paths.push(PathBuf::from(path)),
            }
        }
        let [input, out] = <[PathBuf; 2]>::try_from(paths).map_err(|_| Error::new(USAGE))?;
        Ok(Self {
            input,
            out,
            min_turns,
            json,
        })
    }
}

/// One leaf of a conversation's branch tree, as its own conversation.
#[derive(Debug, Clone)]
pub struct Leaf {
    /// The line it ends: a kept branch's aside, the active line with
    /// `None`.
    pub line: Option<AsideId>,
    /// The line's title.
    pub title: String,
    /// Its last message.
    pub leaf: MessageId,
    /// The path from the first message to the leaf, with the asides on it
    /// and no other branch.
    pub conversation: Conversation,
    /// Its turns: the messages that asked the model for one.
    pub turns: usize,
    /// Its turns after the last point where another line parts from it.
    pub own_turns: usize,
}

/// Every leaf of `conversation`'s branch tree, in the order their last
/// messages were made: the active line and each kept branch (but ranges
/// moved out, which belong back in their line) that no other line
/// continues, each as a conversation of its own.
pub fn leaves(conversation: &Conversation) -> Vec<Leaf> {
    let kept = conversation.branches();
    let ids = std::iter::once((None, conversation.line_title().to_owned())).chain(
        kept.iter()
            .filter(|kept| !kept.branch.moved)
            .map(|kept| (Some(kept.id), kept.branch.title.clone())),
    );
    let lines: Vec<(Option<AsideId>, String, Vec<Entry>)> = ids
        .filter_map(|(id, title)| Some((id, title, conversation.line(id)?)))
        .collect();
    let mut found = Vec::new();
    for (at, (id, title, line)) in lines.iter().enumerate() {
        let others = lines
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != at)
            .map(|(other, (_, _, entries))| (other, entries));
        let mut parts = 0;
        let mut continued = false;
        for (other, entries) in others {
            let shared = shared_prefix(line, entries);
            // Another line holds all of this one: it is no leaf, unless
            // both are the same and this is the first of them.
            continued |= shared == line.len() && (entries.len() > line.len() || other < at);
            parts = parts.max(shared);
        }
        let Some(leaf) = line.last().and_then(|entry| entry.id) else {
            continue;
        };
        if continued {
            continue;
        }
        let Some(mut own) = conversation.line_conversation(*id, line.len()) else {
            continue;
        };
        // The branches kept along the path are other lines.
        let others: Vec<AsideId> = own
            .asides()
            .filter(|(_, aside)| Branch::of(aside).is_some_and(|branch| !branch.moved))
            .map(|(id, _)| id)
            .collect();
        for id in others {
            own.remove_aside(id);
        }
        let requests = |skip| own.entries().skip(skip).filter(|e| e.is_request()).count();
        found.push(Leaf {
            line: *id,
            title: title.clone(),
            leaf,
            turns: requests(0),
            own_turns: requests(parts),
            conversation: own,
        });
    }
    found.sort_by_key(|leaf| leaf.leaf);
    found
}

/// One session written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Written {
    /// Its directory.
    pub path: PathBuf,
    /// Its name, which `rig --resume` takes.
    pub name: String,
    /// Its leaf message's id.
    pub leaf: String,
    /// How many turns it holds.
    pub turns: usize,
}

/// The session read, refused unless it reads whole.
struct Source {
    store: DirStore,
    session: Session,
    main: SavedAgent,
    /// The original session's id: its directory's name.
    id: String,
    /// What its outputs are named after: its name, else its id.
    stem: String,
    meta: SessionMeta,
    /// Its working directory file, if any.
    directory: Option<Vec<u8>>,
}

fn read_source(input: &Path) -> Result<Source, Error> {
    let shown = input.display();
    if !input.is_dir() {
        return Err(Error::new(format!("{shown}: no session directory there")));
    }
    let unreadable = |why: &dyn fmt::Display| Error::new(format!("{shown}: unreadable: {why}"));
    let store = DirStore::open(input).map_err(|why| unreadable(&why))?;
    let session = Session::read(&store).map_err(|why| unreadable(&why))?;
    if let Some(repair) = session.repairs.first() {
        return Err(unreadable(repair));
    }
    let main = session.main_agent().cloned();
    let main = main.ok_or_else(|| unreadable(&"the session has no agent"))?;
    let dir = SessionDir::at(input);
    let id = input
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = dir.name().unwrap_or_else(|| id.clone());
    let meta = SessionMeta::read(&store).unwrap_or_default();
    let directory = fs::read(dir.directory()).ok();
    Ok(Source {
        store,
        session,
        main,
        id,
        stem,
        meta,
        directory,
    })
}

/// Explodes the session in `options.input` into `options.out`: the sessions
/// written, in leaf order.
///
/// # Errors
///
/// Writing nothing, when the input cannot be read whole; when an output
/// cannot be written, the outputs before it are kept and the one being
/// written is removed.
pub fn explode(options: &Options) -> Result<Vec<Written>, Error> {
    let source = read_source(&options.input)?;
    let found = leaves(&source.main.conversation);
    let total = found.len();
    let wanted: Vec<(usize, Leaf)> = (1..)
        .zip(found)
        .filter(|(_, leaf)| leaf.own_turns >= options.min_turns)
        .collect();
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let out = &options.out;
    fs::create_dir_all(out).map_err(|why| Error::new(format!("{}: {why}", out.display())))?;
    let mut taken = names_in(out);
    let mut written = Vec::new();
    for (k, leaf) in wanted {
        let name = free_name(&format!("{}-branch-{k}", source.stem), &taken);
        taken.insert(name.clone());
        let path = write_one(&source, &leaf, out, &name, (k, total))?;
        written.push(Written {
            path,
            name,
            leaf: leaf.leaf.to_string(),
            turns: leaf.turns,
        });
    }
    Ok(written)
}

/// The names of the sessions in `dir`.
fn names_in(dir: &Path) -> BTreeSet<String> {
    let entries = fs::read_dir(dir).into_iter().flatten().flatten();
    entries
        .filter_map(|entry| SessionDir::at(entry.path()).name())
        .collect()
}

/// `name`, or with a numeric suffix the first one not `taken`.
fn free_name(name: &str, taken: &BTreeSet<String>) -> String {
    let mut candidates =
        std::iter::once(name.to_owned()).chain((1..).map(|n| format!("{name}-{n}")));
    candidates
        .find(|candidate| !taken.contains(candidate))
        .unwrap_or_else(|| name.to_owned())
}

/// Writes `leaf`, number `k` of `n`, as the session `name` in a new
/// directory of `out`: built hidden beside it, renamed into place. Its
/// path.
fn write_one(
    source: &Source,
    leaf: &Leaf,
    out: &Path,
    name: &str,
    (k, n): (usize, usize),
) -> Result<PathBuf, Error> {
    let id = SessionId::generate();
    let staging = out.join(format!(".{}.session-explode", id.as_str()));
    let failed = |why: &dyn fmt::Display| {
        let _ = fs::remove_dir_all(&staging);
        Error::new(format!("{}: {why}", staging.display()))
    };
    fs::create_dir(&staging).map_err(|why| failed(&why))?;
    fill(source, leaf, &staging, name, (k, n)).map_err(|why| failed(&why))?;
    let target = free_dir(out, &id);
    // A directory renamed over one that is there would replace it when
    // empty: the check is the guard, and ids are made fresh.
    if target.exists() {
        return Err(failed(&"its session id is taken"));
    }
    fs::rename(&staging, &target).map_err(|why| failed(&why))?;
    Ok(target)
}

/// The directory of `out` for session `id`, or with a numeric suffix (as a
/// rig home's new sessions get) the first that is not there.
fn free_dir(out: &Path, id: &SessionId) -> PathBuf {
    let ids = std::iter::once(id.as_str().to_owned())
        .chain((1..100).map(|n| format!("{}-{n}", id.as_str())));
    let mut dirs = ids.map(|id| out.join(id));
    dirs.find(|dir| !dir.exists())
        .unwrap_or_else(|| out.join(id.as_str()))
}

/// Writes the session of `leaf` into the empty directory `dir`.
fn fill(
    source: &Source,
    leaf: &Leaf,
    dir: &Path,
    name: &str,
    (k, n): (usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    let store = DirStore::open(dir)?;
    let main = &source.main;
    let mut copy = Session {
        agents: vec![SavedAgent {
            id: main.id.clone(),
            parent: None,
            started_ms: main.started_ms,
            conversation: leaf.conversation.clone(),
            saved: main.saved.clone(),
            retired: false,
            rewrites: Vec::new(),
        }],
        saved: source.session.saved.clone(),
        repairs: Vec::new(),
    };
    copy.save(&store)?;
    for file in source.store.files()? {
        if file.starts_with("plugins/") {
            store.write_file(&file, &source.store.read_file(&file)?)?;
        }
    }
    let original = if source.meta.title.is_empty() {
        source.stem.as_str()
    } else {
        source.meta.title.as_str()
    };
    let mut meta = SessionMeta {
        title: format!("{original} [branch {k} of {n}]"),
        ..source.meta.clone()
    };
    meta.relist(&copy, &BTreeSet::new());
    meta.write(&store)?;
    ForkedFrom {
        session: source.id.clone(),
        before: None,
        through: Some((main.id.clone(), leaf.leaf)),
        line: Some(leaf.title.clone()),
        text: None,
    }
    .write(&store)?;
    drop(store);
    let session = SessionDir::at(dir);
    session.set_name(Some(name))?;
    if let Some(directory) = &source.directory {
        fs::write(session.directory(), directory)?;
    }
    session.mark_saved()?;
    Ok(())
}

/// Runs the program on `args` (after its name), printing to `out` one line
/// per session written, or with `--json` a JSON array of them.
///
/// # Errors
///
/// As [`explode`], or on a bad command line.
pub fn run(args: &[String], out: &mut dyn Write) -> Result<(), Error> {
    let options = Options::parse(args)?;
    let written = explode(&options)?;
    if options.json {
        serde_json::to_string_pretty(&written)
            .map_err(Error::new)
            .and_then(|json| writeln!(out, "{json}").map_err(Error::new))
    } else {
        written.iter().try_for_each(|session| {
            let path = session.path.display();
            let (name, leaf, turns) = (&session.name, &session.leaf, session.turns);
            writeln!(out, "{path}\t{name}\tleaf {leaf}\t{turns} turns").map_err(Error::new)
        })
    }
}

#[cfg(test)]
mod tests;
