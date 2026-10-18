//! Watching a workspace's files: [`Watch`], what
//! [`Workspace::watch`](crate::workspace::Workspace::watch) gives.
//! [`Local`](crate::workspace::Local) hears of changes from the system (inotify, FSEvents, …); the default
//! looks every second.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::task::{Context, Poll};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use futures::Stream;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify::{EventKind, RecursiveMode, Watcher as _};
use rig_core::tool::{ContentHash, content_hash};

use crate::MAX_FILE_BYTES;
use crate::workspace::{Listing, Stat, Workspace};

/// How often the default watch looks.
const EVERY: Duration = Duration::from_secs(1);
/// How long a batch waits for the next event before it is sent.
const QUIET: Duration = Duration::from_millis(50);
/// The longest a batch waits, however many events come.
const LONGEST: Duration = Duration::from_millis(500);

/// Batches of paths created, changed or removed under a watched root, each
/// batch in path order, as a [`Stream`]. Dropping it stops the watch.
pub struct Watch {
    /// The batches, as they come.
    batches: UnboundedReceiver<Vec<PathBuf>>,
    /// What feeds them, stopped when dropped.
    _keep: Box<dyn Send>,
}

impl Watch {
    /// A watch whose batches come on `batches`, keeping `keep` (such as the
    /// system watcher that feeds them) until it is dropped: for a
    /// workspace of your own that hears of changes its own way.
    pub fn new(batches: UnboundedReceiver<Vec<PathBuf>>, keep: impl Send + 'static) -> Self {
        Self {
            batches,
            _keep: Box::new(keep),
        }
    }
}

impl Stream for Watch {
    type Item = Vec<PathBuf>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.batches).poll_next(context)
    }
}

impl fmt::Debug for Watch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Watch")
    }
}

/// What a look saw of one file: its size and change time, and its hash
/// when the workspace cannot tell the time.
type Mark = (u64, Option<SystemTime>, Option<ContentHash>);

/// The default watch: looks at what `workspace` lists under `root` every
/// second, on a thread of its own, and sends what changed.
pub(crate) fn poll<W: Workspace + ?Sized>(
    workspace: Arc<W>,
    root: &Path,
    recursive: bool,
    listing: Listing,
) -> io::Result<Watch> {
    let (sender, batches) = unbounded();
    let root = root.to_path_buf();
    let mut before = look(&*workspace, &root, recursive, listing);
    thread::Builder::new()
        .name("rig-watch".to_owned())
        .spawn(move || {
            while !sender.is_closed() {
                thread::sleep(EVERY);
                let now = look(&*workspace, &root, recursive, listing);
                let changed = differences(&before, &now);
                before = now;
                if !changed.is_empty() && sender.unbounded_send(changed).is_err() {
                    break;
                }
            }
        })?;
    Ok(Watch::new(batches, ()))
}

/// The files under `root` that `workspace` lists with `listing`, each with
/// its [`Mark`].
fn look<W: Workspace + ?Sized>(
    workspace: &W,
    root: &Path,
    recursive: bool,
    listing: Listing,
) -> BTreeMap<PathBuf, Mark> {
    let files = workspace.files(root, None, listing).unwrap_or_default();
    files
        .into_iter()
        .filter(|path| recursive || path == root || path.parent() == Some(root))
        .filter_map(|path| match workspace.stat(&path) {
            Ok(Stat::File { size, modified, .. }) => {
                let hash = modified.is_none().then(|| {
                    let bytes = workspace.read(&path, MAX_FILE_BYTES).unwrap_or_default();
                    content_hash(&bytes.bytes)
                });
                Some((path, (size, modified, hash)))
            }
            _ => None,
        })
        .collect()
}

/// The paths created, changed or removed from `before` to `after`, in path
/// order.
fn differences(before: &BTreeMap<PathBuf, Mark>, after: &BTreeMap<PathBuf, Mark>) -> Vec<PathBuf> {
    let changed = after
        .iter()
        .filter(|(path, now)| before.get(path.as_path()) != Some(now))
        .map(|(path, _)| path);
    let removed = before
        .keys()
        .filter(|path| !after.contains_key(path.as_path()));
    let paths: BTreeSet<&PathBuf> = changed.chain(removed).collect();
    paths.into_iter().cloned().collect()
}

/// [`Local`](crate::workspace::Local)'s watch: the system's file events
/// under `root`, filtered as [`Workspace::files`] lists with `listing`,
/// and sent in batches. A `root` that is a file is watched through its
/// directory, so a write that replaces it is seen; one that does not exist
/// yet is looked at every second instead.
pub(crate) fn notified(root: &Path, recursive: bool, listing: Listing) -> io::Result<Watch> {
    let Ok(meta) = fs::metadata(root) else {
        return poll(Arc::new(crate::workspace::Local), root, recursive, listing);
    };
    let (events, raw) = std::sync::mpsc::channel::<Raw>();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        let Ok(event) = event else {
            return;
        };
        if matches!(event.kind, EventKind::Access(_)) {
            return;
        }
        let created = matches!(event.kind, EventKind::Create(_));
        for path in event.paths {
            events.send(Raw { path, created }).ok();
        }
    })
    .map_err(io::Error::other)?;
    let file = (!meta.is_dir()).then(|| root.to_path_buf());
    let (target, mode) = match file.as_deref().and_then(Path::parent) {
        Some(directory) => (directory, RecursiveMode::NonRecursive),
        None if recursive => (root, RecursiveMode::Recursive),
        None => (root, RecursiveMode::NonRecursive),
    };
    watcher.watch(target, mode).map_err(io::Error::other)?;
    let filter = Filter {
        root: root.to_path_buf(),
        file,
        recursive,
        listing,
        ignores: Ignores::new(root),
    };
    let (sender, batches) = unbounded();
    thread::Builder::new()
        .name("rig-watch".to_owned())
        .spawn(move || send_batches(&raw, &sender, filter))?;
    Ok(Watch::new(batches, watcher))
}

/// A path the system's watcher named, and whether it was created.
pub(crate) struct Raw {
    pub(crate) path: PathBuf,
    pub(crate) created: bool,
}

/// Every file under `dir`, when it is a directory, into `files`; symbolic
/// links are not followed.
fn files_in(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => files_in(&path, files),
            Ok(kind) if kind.is_file() => files.push(path),
            _ => {}
        }
    }
}

/// Gathers the paths `raw` names into batches, each sent once no event came
/// for [`QUIET`] or it waited [`LONGEST`], until the watcher or the
/// [`Watch`] is gone. A directory created under a recursive watch is
/// watched only after the watcher has told of it, so the files already in
/// it when its batch ends, by then watched, join the batch.
fn send_batches(raw: &Receiver<Raw>, sender: &UnboundedSender<Vec<PathBuf>>, mut filter: Filter) {
    while let Ok(first) = raw.recv() {
        let mut paths = BTreeSet::new();
        let mut directories = Vec::new();
        let deadline = Instant::now() + LONGEST;
        let mut next = Ok(first);
        while let Ok(Raw { path, created }) = next {
            if created && filter.recursive && filter.file.is_none() && path.is_dir() {
                directories.push(path);
            } else if filter.keeps(&path) {
                paths.insert(path);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            next = match left.is_zero() {
                true => Err(RecvTimeoutError::Timeout),
                false => raw.recv_timeout(QUIET.min(left)),
            };
        }
        let mut found = Vec::new();
        for directory in &directories {
            files_in(directory, &mut found);
        }
        paths.extend(found.into_iter().filter(|path| filter.keeps(path)));
        if !paths.is_empty() && sender.unbounded_send(paths.into_iter().collect()).is_err() {
            return;
        }
        if matches!(next, Err(RecvTimeoutError::Disconnected)) {
            return;
        }
    }
}

/// Which event paths a watch passes on.
struct Filter {
    /// The watched root.
    root: PathBuf,
    /// The watched file, when the root is one.
    file: Option<PathBuf>,
    /// Whether directories under the root count.
    recursive: bool,
    /// Which files count.
    listing: Listing,
    /// The ignore rules, read as they are needed.
    ignores: Ignores,
}

impl Filter {
    /// Whether `path` is one the watch reports: under the root (one level
    /// down unless recursive), no directory, and a file
    /// [`Workspace::files`] would list with the watch's listing.
    fn keeps(&mut self, path: &Path) -> bool {
        if let Some(file) = &self.file {
            return path == file;
        }
        let Ok(below) = path.strip_prefix(&self.root) else {
            return false;
        };
        if path
            .file_name()
            .is_some_and(|name| name == ".gitignore" || name == ".ignore")
        {
            self.ignores.forget(path.parent());
        }
        if below.as_os_str().is_empty()
            || (!self.recursive && below.components().count() > 1)
            || fs::metadata(path).is_ok_and(|meta| meta.is_dir())
        {
            return false;
        }
        self.listing == Listing::All || (self.listing.keeps(below) && !self.ignores.ignored(path))
    }
}

/// The ignore rules [`Workspace::files`] follows on this machine (each
/// directory's `.ignore`, and in a git work tree its `.gitignore` files,
/// `.git/info/exclude` and the user's global excludes), read per directory
/// when first needed.
struct Ignores {
    /// The highest directory whose rules count: the work tree's top, or
    /// the root outside one.
    top: PathBuf,
    /// The watched root: the rules decide about what is below it, never
    /// about the root itself or a directory it is in.
    root: PathBuf,
    /// Whether the root is in a git work tree.
    git: bool,
    /// The repository's and the user's excludes, the weakest rules.
    excludes: Vec<Gitignore>,
    /// Each directory's own rules.
    directories: HashMap<PathBuf, Gitignore>,
}

impl Ignores {
    /// The rules for a watch of `root`.
    fn new(root: &Path) -> Self {
        let top = root.ancestors().find(|dir| dir.join(".git").exists());
        let mut excludes = Vec::new();
        if let Some(top) = top {
            excludes.push(Gitignore::new(top.join(".git/info/exclude")).0);
            excludes.push(Gitignore::global().0);
        }
        Self {
            top: top.unwrap_or(root).to_path_buf(),
            root: root.to_path_buf(),
            git: top.is_some(),
            excludes,
            directories: HashMap::new(),
        }
    }

    /// Reads `directory`'s rules again when next needed.
    fn forget(&mut self, directory: Option<&Path>) {
        if let Some(directory) = directory {
            self.directories.remove(directory);
        }
    }

    /// Whether `path`, a file, or a directory it is in below the root, is
    /// ignored.
    fn ignored(&mut self, path: &Path) -> bool {
        let Ok(below) = path.strip_prefix(&self.top) else {
            return false;
        };
        let parts = below.components().count();
        let above = self
            .root
            .strip_prefix(&self.top)
            .map_or(0, |root| root.components().count());
        let mut at = self.top.clone();
        for (index, part) in below.components().enumerate() {
            let directory = at.clone();
            at.push(part);
            if index >= above && self.decides(&directory, &at, index + 1 < parts) {
                return true;
            }
        }
        false
    }

    /// Whether `candidate`, an entry of `directory`, is ignored by the
    /// rules of `directory` or a directory above it: the deepest rule that
    /// matches decides.
    fn decides(&mut self, directory: &Path, candidate: &Path, is_dir: bool) -> bool {
        for dir in directory.ancestors() {
            let rules = self.rules(dir);
            let below = candidate.strip_prefix(dir).unwrap_or(candidate);
            match rules.matched(below, is_dir) {
                Match::Ignore(_) => return true,
                Match::Whitelist(_) => return false,
                Match::None => {}
            }
            if dir == self.top {
                break;
            }
        }
        let below = candidate.strip_prefix(&self.top).unwrap_or(candidate);
        self.excludes
            .iter()
            .map(|rules| rules.matched(below, is_dir))
            .find(|found| !found.is_none())
            .is_some_and(|found| found.is_ignore())
    }

    /// `dir`'s own rules, read once.
    fn rules(&mut self, dir: &Path) -> &Gitignore {
        let git = self.git;
        self.directories
            .entry(dir.to_path_buf())
            .or_insert_with(|| {
                // The later file wins where both match: `.ignore` over `.gitignore`.
                let mut builder = GitignoreBuilder::new(dir);
                if git {
                    builder.add(dir.join(".gitignore"));
                }
                builder.add(dir.join(".ignore"));
                builder.build().unwrap_or_else(|_| Gitignore::empty())
            })
    }
}

#[cfg(test)]
mod tests;
