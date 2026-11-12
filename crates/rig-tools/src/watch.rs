//! Watching a workspace's files: [`Watch`], what
//! [`Workspace::watch`](crate::workspace::Workspace::watch) gives, a stream
//! of [`WatchEvent`]s. Both ways of watching keep the files they last saw
//! and compare a new listing with it: the default lists every second;
//! [`Local`](crate::workspace::Local) hears of changes from the system
//! (inotify, FSEvents, …) and lists again only where they were, with the
//! same walk [`Workspace::files`] lists with.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, ErrorKind};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::task::{Context, Poll};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use futures::Stream;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use notify::{EventKind, RecursiveMode, Watcher as _};
use rig_core::tool::{ContentHash, content_hash};

use crate::MAX_FILE_BYTES;
use crate::workspace::{Listed, Listing, Stat, Workspace};

/// How often the default watch looks.
const EVERY: Duration = Duration::from_secs(1);
/// How long a batch waits for the next event before it is sent.
const QUIET: Duration = Duration::from_millis(50);
/// The longest a batch waits, however many events come.
const LONGEST: Duration = Duration::from_millis(500);

/// What a [`Watch`] tells, in order: one [`Snapshot`](Self::Snapshot)
/// once it could list, then a [`Changed`](Self::Changed) for each batch of
/// changes, and [`Closed`](Self::Closed) last.
#[derive(Clone, Debug)]
pub enum WatchEvent {
    /// The files under the root when the watch first listed them, in path
    /// order. A root that does not exist has none.
    Snapshot(Vec<PathBuf>),
    /// The files created, changed or removed since the last event, in path
    /// order.
    Changed(Vec<PathBuf>),
    /// Listing the files failed, so what is there is not known. The watch
    /// goes on: the next listing that works is compared with the last one
    /// that did.
    Unavailable(Arc<io::Error>),
    /// The watch stopped; nothing comes after it.
    Closed,
}

impl WatchEvent {
    /// The paths a snapshot or a change names; none for the others.
    pub fn paths(&self) -> &[PathBuf] {
        match self {
            Self::Snapshot(paths) | Self::Changed(paths) => paths,
            Self::Unavailable(_) | Self::Closed => &[],
        }
    }
}

/// The [`WatchEvent`]s of a watched root, as a [`Stream`] that ends after
/// [`WatchEvent::Closed`]. Dropping it stops the watch.
pub struct Watch {
    /// The events, as they come.
    events: UnboundedReceiver<WatchEvent>,
    /// Whether `Closed` was passed on.
    closed: bool,
    /// What feeds them, stopped when dropped.
    _keep: Box<dyn Send>,
}

impl Watch {
    /// A watch whose events come on `events`, keeping `keep` (such as the
    /// system watcher that feeds them) until it is dropped: for a
    /// workspace of your own that hears of changes its own way. Once
    /// `events` ends, the watch tells [`WatchEvent::Closed`].
    pub fn new(events: UnboundedReceiver<WatchEvent>, keep: impl Send + 'static) -> Self {
        Self {
            events,
            closed: false,
            _keep: Box::new(keep),
        }
    }
}

impl Stream for Watch {
    type Item = WatchEvent;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.closed {
            return Poll::Ready(None);
        }
        let next = Pin::new(&mut self.events).poll_next(context);
        if let Poll::Ready(None | Some(WatchEvent::Closed)) = next {
            self.closed = true;
            return Poll::Ready(Some(WatchEvent::Closed));
        }
        next
    }
}

impl fmt::Debug for Watch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Watch")
    }
}

/// What one listing found under the paths it was asked about: each file
/// with what it is compared by, and the paths it could not look into.
struct Look<M> {
    files: BTreeMap<PathBuf, M>,
    failed: Vec<PathBuf>,
}

impl<M> Look<M> {
    /// The listing `listed` of the watch of `root`, each file kept by
    /// `recursive` with its mark from `mark`: `Ok(None)` leaves the file
    /// out, as gone, and an error makes it unknown. A listing that could
    /// not look into the root itself failed.
    fn of(
        listed: Listed,
        root: &Path,
        recursive: bool,
        mut mark: impl FnMut(&Path) -> io::Result<Option<M>>,
    ) -> io::Result<Self> {
        let mut failed = Vec::new();
        for (path, error) in listed.failed {
            if root.starts_with(&path) {
                return Err(error);
            }
            failed.push(path);
        }
        let mut files = BTreeMap::new();
        let watched = listed
            .files
            .into_iter()
            .filter(|path| recursive || path == root || path.parent() == Some(root));
        for path in watched {
            match mark(&path) {
                Ok(Some(mark)) => {
                    files.insert(path, mark);
                }
                Ok(None) => {}
                Err(_) => failed.push(path),
            }
        }
        Ok(Self { files, failed })
    }
}

/// The paths a listing covers: each one and everything below it, none
/// below another.
#[derive(Default)]
struct Scopes(BTreeSet<PathBuf>);

impl Scopes {
    /// Every path under `root`.
    fn all(root: &Path) -> Self {
        Self(BTreeSet::from([root.to_path_buf()]))
    }

    /// Adds `path` and what is below it.
    fn add(&mut self, path: PathBuf) {
        if self.covers(&path) {
            return;
        }
        let below: Vec<PathBuf> = self.below(&path).cloned().collect();
        for inner in below {
            self.0.remove(&inner);
        }
        self.0.insert(path);
    }

    /// Whether `path` is one of the paths or below one.
    fn covers(&self, path: &Path) -> bool {
        path.ancestors().any(|above| self.0.contains(above))
    }

    /// Whether a walk goes into `path`: it is covered, or a covered path is
    /// below it.
    fn leads_to(&self, path: &Path) -> bool {
        self.covers(path) || self.below(path).next().is_some()
    }

    /// The paths at or below `path`, which sort right after it.
    fn below<'a>(&'a self, path: &'a Path) -> impl Iterator<Item = &'a PathBuf> {
        let from: (Bound<&Path>, Bound<&Path>) = (Bound::Included(path), Bound::Unbounded);
        self.0
            .range::<Path, _>(from)
            .take_while(move |inner| inner.starts_with(path))
    }
}

/// The files a watch last saw, with what each is compared by, and whether
/// its last listing failed.
struct Baseline<M> {
    files: Option<BTreeMap<PathBuf, M>>,
    failing: bool,
}

impl<M: PartialEq> Baseline<M> {
    /// Nothing seen yet.
    fn new() -> Self {
        Self {
            files: None,
            failing: false,
        }
    }

    /// Whether a listing of some paths can be compared with it: the watch
    /// listed once, and its last listing worked.
    fn known(&self) -> bool {
        self.files.is_some() && !self.failing
    }

    /// Takes `look`, the listing of what is under `scopes`, and says what
    /// changed: the first listing that works is a snapshot; a later one
    /// names the files it shows that this did not hold, held otherwise, or
    /// that `touched` names, and those under `scopes` it no longer shows,
    /// unless under a path it could not look into. A failed listing
    /// changes nothing and is told once until one works again.
    fn update(
        &mut self,
        scopes: &Scopes,
        look: io::Result<Look<M>>,
        touched: &BTreeSet<PathBuf>,
    ) -> Option<WatchEvent> {
        let look = match look {
            Ok(look) => look,
            Err(error) => {
                let told = std::mem::replace(&mut self.failing, true);
                return (!told).then(|| WatchEvent::Unavailable(Arc::new(error)));
            }
        };
        self.failing = false;
        let Some(files) = self.files.as_mut() else {
            let paths = look.files.keys().cloned().collect();
            self.files = Some(look.files);
            return Some(WatchEvent::Snapshot(paths));
        };
        let unknown = |path: &Path| look.failed.iter().any(|failed| path.starts_with(failed));
        let mut changed = BTreeSet::new();
        for scope in &scopes.0 {
            let from: (Bound<&Path>, Bound<&Path>) = (Bound::Included(scope), Bound::Unbounded);
            let gone: Vec<PathBuf> = files
                .range::<Path, _>(from)
                .map(|(path, _)| path)
                .take_while(|path| path.starts_with(scope))
                .filter(|path| !look.files.contains_key(*path) && !unknown(path))
                .cloned()
                .collect();
            for path in gone {
                files.remove(&path);
                changed.insert(path);
            }
        }
        for (path, mark) in look.files {
            if files.get(&path) != Some(&mark) || touched.contains(&path) {
                changed.insert(path.clone());
            }
            files.insert(path, mark);
        }
        (!changed.is_empty()).then(|| WatchEvent::Changed(changed.into_iter().collect()))
    }
}

/// What the default watch compares a file by: its size and change time,
/// and its hash when the workspace cannot tell the time.
type Mark = (u64, Option<SystemTime>, Option<ContentHash>);

/// The default watch: lists what `workspace` lists under `root` at once
/// and then every second, on a thread of its own, and tells what changed.
pub(crate) fn poll<W: Workspace + ?Sized>(
    workspace: Arc<W>,
    root: &Path,
    recursive: bool,
    listing: Listing,
) -> io::Result<Watch> {
    poll_every(workspace, root, recursive, listing, EVERY)
}

/// [`poll`], listing every `every`.
fn poll_every<W: Workspace + ?Sized>(
    workspace: Arc<W>,
    root: &Path,
    recursive: bool,
    listing: Listing,
    every: Duration,
) -> io::Result<Watch> {
    let (sender, events) = unbounded();
    let root = root.to_path_buf();
    thread::Builder::new()
        .name("rig-watch".to_owned())
        .spawn(move || {
            let (scopes, mut seen) = (Scopes::all(&root), Baseline::new());
            let none = BTreeSet::new();
            while !sender.is_closed() {
                let look = look(&*workspace, &root, recursive, listing);
                let event = seen.update(&scopes, look, &none);
                if event.is_some_and(|event| sender.unbounded_send(event).is_err()) {
                    break;
                }
                thread::sleep(every);
            }
        })?;
    Ok(Watch::new(events, ()))
}

/// What `workspace` lists under `root` with `listing`, each file with its
/// [`Mark`]. A root that is not there has no files.
fn look<W: Workspace + ?Sized>(
    workspace: &W,
    root: &Path,
    recursive: bool,
    listing: Listing,
) -> io::Result<Look<Mark>> {
    let listed = match workspace.files(root, None, listing) {
        Err(error) if error.kind() == ErrorKind::NotFound => Listed::default(),
        listed => listed?,
    };
    Look::of(listed, root, recursive, |path| {
        match workspace.stat(path)? {
            Stat::File { size, modified, .. } => {
                let hash = match modified {
                    Some(_) => None,
                    None => Some(content_hash(&workspace.read(path, MAX_FILE_BYTES)?.bytes)),
                };
                Ok(Some((size, modified, hash)))
            }
            Stat::Dir | Stat::Other | Stat::Missing => Ok(None),
        }
    })
}

/// [`Local`](crate::workspace::Local)'s watch of `root`: the system's file
/// events mark where to list again, and those places are listed with the
/// walk [`Workspace::files`] lists with, so it reports what the default
/// watch would, at once. A `root` that is a file is watched through its
/// directory, so a write that replaces it is seen; one that does not exist
/// yet is looked at every second instead. A relative `root` that exists is
/// resolved against the current directory once, so the paths told are
/// absolute.
pub(crate) fn notified(root: &Path, recursive: bool, listing: Listing) -> io::Result<Watch> {
    let Ok(meta) = fs::metadata(root) else {
        return poll(Arc::new(crate::workspace::Local), root, recursive, listing);
    };
    // The system's events name absolute paths, which a relative root never matches.
    let root = std::path::absolute(root)?;
    let tree = Tree {
        root,
        recursive: recursive && meta.is_dir(),
        listing,
    };
    let (heard, raw) = std::sync::mpsc::channel::<Heard>();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        let heard_now = match event {
            Ok(event) if matches!(event.kind, EventKind::Access(_)) => return,
            Ok(event) if !event.need_rescan() => Heard::Paths(event.paths),
            _ => Heard::Everything,
        };
        heard.send(heard_now).ok();
    })
    .map_err(io::Error::other)?;
    let (target, mode) = match meta.is_dir() {
        false => (
            tree.root.parent().unwrap_or(&tree.root),
            RecursiveMode::NonRecursive,
        ),
        true if tree.recursive => (tree.root.as_path(), RecursiveMode::Recursive),
        true => (tree.root.as_path(), RecursiveMode::NonRecursive),
    };
    watcher.watch(target, mode).map_err(io::Error::other)?;
    for rules in tree.rules_outside() {
        // Missing one leaves its rules' edits unheard, not the files.
        watcher.watch(&rules, RecursiveMode::NonRecursive).ok();
    }
    let (sender, events) = unbounded();
    thread::Builder::new()
        .name("rig-watch".to_owned())
        .spawn(move || send_batches(&raw, &sender, &tree))?;
    Ok(Watch::new(events, watcher))
}

/// What the system's watcher told: paths that changed, or that it may have
/// missed some, so every path may have.
pub(crate) enum Heard {
    Paths(Vec<PathBuf>),
    Everything,
}

/// A watched root as [`Local`](crate::workspace::Local) lists it.
pub(crate) struct Tree {
    pub(crate) root: PathBuf,
    pub(crate) recursive: bool,
    pub(crate) listing: Listing,
}

impl Tree {
    /// What may change which files are listed under the root: an ignore
    /// file in `path`'s directory changes the listing of that directory, a
    /// `.git` there or the excludes in it the listing of its work tree, the
    /// user's global excludes everything; else `path` itself. None when it
    /// is no path the watch lists, nor one above the root.
    fn scope(&self, path: &Path, global: Option<&Path>) -> Option<PathBuf> {
        let name = path.file_name()?;
        let parent = path.parent()?;
        let dir = if Some(path) == global {
            &self.root
        } else if name == ".gitignore" || name == ".ignore" || name == ".git" {
            parent
        } else if name == "exclude"
            && parent.ends_with(".git/info")
            && let Some(top) = parent.parent().and_then(Path::parent)
        {
            top
        } else {
            path
        };
        if self.root.starts_with(dir) {
            return Some(self.root.clone());
        }
        let below = dir.strip_prefix(&self.root).ok()?;
        let near = self.recursive || below.components().count() == 1;
        (near && self.listing.keeps(below)).then(|| dir.to_path_buf())
    }

    /// The directories outside the system's watch whose files hold rules
    /// for the root: those from the root's up to the top of its git work
    /// tree, that top's `.git/info`, and the directory of the user's
    /// global excludes. Outside a work tree only the root's own rules
    /// are heard.
    fn rules_outside(&self) -> Vec<PathBuf> {
        let watched =
            |dir: &Path| dir.starts_with(&self.root) && (self.recursive || dir == self.root);
        let Some(top) = self.root.ancestors().find(|dir| dir.join(".git").exists()) else {
            return Vec::new();
        };
        let above = self
            .root
            .ancestors()
            .skip(1)
            .take_while(|dir| dir.starts_with(top));
        let global = ignore::gitignore::gitconfig_excludes_path();
        let global = global
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        let mut dirs: Vec<PathBuf> = above.map(Path::to_path_buf).collect();
        dirs.push(top.join(".git/info"));
        dirs.extend(global);
        dirs.retain(|dir| !watched(dir) && dir.is_dir());
        dirs
    }

    /// The files under `scopes` that [`Workspace::files`] would list for
    /// the root.
    fn look(&self, scopes: &Arc<Scopes>) -> io::Result<Look<()>> {
        let within = Arc::clone(scopes);
        let mut walk =
            crate::workspace::walk(&self.root, self.listing, move |path| within.leads_to(path));
        if !self.recursive {
            walk.max_depth(Some(1));
        }
        let listed = crate::workspace::listed(&walk);
        Look::of(listed, &self.root, true, |_| Ok(Some(())))
    }
}

/// Gathers what `raw` tells into batches, each ending once nothing came
/// for [`QUIET`] or it waited [`LONGEST`], lists again where the batch
/// says, and sends what changed, until the watcher or the [`Watch`] is
/// gone. The first listing, of everything, is made once the system's
/// watch is armed, so nothing made in between is missed; after one that
/// failed, everything is listed again, every second until one works.
pub(crate) fn send_batches(
    raw: &Receiver<Heard>,
    sender: &UnboundedSender<WatchEvent>,
    tree: &Tree,
) {
    let global = ignore::gitignore::gitconfig_excludes_path();
    let mut seen = Baseline::new();
    let mut heard = Vec::new();
    loop {
        let (mut scopes, mut touched) = (Scopes::default(), BTreeSet::new());
        if !seen.known() {
            scopes = Scopes::all(&tree.root);
        }
        for told in heard.drain(..) {
            match told {
                Heard::Everything => scopes.add(tree.root.clone()),
                Heard::Paths(paths) => {
                    for path in paths {
                        if let Some(scope) = tree.scope(&path, global.as_deref()) {
                            scopes.add(scope);
                        }
                        touched.insert(path);
                    }
                }
            }
        }
        if !scopes.0.is_empty() {
            let scopes = Arc::new(scopes);
            let look = tree.look(&scopes);
            let event = seen.update(&scopes, look, &touched);
            if event.is_some_and(|event| sender.unbounded_send(event).is_err()) {
                return;
            }
        }
        // A listing that failed is tried again every second.
        let first = match seen.known() {
            true => raw.recv().map_err(|_| RecvTimeoutError::Disconnected),
            false => raw.recv_timeout(EVERY),
        };
        let first = match first {
            Ok(first) => first,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        let deadline = Instant::now() + LONGEST;
        let mut next = Ok(first);
        while let Ok(told) = next {
            heard.push(told);
            let left = deadline.saturating_duration_since(Instant::now());
            next = match left.is_zero() {
                true => Err(RecvTimeoutError::Timeout),
                false => raw.recv_timeout(QUIET.min(left)),
            };
        }
        if sender.is_closed() || matches!(next, Err(RecvTimeoutError::Disconnected)) {
            return;
        }
    }
}

#[cfg(test)]
mod tests;
