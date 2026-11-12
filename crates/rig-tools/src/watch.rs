//! Watching a workspace's files: [`Watch`], what
//! [`Workspace::watch`](crate::workspace::Workspace::watch) gives, a stream
//! of [`WatchEvent`]s. Both ways of watching keep the files they last saw,
//! each with its size and change time, and compare a new listing with it:
//! the default lists every second; [`Local`](crate::workspace::Local)
//! hears of changes from the system (inotify, FSEvents, …) and lists again
//! only where they were, with the same walk [`Workspace::files`] lists
//! with.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, ErrorKind};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::task::{Context, Poll};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use futures::Stream;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher as _};
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
/// changes, and [`Closed`](Self::Closed) last. A watch whose first listing
/// fails tells [`Unavailable`](Self::Unavailable) and then `Closed`.
#[derive(Clone, Debug)]
pub enum WatchEvent {
    /// The files under the root when the watch first listed them, in path
    /// order. A root that does not exist has none.
    Snapshot(Vec<PathBuf>),
    /// The files created, changed or removed since the last event, in path
    /// order.
    Changed(Vec<PathBuf>),
    /// Listing the files failed, or the system's watch of a directory put
    /// in the root's place could not be armed, so what is there is not
    /// known. Unless `Closed` follows, the watch goes on: the next listing
    /// that works is compared with the last one that did.
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

    /// Whether it says what the files are: a snapshot or a change. A
    /// failed listing or the end leaves them as they were last told, so
    /// one who reads the files again on news waits for this.
    pub fn names_files(&self) -> bool {
        matches!(self, Self::Snapshot(_) | Self::Changed(_))
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
    /// workspace of your own that hears of changes its own way. Send the
    /// [`WatchEvent::Snapshot`] first, listed once the changes are heard,
    /// so none made in between is missed, and each `Changed` after it; or
    /// `Unavailable` and end when the first listing fails. Once `events`
    /// ends, the watch tells [`WatchEvent::Closed`].
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

/// What a watch compares a file by: its size and change time, and its
/// hash when the workspace cannot tell the time. Both ways of watching
/// compare by it, so a file is changed when they see it changed: also one
/// moved in for another of the same name, or edited while the system's
/// events about it went unheard.
type Mark = (u64, Option<SystemTime>, Option<ContentHash>);

/// What one listing found under the paths it was asked about: each file
/// with its [`Mark`], and the paths it could not look into.
struct Look {
    files: BTreeMap<PathBuf, Mark>,
    failed: Vec<PathBuf>,
}

impl Look {
    /// The listing `listed` of the watch of `root` in `workspace`, each
    /// file kept by `recursive` with its mark: a file gone since is left
    /// out, and one whose mark could not be read is unknown. A listing
    /// that could not look into the root itself failed.
    fn of<W: Workspace + ?Sized>(
        workspace: &W,
        listed: Listed,
        root: &Path,
        recursive: bool,
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
            match mark(workspace, &path) {
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

/// The [`Mark`] of the file `path` in `workspace`; none when it is no
/// file, or gone.
fn mark<W: Workspace + ?Sized>(workspace: &W, path: &Path) -> io::Result<Option<Mark>> {
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
}

/// The paths a listing covers: each one and everything below it, none
/// below another.
#[derive(Clone, Default)]
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

/// The files a watch last saw, each with its [`Mark`], and whether its
/// last listing failed.
struct Baseline {
    files: Option<BTreeMap<PathBuf, Mark>>,
    failing: bool,
}

impl Baseline {
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

    /// Whether the watch ends: its first listing failed.
    fn ended(&self) -> bool {
        self.files.is_none() && self.failing
    }

    /// Whether a file it holds is below `dir`.
    fn holds_below(&self, dir: &Path) -> bool {
        let from: (Bound<&Path>, Bound<&Path>) = (Bound::Excluded(dir), Bound::Unbounded);
        (self.files.iter())
            .flat_map(|files| files.range::<Path, _>(from).next())
            .any(|(path, _)| path.starts_with(dir))
    }

    /// Takes `look`, the listing of what is under `scopes`, and says what
    /// changed: the first listing is a snapshot; a later one names the
    /// files it shows that this did not hold, held with another mark, or
    /// that `touched` names, and those under `scopes` it no longer shows,
    /// unless under a path it could not look into, whose files keep their
    /// marks until a listing sees them again. A failed listing changes
    /// nothing and is told once until one works again; when the first one
    /// fails, the watch [`ended`](Self::ended).
    fn update(
        &mut self,
        scopes: &Scopes,
        look: io::Result<Look>,
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
                if event.is_some_and(|event| sender.unbounded_send(event).is_err()) || seen.ended()
                {
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
) -> io::Result<Look> {
    let listed = match workspace.files(root, None, listing) {
        Err(error) if error.kind() == ErrorKind::NotFound => Listed::default(),
        listed => listed?,
    };
    Look::of(workspace, listed, root, recursive)
}

/// [`Local`](crate::workspace::Local)'s watch of `root`: the system's file
/// events mark where to list again, and those places are listed with the
/// walk [`Workspace::files`] lists with, so it reports what the default
/// watch would, at once. A `root` that is a file is watched through its
/// directory, so a write that replaces it is seen; one that does not exist
/// yet is looked at every second instead. A directory put in place of the
/// one watched, once that is moved or removed, is watched in its turn,
/// looked for every second while none is there; on systems other than
/// Unix only one put in place after none was there. A root that cannot be
/// looked at for a while is [`Unavailable`](WatchEvent::Unavailable),
/// listed again every second. A relative `root` that
/// exists is resolved against the current directory once, so the paths
/// told are absolute.
///
/// # Errors
///
/// When `root` cannot be looked at, other than for not being there, or
/// the system's watch cannot be armed.
pub(crate) fn notified(root: &Path, recursive: bool, listing: Listing) -> io::Result<Watch> {
    let meta = match fs::metadata(root) {
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return poll(Arc::new(crate::workspace::Local), root, recursive, listing);
        }
        meta => meta?,
    };
    // The system's events name absolute paths, which a relative root never matches.
    let root = std::path::absolute(root)?;
    let tree = Tree {
        root,
        recursive: recursive && meta.is_dir(),
        listing,
    };
    let (heard, raw) = std::sync::mpsc::channel::<Heard>();
    let watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
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
    let watcher = Arc::new(Mutex::new(watcher));
    let mut armed = Armed::new(Arc::downgrade(&watcher), target.to_path_buf(), mode)?;
    {
        let mut watcher = watcher.lock().unwrap_or_else(PoisonError::into_inner);
        watcher.watch(target, mode).map_err(io::Error::other)?;
        for rules in tree.rules_outside(&tree.wide_rules()) {
            // Missing one leaves its rules' edits unheard, not the files.
            watcher.watch(&rules, RecursiveMode::NonRecursive).ok();
        }
    }
    let (sender, events) = unbounded();
    thread::Builder::new()
        .name("rig-watch".to_owned())
        .spawn(move || send_batches(&raw, &sender, &tree, &mut armed))?;
    Ok(Watch::new(events, watcher))
}

/// Which directory a path is: where the system tells it, its device,
/// inode and birth time, so one put in its place is another even when it
/// reuses the inode.
#[cfg(unix)]
type Identity = (u64, u64, Option<std::time::SystemTime>);
#[cfg(not(unix))]
type Identity = ();

/// The [`Identity`] of what is at `path`; none when nothing is.
fn identity(path: &Path) -> io::Result<Option<Identity>> {
    match fs::metadata(path) {
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        meta => Ok(Some(identity_of(&meta?))),
    }
}

#[cfg(unix)]
fn identity_of(meta: &fs::Metadata) -> Identity {
    use std::os::unix::fs::MetadataExt as _;
    (meta.dev(), meta.ino(), meta.created().ok())
}

#[cfg(not(unix))]
fn identity_of(_: &fs::Metadata) -> Identity {}

/// The system's watch of a directory, which watches that directory and not
/// its path: one moved or made in its place is unheard until armed again.
pub(crate) struct Armed {
    /// The watcher, kept by the [`Watch`]: gone once the watch is.
    watcher: Weak<Mutex<RecommendedWatcher>>,
    target: PathBuf,
    mode: RecursiveMode,
    /// The directory watched; none while nothing is at `target`.
    at: Option<Identity>,
}

impl Armed {
    /// The watch of `target` with `mode` by `watcher`, of what is at
    /// `target` now, to be armed by the caller.
    ///
    /// # Errors
    ///
    /// When `target` cannot be looked at, other than for not being there.
    pub(crate) fn new(
        watcher: Weak<Mutex<RecommendedWatcher>>,
        target: PathBuf,
        mode: RecursiveMode,
    ) -> io::Result<Self> {
        let at = identity(&target)?;
        Ok(Self {
            watcher,
            target,
            mode,
            at,
        })
    }

    /// Whether a directory is watched; when not, it is looked for.
    fn watching(&self) -> bool {
        self.at.is_some()
    }

    /// Watches what is at the target now, if that is not what it watched:
    /// true when it changed, or cannot be looked at, so what was heard
    /// meanwhile may be missing.
    ///
    /// # Errors
    ///
    /// When a directory is at the target and cannot be watched.
    fn follow(&mut self) -> io::Result<bool> {
        let Ok(now) = identity(&self.target) else {
            return Ok(true);
        };
        if now == self.at {
            return Ok(false);
        }
        let Some(watcher) = self.watcher.upgrade() else {
            return Ok(false);
        };
        let mut watcher = watcher.lock().unwrap_or_else(PoisonError::into_inner);
        // The old directory's watch is gone with it, or watches a path no
        // longer the target's.
        watcher.unwatch(&self.target).ok();
        self.at = now;
        if now.is_some()
            && let Err(error) = watcher.watch(&self.target, self.mode)
        {
            // Removed again since, or not to be looked at: looked for again.
            self.at = identity(&self.target).ok().flatten();
            if self.at.is_some() {
                return Err(io::Error::other(error));
            }
        }
        Ok(true)
    }
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
    /// `.git` there or the excludes in it the listing of its work tree, one
    /// of `wide` (the user's global excludes, the root's work tree's
    /// excludes) everything; else `path` itself. None when it is no path
    /// the watch lists, nor one above the root.
    fn scope(&self, path: &Path, wide: &[PathBuf]) -> Option<PathBuf> {
        let name = path.file_name()?;
        let parent = path.parent()?;
        let dir = if wide.iter().any(|rules| rules == path) {
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

    /// The top of the git work tree the root is in, as the walk finds it:
    /// the nearest directory with a `.git`.
    fn top(&self) -> Option<&Path> {
        self.root.ancestors().find(|dir| dir.join(".git").exists())
    }

    /// The files of rules for the whole root, in a git work tree only, as
    /// the walk reads them: the user's global excludes, and the excludes of
    /// the work tree. A linked work tree or a submodule has a `.git` file
    /// naming its git directory; one whose `commondir` names another keeps
    /// its excludes there.
    fn wide_rules(&self) -> Vec<PathBuf> {
        let Some(top) = self.top() else {
            return Vec::new();
        };
        let mut wide: Vec<PathBuf> = ignore::gitignore::gitconfig_excludes_path()
            .into_iter()
            .collect();
        // The system's events name these without the `..` they may hold.
        let dot_git = top.join(".git");
        let git_dir = match fs::read_to_string(&dot_git) {
            Ok(text) => match text.trim_end().strip_prefix("gitdir: ") {
                Some(dir) => fs::canonicalize(top.join(dir)),
                None => return wide,
            },
            Err(_) => Ok(dot_git),
        };
        let common =
            git_dir.and_then(
                |git_dir| match fs::read_to_string(git_dir.join("commondir")) {
                    Ok(text) => fs::canonicalize(git_dir.join(text.trim_end())),
                    Err(_) => Ok(git_dir),
                },
            );
        wide.extend(common.map(|common| common.join("info/exclude")));
        wide
    }

    /// The directories outside the system's watch whose files hold rules
    /// for the root: those from the root's up to the top of its git work
    /// tree, and those holding `wide`. Outside a work tree only the root's
    /// own rules are heard.
    fn rules_outside(&self, wide: &[PathBuf]) -> Vec<PathBuf> {
        let watched =
            |dir: &Path| dir.starts_with(&self.root) && (self.recursive || dir == self.root);
        let above = self.top().into_iter().flat_map(|top| {
            (self.root.ancestors().skip(1)).take_while(move |dir| dir.starts_with(top))
        });
        let mut dirs: Vec<PathBuf> = above.map(Path::to_path_buf).collect();
        dirs.extend(
            wide.iter()
                .filter_map(|rules| rules.parent())
                .map(Path::to_path_buf),
        );
        dirs.retain(|dir| !watched(dir) && dir.is_dir());
        dirs
    }

    /// The files under `scopes` that [`Workspace::files`] would list for
    /// the root. Each scope is walked from the nearest directory above it
    /// that `visible` says the root's listing goes into, else the root, so
    /// the walk reads the same rules the root's would, and none of the
    /// directories above that one, however large.
    fn look(&self, scopes: Scopes, visible: impl Fn(&Path) -> bool) -> io::Result<Look> {
        let mut starts: BTreeMap<PathBuf, Scopes> = BTreeMap::new();
        for scope in scopes.0 {
            let start = scope
                .ancestors()
                .skip(1)
                .take_while(|dir| dir.starts_with(&self.root) && *dir != self.root)
                .find(|dir| visible(dir))
                .unwrap_or(&self.root)
                .to_path_buf();
            starts.entry(start).or_default().add(scope);
        }
        let mut found = Listed::default();
        for (start, within) in starts {
            let within = Arc::new(within);
            let mut walk =
                crate::workspace::walk(&start, self.listing, move |path| within.leads_to(path));
            if !self.recursive {
                walk.max_depth(Some(1));
            }
            let listed = crate::workspace::listed(&walk);
            found.files.extend(listed.files);
            found.failed.extend(listed.failed);
        }
        Look::of(&crate::workspace::Local, found, &self.root, true)
    }
}

/// Gathers what `raw` tells into batches, each ending once nothing came
/// for [`QUIET`] or it waited [`LONGEST`], lists again where the batch
/// says, and sends what changed, until the watcher or the [`Watch`] is
/// gone. The first listing, of everything, is made once the system's
/// watch is armed, so nothing made in between is missed; the watch ends
/// when it fails. After a later one that failed, everything is listed
/// again, every second until one works; paths a listing could not look
/// into are listed again every second the same way. When another directory is at the
/// target of `armed`, it is watched and everything listed again; while
/// none is there, it is looked for every second. The watch ends when that
/// cannot be done.
pub(crate) fn send_batches(
    raw: &Receiver<Heard>,
    sender: &UnboundedSender<WatchEvent>,
    tree: &Tree,
    armed: &mut Armed,
) {
    let wide = tree.wide_rules();
    let mut seen = Baseline::new();
    let mut heard = Vec::new();
    // Arming a recursive watch walks the tree without reading the events
    // that walk makes, so the system may drop some and ask for a rescan
    // soon after. The first listing covers what came before it.
    let deadline = Instant::now() + LONGEST;
    while let Some(left) = deadline.checked_duration_since(Instant::now())
        && raw.recv_timeout(QUIET.min(left)).is_ok()
    {}
    let mut failed = Scopes::default();
    loop {
        let (mut scopes, mut touched) = (std::mem::take(&mut failed), BTreeSet::new());
        let moved = match armed.follow() {
            Ok(moved) => moved,
            Err(error) => {
                sender
                    .unbounded_send(WatchEvent::Unavailable(Arc::new(error)))
                    .ok();
                return;
            }
        };
        if moved || !seen.known() {
            scopes = Scopes::all(&tree.root);
        }
        for told in heard.drain(..) {
            match told {
                Heard::Everything => scopes.add(tree.root.clone()),
                Heard::Paths(paths) => {
                    for path in paths {
                        if let Some(scope) = tree.scope(&path, &wide) {
                            scopes.add(scope);
                        }
                        touched.insert(path);
                    }
                }
            }
        }
        if !scopes.0.is_empty() {
            // A directory the last listing found files under is listed by
            // the rules as they are: an edit of them is a scope of its own.
            let look = tree.look(scopes.clone(), |dir| seen.holds_below(dir));
            if let Ok(look) = &look {
                for path in &look.failed {
                    failed.add(path.clone());
                }
            }
            let event = seen.update(&scopes, look, &touched);
            if event.is_some_and(|event| sender.unbounded_send(event).is_err()) || seen.ended() {
                return;
            }
        }
        // A listing that failed, in whole or in part, or a target not
        // there, is tried again every second.
        let first = match seen.known() && armed.watching() && failed.0.is_empty() {
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
