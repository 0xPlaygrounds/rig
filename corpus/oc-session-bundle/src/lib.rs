//! OC-12: a portable session bundle.
//!
//! `/bundle [path] [--allow-secrets]` writes the running session to one
//! gzip-compressed tar file (`<session-name>.rigbundle` in the project by
//! default): every log, blob and file of the session's store (all agents,
//! subagents and branches, and every plugin's saved values and session
//! files, exactly as a restore reads them), the project-relative paths and
//! current contents of the files the session's tools changed, and a
//! [`Manifest`]. Secrets found by rig-core's `redact` shapes or as
//! environment-looking assignments (`AWS_SECRET_ACCESS_KEY=…`) are asked
//! about first (redact, keep or cancel); with nobody to ask the bundle is
//! refused unless `--allow-secrets`.
//!
//! `rig bundle open <file> [--project <dir>] [--replace] [--apply|--keep|--apply-file <path>…]`
//! and `/unbundle <file> …` import a bundle as a new saved session of the
//! project: every string of its logs and files that names the original
//! project root is remapped to the new one, the bundled files are written
//! as chosen when they differ on disk or the project is at another commit,
//! plugins named in the manifest that are not installed are listed (their
//! saved values stay in the logs for a later build), and an agent whose
//! model cannot be reached here falls back to the default model.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use rig_core::redact;
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::journal::SessionStore;
use rig_harness::rig_memory::{
    FORMAT, Keep, MODEL, MemoryStore, SavedAgent, Session, SessionMeta, Store, fork, now_ms,
    title_of,
};
use rig_harness::rig_tools::process::Launcher;
use rig_sessions::SessionTitle;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The command and the front that runs it.
const COMMAND: &str = "bundle";
/// A bundle's extension.
const EXTENSION: &str = "rigbundle";
/// The manifest's name in the archive.
const MANIFEST: &str = "manifest.json";
/// Where the store's logs, blobs and files go in the archive.
const LOGS: &str = "store/logs/";
const BLOBS: &str = "store/blobs/";
const FILES: &str = "store/files/";
/// Where the changed project files go in the archive.
const PROJECT: &str = "project/";
/// The plugin's file in an imported session's store: the bundle it came
/// from, for `--replace`.
const MARKER: &str = "plugins/session-bundle/imported.json";
/// The format a manifest names.
const BUNDLE_FORMAT: &str = "rigbundle/1";

/// `/bundle`, `/unbundle` and `rig bundle open`.
#[derive(Default)]
pub struct SessionBundlePlugin;

impl Plugin for SessionBundlePlugin {
    fn build(&self, app: &mut App) {
        let front = Front::new(COMMAND).sessionless();
        app.register_type::<BundleOrigin>()
            .init_resource::<BundleOrigin>()
            .init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .add_message::<BundleWritten>()
            .add_message::<BundleImported>()
            .add_front(front.on_command(
                COMMAND,
                "open <file> [--project <dir>] [--replace] [--apply|--keep|--apply-file <path>…]",
                "Import a session bundle as a new session of a project",
            ))
            .add_command(
                "bundle",
                "Write this session to a bundle: /bundle [path] [--allow-secrets]",
                bundle_command,
            )
            .add_command(
                "unbundle",
                "Import a session bundle: /unbundle <file> [--project <dir>] [--replace] [--apply|--keep]",
                unbundle_command,
            )
            .add_observer(track_changes)
            .add_systems(Startup, record_origin)
            .add_systems(Update, (answered_secrets, answered_files).after(AgentLoop));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(COMMAND) {
            app.add_systems(Startup, open_command_line);
        }
    }
}

/// Where the session started and what its tools changed, saved with the
/// session: the project root, the git commit and branch it started on,
/// and each project-relative path a tool call changed.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved, Default)]
pub struct BundleOrigin {
    /// The project root the session started in.
    pub project_root: String,
    /// The git commit it started on, if the project is a git work tree.
    pub commit: Option<String>,
    /// The git branch it started on, unless detached.
    pub branch: Option<String>,
    /// The project-relative paths the session's tools changed, sorted.
    pub changed: Vec<String>,
}

/// What a bundle says about itself.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    /// The bundle's format, `rigbundle/1`.
    pub format: String,
    /// The bundle's own id: the session and when it was written.
    pub id: String,
    /// The session log format of the bundled store, as rig-memory names it
    /// (the harness exposes no rig version).
    pub session_format: String,
    /// The bundled session's id.
    pub session: String,
    /// The bundled session's name or title.
    pub name: String,
    /// The plugins of the build that wrote it.
    pub plugins: Vec<String>,
    /// The models its agents used, as `vendor/model`.
    pub models: Vec<String>,
    /// The project root it worked in.
    pub project_root: PathBuf,
    /// When it was written, in milliseconds since the Unix epoch.
    pub created_ms: u64,
    /// The git commit the session started on.
    pub commit: Option<String>,
    /// The git branch the session started on.
    pub branch: Option<String>,
    /// The project-relative files bundled.
    pub files: Vec<String>,
}

/// A bundle in memory: the manifest, the session's store and the changed
/// project files.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bundle {
    /// What it says about itself.
    pub manifest: Manifest,
    /// Each log of the store, by name.
    pub logs: BTreeMap<String, Vec<u8>>,
    /// Each blob of the store, by name.
    pub blobs: BTreeMap<String, Vec<u8>>,
    /// Each file of the store, by name.
    pub store_files: BTreeMap<String, Vec<u8>>,
    /// Each changed project file, by project-relative path.
    pub files: BTreeMap<String, Vec<u8>>,
}

impl Bundle {
    /// Writes it to `path` as a gzip-compressed tar archive.
    ///
    /// # Errors
    ///
    /// When the file cannot be written.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let encoder = GzEncoder::new(fs::File::create(path)?, Compression::default());
        let mut archive = tar::Builder::new(encoder);
        let mut add = |name: &str, bytes: &[u8]| -> io::Result<()> {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(self.manifest.created_ms / 1000);
            archive.append_data(&mut header, name, bytes)
        };
        add(MANIFEST, &serde_json::to_vec_pretty(&self.manifest)?)?;
        let parts = [
            (LOGS, &self.logs),
            (BLOBS, &self.blobs),
            (FILES, &self.store_files),
            (PROJECT, &self.files),
        ];
        for (prefix, entries) in parts {
            for (name, bytes) in entries {
                add(&format!("{prefix}{name}"), bytes)?;
            }
        }
        archive.into_inner()?.finish()?.flush()
    }

    /// Reads the bundle at `path`.
    ///
    /// # Errors
    ///
    /// When it cannot be read, is no bundle, or names a file outside its
    /// folders.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut archive = tar::Archive::new(GzDecoder::new(fs::File::open(path)?));
        let mut bundle = Self::default();
        let mut manifest = None;
        for entry in archive.entries()? {
            let mut entry = entry?;
            let name = entry.path()?.to_string_lossy().into_owned();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes)?;
            if name == MANIFEST {
                manifest = Some(serde_json::from_slice::<Manifest>(&bytes)?);
                continue;
            }
            let parts = [
                (LOGS, &mut bundle.logs),
                (BLOBS, &mut bundle.blobs),
                (FILES, &mut bundle.store_files),
                (PROJECT, &mut bundle.files),
            ];
            let mut placed = false;
            for (prefix, entries) in parts {
                if let Some(rest) = name.strip_prefix(prefix) {
                    rig_harness::rig_memory::check_name(rest)?;
                    entries.insert(rest.to_owned(), bytes);
                    placed = true;
                    break;
                }
            }
            if !placed {
                return Err(io::Error::other(format!("{name} is no part of a bundle")));
            }
        }
        bundle.manifest = manifest.ok_or_else(|| io::Error::other("the bundle has no manifest"))?;
        if bundle.manifest.format != BUNDLE_FORMAT {
            let why = format!("the bundle is in format {:?}", bundle.manifest.format);
            return Err(io::Error::other(why));
        }
        Ok(bundle)
    }

    /// Where it holds secrets: `log <name>`, `session file <name>` or
    /// `project file <path>`.
    pub fn secrets(&self) -> Vec<String> {
        let mut found = Vec::new();
        let parts = [
            ("log", &self.logs),
            ("session file", &self.store_files),
            ("project file", &self.files),
        ];
        for (kind, entries) in parts {
            for (name, bytes) in entries {
                if map_bytes(bytes, &redacted).is_some() {
                    found.push(format!("{kind} {name}"));
                }
            }
        }
        found
    }

    /// The same bundle with every secret replaced by `[REDACTED]`.
    pub fn redact(mut self) -> Self {
        for entries in [&mut self.logs, &mut self.store_files, &mut self.files] {
            for bytes in entries.values_mut() {
                if let Some(clean) = map_bytes(bytes, &redacted) {
                    *bytes = clean;
                }
            }
        }
        self
    }

    /// The bundled store with every string naming the original project
    /// root remapped to `root`.
    fn store_at(&self, root: &Path) -> io::Result<MemoryStore> {
        let from = self.manifest.project_root.to_string_lossy().into_owned();
        let to = root.to_string_lossy().into_owned();
        let remap = |text: &str| remap(text, &from, &to);
        let store = MemoryStore::default();
        for (name, bytes) in &self.logs {
            let bytes = map_bytes(bytes, &remap).unwrap_or_else(|| bytes.clone());
            store.append_log(name, &bytes)?;
        }
        for (name, bytes) in &self.blobs {
            store.put_blob(name, bytes)?;
        }
        for (name, bytes) in &self.store_files {
            let bytes = map_bytes(bytes, &remap).unwrap_or_else(|| bytes.clone());
            store.write_file(name, &bytes)?;
        }
        Ok(store)
    }
}

/// A bundle written.
#[derive(Message, Clone, Debug)]
pub struct BundleWritten {
    /// The agent `/bundle` was typed for.
    pub agent: Entity,
    /// Where.
    pub path: PathBuf,
    /// Its manifest.
    pub manifest: Manifest,
}

/// A bundle imported as a session.
#[derive(Message, Clone, Debug)]
pub struct BundleImported {
    /// The new session.
    pub session: SessionId,
    /// Plugins the manifest names that this build lacks; their saved
    /// values stay in the session for a build that has them.
    pub missing_plugins: Vec<String>,
    /// The bundled files written to the project.
    pub applied: Vec<String>,
    /// The bundled files that differ from the project's, left as they are.
    pub kept: Vec<String>,
    /// What else the user should know: another commit, a model that fell
    /// back to the default.
    pub notes: Vec<String>,
}

impl BundleImported {
    /// What it says to the user, a line each.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec![format!("Imported the bundle as session {}.", self.session)];
        if !self.missing_plugins.is_empty() {
            lines.push(format!(
                "Missing plugins, whose saved state is kept for when they are installed: {}.",
                self.missing_plugins.join(", ")
            ));
        }
        if !self.applied.is_empty() {
            lines.push(format!("Wrote bundled files: {}.", self.applied.join(", ")));
        }
        if !self.kept.is_empty() {
            lines.push(format!("Kept local files: {}.", self.kept.join(", ")));
        }
        lines.extend(self.notes.iter().cloned());
        lines.push(format!("Resume it with /resume {}.", self.session));
        lines
    }
}

/// What to do with bundled files that differ from the project's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileChoice {
    /// Write every bundled file.
    Apply,
    /// Keep every local file that differs.
    Keep,
    /// Write only these of the files that differ.
    Only(Vec<String>),
}

/// An import of a bundle into a project, before the files are chosen.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The bundle.
    pub bundle: Bundle,
    /// The project it goes to.
    pub target: PathBuf,
    /// Whether it replaces the session an earlier import of it made.
    pub replace: bool,
    /// The bundled files that differ from the project's on disk.
    pub differing: Vec<String>,
    /// The project's commit when it is not the one the session started on.
    pub other_commit: Option<String>,
}

impl Plan {
    /// Plans importing `bundle` into the project at `target`, asking git
    /// through `launcher` which commit the project is at.
    pub fn new(bundle: Bundle, target: PathBuf, replace: bool, launcher: &Launcher) -> Self {
        let differing = bundle
            .files
            .iter()
            .filter(|(path, bytes)| fs::read(target.join(path)).is_ok_and(|local| &local != *bytes))
            .map(|(path, _)| path.clone())
            .collect();
        let head = git(launcher, &target, &["rev-parse", "HEAD"]);
        let other_commit = match (&bundle.manifest.commit, head) {
            (Some(bundled), Some(head)) if *bundled != head => Some(head),
            _ => None,
        };
        Self {
            bundle,
            target,
            replace,
            differing,
            other_commit,
        }
    }

    /// Whether the user has something to choose.
    pub fn needs_choice(&self) -> bool {
        !self.differing.is_empty() || self.other_commit.is_some()
    }

    /// What differs, for the user.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        if let Some(head) = &self.other_commit {
            lines.push(format!(
                "The project is at {head}, the session started at {}.",
                self.bundle.manifest.commit.as_deref().unwrap_or_default()
            ));
        }
        for path in &self.differing {
            lines.push(format!("{path} differs on disk."));
        }
        lines.join("\n")
    }
}

/// What an import needs of the app.
struct Here<'a> {
    home: &'a Home,
    stores: &'a SessionStores,
    models: Option<&'a Models>,
    installed: Vec<String>,
}

/// Imports `plan` with the files chosen so.
fn import(plan: &Plan, choice: &FileChoice, here: &Here<'_>) -> Result<BundleImported, String> {
    let fail = |failure: &dyn std::fmt::Display| failure.to_string();
    let manifest = &plan.bundle.manifest;
    let staged = plan.bundle.store_at(&plan.target).map_err(|f| fail(&f))?;
    let (session, dir) = match plan
        .replace
        .then(|| imported_before(here, &manifest.id))
        .flatten()
    {
        Some((session, dir)) if dir.runs_elsewhere() => {
            return Err(format!(
                "its session {session} is open; close it to replace it"
            ));
        }
        Some((session, dir)) => {
            here.stores
                .open(&dir)
                .and_then(|store| Ok(store.destroy()?))
                .map_err(|f| fail(&f))?;
            (session, dir)
        }
        None => here.home.new_session().map_err(|f| fail(&f))?,
    };
    let written = write_session(&staged, &dir, plan, here);
    let notes = match written {
        Ok(notes) => notes,
        Err(why) => {
            fs::remove_dir_all(dir.path()).ok();
            return Err(why);
        }
    };
    let mut imported = BundleImported {
        session,
        missing_plugins: manifest
            .plugins
            .iter()
            .filter(|plugin| !here.installed.contains(plugin))
            .cloned()
            .collect(),
        applied: Vec::new(),
        kept: Vec::new(),
        notes,
    };
    if let Some(head) = &plan.other_commit {
        imported.notes.push(format!(
            "The project is at commit {head}, not {}.",
            manifest.commit.as_deref().unwrap_or_default()
        ));
    }
    for (path, bytes) in &plan.bundle.files {
        let differs = plan.differing.contains(path);
        let apply = match choice {
            FileChoice::Apply => true,
            FileChoice::Keep => !differs,
            FileChoice::Only(chosen) => !differs || chosen.contains(path),
        };
        if !apply {
            imported.kept.push(path.clone());
            continue;
        }
        let file = plan.target.join(path);
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent).map_err(|f| fail(&f))?;
        }
        fs::write(&file, bytes).map_err(|f| fail(&f))?;
        imported.applied.push(path.clone());
    }
    Ok(imported)
}

/// The session an earlier import of the bundle `id` made.
fn imported_before(here: &Here<'_>, id: &str) -> Option<(SessionId, SessionDir)> {
    fs::read_dir(here.home.sessions())
        .ok()?
        .filter_map(Result::ok)
        .find_map(|entry| {
            let session: SessionId = entry.file_name().to_str()?.parse().ok()?;
            let dir = here.home.session(&session);
            let store = here.stores.open(&dir).ok()?;
            let marker: Value = serde_json::from_slice(&store.read_file(MARKER).ok()?).ok()?;
            (marker.get("bundle")?.as_str()? == id).then_some((session, dir))
        })
}

/// Writes the staged store into the session at `dir`, as a fork does (each
/// log gets a lineage of its own), with the store's other files, the
/// marker, the working directory and its `/resume` listing; a model this
/// build cannot reach is dropped, so its agent takes the default. What the
/// user should know.
fn write_session(
    staged: &MemoryStore,
    dir: &SessionDir,
    plan: &Plan,
    here: &Here<'_>,
) -> Result<Vec<String>, String> {
    let fail = |failure: &dyn std::fmt::Display| failure.to_string();
    let manifest = &plan.bundle.manifest;
    let store = here.stores.open(dir).map_err(|f| fail(&f))?;
    fork(staged, &*store, None, &Keep).map_err(|f| fail(&f))?;
    for name in staged.files().map_err(|f| fail(&f))? {
        if !name.starts_with("plugins/") {
            let bytes = staged.read_file(&name).map_err(|f| fail(&f))?;
            store.write_file(&name, &bytes).map_err(|f| fail(&f))?;
        }
    }
    let marker = serde_json::json!({ "bundle": manifest.id, "from": manifest.project_root });
    store
        .write_file(MARKER, marker.to_string().as_bytes())
        .map_err(|f| fail(&f))?;
    fs::write(dir.directory(), plan.target.to_string_lossy().as_bytes()).map_err(|f| fail(&f))?;
    let mut notes = Vec::new();
    let mut session = Session::read(&*store).map_err(|f| fail(&f))?;
    for agent in &mut session.agents {
        let Some(model) = agent.model() else {
            continue;
        };
        let usable = here
            .models
            .is_some_and(|models| models.connect(&model).is_ok());
        if !usable {
            agent.state.remove(MODEL);
            notes.push(format!(
                "{model} is not available here: agent {} uses the default model.",
                agent.id
            ));
        }
    }
    session.save(&*store).map_err(|f| fail(&f))?;
    let mut meta = SessionMeta::read(&*store).unwrap_or_default();
    if meta.title.is_empty() {
        meta.title = title_of(&manifest.name);
    }
    meta.labels.insert(
        "bundle".to_owned(),
        format!("from {}", manifest.project_root.display()),
    );
    meta.updated = now_ms();
    meta.write(&*store).map_err(|f| fail(&f))?;
    Ok(notes)
}

/// Records where the session started, on its first start.
fn record_origin(
    first: Option<Res<FirstStart>>,
    root: Res<ProjectRoot>,
    processes: Option<Res<Processes>>,
    mut origin: ResMut<BundleOrigin>,
) {
    if first.is_none() || !origin.project_root.is_empty() {
        return;
    }
    let launcher = processes.map_or_else(Launcher::default, |processes| processes.launcher());
    origin.project_root = root.0.to_string_lossy().into_owned();
    origin.commit = git(&launcher, &root.0, &["rev-parse", "HEAD"]);
    origin.branch = git(&launcher, &root.0, &["rev-parse", "--abbrev-ref", "HEAD"])
        .filter(|branch| branch != "HEAD");
}

/// The trimmed output of `git args` in `dir`, if it succeeded.
fn git(launcher: &Launcher, dir: &Path, args: &[&str]) -> Option<String> {
    let output = launcher
        .clone()
        .in_dir(dir)
        .command("git")
        .args(args)
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    output.status.success().then(|| text.trim().to_owned())
}

/// Adds the project files each settled tool call changed to the origin.
fn track_changes(
    settled: On<Add<Settled>>,
    calls: Query<&Published, ToolCalls>,
    root: Res<ProjectRoot>,
    mut origin: ResMut<BundleOrigin>,
) {
    let Ok(published) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(changed)) = published.0.get::<rig_core::tool::FilesChanged>() else {
        return;
    };
    let mut paths: BTreeSet<String> = origin.changed.iter().cloned().collect();
    let before = paths.len();
    for path in changed.paths() {
        let path = Path::new(path);
        let relative = if path.is_absolute() {
            path.strip_prefix(&root.0).ok()
        } else {
            Some(path)
        };
        if let Some(relative) = relative {
            paths.insert(relative.to_string_lossy().into_owned());
        }
    }
    if paths.len() != before {
        origin.changed = paths.into_iter().collect();
    }
}

/// What `/bundle` reads of the app.
#[derive(bevy_ecs::system::SystemParam)]
struct Running<'w, 's> {
    store: Option<Res<'w, SessionStore>>,
    paths: Option<Res<'w, SessionPaths>>,
    title: Option<Res<'w, SessionTitle>>,
    root: Res<'w, ProjectRoot>,
    origin: Res<'w, BundleOrigin>,
    plugins: Query<'w, 's, &'static Name, With<PluginSource>>,
}

impl Running<'_, '_> {
    /// The running session as a bundle.
    fn bundle(&self) -> Result<Bundle, String> {
        let store = self.store.as_ref().ok_or("this session is not stored")?;
        let store = &*store.0;
        let fail = |failure: io::Error| failure.to_string();
        let mut bundle = Bundle::default();
        for name in store.logs().map_err(fail)? {
            bundle
                .logs
                .insert(name.clone(), store.read_log(&name).map_err(fail)?);
        }
        for name in store.blobs().map_err(fail)? {
            bundle
                .blobs
                .insert(name.clone(), store.blob(&name).map_err(fail)?);
        }
        for name in store.files().map_err(fail)? {
            bundle
                .store_files
                .insert(name.clone(), store.read_file(&name).map_err(fail)?);
        }
        for path in &self.origin.changed {
            // A file the session deleted is not bundled.
            if let Ok(bytes) = fs::read(self.root.0.join(path)) {
                bundle.files.insert(path.clone(), bytes);
            }
        }
        let session = self.paths.as_ref().and_then(|paths| paths.id());
        let session = session.map(|id| id.to_string()).unwrap_or_default();
        let name = self.title.as_deref().map(SessionTitle::shown);
        let name = name.filter(|name| !name.is_empty()).map(str::to_owned);
        let saved = Session::read_lenient(store)
            .map(|(saved, _)| saved)
            .unwrap_or_default();
        let models: BTreeSet<String> = saved.agents.iter().filter_map(SavedAgent::model).collect();
        let created_ms = now_ms();
        let root = match self.origin.project_root.as_str() {
            "" => self.root.0.clone(),
            root => PathBuf::from(root),
        };
        bundle.manifest = Manifest {
            format: BUNDLE_FORMAT.to_owned(),
            id: format!("{session}-{created_ms}"),
            session_format: FORMAT.to_owned(),
            name: name.unwrap_or_else(|| format!("session-{session}")),
            session,
            plugins: self
                .plugins
                .iter()
                .map(|name| name.as_str().to_owned())
                .collect(),
            models: models.into_iter().collect(),
            project_root: root,
            created_ms,
            commit: self.origin.commit.clone(),
            branch: self.origin.branch.clone(),
            files: bundle.files.keys().cloned().collect(),
        };
        Ok(bundle)
    }

    /// Where `/bundle` writes: `path` from the project root, else
    /// `<session-name>.rigbundle` there.
    fn path(&self, path: Option<&str>, manifest: &Manifest) -> PathBuf {
        match path {
            Some(path) => self.root.0.join(path),
            None => {
                let stem: String = manifest
                    .name
                    .chars()
                    .map(|c| {
                        if c.is_alphanumeric() || c == '-' || c == '_' {
                            c
                        } else {
                            '-'
                        }
                    })
                    .collect();
                self.root.0.join(format!("{stem}.{EXTENSION}"))
            }
        }
    }
}

/// On an `Ask` about secrets: the bundle waiting for the answer.
#[derive(Component)]
struct SecretsAsk {
    agent: Entity,
    path: PathBuf,
    bundle: Box<Bundle>,
}

/// `/bundle [path] [--allow-secrets]`.
fn bundle_command(
    In(args): In<CommandArgs>,
    running: Running,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut written: MessageWriter<BundleWritten>,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let allow = words.contains(&"--allow-secrets");
    let path = words.iter().find(|word| !word.starts_with("--")).copied();
    let bundle = match running.bundle() {
        Ok(bundle) => bundle,
        Err(why) => {
            notices.write(Notice::error(
                args.agent,
                format!("Could not bundle: {why}."),
            ));
            return;
        }
    };
    let path = running.path(path, &bundle.manifest);
    let secrets = bundle.secrets();
    if secrets.is_empty() || allow {
        finish_bundle(args.agent, &path, bundle, &mut notices, &mut written);
    } else if answerers.can_answer(Some(args.agent)) {
        let ask = Ask::new("The bundle holds secrets")
            .body(format!("Found in: {}.", secrets.join(", ")))
            .choice("redact", "Redact them")
            .choice("keep", "Keep them")
            .choice("cancel", "Cancel");
        let pending = SecretsAsk {
            agent: args.agent,
            path,
            bundle: Box::new(bundle),
        };
        commands.spawn((ask, ForAgent(args.agent), pending));
    } else {
        notices.write(Notice::error(
            args.agent,
            format!(
                "Not bundled: secrets found in {}; pass --allow-secrets to keep them.",
                secrets.join(", ")
            ),
        ));
    }
}

/// Writes `bundle` to `path` and says so.
fn finish_bundle(
    agent: Entity,
    path: &Path,
    bundle: Bundle,
    notices: &mut MessageWriter<Notice>,
    written: &mut MessageWriter<BundleWritten>,
) {
    match bundle.write(path) {
        Ok(()) => {
            notices.write(Notice::info(
                agent,
                format!("Bundled the session to {}.", path.display()),
            ));
            written.write(BundleWritten {
                agent,
                path: path.to_owned(),
                manifest: bundle.manifest,
            });
        }
        Err(failure) => {
            fs::remove_file(path).ok();
            notices.write(Notice::error(
                agent,
                format!("Could not bundle: {failure}."),
            ));
        }
    }
}

/// Acts on the answer about secrets.
fn answered_secrets(
    asks: Query<(Entity, &Outcome, &SecretsAsk), Added<Outcome>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut written: MessageWriter<BundleWritten>,
) {
    for (ask, outcome, pending) in &asks {
        let bundle = (*pending.bundle).clone();
        match Answer::of(outcome) {
            Some(answer) if answer.is("redact") => {
                finish_bundle(
                    pending.agent,
                    &pending.path,
                    bundle.redact(),
                    &mut notices,
                    &mut written,
                );
            }
            Some(answer) if answer.is("keep") => {
                finish_bundle(
                    pending.agent,
                    &pending.path,
                    bundle,
                    &mut notices,
                    &mut written,
                );
            }
            _ => {
                notices.write(Notice::info(pending.agent, "Not bundled."));
            }
        }
        commands.entity(ask).remove::<SecretsAsk>();
    }
}

/// What `rig bundle open` and `/unbundle` take.
struct OpenArgs {
    file: PathBuf,
    project: Option<PathBuf>,
    replace: bool,
    choice: Option<FileChoice>,
}

impl OpenArgs {
    /// Reads `<file> [--project <dir>] [--replace] [--apply|--keep|--apply-file <path>…]`.
    fn parse<'a>(words: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let mut words = words.into_iter();
        let (mut file, mut project, mut replace, mut choice) = (None, None, false, None);
        let mut only = Vec::new();
        while let Some(word) = words.next() {
            match word {
                "--project" => {
                    project = Some(PathBuf::from(
                        words.next().ok_or("--project needs a directory")?,
                    ))
                }
                "--replace" => replace = true,
                "--apply" => choice = Some(FileChoice::Apply),
                "--keep" => choice = Some(FileChoice::Keep),
                "--apply-file" => {
                    only.push(words.next().ok_or("--apply-file needs a path")?.to_owned())
                }
                flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
                path if file.is_none() => file = Some(PathBuf::from(path)),
                extra => return Err(format!("unexpected {extra}")),
            }
        }
        if !only.is_empty() {
            choice = Some(FileChoice::Only(only));
        }
        let file = file.ok_or("a bundle file is needed")?;
        Ok(Self {
            file,
            project,
            replace,
            choice,
        })
    }
}

/// What an import reads of the app.
#[derive(bevy_ecs::system::SystemParam)]
struct Importer<'w, 's> {
    home: Res<'w, RigHome>,
    stores: Res<'w, SessionStores>,
    models: Option<Res<'w, Models>>,
    processes: Option<Res<'w, Processes>>,
    root: Res<'w, ProjectRoot>,
    plugins: Query<'w, 's, &'static Name, With<PluginSource>>,
}

impl Importer<'_, '_> {
    /// Reads the bundle `args` names and plans importing it into its
    /// project, the running one by default; relative paths are from the
    /// project root.
    fn plan(&self, args: &OpenArgs) -> Result<Plan, String> {
        let base = &self.root.0;
        let file = base.join(&args.file);
        let bundle =
            Bundle::open(&file).map_err(|failure| format!("{}: {failure}", file.display()))?;
        let target = args
            .project
            .as_ref()
            .map_or_else(|| base.clone(), |dir| base.join(dir));
        let launcher = self
            .processes
            .as_ref()
            .map_or_else(Launcher::default, |p| p.launcher());
        Ok(Plan::new(bundle, target, args.replace, &launcher))
    }

    fn import(&self, plan: &Plan, choice: &FileChoice) -> Result<BundleImported, String> {
        let here = Here {
            home: &self.home.0,
            stores: &self.stores,
            models: self.models.as_deref(),
            installed: self
                .plugins
                .iter()
                .map(|name| name.as_str().to_owned())
                .collect(),
        };
        import(plan, choice, &here)
    }
}

/// `rig bundle open <file> …`: imports, prints what it did, and exits.
fn open_command_line(
    invoked: Res<Invoked>,
    importer: Importer,
    mut imported: MessageWriter<BundleImported>,
    mut commands: Commands,
) {
    let words = invoked.command().map_or(&[][..], |(_, words)| words);
    let result = match words.split_first() {
        Some((open, rest)) if open == "open" => OpenArgs::parse(rest.iter().map(String::as_str))
            .and_then(|args| Ok((importer.plan(&args)?, args.choice))),
        _ => Err("usage: rig bundle open <file> [--project <dir>] [--replace] [--apply|--keep|--apply-file <path>…]".to_owned()),
    };
    let reason = match result {
        Err(why) => {
            eprintln!("rig bundle: {why}");
            ExitReason::Failed(2)
        }
        Ok((plan, None)) if plan.needs_choice() => {
            eprintln!(
                "{}\nRun again with --apply (Apply bundled files), --keep (Keep local files) or --apply-file <path> for each file to write.",
                plan.summary()
            );
            ExitReason::Failed(2)
        }
        Ok((plan, choice)) => match importer.import(&plan, &choice.unwrap_or(FileChoice::Apply)) {
            Ok(done) => {
                for line in done.lines() {
                    println!("{line}");
                }
                imported.write(done);
                ExitReason::Quit
            }
            Err(why) => {
                eprintln!("rig bundle: {why}");
                ExitReason::Failed(1)
            }
        },
    };
    commands.trigger(Exit(reason));
}

/// On an `Ask` about differing files: the import waiting for the answer.
#[derive(Component)]
struct FilesAsk {
    agent: Entity,
    plan: Box<Plan>,
    /// Whether this is the per-file question.
    each: bool,
}

/// `/unbundle <file> …`.
fn unbundle_command(
    In(args): In<CommandArgs>,
    importer: Importer,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut imported: MessageWriter<BundleImported>,
) {
    let parsed = OpenArgs::parse(args.args.split_whitespace())
        .and_then(|open| Ok((importer.plan(&open)?, open.choice)));
    let (plan, choice) = match parsed {
        Ok(parsed) => parsed,
        Err(why) => {
            notices.write(Notice::error(args.agent, format!("/unbundle: {why}.")));
            return;
        }
    };
    let choice = match choice {
        Some(choice) => choice,
        None if !plan.needs_choice() => FileChoice::Apply,
        None if answerers.can_answer(Some(args.agent)) => {
            let ask = Ask::new("The bundle differs from this project")
                .body(plan.summary())
                .choice("apply", "Apply bundled files")
                .choice("keep", "Keep local files")
                .choice("each", "Choose for each file");
            let pending = FilesAsk {
                agent: args.agent,
                plan: Box::new(plan),
                each: false,
            };
            commands.spawn((ask, ForAgent(args.agent), pending));
            return;
        }
        None => {
            let why = format!("{}\nPass --apply or --keep.", plan.summary());
            notices.write(Notice::error(args.agent, why));
            return;
        }
    };
    report(
        args.agent,
        importer.import(&plan, &choice),
        &mut notices,
        &mut imported,
    );
}

/// Says what an import did.
fn report(
    agent: Entity,
    result: Result<BundleImported, String>,
    notices: &mut MessageWriter<Notice>,
    imported: &mut MessageWriter<BundleImported>,
) {
    match result {
        Ok(done) => {
            for line in done.lines() {
                notices.write(Notice::info(agent, line));
            }
            imported.write(done);
        }
        Err(why) => {
            notices.write(Notice::error(
                agent,
                format!("Could not import the bundle: {why}."),
            ));
        }
    }
}

/// Acts on the answers about differing files: imports, or asks per file.
fn answered_files(
    asks: Query<(Entity, &Outcome, &FilesAsk), Added<Outcome>>,
    importer: Importer,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut imported: MessageWriter<BundleImported>,
) {
    for (ask, outcome, pending) in &asks {
        commands.entity(ask).remove::<FilesAsk>();
        let choice = match Answer::of(outcome) {
            Some(Answer::Choices(chosen)) if pending.each => FileChoice::Only(chosen),
            Some(answer) if answer.is("apply") => FileChoice::Apply,
            Some(answer) if answer.is("keep") => FileChoice::Keep,
            Some(answer) if answer.is("each") => {
                let mut each = Ask::new("Which bundled files to write?").multiple();
                for path in &pending.plan.differing {
                    each = each.choice(path.clone(), path.clone());
                }
                let next = FilesAsk {
                    agent: pending.agent,
                    plan: pending.plan.clone(),
                    each: true,
                };
                commands.spawn((each, ForAgent(pending.agent), next));
                continue;
            }
            _ => {
                notices.write(Notice::info(pending.agent, "Not imported."));
                continue;
            }
        };
        let result = importer.import(&pending.plan, &choice);
        report(pending.agent, result, &mut notices, &mut imported);
    }
}

/// `text` with every secret redacted: rig-core's shapes, and the value of
/// each environment-looking assignment of a secret (`NAME=value` where the
/// name says it is a secret, key, token or password); `None` when it has
/// none.
fn redacted(text: &str) -> Option<String> {
    let shapes = redact::text(text);
    let mut out = String::with_capacity(shapes.len());
    let mut changed = matches!(shapes, std::borrow::Cow::Owned(_));
    let mut rest: &str = &shapes;
    while let Some(at) = rest.find('=') {
        let (before, after) = rest.split_at(at);
        let after = after.get(1..).unwrap_or_default();
        let name_len = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .count();
        let name = before
            .get(before.len().saturating_sub(name_len)..)
            .unwrap_or_default();
        let value_len = after
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ';' | ','))
            .unwrap_or(after.len());
        let value = after.get(..value_len).unwrap_or_default();
        out.push_str(before);
        out.push('=');
        if secret_name(name) && !value.is_empty() && value != redact::REDACTED {
            out.push_str(redact::REDACTED);
            rest = after.get(value_len..).unwrap_or_default();
            changed = true;
        } else {
            rest = after;
        }
    }
    out.push_str(rest);
    changed.then_some(out)
}

/// Whether an environment variable named `name` holds a secret.
fn secret_name(name: &str) -> bool {
    const SECRET: [&str; 7] = [
        "SECRET",
        "TOKEN",
        "PASSWORD",
        "PASSWD",
        "API_KEY",
        "ACCESS_KEY",
        "PRIVATE_KEY",
    ];
    name.len() > 2
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && SECRET.iter().any(|word| name.contains(word))
}

/// `text` with every mention of the path `from`, as a whole path or the
/// start of one, made `to`; `None` when it has none.
fn remap(text: &str, from: &str, to: &str) -> Option<String> {
    if from.is_empty() || !text.contains(from) {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut changed = false;
    while let Some(at) = rest.find(from) {
        let before = rest.get(..at).unwrap_or_default();
        let tail = rest.get(at + from.len()..).unwrap_or_default();
        let whole = tail
            .chars()
            .next()
            .is_none_or(|c| !(c.is_alphanumeric() || matches!(c, '_' | '-' | '.')));
        out.push_str(before);
        out.push_str(if whole { to } else { from });
        changed |= whole;
        rest = tail;
    }
    out.push_str(rest);
    changed.then_some(out)
}

/// `bytes` with `map` applied to each string: of the JSON document, of
/// each line of a JSON-lines file (as the session logs are), else of the
/// text; `None` when nothing changed or it is no text.
fn map_bytes(bytes: &[u8], map: &dyn Fn(&str) -> Option<String>) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(bytes).ok()?;
    if let Ok(mut value) = serde_json::from_str::<Value>(text) {
        return map_json(&mut value, map)
            .then(|| serde_json::to_vec(&value).ok())
            .flatten();
    }
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let parsed: Vec<Option<Value>> = lines
        .iter()
        .map(|line| serde_json::from_str(line.trim_end()).ok())
        .collect();
    if !lines.is_empty() && parsed.iter().all(Option::is_some) {
        let mut out = String::with_capacity(text.len());
        let mut changed = false;
        for (line, value) in lines.iter().zip(parsed) {
            let mut value = value.unwrap_or_default();
            if map_json(&mut value, map) {
                out.push_str(&serde_json::to_string(&value).ok()?);
                if line.ends_with('\n') {
                    out.push('\n');
                }
                changed = true;
            } else {
                out.push_str(line);
            }
        }
        return changed.then(|| out.into_bytes());
    }
    map(text).map(String::into_bytes)
}

/// Applies `map` to every string of `value`; whether one changed.
fn map_json(value: &mut Value, map: &dyn Fn(&str) -> Option<String>) -> bool {
    match value {
        Value::String(text) => match map(text) {
            Some(mapped) => {
                *text = mapped;
                true
            }
            None => false,
        },
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |changed, item| map_json(item, map) | changed),
        Value::Object(fields) => fields
            .values_mut()
            .fold(false, |changed, item| map_json(item, map) | changed),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}
