//! OC-02: Dependency change guard (one-conversation fresh held-out spec
//! `02-dependency-guard.md`).
//!
//! A Gate system looks at each `shell`, `edit` and `write` call about to
//! run. A `shell` command that installs packages (`cargo add`, `npm install
//! <pkg>`, `pnpm add`, `yarn add`, `pip install`, `uv add`, `go get`), or an
//! `edit`/`write` of a manifest (`Cargo.toml`, `package.json`,
//! `pyproject.toml`, `requirements*.txt`, `go.mod`) whose result has a
//! dependency entry the file had not, is held while the policy files are
//! read (and the manifest, through the project's workspace, off the main
//! thread). Packages the policy denies refuse the call at once; packages it
//! allows, or the user allowed for the session, pass; any other waits for
//! an [`Ask`]: allow, allow for this session, or deny. With nobody to
//! answer, unlisted packages are denied. `/deps` lists what was allowed and
//! denied this session.
//!
//! The policy is `deps.toml` in `RIG_HOME` and in the project's `.rig/`,
//! their lists added together: `allow = ["serde", "tokio*"]`,
//! `deny = ["openssl"]`, where `*` matches any run of characters.

use std::fmt;
use std::path::{Path, PathBuf};

use bevy_log::warn;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_tools::workspace::Workspace;
use rig_harness::rig_tools::{
    Edit, EditArgs, Replacement, Shell, ShellArgs, Write, WriteArgs, edit,
};
use serde::Deserialize;

/// Who decides the calls this plugin refuses.
pub const NAME: &str = "dependency-guard";
/// Runs the call once.
pub const ALLOW: &str = "allow";
/// Runs the call, and lets its packages pass for the rest of the session.
pub const SESSION: &str = "session";
/// Refuses the call.
pub const DENY: &str = "deny";
/// The policy file's name, in `RIG_HOME` and in the project's `.rig/`.
pub const POLICY_FILE: &str = "deps.toml";

/// What the model reads after a refusal: solve it another way.
const INSTEAD: &str = "Solve the task without these packages.";

/// A `deps.toml`: package patterns that pass silently and that are refused.
#[derive(Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct DepsPolicy {
    /// Patterns of packages added without asking.
    pub allow: Vec<String>,
    /// Patterns of packages refused outright; they win over `allow`.
    pub deny: Vec<String>,
}

impl DepsPolicy {
    /// Whether a `deny` pattern matches `name`.
    pub fn denies(&self, name: &str) -> bool {
        self.deny.iter().any(|pattern| matches(pattern, name))
    }

    /// Whether an `allow` pattern matches `name`.
    pub fn allows(&self, name: &str) -> bool {
        self.allow.iter().any(|pattern| matches(pattern, name))
    }

    /// Adds `other`'s patterns to these.
    fn extend(&mut self, other: Self) {
        self.allow.extend(other.allow);
        self.deny.extend(other.deny);
    }
}

/// Whether `pattern`, where `*` matches any run of characters, matches all
/// of `name`, ignoring case.
pub fn matches(pattern: &str, name: &str) -> bool {
    let (pattern, name) = (pattern.to_lowercase(), name.to_lowercase());
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    let Some((last, middle)) = parts.split_last() else {
        return rest.is_empty();
    };
    for part in middle {
        match rest.find(part).and_then(|at| rest.get(at + part.len()..)) {
            Some(after) => rest = after,
            None => return false,
        }
    }
    rest.ends_with(last)
}

/// A package a call would add, with the version it names, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Package {
    /// Its name as written, such as `serde` or `@types/node`.
    pub name: String,
    /// The version or requirement written with it, such as `0.8`.
    pub version: Option<String>,
}

impl Package {
    fn new(name: &str, version: Option<&str>) -> Self {
        let version = version.map(str::trim).filter(|v| !v.is_empty());
        Self {
            name: name.trim().to_owned(),
            version: version.map(str::to_owned),
        }
    }
}

impl fmt::Display for Package {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.version {
            Some(version) => write!(f, "{}@{version}", self.name),
            None => f.write_str(&self.name),
        }
    }
}

/// What was allowed and denied this session, kept with it.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SessionDeps {
    /// Each package that was added, as shown (`name@version`).
    pub allowed: Vec<String>,
    /// The names the user allowed for the rest of the session.
    pub for_session: Vec<String>,
    /// Each package that was refused, as shown.
    pub denied: Vec<String>,
}

impl SessionDeps {
    fn note(list: &mut Vec<String>, packages: &[Package]) {
        for shown in packages.iter().map(Package::to_string) {
            if !list.contains(&shown) {
                list.push(shown);
            }
        }
    }
}

/// Where the policy files are: the user's, on this machine, and the
/// project's, in its workspace.
#[derive(Resource, Clone, Debug)]
struct PolicyFiles {
    home: PathBuf,
}

/// This plugin's Gate systems: they carry out each answer, then check, refuse or ask about each call about to run.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepsSystems;

/// Holds every new dependency for the user's approval unless `deps.toml`
/// allows it.
#[derive(Default)]
pub struct DependencyGuardPlugin;

impl Plugin for DependencyGuardPlugin {
    fn build(&self, app: &mut App) {
        let home = RigHome::of(app.world()).root().join(POLICY_FILE);
        app.insert_resource(PolicyFiles { home })
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<SessionDeps>()
            .configure_sets(AgentPass, DepsSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, (answered, gate).chain().in_set(DepsSystems))
            .add_observer(checked)
            .add_command(
                "deps",
                "List the packages allowed or denied this session",
                list,
            );
    }
}

/// What a call would add, found off the main thread.
#[derive(Clone, Debug, Default)]
struct Found {
    /// The packages new to the project.
    packages: Vec<Package>,
    /// Where they go: a manifest's file name, or the install command.
    target: String,
    /// The policy as read for this call.
    policy: DepsPolicy,
    /// Policy files that could not be read as a policy.
    problems: Vec<String>,
}

/// On a hold entity: the check of the call it holds.
#[derive(Component)]
struct CheckHold;

/// On a call: what it would add, once checked.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked(Result<Found, String>);

/// On a call that may run: allowed by the policy or by the user.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Approved;

/// On an [`Ask`]: the packages it asks about.
#[derive(Component)]
struct Asked(Vec<Package>);

/// What a call would change that may add a dependency.
enum Probe {
    /// An install command and the packages it names.
    Shell {
        command: String,
        packages: Vec<Package>,
    },
    /// A manifest written or edited.
    Manifest {
        path: PathBuf,
        kind: Manifest,
        change: Change,
    },
}

/// The new content of a manifest, or how to make it.
enum Change {
    Write(String),
    Edit(Vec<Replacement>),
}

impl Probe {
    /// What `request` would change that may add a dependency; `footprint`
    /// says which file an `edit` or `write` changes.
    fn of(request: &ToolRequest, footprint: &Footprint) -> Option<Self> {
        let function = &request.call.function;
        let arguments = &function.arguments;
        match function.name.as_str() {
            Shell::NAME => {
                let ShellArgs { command, .. } = ShellArgs::deserialize(arguments).ok()?;
                let packages = installs(&command);
                (!packages.is_empty()).then_some(Self::Shell { command, packages })
            }
            name @ (Edit::NAME | Write::NAME) => {
                let Touches::Writes(Scope::Paths(paths)) = &footprint.0 else {
                    return None;
                };
                let [path] = paths.as_slice() else {
                    return None;
                };
                let kind = Manifest::of(path)?;
                let change = if name == Edit::NAME {
                    Change::Edit(EditArgs::deserialize(arguments).ok()?.edits)
                } else {
                    Change::Write(WriteArgs::deserialize(arguments).ok()?.content)
                };
                Some(Self::Manifest {
                    path: path.clone(),
                    kind,
                    change,
                })
            }
            _ => None,
        }
    }

    /// The packages it adds, under the policy read now. Blocking.
    fn check(self, workspace: &dyn Workspace, home: &Path, project: &Path) -> Found {
        let mut found = Found::default();
        let home = std::fs::read_to_string(home).ok();
        let project = workspace
            .read_text(&project.join(".rig").join(POLICY_FILE))
            .ok();
        for text in [home, project].into_iter().flatten() {
            match toml::from_str::<DepsPolicy>(&text) {
                Ok(policy) => found.policy.extend(policy),
                Err(error) => found.problems.push(error.to_string()),
            }
        }
        match self {
            Self::Shell { command, packages } => {
                found.target = format!("`{command}`");
                found.packages = packages;
            }
            Self::Manifest { path, kind, change } => {
                let before = workspace.read_text(&path).unwrap_or_default();
                let after = match change {
                    Change::Write(content) => Some(content),
                    Change::Edit(edits) => {
                        let shown = path.to_string_lossy();
                        edit::apply(&shown, &before, &edits).ok().map(|it| it.text)
                    }
                };
                found.target = path
                    .file_name()
                    .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
                found.packages = after
                    .and_then(|after| kind.added(&before, &after))
                    .unwrap_or_default();
            }
        }
        found
    }
}

/// Moves a finished check onto its call and lets go of it, so it is gated
/// again. An observer: the loop does not see a task finish, but sees the
/// hold go.
fn checked(
    done: On<Add<Done<Result<Found, String>>>>,
    mut holds: Query<(&HoldOf, &mut Done<Result<Found, String>>), With<CheckHold>>,
    mut commands: Commands,
) {
    let Ok((&HoldOf(call), mut found)) = holds.get_mut(done.entity) else {
        return;
    };
    let found = std::mem::replace(&mut found.0, Err(String::new()));
    if let Ok(found) = &found {
        for problem in &found.problems {
            warn!("{NAME}: a {POLICY_FILE} was left out: {problem}");
        }
    }
    commands.entity(call).insert(Checked(found));
    commands.entity(done.entity).despawn();
}

/// Checks each call that may add a dependency, then refuses, passes or
/// asks about what it adds.
fn gate(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &ForAgent,
            &Footprint,
            Option<&Checked>,
            &mut Decision,
        ),
        (ReadyToStart, ToolCalls, Without<Approved>),
    >,
    answerers: Answerers,
    mut session: ResMut<SessionDeps>,
    (workspace, root, files, wake): (
        Res<ProjectWorkspace>,
        Res<ProjectRoot>,
        Res<PolicyFiles>,
        Res<Wake>,
    ),
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), footprint, checked, mut decision) in &mut calls {
        // A gate before this one decided it already: nothing to do.
        if decision.is_decided() {
            continue;
        }
        let Some(Checked(found)) = checked else {
            let Some(probe) = Probe::of(request, footprint) else {
                continue;
            };
            let (workspace, home, project) =
                (workspace.0.clone(), files.home.clone(), root.0.clone());
            let check = blocking(move || Ok(probe.check(workspace.as_ref(), &home, &project)));
            let work = async move { check.await.map_err(|error| error.to_string()) };
            let pool = IoTaskPool::get_or_init(TaskPool::default);
            commands.spawn((
                Name::new("checking for new dependencies"),
                CheckHold,
                HoldOf(call),
                Running::spawn(pool, &wake, work),
            ));
            continue;
        };
        let found = match found {
            Ok(found) => found,
            Err(error) => {
                let why = format!("{NAME} could not check this call for new dependencies: {error}");
                decision.decide(Decider::plugin(NAME), Outcome::refused(why));
                continue;
            }
        };
        let (denied, rest): (Vec<Package>, Vec<Package>) = found
            .packages
            .iter()
            .cloned()
            .partition(|package| found.policy.denies(&package.name));
        if !denied.is_empty() {
            SessionDeps::note(&mut session.denied, &denied);
            let why = format!("{} refused by {POLICY_FILE}. {INSTEAD}", shown(&denied));
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
            continue;
        }
        let unlisted: Vec<Package> = rest
            .iter()
            .filter(|package| {
                !found.policy.allows(&package.name)
                    && !session
                        .for_session
                        .iter()
                        .any(|name| same(name, &package.name))
            })
            .cloned()
            .collect();
        if unlisted.is_empty() {
            SessionDeps::note(&mut session.allowed, &rest);
            commands.entity(call).insert(Approved);
            continue;
        }
        if !answerers.can_answer(Some(agent)) {
            SessionDeps::note(&mut session.denied, &unlisted);
            let why = format!(
                "Adding {} needs the user's approval, and nobody can give it here. {INSTEAD}",
                shown(&unlisted)
            );
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
            continue;
        }
        let noun = if unlisted.len() == 1 {
            "dependency"
        } else {
            "dependencies"
        };
        let into = if found.target.starts_with('`') {
            "with"
        } else {
            "to"
        };
        let lines: Vec<String> = unlisted.iter().map(|p| format!("- {p}")).collect();
        let ask = Ask::new(format!(
            "Add {noun} {} {into} {}?",
            shown(&unlisted),
            found.target
        ))
        .body(lines.join("\n"))
        .choice(ALLOW, "Allow")
        .choice(SESSION, "Allow for this session")
        .choice(DENY, "Deny");
        commands.spawn((ask, Asked(unlisted), ForAgent(agent), HoldOf(call)));
    }
}

/// Carries out each answer: the call runs, or is refused naming its
/// packages, and the question lets go of it.
fn answered(
    asks: Query<(Entity, &Outcome, &HoldOf, &Asked), (With<Ask>, Added<Outcome>)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut session: ResMut<SessionDeps>,
    mut commands: Commands,
) {
    for (ask, outcome, &HoldOf(call), Asked(packages)) in &asks {
        let answer = Answer::of(outcome);
        if let Some(answer) = answer.filter(|a| a.is(ALLOW) || a.is(SESSION)) {
            if answer.is(SESSION) {
                for package in packages {
                    if !session
                        .for_session
                        .iter()
                        .any(|name| same(name, &package.name))
                    {
                        session.for_session.push(package.name.clone());
                    }
                }
            }
            SessionDeps::note(&mut session.allowed, packages);
            commands.entity(call).insert(Approved);
        } else if let Ok(mut decision) = calls.get_mut(call) {
            SessionDeps::note(&mut session.denied, packages);
            let why = format!("The user denied adding {}. {INSTEAD}", shown(packages));
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}

/// `/deps`: what was allowed and denied this session.
fn list(In(args): In<CommandArgs>, session: Res<SessionDeps>, mut notices: MessageWriter<Notice>) {
    let lines = [
        ("Allowed", &session.allowed),
        ("Allowed for the session", &session.for_session),
        ("Denied", &session.denied),
    ];
    let lines: Vec<String> = lines
        .into_iter()
        .filter(|(_, names)| !names.is_empty())
        .map(|(label, names)| format!("{label}: {}", names.join(", ")))
        .collect();
    let text = if lines.is_empty() {
        "No dependency was allowed or denied this session.".to_owned()
    } else {
        lines.join("\n")
    };
    notices.write(Notice::info(args.agent, text));
}

/// `packages` as the model and the user read them: `` `a@1`, `b` ``.
fn shown(packages: &[Package]) -> String {
    let names: Vec<String> = packages.iter().map(|p| format!("`{p}`")).collect();
    names.join(", ")
}

/// The packages `command` installs that are new dependencies: each
/// install of a package manager it runs, with its package arguments.
pub fn installs(command: &str) -> Vec<Package> {
    let mut packages = Vec::new();
    for segment in command.split([';', '|', '&', '\n']) {
        let words: Vec<&str> = segment
            .split_whitespace()
            .map(|word| word.trim_matches(['\'', '"']))
            .skip_while(|word| *word == "sudo" || (word.contains('=') && !word.starts_with('-')))
            .collect();
        let (ecosystem, arguments) = match words.as_slice() {
            ["cargo", "add", rest @ ..] => (Ecosystem::Cargo, rest),
            ["npm", "install" | "i" | "add" | "in", rest @ ..]
            | ["pnpm" | "yarn", "add", rest @ ..] => (Ecosystem::Npm, rest),
            ["pip" | "pip3", "install", rest @ ..]
            | ["python" | "python3", "-m", "pip", "install", rest @ ..]
            | ["uv", "add", rest @ ..]
            | ["uv", "pip", "install", rest @ ..] => (Ecosystem::Python, rest),
            ["go", "get", rest @ ..] => (Ecosystem::Go, rest),
            _ => continue,
        };
        let mut arguments = arguments.iter();
        while let Some(word) = arguments.next() {
            if word.starts_with('-') {
                if !word.contains('=') && ecosystem.takes_value(word) {
                    arguments.next();
                }
                continue;
            }
            packages.extend(ecosystem.spec(word));
        }
    }
    packages
}

/// A package manager's way of naming packages on its command line.
#[derive(Clone, Copy)]
enum Ecosystem {
    Cargo,
    Npm,
    Python,
    Go,
}

impl Ecosystem {
    /// Whether `flag` takes the next word as its value.
    fn takes_value(self, flag: &str) -> bool {
        let flags: &[&str] = match self {
            Self::Cargo => &[
                "-F",
                "--features",
                "--rename",
                "-p",
                "--package",
                "--manifest-path",
                "--git",
                "--branch",
                "--tag",
                "--rev",
                "--path",
                "--registry",
                "--target",
                "--config",
                "-Z",
            ],
            Self::Npm => &[
                "--registry",
                "--tag",
                "--prefix",
                "-w",
                "--workspace",
                "--omit",
                "--include",
                "--filter",
                "--cwd",
            ],
            Self::Python => &[
                "-r",
                "--requirement",
                "-c",
                "--constraint",
                "-e",
                "--editable",
                "-i",
                "--index-url",
                "--extra-index-url",
                "-t",
                "--target",
                "-f",
                "--find-links",
                "--python",
                "-p",
                "--package",
                "--group",
                "--optional",
                "--extra",
                "--index",
                "--rev",
                "--tag",
                "--branch",
                "--path",
            ],
            Self::Go => &["-modfile"],
        };
        flags.contains(&flag)
    }

    /// The package a package argument names; `None` for a path or URL.
    fn spec(self, word: &str) -> Option<Package> {
        if word.starts_with(['.', '/', '~']) || word.contains("://") {
            return None;
        }
        match self {
            Self::Cargo | Self::Go => {
                let (name, version) = word.split_once('@').unwrap_or((word, ""));
                Some(Package::new(name, Some(version)))
            }
            Self::Npm => match word.rsplit_once('@') {
                Some((name, version)) if !name.is_empty() => {
                    Some(Package::new(name, Some(version)))
                }
                _ => Some(Package::new(word, None)),
            },
            Self::Python => requirement(word),
        }
    }
}

/// A Python requirement (`name[extra]>=1; marker`) as a package.
fn requirement(text: &str) -> Option<Package> {
    let text = text.split(';').next().unwrap_or_default().trim();
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
        .unwrap_or(text.len());
    let (name, rest) = text.split_at_checked(end)?;
    if name.is_empty() {
        return None;
    }
    let rest = rest.trim_start();
    let rest = match rest.strip_prefix('[') {
        Some(extras) => extras.split_once(']').map_or("", |(_, after)| after),
        None => rest,
    };
    let rest = rest.trim();
    let version = rest.strip_prefix("==").unwrap_or(rest);
    Some(Package::new(name, Some(version)))
}

/// The manifests whose dependency entries are watched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Manifest {
    /// `Cargo.toml`.
    Cargo,
    /// `package.json`.
    PackageJson,
    /// `pyproject.toml`.
    PyProject,
    /// `requirements*.txt`.
    Requirements,
    /// `go.mod`.
    GoMod,
}

impl Manifest {
    /// The kind of manifest `path` is, by its file name.
    pub fn of(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        match name {
            "Cargo.toml" => Some(Self::Cargo),
            "package.json" => Some(Self::PackageJson),
            "pyproject.toml" => Some(Self::PyProject),
            "go.mod" => Some(Self::GoMod),
            _ if name.starts_with("requirements") && name.ends_with(".txt") => {
                Some(Self::Requirements)
            }
            _ => None,
        }
    }

    /// The entries of `after` whose package `before` did not have; `None`
    /// when `after` cannot be read as this manifest.
    pub fn added(self, before: &str, after: &str) -> Option<Vec<Package>> {
        let old = self.entries(before).unwrap_or_default();
        let known = |name: &str| old.iter().any(|p| same(&p.name, name));
        let mut added: Vec<Package> = Vec::new();
        for package in self.entries(after)? {
            if !known(&package.name) && !added.iter().any(|p| same(&p.name, &package.name)) {
                added.push(package);
            }
        }
        Some(added)
    }

    /// Every dependency entry of `text`; `None` when it does not parse.
    pub fn entries(self, text: &str) -> Option<Vec<Package>> {
        let mut out = Vec::new();
        match self {
            Self::Cargo => {
                let table: toml::Table = toml::from_str(text).ok()?;
                cargo_tables(&table, &mut out);
                if let Some(workspace) = table.get("workspace").and_then(toml::Value::as_table) {
                    cargo_tables(workspace, &mut out);
                }
                let targets = table.get("target").and_then(toml::Value::as_table);
                for target in targets.into_iter().flat_map(toml::Table::values) {
                    if let Some(target) = target.as_table() {
                        cargo_tables(target, &mut out);
                    }
                }
            }
            Self::PackageJson => {
                let json: serde_json::Value = serde_json::from_str(text).ok()?;
                let keys = [
                    "dependencies",
                    "devDependencies",
                    "peerDependencies",
                    "optionalDependencies",
                ];
                for key in keys {
                    for (name, version) in json
                        .get(key)
                        .and_then(|v| v.as_object())
                        .into_iter()
                        .flatten()
                    {
                        out.push(Package::new(name, version.as_str()));
                    }
                }
            }
            Self::PyProject => {
                let table: toml::Table = toml::from_str(text).ok()?;
                let at = |path: &[&str]| -> Option<&toml::Value> {
                    let (first, rest) = path.split_first()?;
                    rest.iter()
                        .try_fold(table.get(*first)?, |value, key| value.get(key))
                };
                let lists = [
                    at(&["project", "dependencies"])
                        .into_iter()
                        .collect::<Vec<_>>(),
                    tables_of(at(&["project", "optional-dependencies"])),
                    tables_of(at(&["dependency-groups"])),
                ];
                for list in lists.into_iter().flatten() {
                    let items = list.as_array().into_iter().flatten();
                    out.extend(
                        items
                            .filter_map(toml::Value::as_str)
                            .filter_map(requirement),
                    );
                }
                let groups = tables_of(at(&["tool", "poetry", "group"]));
                let poetry = [
                    at(&["tool", "poetry", "dependencies"]),
                    at(&["tool", "poetry", "dev-dependencies"]),
                ];
                let poetry = poetry.into_iter().flatten().chain(
                    groups
                        .into_iter()
                        .filter_map(|group| group.get("dependencies")),
                );
                for deps in poetry.filter_map(toml::Value::as_table) {
                    let deps = deps.iter().filter(|(name, _)| name.as_str() != "python");
                    out.extend(deps.map(|(name, value)| Package::new(name, toml_version(value))));
                }
            }
            Self::Requirements => {
                for line in text.lines() {
                    let line = line.split(" #").next().unwrap_or_default().trim();
                    if line.is_empty() || line.starts_with(['#', '-']) {
                        continue;
                    }
                    out.extend(requirement(line));
                }
            }
            Self::GoMod => {
                let mut block = false;
                for line in text.lines() {
                    let line = line.split("//").next().unwrap_or_default().trim();
                    let entry = if block {
                        block = !line.starts_with(')');
                        line
                    } else if let Some(rest) = line.strip_prefix("require") {
                        let rest = rest.trim();
                        block = rest.starts_with('(');
                        rest
                    } else {
                        continue;
                    };
                    let mut words = entry.split_whitespace();
                    if let (Some(name), Some(version)) = (words.next(), words.next())
                        && name != "("
                        && name != ")"
                    {
                        out.push(Package::new(name, Some(version)));
                    }
                }
            }
        }
        Some(out)
    }
}

/// The dependency tables of a Cargo manifest's `table` (the top, a
/// `[target.*]` or `[workspace]`), into `out`.
fn cargo_tables(table: &toml::Table, out: &mut Vec<Package>) {
    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        let deps = table.get(key).and_then(toml::Value::as_table);
        for (name, value) in deps.into_iter().flatten() {
            out.push(Package::new(name, toml_version(value)));
        }
    }
}

/// A TOML dependency's version: the string, or its table's `version`.
fn toml_version(value: &toml::Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("version").and_then(toml::Value::as_str))
}

/// The values of the table `value` is, if it is one.
fn tables_of(value: Option<&toml::Value>) -> Vec<&toml::Value> {
    let table = value.and_then(toml::Value::as_table);
    table.into_iter().flat_map(toml::Table::values).collect()
}

/// Whether two package names are the same package: case and Python's
/// `_`/`-`/`.` aside.
fn same(a: &str, b: &str) -> bool {
    let normal = |name: &str| name.to_lowercase().replace(['_', '.'], "-");
    normal(a) == normal(b)
}

#[cfg(test)]
mod tests;
