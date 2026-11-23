//! The workspace as a feature-aware dependency graph, read from `cargo
//! metadata`: which workspace packages a package compiles under a set of its
//! features, and which packages and features can observe a change.
//!
//! Only edges between workspace packages are followed. Target-specific
//! dependencies count as live on every target, so a selection is never
//! narrower than any platform's.
use super::{Result, invalid};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Normal,
    Dev,
}

#[derive(Debug)]
struct Dep {
    /// Index of the workspace package this edge reaches.
    target: usize,
    /// The name the depending manifest uses for it (its rename, if any).
    local: String,
    kind: Kind,
    optional: bool,
    default: bool,
    features: Vec<String>,
}

#[derive(Debug)]
pub(super) struct Package {
    pub(super) name: String,
    pub(super) dir: PathBuf,
    /// Whether it has a library target (doctests and docs need one).
    pub(super) lib: bool,
    features: BTreeMap<String, Vec<String>>,
    deps: Vec<Dep>,
}

impl Package {
    /// Every feature the package declares, `default` excluded.
    pub(super) fn feature_names(&self) -> impl Iterator<Item = &str> {
        self.features
            .keys()
            .map(String::as_str)
            .filter(|f| *f != "default")
            .chain(
                // Optional dependencies not named by any `dep:` entry are
                // implicit features of the same name.
                self.deps
                    .iter()
                    .filter(|d| d.optional && d.kind == Kind::Normal)
                    .filter(|d| !self.mentions_dep(&d.local))
                    .map(|d| d.local.as_str()),
            )
    }

    fn mentions_dep(&self, local: &str) -> bool {
        let explicit = format!("dep:{local}");
        self.features
            .values()
            .flatten()
            .any(|item| *item == explicit)
    }
}

/// What a package compiles: each live workspace package and its features.
pub(super) type Resolved = BTreeMap<usize, BTreeSet<String>>;

#[derive(Debug)]
pub(super) struct Graph {
    pub(super) packages: Vec<Package>,
}

/// Which features of a package a run enables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Features {
    /// `--all-features` for that package.
    All,
    /// Its default features plus these.
    Default(BTreeSet<String>),
}

impl Graph {
    pub(super) fn from_metadata(metadata: &Value) -> Result<Self> {
        let raw = metadata["packages"]
            .as_array()
            .ok_or_else(|| invalid("cargo metadata has no packages"))?;
        let dirs: Vec<PathBuf> = raw
            .iter()
            .map(|p| {
                p["manifest_path"]
                    .as_str()
                    .and_then(|m| Path::new(m).parent())
                    .map(Path::to_path_buf)
                    .ok_or_else(|| invalid("a package without a manifest path"))
            })
            .collect::<Result<_>>()?;
        let mut packages = Vec::new();
        for (p, dir) in raw.iter().zip(&dirs) {
            let name = p["name"]
                .as_str()
                .ok_or_else(|| invalid("a package without a name"))?
                .to_owned();
            let features = p["features"]
                .as_object()
                .map(|map| {
                    map.iter()
                        .map(|(k, v)| {
                            let items = v
                                .as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(Value::as_str)
                                .map(str::to_owned)
                                .collect();
                            (k.clone(), items)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut deps = Vec::new();
            for d in p["dependencies"].as_array().into_iter().flatten() {
                // A workspace edge is a path dependency onto a member.
                let Some(target) = d["path"]
                    .as_str()
                    .and_then(|path| dirs.iter().position(|dir| dir == Path::new(path)))
                else {
                    continue;
                };
                let kind = match d["kind"].as_str() {
                    Some("dev") => Kind::Dev,
                    _ => Kind::Normal,
                };
                let local = d["rename"]
                    .as_str()
                    .or_else(|| d["name"].as_str())
                    .ok_or_else(|| invalid(format!("{name}: a dependency without a name")))?
                    .to_owned();
                deps.push(Dep {
                    target,
                    local,
                    kind,
                    optional: d["optional"].as_bool().unwrap_or(false),
                    default: d["uses_default_features"].as_bool().unwrap_or(true),
                    features: d["features"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect(),
                });
            }
            let lib = p["targets"].as_array().into_iter().flatten().any(|t| {
                t["kind"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|k| matches!(k.as_str(), Some("lib" | "rlib" | "proc-macro")))
            });
            packages.push(Package {
                name,
                dir: dir.clone(),
                lib,
                features,
                deps,
            });
        }
        Ok(Self { packages })
    }

    pub(super) fn index(&self, name: &str) -> Option<usize> {
        self.packages.iter().position(|p| p.name == name)
    }

    pub(super) fn name(&self, index: usize) -> &str {
        self.packages.get(index).map_or("", |p| p.name.as_str())
    }

    /// The package whose directory holds `path` (relative to `root`), the
    /// deepest one when packages nest.
    pub(super) fn owner(&self, root: &Path, path: &str) -> Option<usize> {
        let absolute = root.join(path);
        self.packages
            .iter()
            .enumerate()
            .filter(|(_, p)| absolute.starts_with(&p.dir))
            .max_by_key(|(_, p)| p.dir.components().count())
            .map(|(i, _)| i)
    }

    /// What `root` compiles with `features`, and with its dev-dependencies
    /// when `tests` (its test targets are built).
    pub(super) fn resolve(&self, root: usize, features: &Features, tests: bool) -> Resolved {
        let mut state = Resolve {
            graph: self,
            root,
            tests,
            enabled: Resolved::new(),
            deps_on: BTreeSet::new(),
            queue: VecDeque::new(),
        };
        state.queue.push_back(Event::Visit(root));
        match features {
            Features::All => {
                if let Some(package) = self.packages.get(root) {
                    for f in package.features.keys().map(String::as_str) {
                        state.queue.push_back(Event::Feature(root, f.to_owned()));
                    }
                    for f in package.feature_names() {
                        state.queue.push_back(Event::Feature(root, f.to_owned()));
                    }
                }
            }
            Features::Default(extra) => {
                state
                    .queue
                    .push_back(Event::Feature(root, "default".into()));
                for f in extra {
                    state.queue.push_back(Event::Feature(root, f.clone()));
                }
            }
        }
        state.run();
        state.enabled
    }

    /// Every package whose own code (all features, no tests) compiles one of
    /// `changed`: the packages a change can break.
    pub(super) fn dependents(&self, changed: &BTreeSet<usize>) -> BTreeSet<usize> {
        (0..self.packages.len())
            .filter(|&p| {
                self.resolve(p, &Features::All, false)
                    .keys()
                    .any(|live| changed.contains(live))
            })
            .collect()
    }

    /// How `package`'s tests must run to observe `changed`, given the
    /// packages that depend on it (`dependents`): every feature when it is
    /// itself changed; else its default features plus each feature that
    /// brings in a dependent or adds features to a changed package; `None`
    /// when no feature lets its tests observe the change.
    pub(super) fn observing(
        &self,
        package: usize,
        changed: &BTreeSet<usize>,
        dependents: &BTreeSet<usize>,
    ) -> Option<Features> {
        if changed.contains(&package) {
            return Some(Features::All);
        }
        let none = BTreeSet::new();
        let base = self.resolve(package, &Features::Default(none), true);
        let sees_base = base.keys().any(|p| changed.contains(p));
        let mut extra = BTreeSet::new();
        for f in self.packages.get(package)?.feature_names() {
            let with = self.resolve(
                package,
                &Features::Default(BTreeSet::from([f.to_owned()])),
                true,
            );
            let observes = with.iter().any(|(p, features)| match base.get(p) {
                None => dependents.contains(p),
                Some(before) => changed.contains(p) && features != before,
            });
            if observes {
                extra.insert(f.to_owned());
            }
        }
        (sees_base || !extra.is_empty()).then_some(Features::Default(extra))
    }

    /// Paths outside each package that its sources name relative to itself
    /// (`"../other/fixtures"` joined to its manifest directory or written in
    /// an `include_str!`), as repository-relative prefixes: a change under
    /// one changes what that package's tests read. `embedded` marks a path
    /// that library code compiles in (`include_str!`/`include_bytes!` outside
    /// test files): a change there changes the package itself.
    pub(super) fn reads(&self, root: &Path, files: &[String]) -> Vec<Read> {
        let mut out = Vec::new();
        for file in files.iter().filter(|f| f.ends_with(".rs")) {
            let Some(owner) = self.owner(root, file) else {
                continue;
            };
            let Some(package) = self.packages.get(owner) else {
                continue;
            };
            let Ok(text) = std::fs::read_to_string(root.join(file)) else {
                continue;
            };
            let file_dir = root.join(file).parent().map(Path::to_path_buf);
            let test_file = file.ends_with("tests.rs") || file.contains("/tests/");
            for (literal, included) in relative_literals(&text) {
                for base in [Some(package.dir.clone()), file_dir.clone()]
                    .into_iter()
                    .flatten()
                {
                    let Some(target) = normalize(&base.join(&literal)) else {
                        continue;
                    };
                    let Ok(relative) = target.strip_prefix(root) else {
                        continue;
                    };
                    if target.starts_with(&package.dir) || relative.as_os_str().is_empty() {
                        continue;
                    }
                    let read = Read {
                        prefix: relative.to_string_lossy().into_owned(),
                        package: owner,
                        embedded: included && !test_file,
                    };
                    if !out.contains(&read) {
                        out.push(read);
                    }
                }
            }
        }
        out
    }
}

/// A path outside a package that the package reads.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Read {
    pub(super) prefix: String,
    pub(super) package: usize,
    pub(super) embedded: bool,
}

/// String literals that start with `../`, and whether each is the argument
/// of an `include_str!` or `include_bytes!`. The text is tokenized, so
/// comments, escapes and raw strings cannot shift what counts as a literal;
/// text that does not tokenize names nothing.
fn relative_literals(text: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    match text.parse::<proc_macro2::TokenStream>() {
        Ok(tokens) => literals(tokens, false, &mut out),
        Err(error) => eprintln!("a source file does not tokenize ({error}); its reads are unknown"),
    }
    out
}

fn literals(tokens: proc_macro2::TokenStream, included: bool, out: &mut Vec<(String, bool)>) {
    use proc_macro2::TokenTree;
    let mut previous: Vec<String> = Vec::new();
    for tree in tokens {
        match tree {
            TokenTree::Group(group) => {
                let macro_call = matches!(
                    previous.as_slice(),
                    [.., name, bang] if bang == "!" && (name == "include_str" || name == "include_bytes")
                );
                literals(group.stream(), macro_call, out);
                previous.clear();
            }
            TokenTree::Literal(literal) => {
                let text = literal.to_string();
                let value = text
                    .trim_start_matches('r')
                    .trim_matches('#')
                    .strip_prefix('"')
                    .and_then(|rest| rest.strip_suffix('"'));
                if let Some(value) =
                    value.filter(|v| v.starts_with("../") && !v.contains(char::is_whitespace))
                {
                    out.push((value.to_owned(), included));
                }
                previous.clear();
            }
            other => previous.push(other.to_string()),
        }
    }
}

/// `path` with `.` and `..` resolved lexically.
fn normalize(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests;

enum Event {
    Visit(usize),
    Feature(usize, String),
    DepOn(usize, String),
}

struct Resolve<'a> {
    graph: &'a Graph,
    root: usize,
    tests: bool,
    enabled: Resolved,
    deps_on: BTreeSet<(usize, String)>,
    queue: VecDeque<Event>,
}

impl Resolve<'_> {
    fn run(&mut self) {
        while let Some(event) = self.queue.pop_front() {
            match event {
                Event::Visit(p) => self.visit(p),
                Event::Feature(p, f) => self.feature(p, f),
                Event::DepOn(p, local) => self.dep_on(p, local),
            }
        }
    }

    /// The edges of `p` this resolution follows: normal and build edges, and
    /// the root's dev-dependencies when its tests are built.
    fn edges(&self, p: usize) -> impl Iterator<Item = &Dep> {
        let tests = self.tests && p == self.root;
        self.graph
            .packages
            .get(p)
            .into_iter()
            .flat_map(|package| package.deps.iter())
            .filter(move |d| d.kind == Kind::Normal || tests)
    }

    fn visit(&mut self, p: usize) {
        if self.enabled.contains_key(&p) {
            return;
        }
        self.enabled.insert(p, BTreeSet::new());
        let required: Vec<String> = self
            .edges(p)
            .filter(|d| !d.optional)
            .map(|d| d.local.clone())
            .collect();
        for local in required {
            self.queue.push_back(Event::DepOn(p, local));
        }
    }

    fn dep_on(&mut self, p: usize, local: String) {
        if !self.deps_on.insert((p, local.clone())) {
            return;
        }
        let mut events = Vec::new();
        for d in self.edges(p).filter(|d| d.local == local) {
            events.push(Event::Visit(d.target));
            if d.default {
                events.push(Event::Feature(d.target, "default".into()));
            }
            for f in &d.features {
                events.push(Event::Feature(d.target, f.clone()));
            }
        }
        // Weak features (`local?/f`) of already enabled features apply now.
        let weak = format!("{local}?/");
        let pending: Vec<String> = self
            .enabled
            .get(&p)
            .into_iter()
            .flatten()
            .filter_map(|f| self.items(p, f))
            .flatten()
            .filter_map(|item| item.strip_prefix(&weak).map(str::to_owned))
            .collect();
        for f in pending {
            for t in self.targets(p, &local) {
                events.push(Event::Feature(t, f.clone()));
            }
        }
        self.queue.extend(events);
    }

    fn items(&self, p: usize, feature: &str) -> Option<Vec<String>> {
        self.graph
            .packages
            .get(p)
            .and_then(|package| package.features.get(feature))
            .cloned()
    }

    fn targets(&self, p: usize, local: &str) -> Vec<usize> {
        self.edges(p)
            .filter(|d| d.local == local)
            .map(|d| d.target)
            .collect()
    }

    fn feature(&mut self, p: usize, f: String) {
        self.visit(p);
        let fresh = self
            .enabled
            .get_mut(&p)
            .is_some_and(|features| features.insert(f.clone()));
        if !fresh {
            return;
        }
        let Some(items) = self.items(p, &f) else {
            // An implicit feature: the optional dependency of that name.
            self.queue.push_back(Event::DepOn(p, f));
            return;
        };
        for item in items {
            if let Some(local) = item.strip_prefix("dep:") {
                self.queue.push_back(Event::DepOn(p, local.to_owned()));
            } else if let Some((local, feature)) = item.split_once("?/") {
                if self.deps_on.contains(&(p, local.to_owned())) {
                    for t in self.targets(p, local) {
                        self.queue.push_back(Event::Feature(t, feature.to_owned()));
                    }
                }
            } else if let Some((local, feature)) = item.split_once('/') {
                self.queue.push_back(Event::DepOn(p, local.to_owned()));
                for t in self.targets(p, local) {
                    self.queue.push_back(Event::Feature(t, feature.to_owned()));
                }
            } else {
                self.queue.push_back(Event::Feature(p, item));
            }
        }
    }
}
