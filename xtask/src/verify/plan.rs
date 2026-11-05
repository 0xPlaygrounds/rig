//! The plan: from the paths a change touches, which checks run and, for the
//! package-scoped ones, over which packages and features.
//!
//! A changed package selects the tests of every package that can observe it
//! ([`Graph::observing`]), each with only the features that do; a package
//! whose tests read a changed file outside it runs its own tests. Every other
//! check runs when a package it builds compiles a changed one. A shared input
//! (workspace manifest, lockfile, toolchain, Cargo or nextest configuration,
//! xtask, CI, test support, scripts) or a path no rule knows runs every check
//! in full.
use super::{
    Result,
    checks::Check,
    graph::{Features, Graph},
    invalid,
};
#[cfg(test)]
mod tests;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

/// What a selected check runs over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    Full,
    Packages(BTreeMap<String, Features>),
}

/// The selected checks by id; a check that is absent is skipped.
pub(crate) type Plan = BTreeMap<String, Scope>;

/// The checks whose commands take a package scope.
pub(crate) const SCOPED: [&str; 5] = ["tests", "doctests", "docs", "clippy", "service-tests"];

/// Packages the main test run leaves to their own checks: the Docker suites
/// (`service-tests`), the trybuild and proc-macro tests (`derive`) and the
/// replay runner built without the facade's graph (`bus-verification`).
const OWN_CHECK: [&str; 3] = ["rig-service-tests", "rig-derive", "rig-cassette-minimal"];

/// Inputs every check reads; a change to one runs the full plan.
fn shared(path: &str) -> bool {
    [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "clippy.toml",
        "rustfmt.toml",
    ]
    .contains(&path)
        || [
            ".cargo/",
            ".config/",
            ".github/",
            "xtask/",
            "test-support/",
            "scripts/",
        ]
        .iter()
        .any(|prefix| path.starts_with(prefix))
}

/// Repository paths no build or test reads: guards check them, nothing else.
fn inert(path: &str) -> bool {
    (!path.contains('/') && path.ends_with(".md") && path != "README.md")
        || path.starts_with("docs/")
        || path.starts_with("prompts/")
}

/// The facade lives at the repository root; only these root paths are its.
fn facade_path(path: &str) -> bool {
    path == "README.md"
        || ["src/", "tests/", "benches/"]
            .iter()
            .any(|p| path.starts_with(p))
}

/// What a non-scoped check builds, by package and features: it runs when
/// one of them compiles a changed package. `None` for a check with no rule.
fn builds(id: &str) -> Option<Vec<(String, Features)>> {
    let with = |names: &[&str], features: &Features| {
        names
            .iter()
            .map(|n| ((*n).to_owned(), features.clone()))
            .collect::<Vec<_>>()
    };
    let all = Features::All;
    let default = Features::Default(BTreeSet::new());
    Some(match id {
        "core-all" => with(
            &[
                "rig-core",
                "rig-http",
                "rig-reqwest",
                "rig-tungstenite",
                "rig-agent",
                "rig-rmcp",
                "rig-cassette-minimal",
            ],
            &all,
        ),
        "bus-verification" => with(&["rig-cassette-minimal", "rig-cassette"], &all),
        "conformance" => with(
            &[
                "rig-core",
                "rig-candle",
                "rig-gemini-grpc",
                "rig-bedrock",
                "rig-vertexai",
            ],
            &all,
        ),
        "effect-corpus" => {
            let mut out = with(&["rig-cassette", "rig-test-support"], &all);
            out.push((
                "rig".to_owned(),
                Features::Default(BTreeSet::from(["bedrock".to_owned()])),
            ));
            out
        }
        "derive" => with(&["rig-derive"], &all),
        "kernel-only" => with(&["rig-ecs"], &default),
        "default-check" => with(
            &[
                "rig",
                "rig-test-support",
                "rig-cassette",
                "rig-memory",
                "rig-tools",
                "rig-core",
            ],
            &default,
        ),
        "process-tests" => with(
            &["rig-tools", "rig-harness", "rig", "rig-harness-protocol"],
            &default,
        ),
        "rig-claude-cloak-tests" => with(&["rig-claude-cloak"], &default),
        "guard-facade-features" => with(&["rig"], &all),
        "guard-middleware-features" => with(&["rig-reqwest"], &all),
        "guard-agent-stepper" => with(&["rig-agent", "rig-core"], &default),
        "guard-macro-hygiene" => with(&["rig-core"], &default),
        "guard-dependency-rename" => with(&["rig-derive"], &default),
        _ => {
            let package = id
                .strip_prefix("wasm-")
                .or_else(|| id.strip_prefix("native-only-"))?;
            let features = if ["rig-core", "rig-http"].contains(&package) {
                all
            } else {
                default
            };
            vec![(package.to_owned(), features)]
        }
    })
}

/// Checks a member manifest selects: what Cargo resolves (dependency floors)
/// and feature forwarding (the compile guards).
fn manifest_check(id: &str) -> bool {
    id == "dependency-floors" || id.starts_with("guard-")
}

/// Checks that always run: they read the tree, not a build.
fn always(id: &str) -> bool {
    ["fmt", "source-guards", "tooling", "acceptance", "bank"].contains(&id)
}

pub(crate) fn full(all: &[Check]) -> Plan {
    all.iter().map(|c| (c.id.clone(), Scope::Full)).collect()
}

pub(crate) fn plan(
    root: &Path,
    graph: &Graph,
    all: &[Check],
    files: &[String],
    paths: &BTreeSet<String>,
) -> Result<(Plan, Vec<String>)> {
    let mut why = Vec::new();
    if let Some(path) = paths.iter().find(|p| shared(p)) {
        why.push(format!("shared input {path}: every check in full"));
        return Ok((full(all), why));
    }
    let reads = graph.reads(root, files);
    let mut changed = BTreeSet::new();
    let mut readers = BTreeSet::new();
    let mut manifests = false;
    for path in paths {
        if inert(path) {
            continue;
        }
        for read in reads.iter().filter(|r| path.starts_with(r.prefix.as_str())) {
            if read.embedded {
                changed.insert(read.package);
            } else {
                readers.insert(read.package);
            }
        }
        let owner = graph.owner(root, path).filter(|&p| {
            graph
                .packages
                .get(p)
                .is_some_and(|package| package.dir != root)
                || facade_path(path)
        });
        let Some(owner) = owner else {
            why.push(format!("{path} belongs to no package: every check in full"));
            return Ok((full(all), why));
        };
        manifests |= path.ends_with("/Cargo.toml");
        changed.insert(owner);
    }
    let dependents = graph.dependents(&changed);
    let mut tests = BTreeMap::new();
    for p in 0..graph.packages.len() {
        let features = if readers.contains(&p) {
            Some(Features::All)
        } else {
            graph.observing(p, &changed, &dependents)
        };
        if let Some(features) = features {
            tests.insert(graph.name(p).to_owned(), features);
        }
    }
    let name =
        |set: &BTreeSet<usize>| -> Vec<&str> { set.iter().map(|&p| graph.name(p)).collect() };
    why.push(format!(
        "changed packages {:?}; readers {:?}",
        name(&changed),
        name(&readers)
    ));
    let touched: BTreeSet<usize> = changed.union(&readers).copied().collect();
    let mut out = Plan::new();
    for check in all {
        let id = check.id.as_str();
        if always(id) {
            out.insert(id.to_owned(), Scope::Full);
            continue;
        }
        if SCOPED.contains(&id) {
            let scope = scoped(id, graph, &tests);
            if !scope.is_empty() {
                out.insert(id.to_owned(), Scope::Packages(scope));
            }
            continue;
        }
        if manifests && manifest_check(id) {
            out.insert(id.to_owned(), Scope::Full);
            continue;
        }
        if id == "dependency-floors" {
            continue;
        }
        let Some(builds) = builds(id) else {
            why.push(format!("check {id} has no rule: it runs in full"));
            out.insert(id.to_owned(), Scope::Full);
            continue;
        };
        let hit = builds.iter().any(|(package, features)| {
            graph.index(package).is_some_and(|p| {
                graph
                    .resolve(p, features, true)
                    .keys()
                    .any(|live| touched.contains(live))
            })
        });
        if hit {
            out.insert(id.to_owned(), Scope::Full);
        }
    }
    Ok((out, why))
}

/// The packages a scoped check runs over, from the observing test set.
fn scoped(
    id: &str,
    graph: &Graph,
    tests: &BTreeMap<String, Features>,
) -> BTreeMap<String, Features> {
    let lib = |name: &str| {
        graph
            .index(name)
            .and_then(|p| graph.packages.get(p))
            .is_some_and(|p| p.lib)
    };
    tests
        .iter()
        .filter(|(name, features)| match id {
            // Every service suite sits behind a store feature: with none the
            // check would run zero tests.
            "service-tests" => {
                name.as_str() == "rig-service-tests"
                    && **features != Features::Default(BTreeSet::new())
            }
            "tests" => !OWN_CHECK.contains(&name.as_str()),
            "doctests" | "docs" => lib(name) && name.as_str() != "rig-service-tests",
            _ => true,
        })
        .map(|(name, features)| (name.clone(), features.clone()))
        .collect()
}

/// `--features` arguments selecting `features` of each package.
pub(crate) fn package_args(
    graph: &Graph,
    packages: &BTreeMap<String, Features>,
) -> Result<Vec<String>> {
    let mut args = Vec::new();
    for (name, features) in packages {
        let package = graph
            .index(name)
            .and_then(|p| graph.packages.get(p))
            .ok_or_else(|| invalid(format!("the plan names unknown package {name}")))?;
        args.extend(["-p".to_owned(), name.clone()]);
        let list: Vec<String> = match features {
            Features::All => package.feature_names().map(str::to_owned).collect(),
            Features::Default(extra) => extra.iter().cloned().collect(),
        };
        for f in list {
            args.extend(["--features".to_owned(), format!("{name}/{f}")]);
        }
    }
    Ok(args)
}

/// The plan as the CI plan job's output: `{"check": "full" | "pkg pkg:* pkg:a,b"}`.
pub(crate) fn encode(plan: &Plan) -> String {
    let map: serde_json::Map<String, serde_json::Value> = plan
        .iter()
        .map(|(id, scope)| {
            let text = match scope {
                Scope::Full => "full".to_owned(),
                Scope::Packages(packages) => packages
                    .iter()
                    .map(|(name, features)| match features {
                        Features::All => format!("{name}:*"),
                        Features::Default(extra) if extra.is_empty() => name.clone(),
                        Features::Default(extra) => {
                            format!(
                                "{name}:{}",
                                extra.iter().cloned().collect::<Vec<_>>().join(",")
                            )
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            };
            (id.clone(), serde_json::Value::String(text))
        })
        .collect();
    serde_json::Value::Object(map).to_string()
}

pub(crate) fn decode(text: &str) -> Result<Plan> {
    let value: serde_json::Value = serde_json::from_str(text)?;
    let map = value
        .as_object()
        .ok_or_else(|| invalid("the plan is not a JSON object"))?;
    let mut plan = Plan::new();
    for (id, scope) in map {
        let scope = scope
            .as_str()
            .ok_or_else(|| invalid(format!("the plan's scope for {id} is not a string")))?;
        let scope = if scope == "full" {
            Scope::Full
        } else {
            Scope::Packages(
                scope
                    .split_whitespace()
                    .map(|item| match item.split_once(':') {
                        None => (item.to_owned(), Features::Default(BTreeSet::new())),
                        Some((name, "*")) => (name.to_owned(), Features::All),
                        Some((name, list)) => (
                            name.to_owned(),
                            Features::Default(list.split(',').map(str::to_owned).collect()),
                        ),
                    })
                    .collect(),
            )
        };
        plan.insert(id.clone(), scope);
    }
    Ok(plan)
}
