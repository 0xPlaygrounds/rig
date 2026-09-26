use super::*;
use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
};

pub(super) fn changes(root: &Path, opts: &Options) -> Result<BTreeSet<String>> {
    let revision = if let Some(base) = &opts.base {
        let merge = output(root, "git", &["merge-base", base, "HEAD"])?;
        println!("Base: {base}; merge base: {}", merge.trim());
        merge.trim().to_string()
    } else {
        "HEAD".into()
    };
    let mut paths = BTreeSet::new();
    // Union the index and working-tree deltas: a staged edit undone only in
    // the working tree still belongs in the plan. Include both sides of moves.
    for args in [
        vec!["diff", "--name-only", "--no-renames", "-z", &revision],
        vec![
            "diff",
            "--cached",
            "--name-only",
            "--no-renames",
            "-z",
            &revision,
        ],
    ] {
        paths.extend(
            output(root, "git", &args)?
                .split('\0')
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
        );
    }
    paths.extend(execute::untracked_inputs(root)?);
    Ok(paths)
}
/// The two expensive lanes on top of the fast PR set. They have separate
/// triggers (`checks::full_lane`, `checks::floor_lane`) both locally and in
/// CI, so a storage-suite edit no longer resolves dependency floors and a
/// manifest edit no longer waits on the service suites unless it also
/// matches that lane.
#[derive(Clone, Copy)]
struct Lanes {
    full: bool,
    floors: bool,
}
impl Lanes {
    const ALL: Self = Self {
        full: true,
        floors: true,
    };
}
fn broad(all: &[Check], reason: &str, lanes: Lanes) -> Vec<Check> {
    all.iter()
        .filter(|c| match c.id.as_str() {
            "full-tests" => lanes.full,
            "dependency-floors" => lanes.floors,
            _ => true,
        })
        .cloned()
        .map(|mut c| {
            c.reason = reason.into();
            c
        })
        .collect()
}
fn add(out: &mut Vec<Check>, all: &[Check], id: &str, reason: &str) -> Result<()> {
    if out.iter().any(|c| c.id == id) {
        return Ok(());
    }
    let mut c = all
        .iter()
        .find(|c| c.id == id)
        .cloned()
        .ok_or_else(|| invalid(format!("missing required check {id}")))?;
    c.reason = reason.into();
    out.push(c);
    Ok(())
}
/// The provider a path belongs to, whichever package now owns its target:
/// the cassette-backed suites and both corpora live in `rig-cassette`, the
/// remaining live-only suites in the facade.
fn provider(path: &str) -> Option<&str> {
    for prefix in [
        "crates/rig-cassette/fixtures/cassettes/",
        "crates/rig-cassette/tests/providers/",
        "tests/providers/",
    ] {
        if let Some(rest) = path.strip_prefix(prefix) {
            return rest.split('/').next();
        }
    }
    for prefix in ["crates/rig-cassette/tests/", "tests/"] {
        if let Some(name) = path
            .strip_prefix(prefix)
            .filter(|s| !s.contains('/'))
            .and_then(|s| s.strip_suffix(".rs"))
        {
            return Some(name);
        }
    }
    None
}
/// The package whose test target is named `name`, searching the cassette
/// package before the facade so a moved suite is never attributed to its old
/// owner.
fn provider_owner<'a>(packages: &'a [Value], name: &str) -> Option<&'a str> {
    ["rig-cassette", "rig"].into_iter().find(|owner| {
        packages
            .iter()
            .find(|p| p["name"] == *owner)
            .and_then(|p| p["targets"].as_array())
            .is_some_and(|ts| {
                ts.iter().any(|t| {
                    t["name"] == name
                        && t["kind"]
                            .as_array()
                            .is_some_and(|k| k.iter().any(|x| x == "test"))
                })
            })
    })
}
/// Build and verification inputs every package reads. `--changed` broadens to
/// the full plan on them; `--quick` leaves them to CI.
fn shared_input(path: &str) -> bool {
    path == "Cargo.lock"
        || path.ends_with("Cargo.toml")
        || path == "rust-toolchain.toml"
        || [
            ".cargo/",
            ".config/",
            ".github/",
            "xtask/",
            "scripts/",
            "test-support/",
            "src/",
            "crates/rig-cassette/tests/common/",
            "tests/integrations/",
        ]
        .iter()
        .any(|p| path.starts_with(p))
}
fn documentation(path: &str) -> bool {
    ["README.md", "CONTRIBUTING.md", "AGENTS.md", "DEVELOPING.md"].contains(&path)
        || path.starts_with("docs/")
}
/// Whether the minimal runner compiles `path` too. It shares these sources by
/// path, not through a Cargo dependency edge, so reverse-dependency discovery
/// cannot find it. `target` is the provider-style target the path maps to.
fn shared_with_minimal(path: &str, target: Option<(&str, &str)>) -> bool {
    match target {
        Some((owner, name)) => {
            owner == "rig-cassette"
                && matches!(name, "verify" | "world_replay" | "world_replay_world")
        }
        None => [
            "crates/rig-cassette/tests/",
            "crates/rig-cassette/src/effect_log/",
            "crates/rig-cassette/src/agent/replay",
        ]
        .iter()
        .any(|prefix| path.starts_with(prefix)),
    }
}
fn package<'a>(root: &Path, packages: &'a [Value], path: &str) -> Option<&'a Value> {
    let absolute = root.join(path);
    packages
        .iter()
        .filter_map(|p| {
            let dir = Path::new(p["manifest_path"].as_str()?).parent()?;
            absolute
                .starts_with(dir)
                .then_some((dir.components().count(), p))
        })
        .max_by_key(|(depth, _)| *depth)
        .map(|(_, p)| p)
}
fn consumers(packages: &[Value], name: &str) -> BTreeSet<String> {
    let mut result = BTreeSet::from([name.into()]);
    let mut queue = VecDeque::from([name.to_string()]);
    while let Some(next) = queue.pop_front() {
        for p in packages {
            let Some(name) = p["name"].as_str() else {
                continue;
            };
            if p["dependencies"]
                .as_array()
                .is_some_and(|deps| deps.iter().any(|d| d["name"] == next))
                && result.insert(name.into())
            {
                queue.push_back(name.into());
            }
        }
    }
    result
}
fn dynamic(id: String, args: Vec<String>, reason: &str) -> Check {
    Check {
        id,
        steps: vec![Step {
            program: "cargo".into(),
            args,
            env: BTreeMap::new(),
        }],
        reason: reason.into(),
    }
}
fn provider_check(owner: &str, name: &str, reason: &str) -> Check {
    dynamic(
        format!("provider-{name}"),
        [
            "nextest",
            "run",
            "--locked",
            "-p",
            owner,
            "--all-features",
            "--test",
            name,
            "--retries",
            "0",
        ]
        .iter()
        .map(|s| (*s).into())
        .collect(),
        reason,
    )
}

/// Cache-warming aliases: each compiles its source check's artifacts with
/// nextest `--no-run` and executes nothing. cache-warm.yaml runs exactly these.
pub(super) const WARMING: [(&str, &str); 2] = [
    ("default-test-build", "default-tests"),
    ("full-test-build", "full-tests"),
];

pub(super) fn plan(
    root: &Path,
    metadata: &Value,
    opts: &Options,
    paths: &BTreeSet<String>,
    all: &[Check],
) -> Result<Vec<Check>> {
    if opts.mode == Mode::Check {
        let id = opts
            .check
            .as_deref()
            .ok_or_else(|| invalid("missing check ID"))?;
        let mut out = Vec::new();
        // Cache warming: compile exactly the artifacts a test check needs,
        // without executing it or claiming its result. Each alias keeps its
        // source check's package/feature graph so rust-cache entries match.
        if let Some((_, source)) = WARMING.iter().find(|(alias, _)| *alias == id) {
            add(
                &mut out,
                all,
                source,
                "cache warming only: compile test artifacts; executes no tests",
            )?;
            if let Some(check) = out.first_mut() {
                check.id = id.into();
                for step in &mut check.steps {
                    // nextest rejects --retries together with --no-run.
                    if let Some(index) = step.args.iter().position(|a| a == "--retries") {
                        step.args.drain(index..index + 2);
                    }
                    step.args.push("--no-run".into());
                }
            }
            return Ok(out);
        }
        add(&mut out, all, id, "explicit check; always selected")?;
        return Ok(out);
    }
    if opts.mode == Mode::Full {
        return Ok(broad(
            all,
            "exhaustive supported repository verification",
            Lanes::ALL,
        ));
    }
    if matches!(opts.mode, Mode::Pr | Mode::Lanes) {
        // The release-document freeze is a PR policy with its own exemptions
        // in ci.yaml; a lane query must not enforce it.
        if opts.mode == Mode::Pr
            && paths.iter().any(|p| {
                p == "CHANGELOG.md"
                    || p == "MIGRATING.md"
                    || p.starts_with("docs/migrations/")
                    || (p.starts_with("crates/") && p.ends_with("/CHANGELOG.md"))
            })
        {
            return Err(invalid(
                "ordinary PR changes frozen release documents; put release notes in the PR description",
            ));
        }
        // The PR plan must never be narrower than a conservative local
        // fallback (for example an unknown generator or CI configuration).
        let changed = Options {
            mode: Mode::Changed,
            base: opts.base.clone(),
            dry_run: opts.dry_run,
            check: None,
        };
        let local = plan(root, metadata, &changed, paths, all)?;
        let full_triggers: Vec<_> = paths.iter().filter(|p| checks::full_lane(p)).collect();
        let floor_triggers: Vec<_> = paths.iter().filter(|p| checks::floor_lane(p)).collect();
        let fallback = |id: &str| local.iter().find(|check| check.id == id);
        let lanes = Lanes {
            full: !full_triggers.is_empty() || fallback("full-tests").is_some(),
            floors: !floor_triggers.is_empty() || fallback("dependency-floors").is_some(),
        };
        let fallbacks: Vec<_> = ["full-tests", "dependency-floors"]
            .into_iter()
            .filter_map(|id| fallback(id).map(|c| format!("{id}: {}", c.reason)))
            .collect();
        let reason = format!(
            "complete intended PR diff; preserves required lanes; full-lane inputs: {full_triggers:?}; floor-lane inputs: {floor_triggers:?}; conservative fallbacks: {fallbacks:?}"
        );
        let required = broad(all, &reason, lanes);
        return Ok(required);
    }
    let packages = metadata["packages"]
        .as_array()
        .ok_or_else(|| invalid("Cargo metadata missing packages"))?;
    let mut out: Vec<Check> = Vec::new();
    let mut affected = BTreeSet::new();
    for path in paths {
        if path == "DEVELOPING.md" {
            continue;
        }
        if path == "scripts/check-dependency-floors.py"
            || path == "scripts/test_dependency_floors.py"
        {
            // The floor checker and its isolation tests touch nothing else.
            add(&mut out, all, "tooling", "floor checker isolation tests")?;
            add(
                &mut out,
                all,
                "dependency-floors",
                "floor checker changed: resolve and build the lowered graph",
            )?;
            continue;
        }
        if shared_input(path) {
            // Shared inputs broaden to every runtime check. Dependency floors
            // join only for resolver inputs: nextest configuration, shared
            // test support or facade source cannot change what Cargo resolves.
            let lanes = Lanes {
                full: true,
                floors: checks::floor_lane(path)
                    || path.starts_with(".github/")
                    || path.starts_with("scripts/"),
            };
            return Ok(broad(
                all,
                &format!("conservative fallback: shared/build/verification input {path}"),
                lanes,
            ));
        }
        if path.starts_with("examples/candle_wasm_chat/www/") {
            add(
                &mut out,
                all,
                "source-guards",
                "browser worker runtime changed",
            )?;
            add(
                &mut out,
                all,
                "wasm-candle_wasm_chat",
                "browser example Rust/JavaScript boundary",
            )?;
            continue;
        }
        // Agent and world goldens share this prefix. Classify both before
        // provider cassettes and the owning package's generic asset rule.
        if path.starts_with("crates/rig-cassette/fixtures/effects/") {
            add(&mut out, all, "bus-verification", "consumed golden changed")?;
            add(
                &mut out,
                all,
                "default-tests",
                "goldens may be consumed by any original/native provider target",
            )?;
            add(
                &mut out,
                all,
                "ecs-parity",
                "a golden changed: replay the parity lane",
            )?;
            continue;
        }
        if let Some(name) = provider(path) {
            if let Some(owner) = provider_owner(packages, name) {
                if shared_with_minimal(path, Some((owner, name))) {
                    affected.insert("rig-cassette-minimal".to_owned());
                }
                let id = format!("provider-{name}");
                if !out.iter().any(|c| c.id == id) {
                    out.push(provider_check(owner, name,
                        "provider source or cassette: execute complete target with all capabilities, including feature-gated tests and shared safety assertions",
                    ));
                }
                continue;
            }
            // Inside a package, a file that is not a provider target is
            // ordinary package source (the corpus matrices beside the moved
            // suites): let the owning package's rule claim it. At the
            // repository root there is no such owner, so broaden.
            if !path.starts_with("crates/") {
                return Ok(broad(
                    all,
                    &format!("unknown provider/target for {path}"),
                    Lanes::ALL,
                ));
            }
        }
        if shared_with_minimal(path, None) {
            affected.insert("rig-cassette-minimal".to_owned());
        }
        if documentation(path) {
            add(&mut out, all, "docs", "documentation edit")?;
            add(
                &mut out,
                all,
                "doctests",
                "documentation edit; preserve executable documentation",
            )?;
            continue;
        }
        let owner = package(root, packages, path).and_then(|p| p["name"].as_str());
        if let Some(name) = owner.filter(|n| *n != "rig") {
            if !path.ends_with(".rs") && !path.ends_with(".md") {
                return Ok(broad(
                    all,
                    &format!("unmodeled package asset or generator {path}"),
                    Lanes::ALL,
                ));
            }
            affected.insert(name.to_string());
            continue;
        }
        return Ok(broad(
            all,
            &format!("unknown input {path}; cannot safely narrow"),
            Lanes::ALL,
        ));
    }
    if paths.iter().any(|p| p != "DEVELOPING.md") {
        add(&mut out, all, "fmt", "changed files must remain formatted")?;
    }
    // The runtimes and the effective-policy hash: an edit to rig-ecs (the
    // hash, `Materialise`), rig-agent (the runner) or rig-core (the message
    // and error rules) can stale or diverge the parity goldens, which the
    // package's own tests never replay (tests/ecs_parity/README.md).
    if affected
        .iter()
        .any(|name| matches!(name.as_str(), "rig-ecs" | "rig-agent" | "rig-core"))
    {
        add(
            &mut out,
            all,
            "ecs-parity",
            "runtime crate changed: the parity goldens may be stale or diverge",
        )?;
    }
    let mut downstream = BTreeSet::new();
    for name in affected {
        downstream.extend(consumers(packages, &name));
        let package = packages
            .iter()
            .find(|p| p["name"] == name)
            .ok_or_else(|| invalid("package vanished"))?;
        let manifest = PathBuf::from(
            package["manifest_path"]
                .as_str()
                .ok_or_else(|| invalid("missing manifest"))?,
        );
        if manifest.starts_with(root.join("examples")) {
            out.push(dynamic(
                format!("example-{name}"),
                ["test", "--locked", "-p", &name, "--all-targets"]
                    .iter()
                    .map(|s| (*s).into())
                    .collect(),
                "example edit: compile declared features and run test harnesses, not application entry points",
            ));
        } else {
            out.push(dynamic(format!("package-{name}"),["test","--locked","-p",&name,"--all-features"].iter().map(|s|(*s).into()).collect(),"package edit: unit, integration and documentation tests under all package features"));
        }
    }
    if !downstream.is_empty() {
        let mut args = vec!["check".into(), "--locked".into(), "--all-targets".into()];
        for name in downstream {
            args.extend(["-p".into(), name]);
        }
        out.push(dynamic(
            "consumers".into(),
            args,
            "Cargo metadata reverse dependencies, including examples and public API consumers",
        ));
    }
    Ok(out)
}

/// What `--quick` builds, and what the change set affects that it leaves to CI.
pub(super) struct Quick {
    pub(super) plan: Vec<Check>,
    /// One line per shared input, unowned path, reverse dependency or feature
    /// set the plan does not build.
    pub(super) deferred: Vec<String>,
    /// The check ids `--pr` selects for the same change set, given a `--base`.
    pub(super) ci: Option<Vec<String>>,
}

/// The features a quick run enables: a target's `required-features`, or for a
/// whole package the union over its targets, so `--all-targets` skips none of
/// them. The provider suites gate every cell on exactly those features. Wider
/// sets are CI's: `--all-features` on the facade builds every companion crate,
/// and elsewhere adds a second TLS stack or an ONNX runtime download.
fn required_features(package: &Value, target: Option<&str>) -> BTreeSet<String> {
    package["targets"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| target.is_none_or(|name| t["name"] == name))
        .filter_map(|t| t["required-features"].as_array())
        .flatten()
        .filter_map(|f| f.as_str().map(str::to_owned))
        .collect()
}

/// Cargo check, then the `local` nextest profile, for one package (every
/// target) or one test target. Nothing broader, and never recording.
fn quick_check(
    id: String,
    package: &str,
    features: &BTreeSet<String>,
    target: Option<&str>,
    reason: &str,
) -> Check {
    let mut selection = vec!["-p".to_owned(), package.to_owned()];
    if !features.is_empty() {
        let joined: Vec<_> = features.iter().map(String::as_str).collect();
        selection.extend(["--features".into(), joined.join(",")]);
    }
    selection.extend(match target {
        Some(name) => vec!["--test".to_owned(), name.to_owned()],
        None => Vec::new(),
    });
    let cargo = |head: &[&str], tail: &[&str]| Step {
        program: "cargo".into(),
        args: head
            .iter()
            .map(|s| (*s).to_owned())
            .chain(selection.iter().cloned())
            .chain(tail.iter().map(|s| (*s).to_owned()))
            .collect(),
        env: BTreeMap::new(),
    };
    let all_targets: &[&str] = if target.is_none() {
        &["--all-targets"]
    } else {
        &[]
    };
    Check {
        id,
        steps: vec![
            cargo(&["check", "--locked"], all_targets),
            // A live-only suite or an example has no test that runs here;
            // that is not a failure.
            cargo(
                &[
                    "nextest",
                    "run",
                    "--locked",
                    "--profile",
                    "local",
                    "--no-tests=warn",
                ],
                &[],
            ),
        ],
        reason: reason.into(),
    }
}

/// `--quick`: map each changed path to its owning package, or for provider
/// source and cassettes to that provider's test target, with the same helpers
/// as `--changed`. It never escalates. Shared inputs, unowned paths and
/// reverse dependencies are listed for CI instead of built.
pub(super) fn quick(
    root: &Path,
    metadata: &Value,
    opts: &Options,
    paths: &BTreeSet<String>,
    all: &[Check],
) -> Result<Quick> {
    let packages = metadata["packages"]
        .as_array()
        .ok_or_else(|| invalid("Cargo metadata missing packages"))?;
    let find = |name: &str| {
        packages
            .iter()
            .find(|p| p["name"] == name)
            .ok_or_else(|| invalid(format!("package {name} vanished")))
    };
    let mut deferred = Vec::new();
    let mut owned = BTreeSet::new();
    let mut targets = BTreeSet::new();
    let mut minimal = false;
    for path in paths {
        if shared_input(path) {
            deferred.push(format!("{path}: shared build or verification input"));
            continue;
        }
        // nextest runs no doctests, so building a package for a Markdown
        // edit would check nothing that changed.
        if documentation(path) || path.ends_with(".md") {
            deferred.push(format!("{path}: documentation"));
            continue;
        }
        if let Some(name) = provider(path)
            && let Some(owner) = provider_owner(packages, name)
        {
            minimal |= shared_with_minimal(path, Some((owner, name)));
            targets.insert((owner.to_owned(), name.to_owned()));
            continue;
        }
        minimal |= shared_with_minimal(path, None);
        // The facade's own files are its re-exports and its tests' shared
        // modules; `--changed` treats them as unknown too.
        match package(root, packages, path)
            .and_then(|p| p["name"].as_str())
            .filter(|n| *n != "rig")
        {
            Some(name) => {
                owned.insert(name.to_owned());
            }
            None => deferred.push(format!("{path}: no owning package or test target")),
        }
    }
    if minimal && !owned.contains("rig-cassette-minimal") {
        deferred.push("rig-cassette-minimal: compiles the changed replay sources by path".into());
    }
    // A package check already builds every one of its targets.
    targets.retain(|(owner, _)| !owned.contains(owner));
    let mut out = Vec::new();
    for name in &owned {
        let package = find(name)?;
        let features = required_features(package, None);
        out.push(quick_check(
            format!("package-{name}"),
            name,
            &features,
            None,
            "package edit: its targets under default and required features",
        ));
        let table = package["features"].as_object();
        let enabled: BTreeSet<&str> = table
            .and_then(|f| f.get("default"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .chain(features.iter().map(String::as_str))
            .collect();
        let other: Vec<&str> = table
            .into_iter()
            .flat_map(|f| f.keys())
            .map(String::as_str)
            .filter(|f| *f != "default" && !enabled.contains(f))
            .collect();
        if !other.is_empty() {
            deferred.push(format!(
                "{name}: features not enabled here: {}",
                other.join(", ")
            ));
        }
        let dependents: Vec<String> = consumers(packages, name)
            .into_iter()
            .filter(|n| !owned.contains(n))
            .collect();
        if !dependents.is_empty() {
            deferred.push(format!(
                "reverse dependencies of {name}: {}",
                dependents.join(", ")
            ));
        }
    }
    for (owner, name) in &targets {
        out.push(quick_check(
            format!("provider-{name}"),
            owner,
            &required_features(find(owner)?, Some(name)),
            Some(name),
            "provider source or cassette: its test target under its required features",
        ));
    }
    if !out.is_empty() {
        deferred.push(
            "the checked packages' formatting, Clippy, doctests, WASM builds and other feature sets"
                .into(),
        );
    }
    let ci = match &opts.base {
        Some(base) if !paths.is_empty() => {
            let pr = Options {
                mode: Mode::Pr,
                base: Some(base.clone()),
                dry_run: opts.dry_run,
                check: None,
            };
            Some(
                plan(root, metadata, &pr, paths, all)?
                    .into_iter()
                    .map(|c| c.id)
                    .collect(),
            )
        }
        _ => None,
    };
    Ok(Quick {
        plan: out,
        deferred,
        ci,
    })
}
