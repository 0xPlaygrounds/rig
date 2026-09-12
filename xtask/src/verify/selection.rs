use super::*;
use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
};

pub(super) fn changes(
    root: &Path,
    opts: &Options,
    target: Option<&Path>,
) -> Result<BTreeSet<String>> {
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
        for p in output(root, "git", &args)?
            .split('\0')
            .filter(|s| !s.is_empty())
        {
            paths.insert(p.into());
        }
    }
    paths.extend(execute::other_inputs(root, target)?);
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
fn provider(path: &str) -> Option<&str> {
    for prefix in ["tests/cassettes/", "tests/providers/"] {
        if let Some(rest) = path.strip_prefix(prefix) {
            return rest.split('/').next();
        }
    }
    path.strip_prefix("tests/")
        .filter(|s| !s.contains('/'))
        .and_then(|s| s.strip_suffix(".rs"))
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
fn provider_check(name: &str, reason: &str) -> Check {
    dynamic(
        format!("provider-{name}"),
        [
            "nextest",
            "run",
            "--locked",
            "-p",
            "rig",
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
        let warming = [
            ("default-test-build", "default-tests"),
            ("full-test-build", "full-tests"),
        ];
        if let Some((_, source)) = warming.iter().find(|(alias, _)| *alias == id) {
            add(
                &mut out,
                all,
                source,
                "cache warming only: compile test artifacts; executes no tests",
            )?;
            if let Some(check) = out.first_mut() {
                check.id = id.into();
                for step in &mut check.steps {
                    // nextest 0.9.67 rejects --retries with --no-run.
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
    if opts.mode == Mode::Pr {
        if paths.iter().any(|p| {
            p == "CHANGELOG.md"
                || p == "MIGRATING.md"
                || p.starts_with("docs/migrations/")
                || (p.starts_with("crates/") && p.ends_with("/CHANGELOG.md"))
        }) {
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
            reuse: false,
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
        let mut required = broad(all, &reason, lanes);
        // The fast PR lane skips facade-build-tests. Preserve this targeted
        // owner when its generated lock selected it without a full-lane trigger.
        if !lanes.full
            && let Some(facade) = local
                .iter()
                .find(|c| c.id == "provider-tool_facade_features")
        {
            required.push(facade.clone());
        }
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
        if let Some(owner) = preflight::fixture_lock_owner(path) {
            let reason = "generated nested-workspace lockfile: execute its owning fixture tests";
            if owner == "provider-tool_facade_features" {
                if !out.iter().any(|c| c.id == owner) {
                    out.push(provider_check("tool_facade_features", reason));
                }
            } else {
                add(&mut out, all, owner, reason)?;
            }
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
        if path == "Cargo.lock"
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
                "tests/common/",
                "tests/integrations/",
            ]
            .iter()
            .any(|p| path.starts_with(p))
        {
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
        if path.starts_with("tests/ecs_parity/") {
            add(
                &mut out,
                all,
                "tooling",
                "scenario catalog or semantic contract changed",
            )?;
            continue;
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
        if path.starts_with("crates/rig-verify/fixtures/") {
            add(&mut out, all, "bus-verification", "consumed golden changed")?;
            add(
                &mut out,
                all,
                "default-tests",
                "goldens may be consumed by any original/native provider target",
            )?;
            continue;
        }
        if let Some(name) = provider(path) {
            let root_package = packages
                .iter()
                .find(|p| p["name"] == "rig")
                .ok_or_else(|| invalid("missing rig package"))?;
            let target = root_package["targets"].as_array().and_then(|ts| {
                ts.iter().find(|t| {
                    t["name"] == name
                        && t["kind"]
                            .as_array()
                            .is_some_and(|k| k.iter().any(|x| x == "test"))
                })
            });
            if target.is_none() {
                return Ok(broad(
                    all,
                    &format!("unknown provider/target for {path}"),
                    Lanes::ALL,
                ));
            }
            let id = format!("provider-{name}");
            if !out.iter().any(|c| c.id == id) {
                out.push(provider_check(name,
                    "provider source or cassette: execute complete target with all capabilities, including feature-gated tests and shared safety assertions",
                ));
            }
            continue;
        }
        if ["README.md", "CONTRIBUTING.md", "AGENTS.md", "DEVELOPING.md"].contains(&path.as_str())
            || path.starts_with("docs/")
        {
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
