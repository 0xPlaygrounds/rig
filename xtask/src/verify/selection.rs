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
        vec!["ls-files", "--others", "--exclude-standard", "-z"],
    ] {
        for p in output(root, "git", &args)?
            .split('\0')
            .filter(|s| !s.is_empty())
        {
            paths.insert(p.into());
        }
    }
    Ok(paths)
}
fn broad(all: &[Check], reason: &str, full: bool) -> Vec<Check> {
    all.iter()
        .filter(|c| {
            full || !["full-tests", "dependency-floors", "workspace-check"].contains(&c.id.as_str())
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
        add(&mut out, all, id, "explicit check; always selected")?;
        return Ok(out);
    }
    if opts.mode == Mode::Full {
        return Ok(broad(
            all,
            "exhaustive supported repository verification",
            true,
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
        let triggers: Vec<_> = paths.iter().filter(|p| checks::full_lane(p)).collect();
        let fallback = local.iter().find(|check| check.id == "full-tests");
        let needs_full = !triggers.is_empty() || fallback.is_some();
        let reason = format!(
            "complete intended PR diff; preserves required lanes; full-lane inputs: {triggers:?}; {}",
            fallback.map_or("no conservative full fallback", |c| c.reason.as_str())
        );
        return Ok(broad(all, &reason, needs_full));
    }
    let packages = metadata["packages"]
        .as_array()
        .ok_or_else(|| invalid("Cargo metadata missing packages"))?;
    let mut out = Vec::new();
    let mut affected = BTreeSet::new();
    for path in paths {
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
            return Ok(broad(
                all,
                &format!("conservative fallback: shared/build/verification input {path}"),
                true,
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
                    true,
                ));
            }
            let id = format!("provider-{name}");
            if !out.iter().any(|c| c.id == id) {
                out.push(dynamic(
                    id,
                    ["nextest", "run", "--locked", "-p", "rig", "--all-features", "--test", name, "--retries", "0"]
                        .iter().map(|s| (*s).into()).collect(),
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
                    true,
                ));
            }
            affected.insert(name.to_string());
            continue;
        }
        return Ok(broad(
            all,
            &format!("unknown input {path}; cannot safely narrow"),
            true,
        ));
    }
    if !paths.is_empty() {
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
