//! The random-id guard, run by `verify --check source-guards`: nothing the
//! engine mints is random.
//!
//! A run replays from its effect log only if every value the program
//! produces is a function of what it received. Tool-call handles for wires
//! that carry no id derive from the block that assembled the call or the
//! call's index, so randomness is drawn only at the sites listed here, none
//! of which can reach a request, a message or an effect record.

use std::path::Path;

/// Where randomness is allowed to be drawn, and why.
const FASTRAND_SITES: &[(&str, &str)] = &[
    ("crates/rig-core/src/id.rs", "the id generator itself"),
    (
        "crates/rig-ecs/src/retry.rs",
        "retry backoff jitter; a wait, never a request or a record",
    ),
    (
        "crates/rig-core/src/conversation/lineage.rs",
        "a saved log's lineage token; in the session log's header, never a request, a message or an effect record",
    ),
    (
        "crates/rig-core/src/vector_store/lsh.rs",
        "the LSH index's hyperplanes, from a constant seed: the same on every run, never a request or a record",
    ),
    (
        "crates/rig-core/benches/vector_search.rs",
        "seeded benchmark vectors; a bench input, never a request or a record",
    ),
];

/// Where `id::generate` may be called, and why.
const GENERATE_SITES: &[(&str, &str)] = &[
    (
        "crates/rig-core/src/id.rs",
        "the generator's own entry point",
    ),
    (
        "crates/rig-core/src/providers/copilot/mod.rs",
        "an `x-request-id` transport header",
    ),
    (
        "crates/rig-core/src/providers/internal/mod.rs",
        "a dialect's session transport header, without a cache key",
    ),
];

/// Every non-comment line naming `needle` in a non-test `.rs` file under
/// `dir` outside `allowed`, as `path:line: text` with `path` relative to
/// `workspace`.
fn offenders(
    workspace: &Path,
    dir: &Path,
    needle: &str,
    allowed: &[(&str, &str)],
) -> Result<Vec<String>, String> {
    let mut offenders = Vec::new();
    for path in crate::support::files_under(dir, Some("rs"))? {
        let relative = path
            .strip_prefix(workspace)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if relative.contains("/tests/") || relative.ends_with("tests.rs") {
            continue;
        }
        if allowed.iter().any(|(site, _)| *site == relative) {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{relative} is readable: {error}"))?;
        for (number, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            if line.contains(needle) {
                offenders.push(format!("{relative}:{}: {}", number + 1, line.trim()));
            }
        }
    }
    Ok(offenders)
}

fn generator_offenders(workspace: &Path, dir: &Path) -> Result<Vec<String>, String> {
    offenders(workspace, dir, "id::generate", GENERATE_SITES)
}

/// Fail when `fastrand` or `id::generate` is used under `workspace/crates`
/// outside its allowed sites, or when an allowed site no longer exists.
pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    for (site, reason) in FASTRAND_SITES.iter().chain(GENERATE_SITES) {
        if !workspace.join(site).exists() {
            return Err(format!(
                "allowed site `{site}` ({reason}) no longer exists; drop it"
            ));
        }
    }
    let crates = workspace.join("crates");
    let fastrand = offenders(workspace, &crates, "fastrand::", FASTRAND_SITES)?;
    if !fastrand.is_empty() {
        return Err(format!(
            "`fastrand` is drawn outside its allowed sites:\n{}",
            fastrand.join("\n")
        ));
    }
    let generate = generator_offenders(workspace, &crates)?;
    if !generate.is_empty() {
        return Err(format!(
            "`id::generate` is called outside the two transport headers:\n{}",
            generate.join("\n")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
