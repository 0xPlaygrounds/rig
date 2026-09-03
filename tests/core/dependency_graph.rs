//! Dependency-graph invariants for the runtime/transport-agnostic split.
//!
//! The crate boundaries that keep rig usable from non-tokio hosts are
//! enforced here rather than by convention: rig-core and rig-agent carry no
//! runtime or transport, MCP is an rig-core-only leaf, and the facade with
//! only `agent` + `derive` pulls in none of tokio / reqwest / rmcp. Each check
//! asks Cargo for the resolved normal (non-dev, non-build) dependency graph of
//! one package and asserts the forbidden crates are absent.

use std::process::Command;

/// `cargo tree -e normal --prefix none` for `package` with extra `args`,
/// returned as the set of package names in the graph.
fn normal_dependency_names(package: &str, args: &[&str]) -> Vec<String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "tree", "--locked", "-p", package, "-e", "normal", "--prefix", "none",
        ])
        .args(args)
        .output()
        .expect("cargo tree runs");
    assert!(
        output.status.success(),
        "cargo tree -p {package} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("cargo tree output is utf-8")
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

fn assert_absent(package: &str, args: &[&str], forbidden: &[&str]) {
    let names = normal_dependency_names(package, args);
    for crate_name in forbidden {
        assert!(
            !names.iter().any(|name| name == crate_name),
            "`{package}` ({}) must not depend on `{crate_name}` through its normal dependencies",
            if args.is_empty() {
                "default features".to_owned()
            } else {
                args.join(" ")
            }
        );
    }
}

#[test]
fn rig_core_is_runtime_and_transport_free() {
    assert_absent("rig-core", &[], &["tokio", "reqwest"]);
    assert_absent("rig-core", &["--all-features"], &["tokio", "reqwest"]);
}

/// The bus runtime carries no runtime or transport either, and the
/// dependency runs one way: rig-core knows nothing of the crate that
/// drives its handlers.
#[test]
fn rig_bus_is_runtime_free_and_below_rig_core() {
    assert_absent("rig-bus", &[], &["tokio", "reqwest"]);
    assert_absent("rig-effect-log", &[], &["tokio", "reqwest"]);
    assert_absent(
        "rig-core",
        &["--all-features"],
        &["rig-bus", "rig-effect-log"],
    );
    assert_absent("rig-bus", &[], &["rig-effect-log"]);
}

/// With default features on, and — the shape a host that steps `AgentRun`
/// itself depends on — with them off: rig-agent is a runtime-free crate.
#[test]
fn rig_agent_carries_no_runtime_or_mcp() {
    assert_absent("rig-agent", &[], &["tokio", "rmcp"]);
    assert_absent(
        "rig-agent",
        &["--no-default-features"],
        &["tokio", "rmcp", "reqwest"],
    );
}

/// rig-agent's `run` layer and rig-core's transcript invariants never await:
/// they are data and transitions only, so a futures loop and an ECS schedule
/// can step the same code. Checked at the source level because both crates
/// legitimately depend on `futures` elsewhere.
#[test]
fn run_vocabulary_sources_contain_no_async() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let sources = [
        "crates/rig-core/src/transcript.rs",
        "crates/rig-agent/src/run/mod.rs",
        "crates/rig-agent/src/run/output.rs",
        "crates/rig-agent/src/run/patch.rs",
        "crates/rig-agent/src/run/policy.rs",
        "crates/rig-agent/src/run/prepare.rs",
        "crates/rig-agent/src/run/response.rs",
        "crates/rig-agent/src/run/spec.rs",
        "crates/rig-agent/src/run/streamed.rs",
        "crates/rig-agent/src/run/transcript.rs",
    ];
    let mut offenders = Vec::new();
    for relative in sources {
        let path = root.join(relative);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("{relative} is readable: {err}"));
        if text.contains("async fn")
            || text.contains(".await")
            || text.contains("async_stream")
            || text.contains("futures::")
        {
            offenders.push(relative);
        }
    }
    assert!(
        offenders.is_empty(),
        "the run vocabulary and AgentRun must stay sans-IO (no `async fn`/`.await`/`async_stream`/`futures::`); found in: {offenders:?}"
    );
}

#[test]
fn rig_rmcp_depends_on_rig_core_only() {
    assert_absent("rig-rmcp", &[], &["rig-agent"]);
}

#[test]
fn facade_agent_derive_is_runtime_transport_and_mcp_free() {
    assert_absent(
        "rig",
        &["--no-default-features", "--features", "agent,derive"],
        &["tokio", "reqwest", "rmcp"],
    );
}
