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

/// rig-core carries the bundled transport, but only when asked: the `reqwest`
/// feature is what pulls `reqwest` and `tokio` in, so a consumer that does not
/// name it — every wasm and custom-transport build — pays for neither.
/// `--all-features` necessarily includes it, which is why the ceiling here is
/// the default feature set rather than the full one.
#[test]
fn rig_core_is_runtime_and_transport_free_by_default() {
    assert_absent("rig-core", &[], &["tokio", "reqwest"]);
    assert_absent(
        "rig-core",
        &["--no-default-features"],
        &["tokio", "reqwest"],
    );
}

/// Picking a TLS flavor is not the same as asking for the transport. The
/// vector-store crates forward `rustls`/`native-tls` for their own graphs, so
/// a consumer that names one must not silently acquire reqwest and tokio.
#[test]
fn rig_core_tls_flavor_does_not_imply_the_transport() {
    for flavor in ["rustls", "native-tls"] {
        assert_absent(
            "rig-core",
            &["--no-default-features", "--features", flavor],
            &["tokio", "reqwest"],
        );
    }
}

/// Every manifest that pulls rig-core's transport in also names a TLS flavor.
///
/// `rig-reqwest` defaulted to `rustls`; rig-core does not, so a bare
/// `features = ["reqwest"]` compiles and then fails every HTTPS request inside
/// reqwest's connector. A workspace build hides it — feature unification
/// supplies `rustls` from the facade — so read the manifests directly.
///
/// Whitespace is normalized before matching so a wrapped dependency table
/// (rustfmt/taplo will wrap these lines as feature lists grow) cannot silence
/// the guard, and `examples/` and the root facade are covered alongside
/// `crates/`.
#[test]
fn transport_dependents_name_a_tls_flavor() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut manifests = vec![root.join("Cargo.toml")];
    for dir in ["crates", "examples"] {
        let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
            continue;
        };
        for entry in entries {
            let manifest = entry.expect("dir entry").path().join("Cargo.toml");
            if manifest.is_file() {
                manifests.push(manifest);
            }
        }
    }
    manifests.sort();

    let mut offenders = Vec::new();
    for manifest in manifests {
        let text = std::fs::read_to_string(&manifest).expect("readable manifest");
        // Collapse the file so a dependency table spanning several lines reads
        // the same as a one-liner.
        let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
        for (index, _) in flat.match_indices("rig-core = {") {
            let Some(close) = flat[index..].find('}') else {
                continue;
            };
            let entry = &flat[index..index + close];
            if !entry.contains("\"reqwest\"") {
                continue;
            }
            if !(entry.contains("\"rustls\"") || entry.contains("\"native-tls\"")) {
                offenders.push(format!("{}: {entry}", manifest.display()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "rig-core transport dependencies without a TLS flavor: {offenders:#?}"
    );
}

/// The other half of the contract: naming the feature does deliver the
/// transport. Without this, `reqwest` could silently stop being wired up and
/// only a downstream compile error would notice.
#[test]
fn rig_core_reqwest_feature_pulls_the_transport() {
    let names = normal_dependency_names(
        "rig-core",
        &["--no-default-features", "--features", "reqwest"],
    );
    for expected in ["reqwest", "tokio"] {
        assert!(
            names.iter().any(|name| name == expected),
            "`rig-core --features reqwest` must depend on `{expected}`"
        );
    }
}

#[test]
fn rig_agent_carries_no_runtime_or_mcp() {
    assert_absent("rig-agent", &[], &["tokio", "rmcp"]);
}

/// The protocol crate is data and transitions only: no runtime, no transport,
/// no futures, no hooks (which live in rig-agent). A second driver — an ECS
/// plugin — depends on it precisely because of this.
#[test]
fn rig_run_is_pure_protocol() {
    assert_absent(
        "rig-run",
        &[],
        // `futures`/`async-stream` are not listed: rig-core itself depends on
        // them (stream and boxed-future vocabulary), so they are in every
        // rig-run tree; the source-level check below is what keeps the
        // protocol itself from awaiting.
        &["tokio", "reqwest", "rig-agent"],
    );
}

/// Same invariant from the inside: the protocol never awaits.
#[test]
fn rig_run_sources_contain_no_async() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/rig-run/src");
    let mut offenders = Vec::new();
    for entry in std::fs::read_dir(&root).expect("rig-run/src is readable") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).expect("source is utf-8");
            if text.contains("async fn")
                || text.contains(".await")
                || text.contains("async_stream")
                || text.contains("futures::")
            {
                offenders.push(path.display().to_string());
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "rig-run must stay sans-IO (no `async fn`/`.await`/`async_stream`/`futures::`); found in: {offenders:?}"
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
