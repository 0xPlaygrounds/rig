//! Check definitions shared by local plans and CI's independently scheduled jobs.
//! `tests/core/mod.rs` includes this file by path for its dependency and
//! conformance guards, so it must stay std-only and free of
//! `crate::` references.
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Step {
    pub(crate) program: String,
    pub(crate) args: Vec<String>,
    pub(crate) env: BTreeMap<String, String>,
}
impl Step {
    pub(crate) fn new(program: &str, args: &[&str]) -> Self {
        Self {
            program: program.into(),
            args: args.iter().map(|s| (*s).into()).collect(),
            env: BTreeMap::new(),
        }
    }
    pub(crate) fn env(mut self, k: &str, v: &str) -> Self {
        self.env.insert(k.into(), v.into());
        self
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Check {
    pub(crate) id: String,
    pub(crate) steps: Vec<Step>,
    pub(crate) reason: String,
}

fn cargo(args: &[&str]) -> Step {
    Step::new("cargo", args)
}
fn check(id: &str, steps: Vec<Step>) -> Check {
    Check {
        id: id.into(),
        steps,
        reason: String::new(),
    }
}
pub(super) fn all() -> Vec<Check> {
    let mut checks = vec![
        check("fmt", vec![cargo(&["fmt", "--all", "--", "--check"])]),
        check(
            "source-guards",
            vec![
                Step::new(
                    "bash",
                    &[".github/scripts/check-migrating-guide-preamble.sh"],
                ),
                Step::new("@fixture-paths", &[]),
                Step::new(
                    "node",
                    &[
                        "--test",
                        "examples/candle_wasm_chat/www/worker-runtime.test.mjs",
                    ],
                ),
            ],
        ),
        check(
            "tooling",
            vec![
                Step::new("@layout", &[]),
                Step::new("@packaging", &[]),
                Step::new("@wires", &[]),
                Step::new("@bevy-sources", &[]),
                Step::new(
                    "python3",
                    &[
                        "-B",
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "scripts",
                        "-p",
                        "test_dependency_floors.py",
                    ],
                ),
                cargo(&["test", "--locked", "-p", "xtask"]),
                cargo(&[
                    "clippy",
                    "--locked",
                    "-p",
                    "xtask",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                ]),
            ],
        ),
        check(
            "clippy",
            vec![cargo(&[
                "clippy",
                "--locked",
                "--all-features",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ])],
        ),
        check(
            "default-check",
            // A dependency's #[cfg(test)] bodies are not compiled by the
            // facade's test targets after moving helpers into this crate.
            vec![cargo(&[
                "check",
                "--locked",
                "-p",
                "rig",
                "-p",
                "rig-test-support",
                "--tests",
            ])],
        ),
        check(
            "default-tests",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "--features",
                "bedrock",
                "--retries",
                "2",
                "-E",
                "not binary(macro_hygiene) and not (package(rig-cassette) and (binary(verify) or binary(world_replay) or test(/(^|::)(ecs|corpus)_/))) and not (package(rig) and test(golden_pairing))",
            ])],
        ),
        // Parity cells have one lane owner; default-tests excludes them. Each
        // cell asserts its own runtime's record against the cell; no log is
        // compared across runtimes. They
        // moved with the cassette-backed provider suites, so the predicate is
        // `package(rig-cassette)` now, qualified away from the two absorbed
        // verification binaries whose `corpus_` module names would otherwise
        // match it. The golden-pairing guard stays in the facade's `core`
        // target. Excluding them never costs the cassette engine's own unit
        // tests their default owner. Distinct feature/target runs (core-all,
        // wasm, loom) remain separate coverage.
        check(
            "ecs-parity",
            vec![
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "-p",
                    "rig",
                    "-p",
                    "rig-cassette",
                    // Extracted ECS helper regressions retain this graph too.
                    "-p",
                    "rig-test-support",
                    "--features",
                    "bedrock",
                    // The same retries as `test`: a divergence or a stale
                    // golden is deterministic and fails every attempt, while
                    // a handful of cells carry wall-clock deadlines (a 30 s
                    // session budget, a stream drained at the replay's pace)
                    // that a loaded machine can miss once. Without retries
                    // one such miss cancels the lane mid-run.
                    "--retries",
                    "2",
                    "-E",
                    "(package(rig-cassette) and test(/(^|::)(ecs|corpus)_/) and not binary(verify) and not binary(world_replay)) or (package(rig-test-support) and test(/(^|::)(ecs|corpus)_/)) or (package(rig) and test(golden_pairing))",
                ]),
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "-p",
                    "rig-cassette-minimal",
                    "--all-features",
                    "-E",
                    "package(rig-cassette-minimal) and binary(world_replay)",
                    "--retries",
                    "0",
                ]),
                // The default-member graph enables different dependency
                // implementations (including JSON/allocator features). Keep
                // that execution separately from the standalone root graph.
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "--features",
                    "bedrock",
                    "--retries",
                    "2",
                    "-E",
                    "(package(rig-cassette) and test(/(^|::)(ecs|corpus)_/) and not binary(verify) and not binary(world_replay)) or (package(rig) and test(golden_pairing)) or (package(rig-cassette) and binary(world_replay))",
                ]),
            ],
        ),
        check(
            "core-all",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "-p",
                "rig-core",
                "-p",
                "rig-reqwest",
                "-p",
                "rig-tungstenite",
                "-p",
                "rig-agent",
                "-p",
                "rig-rmcp",
                // Keep relocated bus regressions in this feature graph,
                // rather than enabling cassette's HTTP dev-dependencies.
                "-p",
                "rig-cassette-minimal",
                "--all-features",
                "--profile",
                "guards",
                "-E",
                "not binary(macro_hygiene) and (not package(rig-cassette-minimal) or (binary(effect_log) and test(/^agent_replay::/)))",
            ])],
        ),
        check(
            "bus-verification",
            vec![
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "-p",
                    "rig-cassette-minimal",
                    "--all-features",
                    "--retries",
                    "0",
                    "-E",
                    "package(rig-cassette-minimal) and (binary(verify) or binary(effect_log))",
                ]),
                // Preserve the former default sweep's dependency graph and
                // retry policy. The nested runner above preserves the former
                // minimal serde_json graph; this execution also exercises
                // preserve_order/float_roundtrip and the unified allocator.
                // Only the `verify` binary moves out of the default sweep.
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "--features",
                    "bedrock",
                    "--retries",
                    "2",
                    "-E",
                    "package(rig-cassette) and binary(verify)",
                ]),
            ],
        ),
        check(
            "macro-hygiene",
            vec![cargo(&[
                "test",
                "--locked",
                "-p",
                "rig-core",
                "--test",
                "macro_hygiene",
            ])],
        ),
        check(
            "conformance",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "-p",
                "rig-core",
                "-p",
                "rig-tungstenite",
                "-p",
                "rig-candle",
                "-p",
                "rig-gemini-grpc",
                "--all-features",
                // nextest -E filters execution after Cargo compilation. Select
                // the same integration targets before compiling their harnesses.
                "--test",
                "streaming_conformance",
                "--test",
                "streaming_conformance_websocket",
                "--test",
                "driver_adoption",
                "--retries",
                "0",
                "-E",
                "binary(streaming_conformance) + binary(streaming_conformance_websocket) + binary(driver_adoption)",
            ])],
        ),
        check(
            "derive",
            vec![cargo(&["test", "--locked", "-p", "rig-derive"])],
        ),
        check(
            "doctests",
            vec![cargo(&[
                "test",
                "--locked",
                "--doc",
                "--workspace",
                "--all-features",
            ])],
        ),
        check(
            "docs",
            vec![
                cargo(&[
                    "doc",
                    "--locked",
                    "--workspace",
                    "--no-deps",
                    "--all-features",
                ])
                .env("RUSTDOCFLAGS", "-D warnings"),
            ],
        ),
        check(
            "loom",
            vec![
                cargo(&[
                    "test",
                    "--locked",
                    "-p",
                    "rig-agent",
                    "--lib",
                    "--release",
                    "loom_",
                ])
                .env("RUSTFLAGS", "--cfg rig_loom")
                .env("LOOM_MAX_PREEMPTIONS", "3"),
            ],
        ),
        check(
            "full-tests",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "--workspace",
                // The same sources run here through rig-cassette. The nested
                // runner only adds coverage when selected without this graph.
                "--exclude",
                "rig-cassette-minimal",
                "--all-features",
                "--retries",
                "2",
                "-E",
                "not package(rig-derive)",
            ])],
        ),
        // No separate `cargo check --workspace --all-features --all-targets`:
        // full-tests compiles workspace lib, bin, test and example targets
        // under the same feature set (Cargo's default `test` target selection),
        // except the separately executed minimal replay runner. Clippy checks
        // the default members' --all-targets, and every bench has an explicit
        // locked compile-only owner, enforced by the verification planner tests.
        // Existing repository floor checker; new verification tooling is Rust.
        check(
            "dependency-floors",
            vec![
                Step::new("python3", &["scripts/check-dependency-floors.py"])
                    .env("PYTHONUNBUFFERED", "1"),
            ],
        ),
    ];
    for package in [
        "rig-core",
        "rig-cassette",
        "rig-ecs",
        "rig-reqwest",
        "rig-agent",
        "rig",
        "rig-candle",
        "candle_wasm_chat",
    ] {
        let mut steps = vec![cargo(&[
            "check",
            "--locked",
            "--package",
            package,
            "--target",
            "wasm32-unknown-unknown",
        ])];
        if package == "rig-ecs" {
            steps.push(cargo(&[
                "check",
                "--locked",
                "-p",
                package,
                "--no-default-features",
                "--lib",
            ]));
            steps.push(cargo(&[
                "check",
                "--locked",
                "-p",
                package,
                "--no-default-features",
                "--lib",
                "--target",
                "wasm32-unknown-unknown",
            ]));
        }
        if package == "rig-cassette" {
            for features in ["agent", "ecs", "agent,ecs"] {
                steps.push(cargo(&[
                    "check",
                    "--locked",
                    "-p",
                    package,
                    "--no-default-features",
                    "--features",
                    features,
                    "--lib",
                    "--target",
                    "wasm32-unknown-unknown",
                ]));
            }
        }
        if ["rig-core", "rig-ecs"].contains(&package) {
            steps.push(cargo(&[
                "check",
                "--locked",
                "--package",
                package,
                "--all-features",
                "--target",
                "wasm32-unknown-unknown",
            ]));
        }
        checks.push(check(&format!("wasm-{package}"), steps));
    }
    for (package, test) in [
        ("rig-agent", "bus_wasm"),
        ("rig-ecs", "bus_wasm"),
        ("rig-ecs", "run_wasm"),
    ] {
        checks.push(check(
            &format!("wasm-{package}-{test}"),
            vec![
                cargo(&[
                    "test",
                    "--locked",
                    "--package",
                    package,
                    "--target",
                    "wasm32-unknown-unknown",
                    "--test",
                    test,
                ])
                .env(
                    "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER",
                    "wasm-bindgen-test-runner",
                ),
            ],
        ));
    }
    for (package, message) in [
        ("rig-rmcp", "the `rmcp` feature is native-only"),
        (
            "rig-tungstenite",
            "rig-tungstenite is a native websocket backend",
        ),
    ] {
        checks.push(check(
            &format!("native-only-{package}"),
            vec![Step::new("@native-only", &[package, message])],
        ));
    }
    checks
}
/// Inputs whose change runs the full runtime lane (`full-tests`): the
/// service-backed integration suites, the crates they cover, the facade
/// feature-forwarding guard, and the shared build/verification inputs whose
/// effect on those suites cannot be narrowed. CI's `slow.yaml` asks this
/// planner (`verify --lanes`) rather than keeping its own path filter.
pub(super) fn full_lane(path: &str) -> bool {
    [
        "Cargo.toml",
        "Cargo.lock",
        ".github/workflows/slow.yaml",
        "tests/tool_facade_features.rs",
        "tests/integrations.rs",
    ]
    .contains(&path)
        || path.starts_with("tests/integrations/")
        || path.starts_with("test-support/")
        || path.starts_with(".github/actions/")
        // `crates/*/Cargo.toml`: the workspace members' manifests, not the
        // nested compile fixtures beneath them (those have their own owners).
        || path
            .strip_prefix("crates/")
            .and_then(|rest| rest.strip_suffix("/Cargo.toml"))
            .is_some_and(|name| !name.contains('/'))
        || [
            "rig-lancedb",
            "rig-mongodb",
            "rig-neo4j",
            "rig-postgres",
            "rig-qdrant",
            "rig-scylladb",
            "rig-sqlite",
            "rig-vectorize",
        ]
        .iter()
        .any(|p| path.starts_with(&format!("crates/{p}/")))
        || path.starts_with("xtask/")
        || path.starts_with(".config/")
}
/// Inputs whose change runs `dependency-floors`: everything Cargo's resolver
/// reads (manifests, the lockfile, the toolchain, Cargo configuration), the
/// floor checker, and the verification tooling that defines the check.
/// Same single source as `full_lane`. A source-only edit
/// that starts using an API newer than a declared floor is deliberately not
/// a trigger: the scheduled run, the merge queue and the release gate still
/// execute the floors on every landed combination.
pub(super) fn floor_lane(path: &str) -> bool {
    [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "scripts/check-dependency-floors.py",
        "scripts/test_dependency_floors.py",
        ".github/workflows/slow.yaml",
    ]
    .contains(&path)
        || path.ends_with("/Cargo.toml")
        || path.starts_with(".cargo/")
        || path.starts_with(".github/actions/")
        || path.starts_with("xtask/")
}
