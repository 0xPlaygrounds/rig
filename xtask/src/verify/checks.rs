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
/// The rig agent's packages, none of them a default member: the kernel,
/// the core and its libraries, every plugin crate of `plugins/`, and their
/// test support.
pub(crate) const HARNESS_PACKAGES: [&str; 14] = [
    "rig-ecs",
    "rig-tools",
    "rig-harness",
    "rig-harness-protocol",
    "rig-basics",
    "rig-inspect",
    "rig-steel",
    "rig-tui",
    "rig-claude-cloak",
    "rig-traffic",
    "rig-mcp-client",
    "rig-acp",
    "rig-harness-test-support",
    "rig-defaults-test-support",
];
/// The plugin corpus (`corpus/*`), as one cargo package glob: plugins
/// written against the harness's public API to shape and prove the kernel,
/// one per module of a few `publish = false` packages named
/// `corpus-<directory>`, so their tests link a few binaries, not one each.
/// None is a default member; CI builds, lints and tests every one beside
/// [`HARNESS_PACKAGES`], so a kernel change that breaks one fails CI.
pub(crate) const CORPUS_PACKAGES: &str = "corpus-*";
/// cargo with `head`, then `-p` for each of [`HARNESS_PACKAGES`] and
/// [`CORPUS_PACKAGES`], then `tail`.
fn harness(head: &[&str], tail: &[&str]) -> Step {
    let mut args = head.to_vec();
    for package in HARNESS_PACKAGES.into_iter().chain([CORPUS_PACKAGES]) {
        args.extend(["-p", package]);
    }
    args.extend(tail);
    cargo(&args)
}
/// The ignored tests of `package`'s integration binary under `module`, run
/// alone.
fn guard(id: &str, package: &str, module: &str) -> Check {
    let filter = format!("test(/^{module}/)");
    check(
        id,
        vec![cargo(&[
            "nextest",
            "run",
            "--locked",
            "-p",
            package,
            "--test",
            "integration",
            "--run-ignored",
            "only",
            "--no-tests",
            "fail",
            "-E",
            &filter,
        ])],
    )
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
                Step::new("@ecs-boundary", &[]),
                Step::new("@redact-boundary", &[]),
                Step::new("@options-guards", &[]),
                Step::new("@driver-adoption", &[]),
                Step::new("@random-ids", &[]),
                Step::new("@stage-order", &[]),
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
                Step::new("@catalog", &[]),
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
                Step::new(
                    "python3",
                    &[
                        "-B",
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "plugins/rig-claude-cloak/tests/acceptance",
                        "-p",
                        "test_*.py",
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
            vec![
                cargo(&[
                    "clippy",
                    "--locked",
                    "--all-features",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                ]),
                // The rig agent's packages and the plugin corpus are not
                // default members.
                harness(
                    &["clippy", "--locked"],
                    &["--all-targets", "--", "-D", "warnings"],
                ),
                // And their plugins without the terminal view (`tui`).
                harness(
                    &["clippy", "--locked", "--no-default-features"],
                    &["--", "-D", "warnings"],
                ),
            ],
        ),
        check(
            "default-check",
            vec![
                // A dependency's #[cfg(test)] bodies are not compiled by the
                // facade's test targets after moving helpers into this crate.
                cargo(&[
                    "check",
                    "--locked",
                    "-p",
                    "rig",
                    "-p",
                    "rig-test-support",
                    "--tests",
                ]),
                // The divan benches compile here and run nowhere in CI.
                cargo(&[
                    "check",
                    "--locked",
                    "-p",
                    "rig-cassette",
                    "--features",
                    "http,jsonl",
                    "--bench",
                    "wire",
                ]),
                cargo(&[
                    "check",
                    "--locked",
                    "-p",
                    "rig-cassette",
                    "--features",
                    "http",
                    "--bench",
                    "agent",
                ]),
                cargo(&[
                    "check",
                    "--locked",
                    "-p",
                    "rig-memory",
                    "--features",
                    "fs",
                    "--bench",
                    "session",
                ]),
                cargo(&["check", "--locked", "-p", "rig-tools", "--bench", "tools"]),
                cargo(&[
                    "check",
                    "--locked",
                    "-p",
                    "rig-core",
                    "--bench",
                    "vector_search",
                ]),
            ],
        ),
        // The agent's signals, closed-pipe check and process groups, and the
        // launcher's cleanup of what a killed agent left (its groups' members
        // read in rig-harness-protocol), which use each platform's own calls
        // and commands.
        check(
            "process-tests",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "-p",
                "rig-tools",
                "-p",
                "rig-harness",
                "-p",
                "rig",
                "-p",
                "rig-harness-protocol",
                "--lib",
                "--bins",
                "--no-fail-fast",
                "--no-tests",
                "fail",
                "-E",
                "test(/^process::tests::/) | test(/^host::(signals|launcher)::tests::/) \
                 | test(/^launcher::(run|locks|run::stop)::tests::/) \
                 | test(/^tests::a_group_whose_leader_left/)",
            ])],
        ),
        check(
            "rig-claude-cloak-tests",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "-p",
                "rig-claude-cloak",
                "--lib",
                "--no-tests",
                "fail",
            ])],
        ),
        // The effect-corpus cells, in the two dependency graphs they must hold
        // in. Each cell replays its cassette and compares the agent's log
        // with its committed golden. The predicate is `package(rig-cassette)`,
        // qualified away from the `verify` binary whose `corpus_` module names
        // would otherwise match it. The golden-pairing guard stays in the
        // facade's `core` target. Excluding them never costs the cassette
        // engine's own unit tests their default owner. Distinct feature/target
        // runs (core-all, wasm) remain separate coverage.
        check(
            "effect-corpus",
            vec![
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "-p",
                    "rig",
                    "-p",
                    "rig-cassette",
                    // Extracted corpus helper regressions retain this graph too.
                    "-p",
                    "rig-test-support",
                    "--features",
                    "bedrock",
                    // The same retries as `test`: a stale golden is
                    // deterministic and fails every attempt, while a handful
                    // of cells carry wall-clock deadlines (a 30 s session
                    // budget, a stream drained at the replay's pace) that a
                    // loaded machine can miss once. Without retries one such
                    // miss cancels the lane mid-run.
                    "--retries",
                    "2",
                    "-E",
                    "(package(rig-cassette) and test(/(^|::)corpus_/) and not test(/^verify::/)) or (package(rig-test-support) and test(/(^|::)corpus_/)) or (package(rig) and test(golden_pairing))",
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
                    "(package(rig-cassette) and test(/(^|::)corpus_/) and not test(/^verify::/)) or (package(rig) and test(golden_pairing))",
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
                "rig-http",
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
                "not package(rig-cassette-minimal) or test(/^effect_log::agent_replay::/)",
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
                    "package(rig-cassette-minimal) and (test(/^verify::/) or test(/^effect_log::/))",
                ]),
                // Preserve the former default sweep's dependency graph and
                // retry policy. The nested runner above has no native HTTP
                // engine; this execution also exercises the unified
                // allocator. Only the `verify` suite moves out of the
                // default sweep.
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "--features",
                    "bedrock",
                    "--retries",
                    "2",
                    "-E",
                    "package(rig-cassette) and test(/^verify::/)",
                ]),
            ],
        ),
        // The builds that start a nested Cargo: a fixture crate's `cargo check`
        // under each feature set, so they assert what compiles, not what
        // runs. The facade's runs as these steps; the others are ignored
        // tests run on their own.
        //
        // The portable tool facade compiles under each feature set of a
        // standalone consumer (`tests/fixtures/tool_facade`, whose `all_root`
        // names every facade feature), against the workspace's lockfile.
        check("guard-facade-features", {
            let fixture = |args: &[&str]| {
                let mut all = vec![
                    "check",
                    "--quiet",
                    "--manifest-path",
                    "tests/fixtures/tool_facade/Cargo.toml",
                    "--target-dir",
                    "target/tool-facade-fixture",
                ];
                all.extend(args);
                cargo(&all)
            };
            vec![
                Step::new(
                    "@copy",
                    &["Cargo.lock", "tests/fixtures/tool_facade/Cargo.lock"],
                ),
                fixture(&["--no-default-features"]),
                fixture(&["--no-default-features", "--features", "agent"]),
                fixture(&[]),
                fixture(&["--all-features"]),
            ]
        }),
        guard(
            "guard-middleware-features",
            "rig-reqwest",
            "middleware_features::",
        ),
        guard("guard-agent-stepper", "rig", "core::agent_run_stepper::"),
        guard("guard-macro-hygiene", "rig-core", "macro_hygiene::"),
        guard(
            "guard-dependency-rename",
            "rig-derive",
            "dependency_rename::",
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
                "rig-candle",
                "-p",
                "rig-gemini-grpc",
                "-p",
                "rig-bedrock",
                "-p",
                "rig-vertexai",
                "--all-features",
                // nextest -E filters execution after Cargo compilation. Select
                // the integration binaries, and rig-bedrock's one target,
                // before compiling their harnesses.
                "--test",
                "integration",
                "--test",
                "history_conformance",
                "--retries",
                "0",
                "--no-tests",
                "fail",
                "-E",
                "test(/^streaming_conformance::/) + test(/^streaming_conformance_websocket::/) + test(/^history_conformance::/) + binary(history_conformance)",
            ])],
        ),
        // Every request fact Rig sends has a live recording: the corpus
        // and its request snapshots against crates/rig-cassette/fixtures/
        // acceptance.toml. Reads files only.
        check(
            "acceptance",
            vec![cargo(&["xtask", "cassette", "acceptance", "--check"])],
        ),
        // The reply bank is what the corpus gives: crates/rig-cassette/
        // fixtures/bank against the cassettes. Reads files only.
        check(
            "bank",
            vec![cargo(&["xtask", "cassette", "bank", "--check"])],
        ),
        check(
            "derive",
            // nextest runs each test in its own process, as the former
            // one binary per file did; doctests are cargo's.
            vec![
                cargo(&[
                    "nextest",
                    "run",
                    "--locked",
                    "-p",
                    "rig-derive",
                    "--no-tests",
                    "fail",
                ]),
                cargo(&["test", "--locked", "-p", "rig-derive", "--doc"]),
            ],
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
        // The runtime suite under all features: every workspace member's lib,
        // bin, test and example targets (Cargo's default `test` selection),
        // but the packages with checks of their own.
        check(
            "tests",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "--workspace",
                "--exclude",
                "rig-cassette-minimal",
                "--exclude",
                "rig-derive",
                "--exclude",
                "rig-service-tests",
                "--all-features",
                "--retries",
                "2",
            ])],
        ),
        // The Docker-backed storage suites, one feature per store.
        check(
            "service-tests",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "-p",
                "rig-service-tests",
                "--all-features",
                "--retries",
                "2",
                "--no-tests",
                "fail",
            ])],
        ),
        // No separate `cargo check --workspace --all-features --all-targets`:
        // `tests` compiles workspace lib, bin, test and example targets
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
        "rig-http",
        "rig-cassette",
        "rig-reqwest",
        "rig-agent",
        "rig-memory",
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
        if package == "rig-cassette" {
            steps.push(cargo(&[
                "check",
                "--locked",
                "-p",
                package,
                "--no-default-features",
                "--features",
                "agent",
                "--lib",
                "--target",
                "wasm32-unknown-unknown",
            ]));
        }
        // The middleware client's errors cross wasm without `Send`.
        if package == "rig-reqwest" {
            steps.push(cargo(&[
                "check",
                "--locked",
                "-p",
                package,
                "--features",
                "reqwest-middleware",
                "--target",
                "wasm32-unknown-unknown",
            ]));
        }
        // A store that is not `Send`, as one over a JavaScript promise is.
        if package == "rig-memory" {
            steps.push(cargo(&[
                "check",
                "--locked",
                "-p",
                package,
                "--test",
                "integration",
                "--target",
                "wasm32-unknown-unknown",
            ]));
        }
        if ["rig-core", "rig-http"].contains(&package) {
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
    // The kernel alone (rig-ecs has no features): it builds natively and
    // for the web, and a headless app of only the kernel and Bevy's task
    // pools runs a turn with a tool call on a scripted model.
    checks.push(check(
        "kernel-only",
        vec![
            cargo(&["check", "--locked", "--package", "rig-ecs"]),
            cargo(&[
                "check",
                "--locked",
                "--package",
                "rig-ecs",
                "--target",
                "wasm32-unknown-unknown",
            ]),
            cargo(&[
                "nextest",
                "run",
                "--locked",
                "--package",
                "rig-ecs",
                "--test",
                "integration",
                "--no-tests",
                "fail",
                "-E",
                "test(/^kernel_only::/)",
            ]),
        ],
    ));
    for (package, message) in [
        ("rig-rmcp", "the `rmcp` feature is native-only"),
        (
            "rig-tungstenite",
            "rig-tungstenite is a native websocket backend",
        ),
        (
            "rig-acp",
            "rig-acp runs sessions as processes: it needs a native target",
        ),
    ] {
        checks.push(check(
            &format!("native-only-{package}"),
            vec![Step::new("@native-only", &[package, message])],
        ));
    }
    checks
}
/// The CI job (a job id in `.github/workflows/ci.yaml`) that runs check `id`.
pub(crate) fn job(id: &str) -> &'static str {
    match id {
        "fmt" | "source-guards" | "tooling" | "acceptance" | "bank" => "guards",
        "tests" => "test",
        "service-tests" => "services",
        "doctests" => "doctest",
        "docs" => "doc",
        "clippy" => "clippy",
        "core-all" | "bus-verification" | "conformance" | "derive" => "test-guards",
        "effect-corpus" => "effect-corpus",
        "default-check" => "check-default",
        "kernel-only" => "kernel-only",
        "dependency-floors" => "floors",
        "rig-claude-cloak-tests" | "process-tests" => "macos",
        id if id.starts_with("guard-") => "compile-guards",
        _ => "wasm",
    }
}

/// The steps of a package-scoped check (`plan::SCOPED`) over `packages`,
/// given as `-p`/`--features` arguments; `names` are the selected packages.
pub(crate) fn scoped(id: &str, packages: &[String], names: &[&str]) -> Vec<Step> {
    let with = |head: &[&str], tail: &[&str]| {
        let mut args: Vec<&str> = head.to_vec();
        args.extend(packages.iter().map(String::as_str));
        args.extend(tail);
        cargo(&args)
    };
    match id {
        // A package may have no tests; a selected Docker suite that runs
        // none is a gap, not a pass.
        "tests" | "service-tests" => {
            let none = if id == "tests" { "warn" } else { "fail" };
            vec![with(
                &[
                    "nextest",
                    "run",
                    "--locked",
                    "--retries",
                    "2",
                    "--no-tests",
                    none,
                ],
                &[],
            )]
        }
        "doctests" => vec![with(&["test", "--locked", "--doc"], &[])],
        "docs" => {
            vec![with(&["doc", "--locked", "--no-deps"], &[]).env("RUSTDOCFLAGS", "-D warnings")]
        }
        _ => {
            let mut steps = vec![with(
                &["clippy", "--locked", "--all-targets"],
                &["--", "-D", "warnings"],
            )];
            // The rig agent's plugins also build without the terminal view.
            let prefix = CORPUS_PACKAGES.trim_end_matches('*');
            let harness: Vec<&str> = names
                .iter()
                .copied()
                .filter(|n| HARNESS_PACKAGES.contains(n) || n.starts_with(prefix))
                .flat_map(|n| ["-p", n])
                .collect();
            if !harness.is_empty() {
                let mut args = vec!["clippy", "--locked", "--no-default-features"];
                args.extend(harness);
                args.extend(["--", "-D", "warnings"]);
                steps.push(cargo(&args));
            }
            steps
        }
    }
}
