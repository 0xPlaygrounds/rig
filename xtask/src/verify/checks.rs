//! Check definitions shared by local plans and CI's independently scheduled jobs.
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
                Step::new("bash", &[".github/scripts/check-toolchain-pin.sh"]),
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
                Step::new("@scenarios", &[]),
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
            vec![cargo(&["check", "--locked", "-p", "rig", "--tests"])],
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
                "not binary(macro_hygiene)",
            ])],
        ),
        check(
            "scenario-registrations",
            // Match default-tests' package/feature graph; narrowing to -p rig
            // recompiles root binaries after the default-member sweep.
            vec![Step::new(
                "@registrations",
                &[
                    "nextest",
                    "list",
                    "--locked",
                    "--features",
                    "bedrock",
                    "--message-format",
                    "json",
                ],
            )],
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
                "--all-features",
                "--profile",
                "guards",
                "-E",
                "not binary(macro_hygiene)",
            ])],
        ),
        check(
            "bus-verification",
            vec![cargo(&[
                "nextest",
                "run",
                "--locked",
                "-p",
                "rig-verify",
                "--all-features",
                "--retries",
                "0",
            ])],
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
                "--all-features",
                "--retries",
                "2",
                "-E",
                "not package(rig-derive)",
            ])],
        ),
        check(
            "workspace-check",
            vec![cargo(&[
                "check",
                "--locked",
                "--workspace",
                "--all-features",
                "--all-targets",
            ])],
        ),
        // Existing repository floor checker; new verification tooling is Rust.
        check(
            "dependency-floors",
            vec![Step::new(
                "python3",
                &["scripts/check-dependency-floors.py"],
            )],
        ),
    ];
    for package in [
        "rig-core",
        "rig-effect-log",
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
pub(super) fn full_lane(path: &str) -> bool {
    [
        "Cargo.toml",
        "Cargo.lock",
        ".github/workflows/nightly.yaml",
        "scripts/check-dependency-floors.py",
        "tests/tool_facade_features.rs",
        "tests/integrations.rs",
    ]
    .contains(&path)
        || path.starts_with("tests/integrations/")
        || path.starts_with("test-support/")
        || (path.starts_with("crates/") && path.ends_with("/Cargo.toml"))
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
