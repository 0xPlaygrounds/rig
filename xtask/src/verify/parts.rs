//! `full-tests` split into parts that CI runs as separate jobs. Every part
//! selects the whole workspace under all features and narrows only by target,
//! so each test is built with exactly the dependency features of the single
//! run. A `-p` subset would resolve its own features (for example dropping
//! `serde_json/preserve_order` or `tokio/test-util`) and test something else.
//!
//! Lib unittests can only be split by excluding packages, because `--lib`
//! takes every selected package. A lib part keeps [`CORE`] selected, the
//! packages whose own features or dev-dependencies the rest of the graph
//! needs, and runs each of their tests in one part only.
use super::*;
use std::collections::BTreeSet;

/// The packages every lib part keeps selected so that it resolves the
/// dependency features of the single run. The planner test checks this
/// against `cargo tree`; a new package that changes the resolution when left
/// out belongs here.
pub(crate) const CORE: [&str; 20] = [
    "candle_wasm_chat",
    "corpus-base",
    "corpus-oc",
    "corpus-sp",
    "rig",
    "rig-agent",
    "rig-cassette",
    "rig-core",
    "rig-defaults-test-support",
    "rig-derive",
    "rig-fastembed",
    "rig-harness",
    "rig-helixdb",
    "rig-lancedb",
    "rig-milvus",
    "rig-postgres",
    "rig-sqlite",
    "rig-steel",
    "rig-typesafeai",
    "rig-vectorize",
];

/// The selection every part starts from: the minimal replay runner reuses
/// rig-cassette's sources and only adds coverage when built outside this
/// graph.
const BASE: [&str; 9] = [
    "nextest",
    "run",
    "--locked",
    "--workspace",
    "--exclude",
    "rig-cassette-minimal",
    "--all-features",
    "--retries",
    "2",
];

/// rig-derive's tests need the dylib component and run in their own job.
const NOT_DERIVE: &str = "not package(rig-derive)";

enum Targets {
    /// Integration test targets by name.
    Tests(&'static [&'static str]),
    /// Every integration test target of one package.
    PackageTests(&'static str),
    /// Every bin unittest and the named integration tests, and every
    /// example built (not tested) the way `cargo test` builds it.
    BinsExamples(&'static [&'static str]),
    /// Every integration test target no other part names.
    OtherTests,
    /// The lib unittests of these packages, and of the [`CORE`] packages
    /// listed second, which only this part runs.
    Libs(&'static [&'static str], &'static [&'static str]),
    /// The lib unittests of every package no other part names, and of the
    /// [`CORE`] packages no other part runs.
    OtherLibs,
}

struct Part {
    name: &'static str,
    targets: Targets,
    partition: Option<&'static str>,
}

/// Balanced by compile and test time on the CI runners: the facade guard
/// runs four nested `cargo check`s and gets a runner to itself, and the wire
/// test holds every test thread of its half of `rig::core`.
const PARTS: [Part; 10] = [
    Part {
        name: "core-1",
        targets: Targets::Tests(&["core"]),
        partition: Some("count:1/2"),
    },
    Part {
        name: "core-2",
        targets: Targets::Tests(&["core"]),
        partition: Some("count:2/2"),
    },
    Part {
        name: "facade",
        targets: Targets::Tests(&["tool_facade_features"]),
        partition: None,
    },
    Part {
        name: "cassette-1",
        targets: Targets::PackageTests("rig-cassette"),
        partition: Some("count:1/2"),
    },
    Part {
        name: "cassette-2",
        targets: Targets::PackageTests("rig-cassette"),
        partition: Some("count:2/2"),
    },
    Part {
        name: "guards",
        targets: Targets::BinsExamples(&["middleware_features", "macro_hygiene"]),
        partition: None,
    },
    Part {
        name: "other",
        targets: Targets::OtherTests,
        partition: None,
    },
    Part {
        name: "lib-1",
        targets: Targets::Libs(
            &[
                "corpus-rf-01",
                "rig-basics",
                "rig-bedrock",
                "rig-claude-cloak",
                "rig-harness-protocol",
                "rig-harness-test-support",
                "rig-memory",
                "rig-mongodb",
                "rig-neo4j",
                "rig-s3vectors",
                "rig-scylladb",
                "rig-vertexai",
            ],
            &[
                "candle_wasm_chat",
                "corpus-sp",
                "rig",
                "rig-agent",
                "rig-defaults-test-support",
                "rig-fastembed",
                "rig-helixdb",
                "rig-milvus",
                "rig-postgres",
                "rig-steel",
                "rig-vectorize",
            ],
        ),
        partition: None,
    },
    Part {
        name: "lib-2",
        targets: Targets::Libs(
            &[
                "corpus-ct",
                "corpus-fh",
                "corpus-rf-03",
                "corpus-rf-05",
                "corpus-rf-08",
                "corpus-shared",
                "rig-acp",
                "rig-ecs",
                "rig-gemini-grpc",
                "rig-http",
                "rig-mcp-client",
                "rig-qdrant",
                "rig-reqwest",
                "rig-tools",
                "rig-traffic",
                "rig-tungstenite",
            ],
            &[
                "corpus-base",
                "corpus-oc",
                "rig-cassette",
                "rig-harness",
                "rig-lancedb",
                "rig-sqlite",
                "rig-typesafeai",
            ],
        ),
        partition: None,
    },
    Part {
        name: "lib-3",
        targets: Targets::OtherLibs,
        partition: None,
    },
];

/// The members' names with their integration test targets and whether they
/// have a library, without the excluded minimal runner.
fn members(metadata: &Value) -> Result<Vec<(String, Vec<String>, bool)>> {
    let packages = metadata["packages"]
        .as_array()
        .ok_or_else(|| invalid("cargo metadata has no packages"))?;
    Ok(packages
        .iter()
        .filter_map(|p| {
            let name = p["name"].as_str()?;
            let targets = p["targets"].as_array().map(Vec::as_slice).unwrap_or(&[]);
            let has = |t: &Value, kinds: &[&str]| {
                t["kind"]
                    .as_array()
                    .is_some_and(|k| k.iter().any(|x| kinds.iter().any(|kind| x == kind)))
            };
            let tests = targets
                .iter()
                .filter(|t| has(t, &["test"]))
                .filter_map(|t| t["name"].as_str().map(str::to_owned))
                .collect();
            let lib = targets
                .iter()
                .any(|t| has(t, &["lib", "rlib", "cdylib", "proc-macro"]));
            (name != "rig-cassette-minimal").then(|| (name.to_owned(), tests, lib))
        })
        .collect())
}

fn packages_filter(packages: &[&str]) -> String {
    packages
        .iter()
        .map(|p| format!("package({p})"))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// The steps of `full-tests`, one per part, each tagged with its part name.
pub(super) fn steps(metadata: &Value) -> Result<Vec<Step>> {
    let members = members(metadata)?;
    let owners = |test: &str| {
        members
            .iter()
            .filter(|(_, t, _)| t.iter().any(|x| x == test))
            .count()
    };
    let package_tests = |package: &str| -> Result<Vec<String>> {
        members
            .iter()
            .find(|(name, _, _)| name == package)
            .map(|(_, tests, _)| tests.clone())
            .ok_or_else(|| invalid(format!("full-tests part names unknown package {package}")))
    };
    let mut named = BTreeSet::new();
    let mut grouped = BTreeSet::new();
    let mut run_core = BTreeSet::new();
    for part in &PARTS {
        match part.targets {
            Targets::Tests(tests) | Targets::BinsExamples(tests) => {
                named.extend(tests.iter().map(|t| (*t).to_owned()))
            }
            Targets::PackageTests(package) => named.extend(package_tests(package)?),
            Targets::Libs(group, core) => {
                grouped.extend(group.iter().copied());
                run_core.extend(core.iter().copied());
            }
            Targets::OtherTests | Targets::OtherLibs => {}
        }
    }
    for test in &named {
        // A name two packages share builds both targets, so it belongs to one part.
        if owners(test) != 1 {
            return Err(invalid(format!(
                "full-tests names test target {test}, which {} packages have",
                owners(test)
            )));
        }
    }
    for package in CORE.iter().chain(&grouped) {
        if !members.iter().any(|(name, _, lib)| name == package && *lib) {
            return Err(invalid(format!(
                "full-tests lib parts name {package}, which is not a member with a library"
            )));
        }
    }
    let libs: Vec<&str> = members
        .iter()
        .filter(|(_, _, lib)| *lib)
        .map(|(name, _, _)| name.as_str())
        .collect();
    let mut steps = Vec::new();
    for part in &PARTS {
        let tagged = |args: Vec<String>| {
            let mut step = Step::new("cargo", &[]);
            step.args = args;
            step.part = Some(part.name.to_owned());
            step
        };
        {
            let mut args: Vec<String> = BASE.iter().map(|a| (*a).to_owned()).collect();
            let mut test = |name: &str| args.extend(["--test".to_owned(), name.to_owned()]);
            let filter = match part.targets {
                Targets::Tests(tests) => {
                    tests.iter().for_each(|t| test(t));
                    NOT_DERIVE.to_owned()
                }
                Targets::PackageTests(package) => {
                    package_tests(package)?.iter().for_each(|t| test(t));
                    NOT_DERIVE.to_owned()
                }
                Targets::BinsExamples(tests) => {
                    tests.iter().for_each(|t| test(t));
                    args.push("--bins".to_owned());
                    // `--examples` would compile them as tests; the default
                    // selection only builds them, under the test profile.
                    let mut build: Vec<String> = ["build", "--profile", "test"]
                        .into_iter()
                        .chain(BASE.iter().skip(2).take(5).copied())
                        .map(str::to_owned)
                        .collect();
                    build.push("--examples".to_owned());
                    steps.push(tagged(build));
                    NOT_DERIVE.to_owned()
                }
                Targets::OtherTests => {
                    let mut other: BTreeSet<&str> = BTreeSet::new();
                    for (_, tests, _) in &members {
                        other.extend(
                            tests
                                .iter()
                                .map(String::as_str)
                                .filter(|t| !named.contains(*t)),
                        );
                    }
                    other.iter().for_each(|t| test(t));
                    NOT_DERIVE.to_owned()
                }
                Targets::Libs(group, core) => {
                    for package in libs
                        .iter()
                        .filter(|p| !CORE.contains(p) && !group.contains(p))
                    {
                        args.extend(["--exclude".to_owned(), (*package).to_owned()]);
                    }
                    args.push("--lib".to_owned());
                    packages_filter(&group.iter().chain(core).copied().collect::<Vec<_>>())
                }
                Targets::OtherLibs => {
                    for package in &grouped {
                        args.extend(["--exclude".to_owned(), (*package).to_owned()]);
                    }
                    args.push("--lib".to_owned());
                    let elsewhere: Vec<&str> =
                        run_core.iter().copied().chain(["rig-derive"]).collect();
                    format!("not ({})", packages_filter(&elsewhere))
                }
            };
            if let Some(partition) = part.partition {
                args.extend(["--partition".to_owned(), partition.to_owned()]);
            }
            args.extend(["-E".to_owned(), filter]);
            steps.push(tagged(args));
        }
    }
    Ok(steps)
}
