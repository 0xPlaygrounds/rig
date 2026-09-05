//! Process evidence for the reproduction, regression and independent acceptance.

use std::{collections::BTreeMap, path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use super::super::Error;
use super::{
    process::{Mode, Output, Sandbox},
    project::{self, Project},
};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Initial,
    Regression,
    Final,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Step {
    pub command: Vec<String>,
    pub output: Output,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Report {
    pub phase: Phase,
    pub project_digest: String,
    /// For diagnostic phases, acceptance means an executed behavioral failure.
    pub accepted: bool,
    pub reason: String,
    pub steps: Vec<Step>,
}

impl Report {
    /// Serialized acceptance is a cache of the retained evidence, never an
    /// authority that can turn a failed process into a successful repair.
    pub fn verify(&self) -> Result<(), Error> {
        if self.accepted != self.proven() {
            return Err(Error::Invariant(
                "validation acceptance contradicts retained process evidence".into(),
            ));
        }
        Ok(())
    }

    fn proven(&self) -> bool {
        let mut steps = self.steps.iter();
        let Some(compile) = steps.next() else {
            return false;
        };
        let mut command = vec![
            "cargo",
            "test",
            "--offline",
            "--locked",
            "--no-run",
            "--message-format=json",
            "--test",
            if self.phase == Phase::Regression {
                "regression"
            } else {
                "pagination"
            },
        ];
        if self.phase == Phase::Final {
            command.extend(["--test", "regression"]);
        }
        if !command_is(compile, &command)
            || !successful(&compile.output)
            || !compile
                .output
                .stdout
                .lines()
                .any(|line| line == "build-finished success=true")
        {
            return false;
        }
        let names: &[&str] = match self.phase {
            Phase::Initial => &["pagination"],
            Phase::Regression => &["regression"],
            Phase::Final => &["pagination", "regression"],
        };
        for name in names {
            let (Some(list), Some(run)) = (steps.next(), steps.next()) else {
                return false;
            };
            let executable = format!("<test:{name}>");
            let listed: Vec<_> = list
                .output
                .stdout
                .lines()
                .filter_map(|line| line.strip_suffix(": test").map(str::to_owned))
                .collect();
            if !command_is(list, &[&executable, "--list", "--format=terse"])
                || !successful(&list.output)
                || !command_is(run, &[&executable, "--test-threads=1"])
                || !accounted(&run.output, &listed, self.phase != Phase::Final)
            {
                return false;
            }
        }
        if self.phase == Phase::Final {
            let (Some(format), Some(compile), Some(oracle)) =
                (steps.next(), steps.next(), steps.next())
            else {
                return false;
            };
            if !command_is(
                format,
                &[
                    "rustfmt",
                    "--edition=2024",
                    "--check",
                    "src/lib.rs",
                    "tests/regression.rs",
                ],
            ) || !successful(&format.output)
                || !command_is(
                    compile,
                    &[
                        "rustc",
                        "--edition=2024",
                        "--extern",
                        "page_window=<production-library>",
                        "<compile>/contract_oracle.rs",
                        "-o",
                        "<compile>/target/contract_oracle",
                    ],
                )
                || !successful(&compile.output)
                || !command_is(oracle, &["<test:contract_oracle>"])
                || !successful(&oracle.output)
                || !oracle.output.stderr.is_empty()
                || oracle.output.stdout != oracle_expected()
            {
                return false;
            }
        }
        steps.next().is_none()
    }
}

fn command_is(step: &Step, expected: &[&str]) -> bool {
    step.command
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
}

fn successful(output: &Output) -> bool {
    output.code == Some(0) && !output.timed_out && !output.output_limit
}

fn elapsed(value: &str) -> bool {
    let decimal = |value: &str| {
        !value.is_empty()
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte == b'.')
            && value.bytes().filter(|byte| *byte == b'.').count() <= 1
            && value
                .parse::<f64>()
                .is_ok_and(|number| number.is_finite() && number >= 0.0)
    };
    let parts: Vec<_> = value.split_whitespace().collect();
    match parts.as_slice() {
        [seconds] => seconds.strip_suffix('s').is_some_and(decimal),
        [minutes, seconds] => {
            minutes.strip_suffix('m').is_some_and(|value| {
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
            }) && seconds.strip_suffix('s').is_some_and(decimal)
        }
        _ => false,
    }
}

/// Preserve diagnostics while replacing only known paths, timings and libtest's
/// incidental panic thread identifier. Compiler JSON is handled separately.
fn normalize(text: &str, project: &Project, sandbox: &Sandbox) -> String {
    let mut text = text.to_owned();
    for (path, replacement) in [
        (project.root().to_path_buf(), "<project>"),
        (sandbox.compile.path().to_path_buf(), "<compile>"),
        (sandbox.runtime_root().to_path_buf(), "<runtime>"),
        (sandbox.toolchain.clone(), "<toolchain>"),
    ] {
        if let Ok(path) = path.canonicalize() {
            text = text.replace(path.to_string_lossy().as_ref(), replacement);
        }
        text = text.replace(path.to_string_lossy().as_ref(), replacement);
    }
    text.lines()
        .map(|line| {
            if line.trim_start().starts_with("Finished `")
                && line.contains("` profile [")
                && let Some((prefix, duration)) = line.rsplit_once(" target(s) in ")
                && elapsed(duration)
            {
                return prefix.to_owned() + " target(s) in <elapsed>";
            }
            if line.starts_with("test result: ")
                && let Some((prefix, duration)) = line.rsplit_once("; finished in ")
                && elapsed(duration)
            {
                return prefix.to_owned() + "; finished in <elapsed>";
            }
            if line.starts_with("thread '")
                && let Some((prefix, tail)) = line.rsplit_once("' (")
                && let Some((id, suffix)) = tail.split_once(") panicked at ")
                && !id.is_empty()
                && id.bytes().all(|byte| byte.is_ascii_digit())
            {
                return format!("{prefix}' (<thread>) panicked at {suffix}");
            }
            line.to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
        + if text.ends_with('\n') { "\n" } else { "" }
}

fn record(
    report: &mut Report,
    command: Vec<String>,
    mut output: Output,
    project: &Project,
    sandbox: &Sandbox,
) {
    output.stdout = normalize(&output.stdout, project, sandbox);
    output.stderr = normalize(&output.stderr, project, sandbox);
    report.steps.push(Step { command, output });
}

/// Require libtest to account for every listed test, with none ignored/filtered.
/// Exit zero alone is insufficient evidence (including an empty regression).
fn accounted(output: &Output, names: &[String], require_failure: bool) -> bool {
    if output.timed_out || output.output_limit || names.is_empty() {
        return false;
    }
    let mut passed = 0;
    let mut failed = 0;
    for name in names {
        let pass = format!("test {name} ... ok");
        let fail = format!("test {name} ... FAILED");
        let pass_count = output.stdout.lines().filter(|line| *line == pass).count();
        let fail_count = output.stdout.lines().filter(|line| *line == fail).count();
        if pass_count + fail_count != 1 {
            return false;
        }
        passed += pass_count;
        failed += fail_count;
    }
    let status = if failed == 0 { "ok" } else { "FAILED" };
    let summary = format!(
        "test result: {status}. {passed} passed; {failed} failed; 0 ignored; 0 measured; 0 filtered out;"
    );
    output
        .stdout
        .lines()
        .filter(|line| line.starts_with(&summary))
        .count()
        == 1
        && if require_failure {
            failed > 0 && output.code == Some(101)
        } else {
            failed == 0 && output.code == Some(0)
        }
}

fn run_test(
    report: &mut Report,
    project: &Project,
    sandbox: &Sandbox,
    name: &str,
    binary: &std::path::Path,
    require_failure: bool,
) -> Result<bool, Error> {
    let list = sandbox.run(
        project.root(),
        binary,
        &["--list".into(), "--format=terse".into()],
        Mode::Test,
        Duration::from_secs(10),
    )?;
    let names: Vec<_> = list
        .stdout
        .lines()
        .filter_map(|line| line.strip_suffix(": test").map(str::to_owned))
        .collect();
    let listed = successful(&list)
        && !names.is_empty()
        && names
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == names.len();
    record(
        report,
        vec![
            format!("<test:{name}>"),
            "--list".into(),
            "--format=terse".into(),
        ],
        list,
        project,
        sandbox,
    );
    if !listed {
        return Ok(false);
    }
    let output = sandbox.run(
        project.root(),
        binary,
        &["--test-threads=1".into()],
        Mode::Test,
        Duration::from_secs(10),
    )?;
    let accepted = accounted(&output, &names, require_failure);
    record(
        report,
        vec![format!("<test:{name}>"), "--test-threads=1".into()],
        output,
        project,
        sandbox,
    );
    Ok(accepted)
}

/// Format only a scratch proposal before constructing its approved diff. Replay
/// observes the returned data; it never invokes this external process.
pub(super) fn format_patch(
    project: &Project,
    contents: &str,
    deadline: std::time::Instant,
) -> Result<Step, Error> {
    let sandbox = Sandbox::new_until(Some(deadline))?;
    let input = sandbox.compile.path().join("proposal.rs");
    std::fs::write(&input, contents)?;
    let input = input.canonicalize()?;
    let mut output = sandbox.run(
        project.root(),
        &sandbox.binary("rustfmt"),
        &[
            "--edition=2024".into(),
            "--emit=stdout".into(),
            input.display().to_string(),
        ],
        Mode::Compile,
        Duration::from_secs(10),
    )?;
    output.stderr = normalize(&output.stderr, project, &sandbox);
    if !successful(&output) {
        return Err(Error::Invariant(format!(
            "proposal formatting failed (code {:?}, timed_out {}, output_limit {}): {}",
            output.code, output.timed_out, output.output_limit, output.stderr
        )));
    }
    // Remove rustfmt's filename header, retaining the formatted Rust verbatim.
    output.stdout = output
        .stdout
        .strip_prefix(&format!("{}:\n\n", input.display()))
        .ok_or_else(|| {
            Error::Invariant("proposal formatter omitted its expected filename header".into())
        })?
        .to_owned();
    Ok(Step {
        command: vec![
            "rustfmt".into(),
            "--edition=2024".into(),
            "--emit=stdout".into(),
            "<proposal>".into(),
        ],
        output,
    })
}

pub(super) fn validate(project: &Project, phase: Phase) -> Result<Report, Error> {
    validate_until(project, phase, None)
}

pub(super) fn validate_until(
    project: &Project,
    phase: Phase,
    deadline: Option<std::time::Instant>,
) -> Result<Report, Error> {
    let image = project.image()?;
    if phase != Phase::Final && image.get("src/lib.rs") != project::initial().get("src/lib.rs") {
        return Err(Error::Invariant(
            "reproduction requires unchanged buggy production source".into(),
        ));
    }
    let mut report = Report {
        phase,
        project_digest: project::digest(&image)?,
        accepted: false,
        reason: "validation did not establish the required behavior".into(),
        steps: vec![],
    };
    if phase != Phase::Initial && !image.contains_key("tests/regression.rs") {
        report.reason = "a regression file is required".into();
        return Ok(report);
    }
    let sandbox = Sandbox::new_until(deadline)?;
    let mut arguments: Vec<String> = [
        "test",
        "--offline",
        "--locked",
        "--no-run",
        "--message-format=json",
        "--test",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    arguments.push(
        if phase == Phase::Regression {
            "regression"
        } else {
            "pagination"
        }
        .into(),
    );
    if phase == Phase::Final {
        arguments.extend(["--test".into(), "regression".into()]);
    }
    let mut compiled = sandbox.run(
        project.root(),
        &sandbox.binary("cargo"),
        &arguments,
        Mode::Compile,
        Duration::from_secs(30),
    )?;
    let mut artifacts = BTreeMap::<String, PathBuf>::new();
    let mut library = None;
    let mut diagnostics = String::new();
    for line in compiled.stdout.lines() {
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => {
                diagnostics.push_str(line);
                diagnostics.push('\n');
                continue;
            }
        };
        match value.get("reason").and_then(serde_json::Value::as_str) {
            Some("compiler-artifact") => {
                if value
                    .pointer("/target/name")
                    .and_then(serde_json::Value::as_str)
                    == Some("page_window")
                    && value
                        .pointer("/profile/test")
                        .and_then(serde_json::Value::as_bool)
                        == Some(false)
                {
                    for path in value
                        .get("filenames")
                        .and_then(serde_json::Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(serde_json::Value::as_str)
                    {
                        if path.ends_with(".rlib") && library.replace(PathBuf::from(path)).is_some()
                        {
                            return Err(Error::Invariant(
                                "duplicate production library artifact".into(),
                            ));
                        }
                    }
                }
                if let (Some(name), Some(path)) = (
                    value
                        .pointer("/target/name")
                        .and_then(serde_json::Value::as_str),
                    value.get("executable").and_then(serde_json::Value::as_str),
                ) && artifacts.insert(name.into(), path.into()).is_some()
                {
                    return Err(Error::Invariant(
                        "duplicate test executable from Cargo".into(),
                    ));
                }
            }
            Some("compiler-message") => {
                diagnostics.push_str(
                    value
                        .pointer("/message/rendered")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(line),
                );
                if !diagnostics.ends_with('\n') {
                    diagnostics.push('\n');
                }
            }
            Some("build-finished") => {
                diagnostics.push_str(&format!(
                    "build-finished success={}\n",
                    value.get("success").unwrap_or(&serde_json::Value::Null)
                ));
            }
            _ => {
                diagnostics.push_str(line);
                diagnostics.push('\n');
            }
        }
    }
    compiled.stdout = diagnostics;
    let compile_ok = successful(&compiled);
    let mut command = vec!["cargo".into()];
    command.extend(arguments);
    record(&mut report, command, compiled, project, &sandbox);
    if !compile_ok {
        report.reason = "compilation failed; this is not a behavioral regression".into();
        return Ok(report);
    }
    let required: &[&str] = match phase {
        Phase::Initial => &["pagination"],
        Phase::Regression => &["regression"],
        Phase::Final => &["pagination", "regression"],
    };
    for name in required {
        let Some(binary) = artifacts.get(*name) else {
            report.reason = format!("missing test executable: {name}");
            return Ok(report);
        };
        if !run_test(
            &mut report,
            project,
            &sandbox,
            name,
            binary,
            phase != Phase::Final,
        )? {
            return Ok(report);
        }
    }
    if phase == Phase::Final {
        let arguments: Vec<String> = [
            "--edition=2024",
            "--check",
            "src/lib.rs",
            "tests/regression.rs",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let output = sandbox.run(
            project.root(),
            &sandbox.binary("rustfmt"),
            &arguments,
            Mode::Compile,
            Duration::from_secs(10),
        )?;
        let ok = successful(&output);
        let mut command = vec!["rustfmt".into()];
        command.extend(arguments);
        record(&mut report, command, output, project, &sandbox);
        if !ok {
            report.reason = "formatting check failed".into();
            return Ok(report);
        }
        let Some(library) = library else {
            report.reason = "missing production library artifact".into();
            return Ok(report);
        };
        if !oracle(&mut report, project, &sandbox, &library)? {
            report.reason = "independent contract validation failed".into();
            return Ok(report);
        }
    }
    if project::digest(&project.image()?)? != report.project_digest {
        return Err(Error::Invariant("validation changed project files".into()));
    }
    report.accepted = true;
    report.reason = if phase == Phase::Final {
        "original tests, regression, formatting and independent contract passed"
    } else {
        "behavioral failure reproduced with executed tests and unchanged source"
    }
    .into();
    Ok(report)
}

fn oracle(
    report: &mut Report,
    project: &Project,
    sandbox: &Sandbox,
    library: &std::path::Path,
) -> Result<bool, Error> {
    let library = library.canonicalize()?;
    if !library.starts_with(sandbox.target().canonicalize()?) {
        return Err(Error::Invariant(
            "oracle library is outside compiler artifacts".into(),
        ));
    }
    let source = sandbox.compile.path().join("contract_oracle.rs");
    std::fs::write(&source, include_str!("validation/oracle.rs.txt"))?;
    let binary = sandbox.target().join("contract_oracle");
    let arguments = vec![
        "--edition=2024".into(),
        "--extern".into(),
        format!("page_window={}", library.display()),
        source.display().to_string(),
        "-o".into(),
        binary.display().to_string(),
    ];
    let output = sandbox.run(
        project.root(),
        &sandbox.binary("rustc"),
        &arguments,
        Mode::Compile,
        Duration::from_secs(30),
    )?;
    let ok = successful(&output);
    record(
        report,
        vec![
            "rustc".into(),
            "--edition=2024".into(),
            "--extern".into(),
            "page_window=<production-library>".into(),
            "<compile>/contract_oracle.rs".into(),
            "-o".into(),
            "<compile>/target/contract_oracle".into(),
        ],
        output,
        project,
        sandbox,
    );
    if !ok {
        return Ok(false);
    }
    let output = sandbox.run(
        project.root(),
        &binary,
        &[],
        Mode::Test,
        Duration::from_secs(10),
    )?;
    // The trusted host computes the expected values. The child must return the
    // complete behavior transcript; a test summary or incomplete result cannot
    // satisfy this comparison. This finite native check still requires source
    // review to reject code that detects the harness and forges all outputs.
    let expected = oracle_expected();
    let accepted = successful(&output) && output.stdout == expected && output.stderr.is_empty();
    record(
        report,
        vec!["<test:contract_oracle>".into()],
        output,
        project,
        sandbox,
    );
    Ok(accepted)
}

fn oracle_expected() -> String {
    let mut expected = String::new();
    for length in [0_usize, 1, 3, 9, 17] {
        let items: Vec<_> = (0..length).map(|index| format!("value-{index}")).collect();
        for offset in [
            0,
            1,
            length / 2,
            length,
            length + 1,
            usize::MAX - 1,
            usize::MAX,
        ] {
            for limit in [
                0,
                1,
                length / 2,
                length,
                length + 1,
                usize::MAX - 1,
                usize::MAX,
            ] {
                let values: Vec<_> = items.iter().skip(offset).take(limit).collect();
                expected.push_str(&format!("length={length} offset={offset} limit={limit} borrowed=true values={values:?}\n"));
            }
        }
    }
    expected
}
