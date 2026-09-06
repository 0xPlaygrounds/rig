//! Exact-ID cassette replay with durable evidence, including failed attempts.

use super::*;
use quote::ToTokens;
use serde_json::json;
use std::{collections::BTreeSet, process::Stdio};

#[cfg(test)]
mod tests;

fn normalized_source(source: &str, permit_sibling_visibility: bool) -> Result<String> {
    let mut syntax = syn::parse_file(source)?;
    // Widening visibility can make rustfmt wrap a signature and add a trailing
    // comma. Normalize that optional punctuation on both sides, keeping every
    // argument, attribute, type and function body intact.
    for item in &mut syntax.items {
        if let syn::Item::Fn(item) = item
            && item.sig.inputs.trailing_punct()
        {
            let last = item
                .sig
                .inputs
                .pop()
                .ok_or("trailing punctuation without an argument")?
                .into_value();
            item.sig.inputs.push_value(last);
        }
    }
    if permit_sibling_visibility {
        for item in &mut syntax.items {
            if let syn::Item::Impl(item) = item {
                for member in &mut item.items {
                    if let syn::ImplItem::Fn(method) = member
                        && matches!(&method.vis, syn::Visibility::Restricted(v) if v.path.is_ident("super"))
                    {
                        method.vis = syn::Visibility::Inherited;
                    }
                }
            }
            let visibility = match item {
                syn::Item::Const(item) => &mut item.vis,
                syn::Item::Fn(item) => &mut item.vis,
                syn::Item::Enum(item) => &mut item.vis,
                syn::Item::Struct(item) => {
                    for field in &mut item.fields {
                        if matches!(&field.vis, syn::Visibility::Restricted(v) if v.path.is_ident("super"))
                        {
                            field.vis = syn::Visibility::Inherited;
                        }
                    }
                    &mut item.vis
                }
                _ => continue,
            };
            if matches!(visibility, syn::Visibility::Restricted(v) if v.path.is_ident("super")) {
                *visibility = syn::Visibility::Inherited;
            }
        }
    }
    Ok(syntax.into_token_stream().to_string())
}

fn check_originals(spec: &Value, baseline: &Path, candidate: &Path) -> Result<()> {
    for name in strings(spec, "original_sources")? {
        let original = normalized_source(&fs::read_to_string(checked(baseline, &name)?)?, true)?;
        let native = normalized_source(&fs::read_to_string(checked(candidate, &name)?)?, true)?;
        if original != native {
            return Err(
                format!("original source changed beyond sibling visibility: {name}").into(),
            );
        }
    }
    for name in strings(spec, "fixtures")? {
        if fs::read(checked(baseline, &name)?)? != fs::read(checked(candidate, &name)?)? {
            return Err(format!("baseline fixture changed: {name}").into());
        }
    }
    Ok(())
}

fn selected_ids(listing: &Value) -> Result<BTreeSet<String>> {
    let suites = field(listing, "rust-suites")
        .as_object()
        .ok_or("missing rust-suites")?;
    let mut selected = BTreeSet::new();
    for (binary, suite) in suites {
        for (name, case) in suite["testcases"].as_object().ok_or("missing testcases")? {
            if field(field(case, "filter-match"), "status") == "matches"
                && field(case, "ignored") == false
            {
                selected.insert(format!("{binary}${name}"));
            }
        }
    }
    Ok(selected)
}

fn outcomes(lines: &str, expected: &BTreeSet<String>) -> Result<BTreeMap<String, Value>> {
    let mut results = BTreeMap::new();
    for line in lines.lines() {
        let event: Value = serde_json::from_str(line)?;
        if field(&event, "type") != "test" || field(&event, "event") == "started" {
            continue;
        }
        let name = text(&event, "name")?.to_owned();
        // nextest's JSON reporter includes filtered-out ignored registrations.
        if !expected.contains(&name) && field(&event, "event") == "ignored" {
            continue;
        }
        if !expected.contains(&name) || results.insert(name.clone(), event).is_some() {
            return Err(format!("unexpected or duplicate terminal outcome: {name}").into());
        }
    }
    if results.keys().cloned().collect::<BTreeSet<_>>() != *expected {
        return Err("missing terminal outcomes".into());
    }
    Ok(results)
}

fn command(root: &Path, args: &[String]) -> Result<std::process::Command> {
    let (program, args) = args.split_first().ok_or("empty command")?;
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(root)
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env_remove("RIG_REGENERATE_GOLDEN")
        .env("NEXTEST_EXPERIMENTAL_LIBTEST_JSON", "1");
    for (name, _) in std::env::vars_os() {
        let spelling = name.to_string_lossy();
        if [
            "_API_KEY",
            "_TOKEN",
            "_SECRET",
            "_ACCESS_KEY_ID",
            "_SECRET_ACCESS_KEY",
        ]
        .iter()
        .any(|s| spelling.ends_with(s))
        {
            command.env_remove(name);
        }
    }
    Ok(command)
}

fn options(spec: &Value, surface: &str, target: &Path) -> Result<(Vec<String>, BTreeSet<String>)> {
    let cells = array(spec, "cells")?;
    let expected: BTreeSet<_> = cells
        .iter()
        .map(|c| text(c, surface).map(str::to_owned))
        .collect::<Result<_>>()?;
    if expected.is_empty() || expected.len() != cells.len() {
        return Err("empty or duplicate batch IDs".into());
    }
    let mut filters = Vec::new();
    for identity in &expected {
        let (binary, name) = identity.split_once('$').ok_or("invalid binary/test ID")?;
        if binary.is_empty()
            || name.is_empty()
            || !binary
                .chars()
                .chain(name.chars())
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | ':'))
        {
            return Err("unsupported exact filter identifier".into());
        }
        filters.push(format!("(binary_id(={binary}) & test(={name}))"));
    }
    let mut args = vec![
        "--locked".into(),
        "-p".into(),
        text(spec, "package")?.into(),
        "--target-dir".into(),
        target.to_string_lossy().into_owned(),
    ];
    for target in strings(spec, "targets")? {
        args.extend(["--test".into(), target]);
    }
    let features = strings(spec, "features")?;
    if !features.is_empty() {
        args.extend(["--features".into(), features.join(",")]);
    }
    args.extend(["-E".into(), filters.join(" | ")]);
    Ok((args, expected))
}

/// Persist the artifact before propagating any selection, compilation, parse,
/// process, outcome, or input-integrity failure to the latest-attempt report.
fn run_surface(
    spec: &Value,
    surface: &str,
    root: &Path,
    evidence: &Path,
) -> Result<(String, bool)> {
    let before = source_index(root)?;
    let mut artifact = json!({
        "schema": 2, "surface": surface, "revision": capture(root, &["git", "rev-parse", "HEAD"])?.trim(),
        "source_index": store(&evidence.join("provenance"), &before)?,
        "batch": store(&evidence.join("batches"), spec)?,
        "toolchain": capture(root, &["rustc", "-Vv"])?,
        "nextest": capture(root, &["cargo", "nextest", "--version"])?,
        "environment": {"RIG_PROVIDER_TEST_MODE": "replay", "credentials": "common credential suffixes removed; not an OS network barrier"},
        "status": "incomplete", "results": [], "logs": {}
    });
    let attempt = (|| -> Result<()> {
        let (args, expected) = options(spec, surface, &root.join("target"))?;
        eprintln!("{surface}: compile and select {} exact IDs", expected.len());
        let list_args: Vec<_> = [
            vec!["cargo".into(), "nextest".into(), "list".into()],
            args.clone(),
            vec!["--message-format".into(), "json".into()],
        ]
        .concat();
        put(&mut artifact, &["listing_command"], json!(list_args))?;
        let listing = command(root, &list_args)?.output()?;
        put(
            &mut artifact,
            &["logs", "listing_stdout"],
            json!(store_raw(&evidence.join("logs"), &listing.stdout, "json")?),
        )?;
        put(
            &mut artifact,
            &["logs", "listing_stderr"],
            json!(store_raw(&evidence.join("logs"), &listing.stderr, "log")?),
        )?;
        put(
            &mut artifact,
            &["listing_exit_code"],
            json!(listing.status.code()),
        )?;
        if !listing.status.success() {
            return Err("listing/build failed; see retained compiler log".into());
        }
        let listing: Value = serde_json::from_slice(&listing.stdout)?;
        put(
            &mut artifact,
            &["listing"],
            json!(store(&evidence.join("listings"), &listing)?),
        )?;
        if selected_ids(&listing)? != expected {
            return Err("compiled selection differs from batch IDs".into());
        }
        let run_args: Vec<_> = [
            vec!["cargo".into(), "nextest".into(), "run".into()],
            args,
            [
                "--retries",
                "0",
                "--no-fail-fast",
                "--message-format",
                "libtest-json-plus",
                "--success-output",
                "immediate",
            ]
            .map(str::to_owned)
            .to_vec(),
        ]
        .concat();
        put(&mut artifact, &["command"], json!(run_args))?;
        let result = command(root, &run_args)?.stdin(Stdio::null()).output()?;
        put(&mut artifact, &["exit_code"], json!(result.status.code()))?;
        put(
            &mut artifact,
            &["logs", "stdout"],
            json!(store_raw(&evidence.join("logs"), &result.stdout, "jsonl")?),
        )?;
        put(
            &mut artifact,
            &["logs", "stderr"],
            json!(store_raw(&evidence.join("logs"), &result.stderr, "log")?),
        )?;
        let events = outcomes(std::str::from_utf8(&result.stdout)?, &expected)?;
        let markers = strings(spec, "semantic_skip_markers")?;
        let mut results = Vec::new();
        let mut passed = result.status.success();
        for cell in array(spec, "cells")? {
            let id = text(cell, surface)?;
            let event = events.get(id).ok_or("missing selected result")?;
            let output = format!(
                "{}{}",
                field(event, "stdout").as_str().unwrap_or_default(),
                field(event, "stderr").as_str().unwrap_or_default()
            );
            let status = if markers.iter().any(|m| output.contains(m)) {
                "semantic_skip"
            } else {
                text(event, "event")?
            };
            let expected = field(cell, "expected").as_str().unwrap_or("ok");
            passed &= status == expected;
            results.push(json!({"id": id, "status": status, "expected": expected, "seconds": field(event, "exec_time")}));
        }
        put(&mut artifact, &["results"], json!(results))?;
        if !passed {
            return Err("failed or semantically skipped selected case".into());
        }
        Ok(())
    })();
    let (name, passed) = finish_artifact(artifact, &before, source_index(root), attempt, evidence)?;
    eprintln!(
        "{surface}: {} ({name})",
        if passed {
            "expected outcomes"
        } else {
            "incomplete"
        }
    );
    Ok((name, passed))
}

fn finish_artifact(
    mut artifact: Value,
    before: &Value,
    after: Result<Value>,
    attempt: Result<()>,
    evidence: &Path,
) -> Result<(String, bool)> {
    let unchanged = match after {
        Ok(after) => {
            put(
                &mut artifact,
                &["source_index_after"],
                json!(store(&evidence.join("provenance"), &after)?),
            )?;
            if *before != after {
                put(
                    &mut artifact,
                    &["input_error"],
                    json!("source or fixtures changed during execution"),
                )?;
            }
            *before == after
        }
        Err(error) => {
            put(
                &mut artifact,
                &["input_error"],
                json!(format!("post-run source inspection failed: {error}")),
            )?;
            false
        }
    };
    let passed = attempt.is_ok() && unchanged;
    put(
        &mut artifact,
        &["status"],
        json!(if passed { "executed" } else { "incomplete" }),
    )?;
    if let Err(error) = attempt {
        put(&mut artifact, &["error"], json!(error.to_string()))?;
    }
    let name = store(&evidence.join("runs"), &artifact)?;
    Ok((name, passed))
}

pub(crate) fn run(candidate: &Path, args: Vec<String>) -> Result<()> {
    let [batch, baseline] = args.as_slice() else {
        return Err("usage: cargo xtask parity-batch <batch.json> <immutable-baseline>".into());
    };
    let spec = read(Path::new(batch))?;
    let name = text(&spec, "name")?;
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err("invalid batch name".into());
    }
    let candidate = candidate.canonicalize()?;
    let evidence = candidate.join("tests/ecs_parity/evidence");
    fs::create_dir_all(&evidence)?;
    let report_path = evidence.join(format!("{name}-report.json"));
    if report_path.exists() {
        store(&evidence.join("reports"), &read(&report_path)?)?;
    }
    let mut report = json!({"schema": 2, "batch": name, "cells": array(&spec, "cells")?.len(), "runs": {}, "status": "running", "assertion_review": "required separately"});
    write(&report_path, &report)?;
    let attempt = (|| -> Result<()> {
        let baseline = Path::new(baseline).canonicalize()?;
        fs::create_dir_all(baseline.join("target"))?;
        fs::create_dir_all(candidate.join("target"))?;
        if baseline == candidate
            || baseline.join("target").canonicalize()? == candidate.join("target").canonicalize()?
        {
            return Err(
                "baseline and candidate require distinct checkouts and target directories".into(),
            );
        }
        if capture(&baseline, &["git", "rev-parse", "HEAD"])?.trim()
            != text(&spec, "baseline_revision")?
        {
            return Err("wrong immutable baseline revision".into());
        }
        if !capture(&baseline, &["git", "status", "--porcelain"])?
            .trim()
            .is_empty()
        {
            return Err("baseline has modified or untracked files".into());
        }
        check_originals(&spec, &baseline, &candidate)?;
        for (surface, root) in [("original", &baseline), ("native", &candidate)] {
            let (run, passed) = run_surface(&spec, surface, root, &evidence)?;
            put(&mut report, &["runs", surface], json!(run))?;
            write(&report_path, &report)?;
            if !passed {
                return Err(
                    format!("{surface} batch incomplete; inspect retained run evidence").into(),
                );
            }
        }
        check_originals(&spec, &baseline, &candidate)?;
        Ok(())
    })();
    put(
        &mut report,
        &["status"],
        json!(if attempt.is_ok() {
            "executed"
        } else {
            "incomplete"
        }),
    )?;
    if let Err(error) = &attempt {
        put(&mut report, &["error"], json!(error.to_string()))?;
    }
    write(&report_path, &report)?;
    store(&evidence.join("reports"), &report)?;
    attempt
}
