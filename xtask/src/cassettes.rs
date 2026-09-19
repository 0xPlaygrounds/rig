//! `cargo xtask cassettes`: the provenance ledger for committed provider
//! cassettes, and `check-cassette-provenance`, the guard that keeps it honest.
//!
//! A committed cassette is evidence. Evidence with no stated origin is worth
//! nothing, so every fixture under `crates/rig-cassette/fixtures/cassettes/`
//! must be declared in `crates/rig-cassette/fixtures/scenarios.json` as one of:
//!
//! * **live** — recorded against the real provider;
//! * **unrecorded** — live by intent, not captured yet because the producing
//!   test is `#[ignore]`d; no file on disk;
//! * **derived** — hand-built from named live recordings, with a reason and a
//!   rebuild recipe;
//! * **scripted** — no fixture at all; a test module injects the behaviour.
//!
//! The guard is deliberately two-sided. A fixture with no declaration fails,
//! and a declaration with no fixture fails, because either direction alone
//! degrades into a rubber stamp: a one-sided "everything declared exists"
//! check passes an empty manifest, and a one-sided "everything on disk is
//! declared" check passes a manifest that claims recordings nobody has. The
//! same reasoning drives the vacuity floor at the bottom of `check`.
//!
//! `plan` is the non-recording half of `record`: it resolves the same
//! selection and prints the exact `cargo test` invocations, touching no
//! network, no credentials and no fixture. Scenario-to-test resolution is
//! structural — `syn` finds the enclosing `#[tokio::test]`/`#[test]` function
//! or matrix row that references the scenario — so a renamed test surfaces as
//! an unresolved scenario rather than as a plausible-looking command that
//! matches nothing.

#[path = "../../test-support/rig-test-support/src/provenance/manifest.rs"]
mod manifest;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::TokenStream;
use syn::parse::{ParseStream, Parser};
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, ExprCall, ExprLit, Item, ItemFn, Lit, LitStr, Token, parenthesized};

use manifest::{Category, Manifest, ProviderScenarios};

/// Cassette fixtures, relative to the workspace root.
const FIXTURE_ROOT: &str = "crates/rig-cassette/fixtures/cassettes";
/// Root of the cassette package's test tree, relative to the workspace root.
const TESTS_ROOT: &str = "crates/rig-cassette/tests";
/// The cassette package, relative to the workspace root.
const CASSETTE_CRATE: &str = "crates/rig-cassette";
/// Recording entry points that bypass the provenance-carrying wrappers.
const RAW_ENTRY_POINTS: &[&str] = &["start", "start_via", "start_at"];
/// The only places allowed to call a raw entry point: the engine itself and
/// the one helper module every provider suite goes through.
const RAW_ENTRY_POINT_HOMES: &[&str] = &[
    "crates/rig-cassette/src/",
    "test-support/rig-test-support/src/cassettes.rs",
];
/// Matrix macros whose rows declare a scenario and generate a test.
const MATRIX_MACROS: &[&str] = &["golden_matrix", "resume_matrix", "case_matrix"];
/// Floors below which the walks cannot be trusted to have found anything.
const MIN_FIXTURES: usize = 1000;
const MIN_PROVIDERS: usize = 10;

pub(crate) const USAGE: &str = "\
usage: cargo xtask cassettes <subcommand>

subcommands:
  list [--provider P] [--provenance live|derived|scripted] [--json]
        one row per declared scenario: provenance, fixture, sources, reason
  plan [--provider P] [--scenario P/SCENARIO]
        print the recording commands for the selected live scenarios and the
        derived/scripted scenarios excluded from them; records nothing
  record [--provider P] [--scenario P/SCENARIO]
        run exactly the commands `plan` prints, with RIG_PROVIDER_TEST_MODE=record
";

/// Dispatch one `cargo xtask cassettes` invocation.
pub(crate) fn run(workspace_root: &Path, args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("list") => list(workspace_root, &ListOptions::parse(args)?),
        Some("plan") => plan(workspace_root, &Selection::parse(args)?),
        Some("record") => record(workspace_root, &Selection::parse(args)?),
        Some(other) => Err(format!("unknown cassettes subcommand {other:?}\n{USAGE}")),
        None => Err(format!("cassettes requires a subcommand\n{USAGE}")),
    }
}

// ---------------------------------------------------------------- options

#[derive(Debug, Default)]
struct ListOptions {
    provider: Option<String>,
    provenance: Option<Category>,
    json: bool,
}

impl ListOptions {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut out = Self::default();
        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--provider" => out.provider = Some(value(&mut args, "--provider", "a provider")?),
                "--provenance" => {
                    let text = value(&mut args, "--provenance", "live, derived or scripted")?;
                    out.provenance = Some(Category::parse(&text).ok_or_else(|| {
                        format!("unknown provenance {text:?}; expected live, derived or scripted")
                    })?);
                }
                "--json" => out.json = true,
                _ => return Err(format!("unknown cassettes list option {arg}\n{USAGE}")),
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Default)]
struct Selection {
    provider: Option<String>,
    scenario: Option<String>,
}

impl Selection {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut out = Self::default();
        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--provider" => out.provider = Some(value(&mut args, "--provider", "a provider")?),
                "--scenario" => {
                    out.scenario = Some(value(&mut args, "--scenario", "provider/scenario")?);
                }
                _ => return Err(format!("unknown cassettes option {arg}\n{USAGE}")),
            }
        }
        if let Some(scenario) = &out.scenario
            && !scenario.contains('/')
        {
            return Err(format!(
                "--scenario takes a fully qualified provider/scenario id, not {scenario:?}"
            ));
        }
        Ok(out)
    }
}

fn value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
    expected: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires {expected}"))
}

// ------------------------------------------------------------------- list

/// One printable scenario, whatever its provenance.
struct Row {
    provider: String,
    /// Fully qualified id: `provider/scenario`, or `provider:family` for a
    /// scripted family, which owns no scenario path of its own.
    id: String,
    category: Category,
    /// Repository-relative fixture path, absent for scripted and unrecorded.
    fixture: Option<String>,
    sources: Vec<String>,
    reason: Option<String>,
    /// Live by intent, no committed recording yet.
    unrecorded: bool,
}

fn rows(manifest: &Manifest) -> Vec<Row> {
    let mut out = Vec::new();
    for provider in &manifest.providers {
        for scenario in &provider.live {
            out.push(Row {
                provider: provider.provider.clone(),
                id: format!("{}/{scenario}", provider.provider),
                category: Category::Live,
                fixture: Some(fixture_path(&provider.provider, scenario)),
                sources: Vec::new(),
                reason: None,
                unrecorded: false,
            });
        }
        for entry in &provider.unrecorded {
            out.push(Row {
                provider: provider.provider.clone(),
                id: format!("{}/{}", provider.provider, entry.scenario),
                category: Category::Live,
                fixture: None,
                sources: Vec::new(),
                reason: Some(entry.reason.clone()),
                unrecorded: true,
            });
        }
        for entry in &provider.derived {
            out.push(Row {
                provider: provider.provider.clone(),
                id: format!("{}/{}", provider.provider, entry.scenario),
                category: Category::Derived,
                fixture: Some(fixture_path(&provider.provider, &entry.scenario)),
                sources: entry.sources.clone(),
                reason: Some(entry.reason.clone()),
                unrecorded: false,
            });
        }
        for entry in &provider.scripted {
            out.push(Row {
                provider: provider.provider.clone(),
                id: format!("{}:{}", provider.provider, entry.family),
                category: Category::Scripted,
                fixture: None,
                sources: entry.sources.clone(),
                reason: Some(entry.reason.clone()),
                unrecorded: false,
            });
        }
    }
    out
}

fn list(workspace_root: &Path, options: &ListOptions) -> Result<(), String> {
    let manifest = Manifest::load(workspace_root)?;
    if let Some(provider) = &options.provider {
        known_provider(&manifest, provider)?;
    }
    let selected: Vec<Row> = rows(&manifest)
        .into_iter()
        .filter(|row| {
            options
                .provider
                .as_deref()
                .is_none_or(|p| row.provider == p)
        })
        .filter(|row| options.provenance.is_none_or(|c| row.category == c))
        .collect();

    if options.json {
        let payload: Vec<serde_json::Value> = selected
            .iter()
            .map(|row| {
                serde_json::json!({
                    "provider": row.provider,
                    "id": row.id,
                    "provenance": row.category.as_str(),
                    "unrecorded": row.unrecorded,
                    "fixture": row.fixture,
                    "sources": row.sources,
                    "reason": row.reason,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&payload)
                .map_err(|error| format!("could not render JSON: {error}"))?
        );
        return Ok(());
    }

    let width = selected.iter().map(|row| row.id.len()).max().unwrap_or(0);
    for row in &selected {
        let fixture = match (&row.fixture, row.unrecorded) {
            (Some(path), _) => path.clone(),
            (None, true) => "- (unrecorded)".to_owned(),
            (None, false) => "-".to_owned(),
        };
        println!("{:<9} {:<width$}  {fixture}", row.category.as_str(), row.id);
        if !row.sources.is_empty() {
            println!("{:width$}  sources: {}", "", row.sources.join(", "));
        }
        if let Some(reason) = &row.reason {
            println!("{:width$}  reason: {reason}", "");
        }
    }
    println!(
        "{} scenario(s): {} live ({} unrecorded), {} derived, {} scripted",
        selected.len(),
        selected
            .iter()
            .filter(|r| r.category == Category::Live)
            .count(),
        selected.iter().filter(|r| r.unrecorded).count(),
        selected
            .iter()
            .filter(|r| r.category == Category::Derived)
            .count(),
        selected
            .iter()
            .filter(|r| r.category == Category::Scripted)
            .count(),
    );
    Ok(())
}

// --------------------------------------------------------- plan / record

/// A live scenario and the test that records it.
#[derive(Debug)]
struct Recording {
    id: String,
    /// The `cargo` arguments, without the leading program name.
    args: Vec<String>,
    /// Further tests that reference the same scenario; recording one is
    /// enough, but a reader deserves to know the others exist.
    also: Vec<String>,
}

impl Recording {
    fn display(&self) -> String {
        format!(
            "RIG_PROVIDER_TEST_MODE=record cargo {}",
            self.args.join(" ")
        )
    }
}

/// A scenario the recorder must not touch, and why.
#[derive(Debug)]
struct Excluded {
    id: String,
    category: Category,
    reason: String,
}

fn resolve(
    workspace_root: &Path,
    manifest: &Manifest,
    selection: &Selection,
) -> Result<(Vec<Recording>, Vec<Excluded>), String> {
    if let Some(provider) = &selection.provider {
        known_provider(manifest, provider)?;
    }
    if let Some(id) = &selection.scenario {
        refuse_non_live(manifest, id)?;
    }

    let mut recordings = Vec::new();
    let mut excluded = Vec::new();
    let mut unresolved = Vec::new();
    for provider in &manifest.providers {
        if selection
            .provider
            .as_deref()
            .is_some_and(|p| p != provider.provider)
        {
            continue;
        }
        let wanted = |id: &str| {
            selection
                .scenario
                .as_deref()
                .is_none_or(|selected| selected == id)
        };

        let needs_discovery = provider
            .live_scenarios()
            .iter()
            .any(|scenario| wanted(&format!("{}/{scenario}", provider.provider)));
        if !needs_discovery {
            continue;
        }
        let discovered = discover(workspace_root, provider)?;

        for scenario in &provider.live {
            let id = format!("{}/{scenario}", provider.provider);
            if !wanted(&id) {
                continue;
            }
            match recording(provider, scenario, &id, &discovered, false) {
                Some(found) => recordings.push(found),
                None => unresolved.push(id),
            }
        }
        for entry in &provider.unrecorded {
            let id = format!("{}/{}", provider.provider, entry.scenario);
            if !wanted(&id) {
                continue;
            }
            match recording(provider, &entry.scenario, &id, &discovered, true) {
                Some(found) => recordings.push(found),
                None => unresolved.push(id),
            }
        }
        for entry in &provider.derived {
            excluded.push(Excluded {
                id: format!("{}/{}", provider.provider, entry.scenario),
                category: Category::Derived,
                reason: entry.reason.clone(),
            });
        }
        for entry in &provider.scripted {
            excluded.push(Excluded {
                id: format!("{}:{}", provider.provider, entry.family),
                category: Category::Scripted,
                reason: entry.reason.clone(),
            });
        }
    }

    if !unresolved.is_empty() {
        return Err(format!(
            "no test references these declared scenarios, so there is no command that records \
             them; fix the declaration or the test:\n  {}",
            unresolved.join("\n  ")
        ));
    }
    if recordings.is_empty() {
        return Err(match &selection.scenario {
            Some(id) => format!("{id} is not a declared live scenario"),
            None => "the selection matched no live scenario".to_owned(),
        });
    }
    Ok((recordings, excluded))
}

/// The recording command for one scenario, or `None` when no test claims it.
fn recording(
    provider: &ProviderScenarios,
    scenario: &str,
    id: &str,
    discovered: &Discovery,
    unrecorded: bool,
) -> Option<Recording> {
    let table = if unrecorded {
        &discovered.ignored_tests
    } else {
        &discovered.tests
    };
    let mut paths = table.get(scenario)?.iter();
    let first = paths.next()?;
    let mut args = vec![
        "test".to_owned(),
        "-p".to_owned(),
        "rig-cassette".to_owned(),
        "--all-features".to_owned(),
        "--test".to_owned(),
        provider.provider.clone(),
        "--".to_owned(),
        "--exact".to_owned(),
        first.clone(),
        "--nocapture".to_owned(),
        "--test-threads=1".to_owned(),
    ];
    // An unrecorded scenario's producer is `#[ignore]`d by definition: libtest
    // will not run it without being told to.
    if unrecorded {
        args.push("--ignored".to_owned());
    }
    Some(Recording {
        id: id.to_owned(),
        args,
        also: paths.cloned().collect(),
    })
}

/// Recording a derived or scripted scenario would destroy the very thing the
/// declaration exists to protect, so the selection is rejected before any
/// command is built.
fn refuse_non_live(manifest: &Manifest, id: &str) -> Result<(), String> {
    let Some((provider_name, scenario)) = id.split_once('/') else {
        return Err(format!(
            "--scenario takes a fully qualified provider/scenario id, not {id:?}"
        ));
    };
    let provider = known_provider(manifest, provider_name)?;
    if let Some(entry) = provider.derived.iter().find(|d| d.scenario == scenario) {
        return Err(format!(
            "{id} is derived, not recorded: {}\nrebuild it instead: {}",
            entry.reason, entry.rebuild
        ));
    }
    if let Some(entry) = provider
        .scripted
        .iter()
        .find(|s| s.family == scenario || s.cases.iter().any(|case| case == scenario))
    {
        return Err(format!(
            "{id} belongs to the scripted family {:?}, which has no fixture to record: {}",
            entry.family, entry.reason
        ));
    }
    if !provider.live_scenarios().contains(&scenario) {
        return Err(format!(
            "{id} is not declared in {}; declare it before recording it",
            manifest::MANIFEST_PATH
        ));
    }
    Ok(())
}

fn known_provider<'a>(
    manifest: &'a Manifest,
    provider: &str,
) -> Result<&'a ProviderScenarios, String> {
    manifest.provider(provider).ok_or_else(|| {
        format!(
            "unknown provider {provider:?}; declared providers: {}",
            manifest
                .providers
                .iter()
                .map(|p| p.provider.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

fn plan(workspace_root: &Path, selection: &Selection) -> Result<(), String> {
    let manifest = Manifest::load(workspace_root)?;
    let (recordings, excluded) = resolve(workspace_root, &manifest, selection)?;
    println!(
        "Recording plan: {} live scenario(s). Preview only: no network, no credentials, no \
         fixture touched.",
        recordings.len()
    );
    for entry in &recordings {
        println!("SELECT {}", entry.id);
        println!("  {}", entry.display());
        for other in &entry.also {
            println!("  (also referenced by {other})");
        }
    }
    for entry in &excluded {
        println!(
            "EXCLUDE {} ({}): {}",
            entry.id,
            entry.category.as_str(),
            entry.reason
        );
    }
    Ok(())
}

fn record(workspace_root: &Path, selection: &Selection) -> Result<(), String> {
    let manifest = Manifest::load(workspace_root)?;
    let (recordings, excluded) = resolve(workspace_root, &manifest, selection)?;
    println!(
        "Recording {} live scenario(s); {} derived/scripted scenario(s) excluded.",
        recordings.len(),
        excluded.len()
    );
    for entry in &recordings {
        println!("RECORD {}\n  {}", entry.id, entry.display());
        let status = Command::new("cargo")
            .args(&entry.args)
            .current_dir(workspace_root)
            .env("RIG_PROVIDER_TEST_MODE", "record")
            .status()
            .map_err(|error| format!("could not run cargo for {}: {error}", entry.id))?;
        if !status.success() {
            return Err(format!("recording {} failed: {status}", entry.id));
        }
    }
    Ok(())
}

// -------------------------------------------------------------- discovery

/// What an AST scan of one provider's test sources found.
#[derive(Debug, Default)]
struct Discovery {
    /// Scenarios referenced by tests that actually run. This is the set the
    /// fixture corpus is checked against, so it follows `cassette_safety`'s
    /// rules exactly: `#[ignore]`d functions and matrix rows contribute
    /// nothing.
    scenarios: BTreeSet<String>,
    /// Scenarios referenced only by `#[ignore]`d tests.
    ignored: BTreeSet<String>,
    /// Scenario to the test paths that reach it.
    tests: BTreeMap<String, BTreeSet<String>>,
    /// The same, for `#[ignore]`d tests: what `--ignored` would run.
    ignored_tests: BTreeMap<String, BTreeSet<String>>,
}

fn discover(workspace_root: &Path, provider: &ProviderScenarios) -> Result<Discovery, String> {
    let dir = workspace_root
        .join(CASSETTE_CRATE)
        .join(&provider.source_dir);
    if !dir.is_dir() {
        return Err(format!(
            "{}/{}: declared source_dir does not exist",
            CASSETTE_CRATE, provider.source_dir
        ));
    }
    let wrappers: BTreeSet<&str> = provider.wrappers.iter().map(String::as_str).collect();
    let mut files = Vec::new();
    collect_rust_files(&dir, &mut files)?;
    files.sort();

    let mut out = Discovery::default();
    for file in &files {
        let prefix = module_path(workspace_root, file)?;
        let source = std::fs::read_to_string(file)
            .map_err(|error| format!("{}: {error}", display(workspace_root, file)))?;
        let parsed = syn::parse_file(&source)
            .map_err(|error| format!("{}: {error}", display(workspace_root, file)))?;
        let mut scan = FileScan::default();
        scan.items(&parsed.items, &prefix, &wrappers)
            .map_err(|error| format!("{}: {error}", display(workspace_root, file)))?;
        scan.finish(&mut out);
    }
    Ok(out)
}

/// One source file's functions, matrix rows, and the calls between them.
#[derive(Default)]
struct FileScan {
    /// Function name to what its body references directly.
    functions: BTreeMap<String, FnScan>,
    /// Test functions: name, module-qualified path, and whether ignored.
    entry_points: Vec<(String, String, bool)>,
    /// Matrix rows: scenario, test path, ignored.
    matrix: Vec<(String, String, bool)>,
}

#[derive(Default)]
struct FnScan {
    scenarios: BTreeSet<String>,
    calls: BTreeSet<String>,
    ignored: bool,
}

impl FileScan {
    fn items(
        &mut self,
        items: &[Item],
        prefix: &str,
        wrappers: &BTreeSet<&str>,
    ) -> Result<(), String> {
        for item in items {
            match item {
                Item::Fn(function) => self.function(function, prefix, wrappers),
                Item::Mod(module) => {
                    if let Some((_, inner)) = &module.content {
                        let nested = format!("{prefix}::{}", module.ident);
                        self.items(inner, &nested, wrappers)?;
                    }
                }
                Item::Macro(macro_item) => {
                    let Some(name) = macro_item.mac.path.segments.last() else {
                        continue;
                    };
                    if !MATRIX_MACROS.iter().any(|m| name.ident == m) {
                        continue;
                    }
                    let matrix = parse_matrix(macro_item.mac.tokens.clone())
                        .map_err(|error| format!("could not read {} rows: {error}", name.ident))?;
                    if !matrix
                        .wrapper
                        .as_deref()
                        .is_some_and(|w| wrappers.contains(w))
                    {
                        continue;
                    }
                    for row in matrix.rows {
                        self.matrix.push((
                            row.scenario,
                            format!("{prefix}::{}", row.name),
                            row.ignored,
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn function(&mut self, function: &ItemFn, prefix: &str, wrappers: &BTreeSet<&str>) {
        let name = function.sig.ident.to_string();
        let ignored = has_attribute(&function.attrs, "ignore");
        let mut visitor = FnVisitor {
            wrappers,
            scan: FnScan {
                ignored,
                ..FnScan::default()
            },
        };
        visitor.visit_block(&function.block);
        if is_test(&function.attrs) {
            self.entry_points
                .push((name.clone(), format!("{prefix}::{name}"), ignored));
        }
        self.functions.insert(name, visitor.scan);
    }

    /// Fold this file's findings into the provider-wide discovery, resolving
    /// one helper level at a time: a test that calls `fixture()` which calls
    /// the wrapper still owns that scenario.
    fn finish(self, out: &mut Discovery) {
        for (scenario, test, ignored) in &self.matrix {
            let (set, table) = if *ignored {
                (&mut out.ignored, &mut out.ignored_tests)
            } else {
                (&mut out.scenarios, &mut out.tests)
            };
            set.insert(scenario.clone());
            table
                .entry(scenario.clone())
                .or_default()
                .insert(test.clone());
        }
        for scan in self.functions.values() {
            if scan.ignored {
                continue;
            }
            out.scenarios.extend(scan.scenarios.iter().cloned());
        }
        for (name, test, ignored) in &self.entry_points {
            let mut seen = BTreeSet::new();
            let reached = self.reachable(name, &mut seen);
            let (set, table) = if *ignored {
                (&mut out.ignored, &mut out.ignored_tests)
            } else {
                (&mut out.scenarios, &mut out.tests)
            };
            for scenario in reached {
                set.insert(scenario.clone());
                table.entry(scenario).or_default().insert(test.clone());
            }
        }
    }

    fn reachable(&self, name: &str, seen: &mut BTreeSet<String>) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        if !seen.insert(name.to_owned()) {
            return out;
        }
        let Some(scan) = self.functions.get(name) else {
            return out;
        };
        out.extend(scan.scenarios.iter().cloned());
        for callee in &scan.calls {
            out.extend(self.reachable(callee, seen));
        }
        out
    }
}

struct FnVisitor<'a> {
    wrappers: &'a BTreeSet<&'a str>,
    scan: FnScan,
}

impl<'ast> Visit<'ast> for FnVisitor<'_> {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref()
            && let Some(last) = path.path.segments.last()
        {
            let name = last.ident.to_string();
            if self.wrappers.contains(name.as_str()) {
                if let Some(scenario) = node.args.first().and_then(scenario_literal) {
                    self.scan.scenarios.insert(scenario);
                }
            } else {
                self.scan.calls.insert(name);
            }
        }
        visit::visit_expr_call(self, node);
    }
}

/// The scenario a wrapper's first argument names, through the literal
/// `CassetteSpec` builder chains the suites use.
fn scenario_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(scenario),
            ..
        }) => Some(scenario.value()),
        Expr::Call(call) => {
            let Expr::Path(path) = call.func.as_ref() else {
                return None;
            };
            let mut segments = path.path.segments.iter().rev();
            match (segments.next(), segments.next()) {
                (Some(method), Some(receiver))
                    if method.ident == "new" && receiver.ident == "CassetteSpec" =>
                {
                    call.args.first().and_then(scenario_literal)
                }
                _ => None,
            }
        }
        Expr::MethodCall(call) => scenario_literal(&call.receiver),
        Expr::Paren(paren) => scenario_literal(&paren.expr),
        Expr::Reference(reference) => scenario_literal(&reference.expr),
        _ => None,
    }
}

fn is_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let path: Vec<String> = attr
            .path()
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        path == ["test"] || path == ["tokio", "test"]
    })
}

fn has_attribute(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident(name))
}

struct Matrix {
    wrapper: Option<String>,
    rows: Vec<MatrixRow>,
}

struct MatrixRow {
    name: String,
    scenario: String,
    ignored: bool,
}

/// Read the rows of a `golden_matrix!`/`resume_matrix!`/`case_matrix!`
/// invocation. Only the shape the row names and scenarios live in is parsed:
/// the macros' own definitions validate everything else, and duplicating that
/// validation here would make this guard fail for reasons that are not its
/// business.
fn parse_matrix(tokens: TokenStream) -> Result<Matrix, String> {
    let parser = |input: ParseStream<'_>| -> syn::Result<Matrix> {
        let mut wrapper = None;
        loop {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let path: syn::Path = input.parse()?;
            if key == "wrapper" {
                wrapper = path.segments.last().map(|s| s.ident.to_string());
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                continue;
            }
            input.parse::<Token![;]>()?;
            break;
        }
        let mut rows = Vec::new();
        while !input.is_empty() {
            let attrs = input.call(Attribute::parse_outer)?;
            let name: syn::Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let mut scenario = None;
            if input.peek(syn::token::Paren) {
                let arguments;
                parenthesized!(arguments in input);
                let literal: LitStr = arguments.parse()?;
                scenario = Some(literal.value());
                arguments.parse::<TokenStream>()?;
            } else {
                while !input.is_empty() && !input.peek(Token![;]) {
                    input.parse::<proc_macro2::TokenTree>()?;
                }
            }
            input.parse::<Token![;]>()?;
            if let Some(scenario) = scenario {
                rows.push(MatrixRow {
                    name: name.to_string(),
                    scenario,
                    ignored: has_attribute(&attrs, "ignore"),
                });
            }
        }
        Ok(Matrix { wrapper, rows })
    };
    parser.parse2(tokens).map_err(|error| error.to_string())
}

/// The module path rustc gives a test source. Each provider's integration test
/// root redirects `providers/<provider>/mod.rs` to a module named after the
/// provider (`#[path = "providers/openai/mod.rs"] mod openai;`), so the
/// `providers/` directory itself contributes no segment.
fn module_path(workspace_root: &Path, file: &Path) -> Result<String, String> {
    let tests = workspace_root.join(TESTS_ROOT);
    let relative = file
        .strip_prefix(&tests)
        .map_err(|_| format!("{} is outside {TESTS_ROOT}", file.display()))?;
    let mut segments: Vec<String> = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();
    if segments.first().is_some_and(|first| first == "providers") {
        segments.remove(0);
    }
    if let Some(last) = segments.last_mut() {
        *last = last.trim_end_matches(".rs").to_owned();
    }
    if segments.last().is_some_and(|last| last == "mod") {
        segments.pop();
    }
    Ok(segments.join("::"))
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("could not read {}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("could not read an entry in {}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_rust_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn fixture_path(provider: &str, scenario: &str) -> String {
    format!("{FIXTURE_ROOT}/{provider}/{scenario}.yaml")
}

fn display(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

// ------------------------------------------------------------------ check

/// The ownership guard: every fixture declared, every declaration real, every
/// derivation traceable, and no way to record around the declaration.
pub(crate) fn check(workspace_root: &Path) -> Result<(), String> {
    let manifest = Manifest::load(workspace_root)?;
    let mut offenders: Vec<(String, String)> = Vec::new();

    let fixtures = fixtures_on_disk(workspace_root)?;
    let providers_on_disk: BTreeSet<&str> = fixtures.keys().map(|(p, _)| p.as_str()).collect();
    // A walk that found nothing must not report success: an empty or misplaced
    // corpus would otherwise satisfy every rule below.
    if fixtures.len() < MIN_FIXTURES {
        return Err(format!(
            "the fixture walk found only {} cassette(s) under {FIXTURE_ROOT}; refusing to pass \
             vacuously",
            fixtures.len()
        ));
    }
    if providers_on_disk.len() < MIN_PROVIDERS {
        return Err(format!(
            "the fixture walk found only {} provider director(ies) under {FIXTURE_ROOT}; refusing \
             to pass vacuously",
            providers_on_disk.len()
        ));
    }

    let mut declared: BTreeMap<String, Category> = BTreeMap::new();
    let mut unrecorded: BTreeSet<String> = BTreeSet::new();
    for provider in &manifest.providers {
        for scenario in &provider.live {
            declared.insert(format!("{}/{scenario}", provider.provider), Category::Live);
        }
        for entry in &provider.derived {
            declared.insert(
                format!("{}/{}", provider.provider, entry.scenario),
                Category::Derived,
            );
        }
        for entry in &provider.unrecorded {
            unrecorded.insert(format!("{}/{}", provider.provider, entry.scenario));
        }
        // A scripted family owns no fixture; sharing its name with a recorded
        // scenario would make one id mean two things.
        for entry in &provider.scripted {
            let id = format!("{}/{}", provider.provider, entry.family);
            if declared.contains_key(&id) || unrecorded.contains(&id) {
                offenders.push((
                    manifest::MANIFEST_PATH.to_owned(),
                    format!("{id} is both a scripted family and a fixture-backed scenario"),
                ));
            }
        }
    }

    // (a) two-sided fixture ownership, plus the unrecorded rule: an id that
    // claims to have no recording must not have one.
    for ((provider, scenario), path) in &fixtures {
        let id = format!("{provider}/{scenario}");
        if unrecorded.contains(&id) {
            offenders.push((
                path.clone(),
                "declared unrecorded but a fixture exists; move it to \"live\"".to_owned(),
            ));
        } else if !declared.contains_key(&id) {
            offenders.push((
                path.clone(),
                format!(
                    "no provenance declared; add {id} to {}",
                    manifest::MANIFEST_PATH
                ),
            ));
        }
    }
    for (id, category) in &declared {
        let Some((provider, scenario)) = id.split_once('/') else {
            continue;
        };
        if !fixtures.contains_key(&(provider.to_owned(), scenario.to_owned())) {
            offenders.push((
                fixture_path(provider, scenario),
                format!(
                    "declared {} but the fixture does not exist",
                    category.as_str()
                ),
            ));
        }
    }

    check_sources(&manifest, &declared, &unrecorded, &mut offenders);
    check_discovery(workspace_root, &manifest, &unrecorded, &mut offenders)?;
    check_scripted_modules(workspace_root, &manifest, &mut offenders);
    check_raw_entry_points(workspace_root, &mut offenders)?;

    offenders.sort();
    offenders.dedup();
    if offenders.is_empty() {
        println!(
            "ok: {} cassette(s) across {} provider(s) each declare their provenance",
            fixtures.len(),
            providers_on_disk.len()
        );
        return Ok(());
    }
    let mut message = String::from(
        "cassette provenance is not sound; each line names the file and what is wrong with it:\n",
    );
    for (path, why) in &offenders {
        message.push_str("  ");
        message.push_str(path);
        message.push_str(": ");
        message.push_str(why);
        message.push('\n');
    }
    Err(message)
}

/// Every committed cassette, keyed by `(provider, scenario)`.
fn fixtures_on_disk(workspace_root: &Path) -> Result<BTreeMap<(String, String), String>, String> {
    let root = workspace_root.join(FIXTURE_ROOT);
    let mut files = Vec::new();
    if root.is_dir() {
        collect_yaml_files(&root, &mut files)?;
    }
    let mut out = BTreeMap::new();
    for file in files {
        let relative = file
            .strip_prefix(&root)
            .map_err(|_| format!("{} escaped {FIXTURE_ROOT}", file.display()))?;
        let mut segments: Vec<String> = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect();
        if segments.len() < 2 {
            return Err(format!(
                "{}: a cassette must live under a provider directory",
                display(workspace_root, &file)
            ));
        }
        let provider = segments.remove(0);
        if let Some(last) = segments.last_mut() {
            *last = last.trim_end_matches(".yaml").to_owned();
        }
        let scenario = segments.join("/");
        out.insert(
            (provider, scenario),
            display(workspace_root, &file).replace('\\', "/"),
        );
    }
    Ok(out)
}

fn collect_yaml_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("could not read {}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("could not read an entry in {}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_yaml_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "yaml") {
            out.push(path);
        }
    }
    Ok(())
}

/// (c) Every `sources` entry names a real recording, nothing derives from
/// itself, and the derivation graph terminates.
fn check_sources(
    manifest: &Manifest,
    declared: &BTreeMap<String, Category>,
    unrecorded: &BTreeSet<String>,
    offenders: &mut Vec<(String, String)>,
) {
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for provider in &manifest.providers {
        for entry in &provider.derived {
            let id = format!("{}/{}", provider.provider, entry.scenario);
            for source in &entry.sources {
                if *source == id {
                    offenders.push((
                        manifest::MANIFEST_PATH.to_owned(),
                        format!("{id} is derived from itself"),
                    ));
                } else if !declared.contains_key(source) {
                    // A derivation needs bytes to start from, so an unrecorded
                    // source is as unusable as a nonexistent one.
                    offenders.push((
                        manifest::MANIFEST_PATH.to_owned(),
                        format!("{id} is derived from {source}, which is not a recorded scenario"),
                    ));
                }
            }
            edges.insert(id, entry.sources.clone());
        }
        for entry in &provider.scripted {
            let id = format!("{}:{}", provider.provider, entry.family);
            for source in &entry.sources {
                // A script starts from a scenario it replays or mutates at
                // runtime; that scenario may legitimately be one nobody has
                // recorded yet.
                if !declared.contains_key(source) && !unrecorded.contains(source) {
                    offenders.push((
                        manifest::MANIFEST_PATH.to_owned(),
                        format!("{id} scripts {source}, which is not a declared scenario"),
                    ));
                }
            }
        }
    }

    // Depth-first over derived-from-derived edges; a scenario reachable from
    // itself has no base recording anywhere underneath it.
    for start in edges.keys() {
        let mut stack = vec![start.clone()];
        let mut seen = BTreeSet::new();
        while let Some(current) = stack.pop() {
            for next in edges.get(&current).into_iter().flatten() {
                if next == start {
                    offenders.push((
                        manifest::MANIFEST_PATH.to_owned(),
                        format!("{start} is in a derivation cycle through {current}"),
                    ));
                    stack.clear();
                    break;
                }
                if seen.insert(next.clone()) {
                    stack.push(next.clone());
                }
            }
        }
    }
}

/// (d) The declaration and the test sources describe the same set.
fn check_discovery(
    workspace_root: &Path,
    manifest: &Manifest,
    unrecorded: &BTreeSet<String>,
    offenders: &mut Vec<(String, String)>,
) -> Result<(), String> {
    for provider in &manifest.providers {
        let discovered = discover(workspace_root, provider)?;
        let declared: BTreeSet<&str> = provider.fixture_scenarios().into_iter().collect();
        for scenario in &discovered.scenarios {
            if !declared.contains(scenario.as_str()) {
                offenders.push((
                    format!("{CASSETTE_CRATE}/{}", provider.source_dir),
                    format!(
                        "a test records {}/{scenario}, which is not declared in {}",
                        provider.provider,
                        manifest::MANIFEST_PATH
                    ),
                ));
            }
        }
        for scenario in declared {
            let id = format!("{}/{scenario}", provider.provider);
            // An unrecorded scenario is invisible here by construction: its
            // producing test is `#[ignore]`d, and discovery skips those.
            if unrecorded.contains(&id) {
                continue;
            }
            if !discovered.scenarios.contains(scenario) {
                offenders.push((
                    manifest::MANIFEST_PATH.to_owned(),
                    format!(
                        "{id} is declared but no test under {CASSETTE_CRATE}/{} uses it",
                        provider.source_dir
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// (f) A scripted family must be implemented by the module it names.
fn check_scripted_modules(
    workspace_root: &Path,
    manifest: &Manifest,
    offenders: &mut Vec<(String, String)>,
) {
    for provider in &manifest.providers {
        for entry in &provider.scripted {
            // `module` shares `source_dir`'s base: the cassette package, not
            // the workspace root.
            let relative = format!("{CASSETTE_CRATE}/{}", entry.module);
            let path = workspace_root.join(&relative);
            let Ok(source) = std::fs::read_to_string(&path) else {
                offenders.push((
                    relative.clone(),
                    format!(
                        "scripted family {}/{} names a module that does not exist",
                        provider.provider, entry.family
                    ),
                ));
                continue;
            };
            if !source.contains(&entry.family) {
                offenders.push((
                    relative.clone(),
                    format!(
                        "does not construct the scripted family {:?} it is declared for",
                        entry.family
                    ),
                ));
            }
            for case in &entry.cases {
                if !source.contains(case) {
                    offenders.push((
                        relative.clone(),
                        format!(
                            "scripted family {:?} declares the case {case:?}, which this module \
                             does not generate",
                            entry.family
                        ),
                    ));
                }
            }
        }
    }
}

/// (e) Nothing outside the engine and the one shared helper may open a
/// cassette directly: the raw entry points take no provenance, so a test
/// calling them records around the whole ledger.
fn check_raw_entry_points(
    workspace_root: &Path,
    offenders: &mut Vec<(String, String)>,
) -> Result<(), String> {
    let mut files = Vec::new();
    collect_workspace_rust_files(workspace_root, &mut files)?;
    files.sort();
    for file in &files {
        let relative = display(workspace_root, file).replace('\\', "/");
        if RAW_ENTRY_POINT_HOMES
            .iter()
            .any(|home| relative.starts_with(home))
        {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(file) else {
            continue;
        };
        if !source.contains("ProviderCassette") {
            continue;
        }
        let parsed = syn::parse_file(&source)
            .map_err(|error| format!("could not parse {relative}: {error}"))?;
        let mut visitor = EntryPointVisitor::default();
        visitor.visit_file(&parsed);
        for found in visitor.found {
            offenders.push((
                relative.clone(),
                format!(
                    "calls ProviderCassette::{found} directly; go through the provenance-carrying \
                     wrappers in test-support/rig-test-support/src/cassettes.rs"
                ),
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct EntryPointVisitor {
    found: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for EntryPointVisitor {
    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        let mut segments = node.path.segments.iter().rev();
        if let (Some(method), Some(receiver)) = (segments.next(), segments.next())
            && receiver.ident == "ProviderCassette"
            && RAW_ENTRY_POINTS.iter().any(|name| method.ident == name)
        {
            self.found.insert(method.ident.to_string());
        }
        visit::visit_expr_path(self, node);
    }
}

/// Every `.rs` file in the workspace, skipping build output and version
/// control. The entry-point guard is repository-wide by definition: the point
/// is that no file anywhere has its own way in.
fn collect_workspace_rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    const SKIP: &[&str] = &["target", ".git", "node_modules", ".jj", "pkg"];
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("could not read {}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("could not read an entry in {}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            if SKIP.contains(&name.as_str()) {
                continue;
            }
            collect_workspace_rust_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}
