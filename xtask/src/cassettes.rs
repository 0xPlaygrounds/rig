//! Repository provenance policy and credential-free recording plans.
//! Literal scenario declarations in explicitly approved live modules are shared
//! with the runtime resolver. Source inspection is syntactic, not a compiler;
//! recording additionally validates exact test identities against libtest.

#[path = "../../test-support/rig-test-support/src/scenario_registry.rs"]
mod discovery;
#[path = "../../test-support/rig-test-support/src/provenance/manifest.rs"]
mod manifest;
#[cfg(test)]
mod tests;

use discovery::{Discovery, collect_files, discover};
#[cfg(test)]
use discovery::{module_path, parse_matrix};
use manifest::{Category, Manifest, ProviderScenarios, RECORD_SCOPE_ENV};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::process::Command;
use syn::visit::{self, Visit};

const FIXTURE_ROOT: &str = "crates/rig-cassette/fixtures/cassettes";
const CASSETTE_CRATE: &str = "crates/rig-cassette";
const MIN_FIXTURES: usize = 1000;
const MIN_PROVIDERS: usize = 10;

pub(crate) const USAGE: &str = "\
usage: cargo xtask cassettes <list|plan|record> [--provider P]
  list [--provenance live|derived|scripted] [--json]
  plan|record [--scenario provider/scenario]
plan is credential-free and never contacts providers; record executes that plan.";

pub(crate) fn run(root: &Path, args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let command = args.next().ok_or(USAGE)?;
    let options = Options::parse(args, &command)?;
    match command.as_str() {
        "list" => list(root, &options),
        "plan" => plan(root, &options),
        "record" => record(root, &options),
        _ => Err(USAGE.to_owned()),
    }
}

#[derive(Debug, Default)]
struct Options {
    provider: Option<String>,
    scenario: Option<String>,
    provenance: Option<Category>,
    json: bool,
}

impl Options {
    fn parse(mut args: impl Iterator<Item = String>, command: &str) -> Result<Self, String> {
        let mut out = Self::default();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--provider" => out.provider = Some(value(&mut args, &arg)?),
                "--scenario" if command != "list" => out.scenario = Some(value(&mut args, &arg)?),
                "--provenance" if command == "list" => {
                    out.provenance = Some(
                        Category::parse(&value(&mut args, &arg)?)
                            .ok_or("expected live, derived or scripted")?,
                    )
                }
                "--json" if command == "list" => out.json = true,
                _ => return Err(format!("unknown cassettes {command} option {arg}\n{USAGE}")),
            }
        }
        if out.scenario.as_deref().is_some_and(|s| !s.contains('/')) {
            return Err("--scenario requires provider/scenario".into());
        }
        Ok(out)
    }
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

#[derive(Debug, serde::Serialize)]
struct Row {
    provider: String,
    id: String,
    #[serde(rename = "provenance")]
    category: Category,
    fixture: Option<String>,
    sources: Vec<String>,
    reason: Option<String>,
    unrecorded: bool,
}

fn rows(manifest: &Manifest) -> Vec<Row> {
    let mut rows = Vec::new();
    for p in &manifest.providers {
        let mut add = |scenario: &str,
                       category,
                       sources: &[String],
                       reason: Option<&str>,
                       has_fixture: bool| {
            let separator = if category == Category::Scripted && !has_fixture {
                ':'
            } else {
                '/'
            };
            rows.push(Row {
                provider: p.provider.clone(),
                id: format!("{}{separator}{scenario}", p.provider),
                category,
                fixture: has_fixture.then(|| fixture_path(&p.provider, scenario)),
                sources: sources.to_vec(),
                reason: reason.map(str::to_owned),
                unrecorded: category == Category::Live && !has_fixture,
            });
        };
        for s in &p.live {
            add(s, Category::Live, &[], None, true);
        }
        for s in &p.unrecorded {
            add(&s.scenario, Category::Live, &[], Some(&s.reason), false);
        }
        for s in &p.derived {
            add(
                &s.scenario,
                Category::Derived,
                &s.sources,
                Some(&s.reason),
                true,
            );
        }
        for s in &p.scripted {
            add(
                s.fixture.as_deref().unwrap_or(&s.family),
                Category::Scripted,
                &s.sources,
                Some(&s.reason),
                s.fixture.is_some(),
            );
        }
    }
    rows
}

fn list(root: &Path, options: &Options) -> Result<(), String> {
    let manifest = discovery::load(root).map_err(|e| e.to_string())?;
    if let Some(p) = &options.provider {
        known_provider(&manifest, p)?;
    }
    let selected: Vec<_> = rows(&manifest)
        .into_iter()
        .filter(|r| {
            options.provider.as_deref().is_none_or(|p| r.provider == p)
                && options.provenance.is_none_or(|c| r.category == c)
        })
        .collect();
    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&selected).map_err(|e| e.to_string())?
        );
    } else {
        for r in &selected {
            println!(
                "{} {} {}",
                r.category.as_str(),
                r.id,
                r.fixture
                    .as_deref()
                    .unwrap_or(if r.unrecorded { "- (unrecorded)" } else { "-" })
            );
            if !r.sources.is_empty() {
                println!("  sources: {}", r.sources.join(", "));
            }
            if let Some(reason) = &r.reason {
                println!("  reason: {reason}");
            }
        }
        println!("{} scenario(s)", selected.len());
    }
    Ok(())
}

#[derive(Debug)]
struct Recording {
    provider: String,
    test: String,
    ignored: bool,
    scenarios: BTreeSet<String>,
}

impl Recording {
    fn ids(&self) -> String {
        self.scenarios.iter().cloned().collect::<Vec<_>>().join(",")
    }

    fn args(&self) -> Vec<String> {
        let mut args: Vec<String> = [
            "test",
            "-p",
            "rig-cassette",
            "--all-features",
            "--test",
            &self.provider,
            "--",
            "--exact",
            &self.test,
            "--nocapture",
            "--test-threads=1",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        if self.ignored {
            args.push("--ignored".into());
        }
        args
    }

    fn display(&self) -> String {
        format!(
            "RIG_PROVIDER_TEST_MODE=record {RECORD_SCOPE_ENV}={} cargo {}",
            self.ids(),
            self.args().join(" ")
        )
    }
}

fn resolve(
    root: &Path,
    manifest: &Manifest,
    selection: &Options,
) -> Result<(Vec<Recording>, Vec<Row>), String> {
    if let Some(p) = &selection.provider {
        known_provider(manifest, p)?;
    }
    if let Some(id) = &selection.scenario {
        refuse_non_live(manifest, id)?;
    }
    let mut commands = BTreeMap::new();
    let excluded = rows(manifest)
        .into_iter()
        .filter(|r| {
            r.category != Category::Live
                && selection
                    .provider
                    .as_deref()
                    .is_none_or(|p| p == r.provider)
        })
        .collect();
    for p in &manifest.providers {
        if selection
            .provider
            .as_deref()
            .is_some_and(|name| name != p.provider)
        {
            continue;
        }
        let discovered = discover(root, p)?;
        for scenario in p.live_scenarios() {
            let id = format!("{}/{scenario}", p.provider);
            if selection.scenario.as_deref().is_some_and(|s| s != id) {
                continue;
            }
            if !p.recordable(root, scenario) {
                return Err(format!(
                    "{id} has no capture; explicitly declare it unrecorded before recording"
                ));
            }
            let unrecorded = p.unrecorded.iter().any(|s| s.scenario == scenario);
            let entry = recording(p, scenario, &discovered, unrecorded)
                .ok_or_else(|| format!("no test references declared scenario {id}"))?;
            // A test is the atomic recording unit. Never silently run a second
            // scenario outside the selection, or a mixed live/derived test.
            for other in &entry.scenarios {
                refuse_non_live(manifest, other)?;
                if selection.scenario.as_deref().is_some_and(|s| s != other) {
                    return Err(format!(
                        "{} also records {other}; select the provider instead",
                        entry.display()
                    ));
                }
            }
            commands.entry(entry.args()).or_insert(entry);
        }
    }
    if commands.is_empty() {
        return Err("selection matched no declared live scenario".into());
    }
    Ok((commands.into_values().collect(), excluded))
}

fn recording(
    p: &ProviderScenarios,
    scenario: &str,
    discovered: &Discovery,
    unrecorded: bool,
) -> Option<Recording> {
    let table = if unrecorded {
        &discovered.ignored_tests
    } else {
        &discovered.tests
    };
    let test = table.get(scenario)?.first()?;
    let scenarios = table
        .iter()
        .filter(|(_, tests)| tests.contains(test))
        .map(|(s, _)| format!("{}/{s}", p.provider))
        .collect();
    Some(Recording {
        provider: p.provider.clone(),
        test: test.clone(),
        ignored: unrecorded,
        scenarios,
    })
}

fn refuse_non_live(manifest: &Manifest, id: &str) -> Result<(), String> {
    let (provider, scenario) = id
        .split_once('/')
        .ok_or("--scenario requires provider/scenario")?;
    let p = known_provider(manifest, provider)?;
    if let Some(s) = p.derived.iter().find(|s| s.scenario == scenario) {
        return Err(format!(
            "{id} is derived, not recorded: {}\nrebuild it instead: {}",
            s.reason, s.rebuild
        ));
    }
    if let Some(s) = p.scripted.iter().find(|s| {
        s.family == scenario
            || s.fixture.as_deref() == Some(scenario)
            || s.cases.iter().any(|c| c == scenario)
    }) {
        return Err(format!(
            "{id} belongs to scripted family {}: {}",
            s.family, s.reason
        ));
    }
    if !p.live_scenarios().contains(&scenario) {
        return Err(format!(
            "{id} is not declared in {}",
            manifest::MANIFEST_PATH
        ));
    }
    Ok(())
}

fn known_provider<'a>(
    manifest: &'a Manifest,
    provider: &str,
) -> Result<&'a ProviderScenarios, String> {
    manifest
        .provider(provider)
        .ok_or_else(|| format!("unknown provider {provider:?}"))
}

fn plan(root: &Path, selection: &Options) -> Result<(), String> {
    let (recordings, excluded) = validated_plan(root, selection).map_err(|e| e.to_string())?;
    for entry in &recordings {
        println!("SELECT {}\n  {}", entry.ids(), entry.display());
    }
    for entry in &excluded {
        println!(
            "EXCLUDE {} ({}): {}",
            entry.id,
            entry.category.as_str(),
            entry.reason.as_deref().unwrap_or_default()
        );
    }
    Ok(())
}

fn validated_plan(root: &Path, selection: &Options) -> anyhow::Result<(Vec<Recording>, Vec<Row>)> {
    let manifest = discovery::load(root)?;
    let plan = resolve(root, &manifest, selection).map_err(anyhow::Error::msg)?;
    let providers: BTreeSet<_> = plan.0.iter().map(|r| r.provider.as_str()).collect();
    for provider in providers {
        let inventory = |ignored: bool| -> anyhow::Result<BTreeSet<String>> {
            let mut command = Command::new("cargo");
            command.args([
                "test",
                "--offline",
                "--locked",
                "-p",
                "rig-cassette",
                "--all-features",
                "--test",
                provider,
                "--",
                "--list",
            ]);
            if ignored {
                command.arg("--ignored");
            }
            let output = command
                .current_dir(root)
                .env("RIG_PROVIDER_TEST_MODE", "replay")
                .output()?;
            anyhow::ensure!(
                output.status.success(),
                "cannot list {provider}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            Ok(String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| line.strip_suffix(": test").map(str::to_owned))
                .collect())
        };
        let all = inventory(false)?;
        let ignored = inventory(true)?;
        for entry in plan.0.iter().filter(|r| r.provider == provider) {
            validate_test(entry, &all, &ignored)?;
        }
    }
    Ok(plan)
}

fn validate_test(
    entry: &Recording,
    all: &BTreeSet<String>,
    ignored: &BTreeSet<String>,
) -> anyhow::Result<()> {
    let test = &entry.test;
    anyhow::ensure!(
        all.contains(test) && ignored.contains(test) == entry.ignored,
        "exact test {test} is missing or its ignored status changed"
    );
    Ok(())
}

fn record(root: &Path, selection: &Options) -> Result<(), String> {
    let (recordings, _) = validated_plan(root, selection).map_err(|e| e.to_string())?;
    for entry in &recordings {
        println!("RECORD {}\n  {}", entry.ids(), entry.display());
        let status = Command::new("cargo")
            .args(entry.args())
            .current_dir(root)
            .env("RIG_PROVIDER_TEST_MODE", "record")
            .env(RECORD_SCOPE_ENV, entry.ids())
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("recording {} failed: {status}", entry.ids()));
        }
    }
    Ok(())
}

fn fixture_path(provider: &str, scenario: &str) -> String {
    format!("{FIXTURE_ROOT}/{provider}/{scenario}.yaml")
}

fn display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let manifest = discovery::load(root).map_err(|e| e.to_string())?;
    let fixtures = fixtures_on_disk(root)?;
    let providers: BTreeSet<_> = fixtures.keys().map(|(p, _)| p).collect();
    if fixtures.len() < MIN_FIXTURES || providers.len() < MIN_PROVIDERS {
        return Err(format!(
            "found {} fixtures across {} providers; refusing to pass vacuously",
            fixtures.len(),
            providers.len()
        ));
    }
    let mut offenders = Vec::new();
    let rows = rows(&manifest);
    let declared: BTreeMap<_, _> = rows
        .iter()
        .filter(|r| r.fixture.is_some())
        .map(|r| (r.id.clone(), r.category))
        .collect();
    let unrecorded: BTreeSet<_> = rows
        .iter()
        .filter(|r| r.unrecorded)
        .map(|r| r.id.clone())
        .collect();
    for ((p, s), path) in &fixtures {
        let id = format!("{p}/{s}");
        if unrecorded.contains(&id) {
            offenders.push((
                path.clone(),
                "declared unrecorded but a fixture exists; move it to live".into(),
            ));
        } else if !declared.contains_key(&id) {
            offenders.push((path.clone(), format!("no provenance declared for {id}")));
        }
    }
    for row in &rows {
        if let Some(path) = &row.fixture
            && !root.join(path).is_file()
        {
            offenders.push((
                path.clone(),
                format!(
                    "declared {} but the fixture does not exist",
                    row.category.as_str()
                ),
            ));
        }
    }
    for p in &manifest.providers {
        for s in &p.scripted {
            let id = format!("{}/{}", p.provider, s.family);
            if (s.fixture.as_deref() != Some(s.family.as_str()) && declared.contains_key(&id))
                || unrecorded.contains(&id)
            {
                offenders.push((
                    manifest::MANIFEST_PATH.into(),
                    format!("{id} is both a scripted family and a fixture-backed scenario"),
                ));
            }
        }
    }
    check_sources(&manifest, &declared, &unrecorded, &mut offenders);
    check_discovery(root, &manifest, &mut offenders)?;
    check_scripted_modules(root, &manifest, &mut offenders);
    check_raw_entry_points(root, &mut offenders)?;
    offenders.sort();
    offenders.dedup();
    if !offenders.is_empty() {
        return Err(offenders
            .into_iter()
            .map(|(path, why)| format!("{path}: {why}"))
            .collect::<Vec<_>>()
            .join("\n"));
    }
    println!(
        "ok: {} cassette(s) across {} provider(s)",
        fixtures.len(),
        providers.len()
    );
    Ok(())
}

fn fixtures_on_disk(root: &Path) -> Result<BTreeMap<(String, String), String>, String> {
    let dir = root.join(FIXTURE_ROOT);
    let mut files = Vec::new();
    if dir.is_dir() {
        collect_files(&dir, "yaml", &mut files)?;
    }
    let mut out = BTreeMap::new();
    for file in files {
        let relative = file
            .strip_prefix(&dir)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        let (p, s) = relative
            .split_once('/')
            .ok_or_else(|| format!("{relative}: cassette must live under a provider"))?;
        out.insert(
            (p.to_owned(), s.trim_end_matches(".yaml").to_owned()),
            display(root, &file),
        );
    }
    Ok(out)
}

fn check_sources(
    manifest: &Manifest,
    declared: &BTreeMap<String, Category>,
    unrecorded: &BTreeSet<String>,
    offenders: &mut Vec<(String, String)>,
) {
    for p in &manifest.providers {
        for s in &p.derived {
            let id = format!("{}/{}", p.provider, s.scenario);
            for source in &s.sources {
                let why = if *source == id {
                    Some(format!("{id} is derived from itself"))
                } else if declared.get(source) != Some(&Category::Live) {
                    Some(format!(
                        "{id} is derived from {source}, which is not a recorded scenario with live provenance; derived chains and derivation cycles are forbidden"
                    ))
                } else {
                    None
                };
                if let Some(why) = why {
                    offenders.push((manifest::MANIFEST_PATH.into(), why));
                }
            }
        }
        for s in &p.scripted {
            for source in &s.sources {
                if !declared.contains_key(source) && !unrecorded.contains(source) {
                    offenders.push((
                        manifest::MANIFEST_PATH.into(),
                        format!(
                            "{}:{} scripts undeclared source {source}",
                            p.provider, s.family
                        ),
                    ));
                }
            }
        }
    }
}

fn check_discovery(
    root: &Path,
    manifest: &Manifest,
    offenders: &mut Vec<(String, String)>,
) -> Result<(), String> {
    for p in &manifest.providers {
        let discovered = discover(root, p)?;
        let declared: BTreeSet<_> = p.fixture_scenarios().into_iter().collect();
        for s in discovered.tests.keys() {
            if !declared.contains(s.as_str()) {
                offenders.push((
                    p.source_dir.clone(),
                    format!("a test records {}/{s}, which is not declared", p.provider),
                ));
            }
        }
        for s in declared {
            if !discovered.tests.contains_key(s) {
                offenders.push((
                    manifest::MANIFEST_PATH.into(),
                    format!("{}/{s} is declared but no test uses it", p.provider),
                ));
            }
        }
        for s in &p.unrecorded {
            if !discovered.ignored_tests.contains_key(&s.scenario) {
                offenders.push((
                    manifest::MANIFEST_PATH.into(),
                    format!("{}/{} has no ignored producer", p.provider, s.scenario),
                ));
            }
        }
    }
    Ok(())
}

fn check_scripted_modules(root: &Path, manifest: &Manifest, offenders: &mut Vec<(String, String)>) {
    for p in &manifest.providers {
        for s in &p.scripted {
            let relative = format!("{CASSETTE_CRATE}/{}", s.module);
            let Ok(source) = std::fs::read_to_string(root.join(&relative)) else {
                offenders.push((
                    relative,
                    format!(
                        "scripted family {} names a module that does not exist",
                        s.family
                    ),
                ));
                continue;
            };
            // Token membership excludes comments but is only a convention
            // check. The runtime ScriptedFamily enforces source permissions.
            let tokens = source
                .parse::<proc_macro2::TokenStream>()
                .map(token_words)
                .unwrap_or_default();
            for name in &s.cases {
                if !tokens.contains(name) {
                    offenders.push((
                        relative.clone(),
                        format!(
                            "does not construct the scripted family {:?} or declared case {name:?}",
                            s.family
                        ),
                    ));
                }
            }
        }
    }
}

fn token_words(tokens: proc_macro2::TokenStream) -> BTreeSet<String> {
    let mut words = BTreeSet::new();
    for token in tokens {
        match token {
            proc_macro2::TokenTree::Group(group) => words.extend(token_words(group.stream())),
            proc_macro2::TokenTree::Ident(ident) => {
                words.insert(ident.to_string());
            }
            proc_macro2::TokenTree::Literal(literal) => {
                if let Ok(s) = syn::parse_str::<syn::LitStr>(&literal.to_string()) {
                    words.insert(s.value());
                }
            }
            _ => {}
        }
    }
    words
}

fn check_raw_entry_points(
    root: &Path,
    offenders: &mut Vec<(String, String)>,
) -> Result<(), String> {
    let mut files = Vec::new();
    collect_files(root, "rs", &mut files)?;
    for file in files {
        let relative = display(root, &file).replace('\\', "/");
        if relative.starts_with("crates/rig-cassette/src/")
            || relative == "test-support/rig-test-support/src/cassettes.rs"
        {
            continue;
        }
        let source = std::fs::read_to_string(&file).map_err(|e| e.to_string())?;
        let parsed = syn::parse_file(&source).map_err(|e| format!("{relative}: {e}"))?;
        let mut visitor = EntryPointVisitor::default();
        visitor.visit_file(&parsed);
        for method in visitor.found {
            offenders.push((
                relative.clone(),
                format!(
                    "calls ProviderCassette::{method} directly; use the declared provider wrappers"
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
            && ["start", "start_via", "start_at"]
                .iter()
                .any(|name| method.ident == name)
        {
            self.found.insert(method.ident.to_string());
        }
        visit::visit_expr_path(self, node);
    }

    fn visit_use_rename(&mut self, node: &'ast syn::UseRename) {
        if node.ident == "ProviderCassette" {
            self.found.insert(format!("alias {}", node.rename));
        }
        visit::visit_use_rename(self, node);
    }

    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        if let syn::Type::Path(path) = node.ty.as_ref()
            && path
                .path
                .segments
                .last()
                .is_some_and(|s| s.ident == "ProviderCassette")
        {
            self.found.insert(format!("type alias {}", node.ident));
        }
        visit::visit_item_type(self, node);
    }
}
