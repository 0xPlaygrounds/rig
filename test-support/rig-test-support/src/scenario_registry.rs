//! Syntactic inventory of explicit cassette calls and matrix rows. This is
//! intentionally not Rust name resolution: executable test names are checked
//! against libtest before recording. Runtime authorization uses the same scan.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use proc_macro2::TokenStream;
use syn::parse::{ParseStream, Parser};
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, ExprCall, ExprLit, Item, Lit, LitStr, Token, parenthesized};

use super::manifest::{Manifest, ProviderScenarios};

pub(crate) fn load(root: &Path) -> anyhow::Result<Manifest> {
    let mut manifest = Manifest::load(root).map_err(anyhow::Error::msg)?;
    for provider in &mut manifest.providers {
        let mut live = BTreeSet::new();
        for module in &provider.live_modules {
            let file = root
                .join("crates/rig-cassette")
                .join(&provider.source_dir)
                .join(module);
            let discovered = discover_file(root, provider, &file)?;
            if discovered.tests.is_empty() {
                anyhow::bail!("{}: live module declares no scenarios", file.display());
            }
            live.extend(discovered.tests.into_keys());
        }
        provider.live.extend(live);
        provider.validate().map_err(anyhow::Error::msg)?;
    }
    Ok(manifest)
}

#[derive(Debug, Default)]
pub(crate) struct Discovery {
    pub(crate) tests: BTreeMap<String, BTreeSet<String>>,
    pub(crate) ignored_tests: BTreeMap<String, BTreeSet<String>>,
}

impl Discovery {
    fn insert(&mut self, scenario: String, test: String, ignored: bool) {
        let table = if ignored {
            &mut self.ignored_tests
        } else {
            &mut self.tests
        };
        table.entry(scenario).or_default().insert(test);
    }

    fn extend(&mut self, other: Self) {
        for (ignored, table) in [(false, other.tests), (true, other.ignored_tests)] {
            for (scenario, tests) in table {
                for test in tests {
                    self.insert(scenario.clone(), test, ignored);
                }
            }
        }
    }
}

pub(crate) fn discover(root: &Path, provider: &ProviderScenarios) -> Result<Discovery, String> {
    let mut files = Vec::new();
    collect_files(
        &root.join("crates/rig-cassette").join(&provider.source_dir),
        "rs",
        &mut files,
    )?;
    let mut out = Discovery::default();
    for file in files {
        out.extend(discover_file(root, provider, &file).map_err(|error| error.to_string())?);
    }
    Ok(out)
}

pub(crate) fn discover_file(
    root: &Path,
    provider: &ProviderScenarios,
    file: &Path,
) -> anyhow::Result<Discovery> {
    use anyhow::Context;
    let source = std::fs::read_to_string(file).with_context(|| file.display().to_string())?;
    let parsed = syn::parse_file(&source).with_context(|| file.display().to_string())?;
    let wrappers = provider.wrappers.iter().map(String::as_str).collect();
    let mut scan = FileScan::default();
    scan.items(
        &parsed.items,
        &module_path(root, file).map_err(anyhow::Error::msg)?,
        &wrappers,
    )
    .map_err(anyhow::Error::msg)?;
    let mut out = Discovery::default();
    for (scenario, test, ignored) in &scan.matrix {
        out.insert(scenario.clone(), test.clone(), *ignored);
    }
    for (name, ignored) in &scan.entry_points {
        for scenario in scan.reachable(name, &mut BTreeSet::new())? {
            out.insert(scenario, name.clone(), *ignored);
        }
    }
    Ok(out)
}

#[derive(Default)]
struct FileScan {
    functions: BTreeMap<String, FnScan>,
    entry_points: Vec<(String, bool)>,
    matrix: Vec<(String, String, bool)>,
}

#[derive(Default)]
struct FnScan {
    scenarios: BTreeSet<String>,
    calls: BTreeSet<String>,
    dynamic_scenario: bool,
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
                Item::Fn(function) => {
                    let name = format!("{prefix}::{}", function.sig.ident);
                    let mut visitor = FnVisitor {
                        wrappers,
                        prefix,
                        scan: FnScan::default(),
                    };
                    visitor.visit_block(&function.block);
                    if function.attrs.iter().any(|attr| {
                        attr.path()
                            .segments
                            .last()
                            .is_some_and(|s| s.ident == "test")
                    }) {
                        self.entry_points
                            .push((name.clone(), has_attribute(&function.attrs, "ignore")));
                    }
                    self.functions.insert(name, visitor.scan);
                }
                Item::Mod(module) => {
                    if let Some((_, inner)) = &module.content {
                        self.items(inner, &format!("{prefix}::{}", module.ident), wrappers)?;
                    }
                }
                Item::Macro(item)
                    if item.mac.path.segments.last().is_some_and(|s| {
                        ["golden_matrix", "resume_matrix", "case_matrix"]
                            .iter()
                            .any(|m| s.ident == m)
                    }) =>
                {
                    let matrix = parse_matrix(item.mac.tokens.clone())?;
                    if matrix
                        .wrapper
                        .as_deref()
                        .is_some_and(|w| wrappers.contains(w))
                    {
                        for row in matrix.rows {
                            self.matrix.push((
                                row.scenario,
                                format!("{prefix}::{}", row.name),
                                row.ignored,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn reachable(
        &self,
        name: &str,
        seen: &mut BTreeSet<String>,
    ) -> anyhow::Result<BTreeSet<String>> {
        let mut out = BTreeSet::new();
        if seen.insert(name.to_owned())
            && let Some(scan) = self.functions.get(name)
        {
            anyhow::ensure!(
                !scan.dynamic_scenario,
                "{name}: cassette wrapper requires a literal scenario; dynamic recording inventories are unsupported"
            );
            out.extend(scan.scenarios.iter().cloned());
            for callee in &scan.calls {
                out.extend(self.reachable(callee, seen)?);
            }
        }
        Ok(out)
    }
}

struct FnVisitor<'a> {
    wrappers: &'a BTreeSet<&'a str>,
    prefix: &'a str,
    scan: FnScan,
}

impl<'ast> Visit<'ast> for FnVisitor<'_> {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref()
            && let Some(last) = path.path.segments.last()
        {
            if self.wrappers.contains(last.ident.to_string().as_str()) {
                if let Some(scenario) = node.args.first().and_then(scenario_literal) {
                    self.scan.scenarios.insert(scenario);
                } else {
                    self.scan.dynamic_scenario = true;
                }
            } else {
                let path = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                let name = if let Some(path) = path.strip_prefix("super::") {
                    format!(
                        "{}::{path}",
                        self.prefix
                            .rsplit_once("::")
                            .map_or(self.prefix, |(parent, _)| parent)
                    )
                } else {
                    format!(
                        "{}::{}",
                        self.prefix,
                        path.strip_prefix("self::").unwrap_or(&path)
                    )
                };
                self.scan.calls.insert(name);
            }
        }
        visit::visit_expr_call(self, node);
    }
}

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

fn has_attribute(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident(name))
}

pub(crate) struct Matrix {
    pub(crate) wrapper: Option<String>,
    pub(crate) rows: Vec<MatrixRow>,
}

pub(crate) struct MatrixRow {
    pub(crate) name: String,
    pub(crate) scenario: String,
    pub(crate) ignored: bool,
}

pub(crate) fn parse_matrix(tokens: TokenStream) -> Result<Matrix, String> {
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

pub(crate) fn module_path(root: &Path, file: &Path) -> Result<String, String> {
    let tests = root.join("crates/rig-cassette/tests");
    // Several provider roots mount cassette/*.rs directly, without a cassette
    // module. Respect those declarations rather than inventing an extra segment.
    for directory in file
        .ancestors()
        .skip(1)
        .take_while(|dir| dir.starts_with(&tests))
    {
        let owner = directory.join("mod.rs");
        if owner == file {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(&owner) else {
            continue;
        };
        let parsed = syn::parse_file(&source).map_err(|e| e.to_string())?;
        for item in parsed.items {
            if let Item::Mod(module) = item
                && let Some(attr) = module.attrs.iter().find(|a| a.path().is_ident("path"))
                && let syn::Meta::NameValue(meta) = &attr.meta
                && let Expr::Lit(ExprLit {
                    lit: Lit::Str(path),
                    ..
                }) = &meta.value
                && directory.join(path.value()) == file
            {
                return Ok(format!("{}::{}", module_path(root, &owner)?, module.ident));
            }
        }
    }
    let relative = file
        .strip_prefix(&tests)
        .map_err(|_| format!("{} is outside {}", file.display(), tests.display()))?;
    let mut segments: Vec<String> = relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
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

pub(crate) fn collect_files(
    dir: &Path,
    extension: &str,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            if !path.file_name().is_some_and(|n| {
                ["target", ".git", "node_modules", ".jj", "pkg"]
                    .iter()
                    .any(|skip| n == *skip)
            }) {
                collect_files(&path, extension, out)?;
            }
        } else if path.extension().is_some_and(|ext| ext == extension) {
            out.push(path);
        }
    }
    Ok(())
}
