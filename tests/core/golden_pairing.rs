//! Original golden logs have exactly one producer. Native parity checks are
//! consumers of that original plus compact fixed scope/program expectations.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
mod tests;
use std::path::Path;
use syn::visit::{self, Visit};

use rig_test_support::matrix_registry;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures() -> Vec<String> {
    let mut names: Vec<_> = std::fs::read_dir(root().join("crates/rig-verify/fixtures"))
        .expect("the corpus directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter_map(|name| name.strip_suffix(".effects.json").map(str::to_owned))
        .collect();
    names.sort();
    names
}

#[derive(Default)]
struct Sites {
    producers: BTreeMap<String, Vec<String>>,
    native: BTreeMap<String, Vec<String>>,
    references: BTreeMap<String, Vec<String>>,
    failures: Vec<String>,
    file: String,
}

impl<'ast> Visit<'ast> for Sites {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if !node.attrs.iter().any(|attr| attr.path().is_ident("ignore")) {
            visit::visit_item_fn(self, node);
        }
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if node
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "golden_matrix")
        {
            match syn::parse2::<matrix_registry::GoldenMatrix>(node.tokens.clone()) {
                Ok(matrix) => {
                    if matrix
                        .oracle
                        .segments
                        .last()
                        .is_none_or(|s| s.ident != "golden_effects")
                    {
                        self.failures
                            .push(format!("{}: unknown matrix oracle", self.file));
                        return;
                    }
                    let helper = matrix
                        .oracle
                        .segments
                        .iter()
                        .rev()
                        .nth(1)
                        .map(|s| s.ident.to_string());
                    let destination = match helper.as_deref() {
                        Some("goldens") => &mut self.producers,
                        Some("ecs_goldens") => &mut self.native,
                        _ => {
                            self.failures
                                .push(format!("{}: unknown matrix oracle", self.file));
                            return;
                        }
                    };
                    for row in matrix.rows.into_iter().filter(|row| !row.ignored) {
                        destination
                            .entry(row.golden.value())
                            .or_default()
                            .push(self.file.clone());
                    }
                }
                Err(error) => self.failures.push(format!("{}: {error}", self.file)),
            }
        }
        visit::visit_macro(self, node);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            let parts: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            let helper = parts.iter().rev().nth(1).map(String::as_str);
            let function = parts.last().map(String::as_str);
            let destination = match (helper, function) {
                (Some("goldens"), Some("golden_effects")) => Some(&mut self.producers),
                (Some("ecs_goldens"), Some("golden_effects")) => Some(&mut self.native),
                (Some("ecs_goldens"), Some("compare_to_original")) => Some(&mut self.references),
                (_, Some("golden_effects" | "compare_to_original")) => {
                    self.failures.push(format!(
                        "{}: qualify the golden helper with goldens or ecs_goldens",
                        self.file
                    ));
                    None
                }
                _ => None,
            };
            if let Some(destination) = destination {
                if let Some(syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(name),
                    ..
                })) = call.args.first()
                {
                    destination
                        .entry(name.value())
                        .or_default()
                        .push(self.file.clone());
                } else {
                    self.failures.push(format!(
                        "{}: golden name must be a string literal",
                        self.file
                    ));
                }
            }
        }
        visit::visit_expr_call(self, call);
    }
}

fn sites() -> Sites {
    let mut sites = Sites::default();
    let mut pending = vec![root().join("tests/providers"), root().join("tests/core")];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("a tests directory") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).expect("source");
                let syntax = syn::parse_file(&source).expect("valid Rust source");
                sites.file = path
                    .strip_prefix(root())
                    .expect("under root")
                    .display()
                    .to_string();
                sites.visit_file(&syntax);
            }
        }
    }
    sites
}

fn pairing_problems(
    fixtures: &[String],
    sites: &Sites,
    identities: &BTreeSet<String>,
) -> Vec<String> {
    let mut problems = sites.failures.clone();
    for name in fixtures {
        match sites.producers.get(name).map(Vec::as_slice) {
            Some([_]) => {}
            Some(locations) => problems.push(format!(
                "golden `{name}` has {} producers: {locations:?}",
                locations.len()
            )),
            None => problems.push(format!("golden `{name}` has no producer")),
        }
    }
    for (name, locations) in sites
        .producers
        .iter()
        .chain(&sites.native)
        .chain(&sites.references)
    {
        if !fixtures.contains(name) {
            problems.push(format!("{locations:?} names missing golden `{name}`"));
        }
    }
    let native: BTreeSet<_> = sites.native.keys().cloned().collect();
    for missing in native.difference(identities) {
        problems.push(format!("native `{missing}` has no identity expectation"));
    }
    for orphan in identities.difference(&native) {
        problems.push(format!("native identity `{orphan}` has no consumer"));
    }
    problems
}

#[test]
fn every_golden_has_exactly_one_producer_and_every_producer_a_golden() {
    let fixtures = fixtures();
    assert!(!fixtures.is_empty(), "the corpus is not empty");
    let identities: BTreeMap<String, serde_json::Value> = serde_json::from_str(include_str!(
        "../../test-support/rig-test-support/src/ecs_goldens/identities.json"
    ))
    .expect("native identity expectations");
    let problems = pairing_problems(&fixtures, &sites(), &identities.into_keys().collect());
    assert!(
        problems.is_empty(),
        "goldens and producers are not paired:\n{}",
        problems.join("\n")
    );
}
