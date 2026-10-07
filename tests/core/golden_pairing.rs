//! Every golden log has exactly one producer, and a producer whose golden
//! the cassette prune replaced checks in-test.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::visit::{self, Visit};

use rig_test_support::matrix_registry::Matrix;

#[path = "golden_pairing/tests.rs"]
mod tests;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures(suffix: &str) -> Vec<String> {
    let directory = root().join("crates/rig-cassette/fixtures/effects");
    let mut names: Vec<_> = std::fs::read_dir(directory)
        .expect("the corpus directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter_map(|name| name.strip_suffix(suffix).map(str::to_owned))
        .collect();
    names.sort();
    names
}

type Producers = BTreeMap<String, Vec<String>>;

#[derive(Default)]
struct Sites {
    agent: Producers,
    failures: Vec<String>,
    file: String,
    ignored: bool,
    test: Option<String>,
}

impl Sites {
    fn fail(&mut self, message: impl std::fmt::Display) {
        self.failures.push(format!("{}: {message}", self.file));
    }

    fn register(&mut self, name: &str, ignored: bool) {
        if ignored {
            return;
        }
        let location = self.test.as_ref().map_or_else(
            || self.file.clone(),
            |test| format!("{}::{test}", self.file),
        );
        self.agent
            .entry(name.to_owned())
            .or_default()
            .push(location);
    }
}

impl<'ast> Visit<'ast> for Sites {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let is_test = node.attrs.iter().any(|attr| {
            attr.path()
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "test")
        });
        self.ignored = node.attrs.iter().any(|attr| attr.path().is_ident("ignore"));
        self.test = is_test.then(|| node.sig.ident.to_string());
        visit::visit_item_fn(self, node);
        self.test = None;
        self.ignored = false;
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        match Matrix::of(node) {
            Some(Ok(Matrix::Golden(matrix))) => {
                let parts: Vec<_> = matrix
                    .oracle
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                if parts != ["crate", "goldens", "golden_effects"] {
                    self.fail("unknown agent matrix oracle");
                }
                for row in matrix.rows {
                    self.register(&row.golden.value(), row.ignored);
                }
            }
            Some(Ok(Matrix::Case(_))) | None => {}
            Some(Err(error)) => self.fail(error),
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
            if parts.last().map(String::as_str) == Some("golden_effects") {
                if parts.iter().rev().nth(1).map(String::as_str) != Some("goldens") {
                    self.fail("qualify the golden helper with goldens");
                }
                if let Some(syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(name),
                    ..
                })) = call.args.first()
                {
                    self.register(&name.value(), self.ignored);
                } else {
                    self.fail("golden name must be a string literal");
                }
            }
        }
        visit::visit_expr_call(self, call);
    }
}

fn sites() -> Sites {
    let mut sites = Sites::default();
    let mut pending = vec![
        root().join("crates/rig-cassette/tests/providers"),
        root().join("tests/providers"),
        root().join("tests/core"),
    ];
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

/// The goldens the cassette prune replaced by an in-test check, by label:
/// their producers name a golden no file holds.
fn checked_in_test() -> BTreeSet<String> {
    let text = std::fs::read_to_string(root().join("crates/rig-cassette/coverage/pruned.tsv"))
        .unwrap_or_default();
    text.lines()
        .filter_map(|line| {
            let mut columns = line.split('\t');
            match (columns.next(), columns.next(), columns.next()) {
                (Some("golden"), Some(label), Some(covered)) if covered.starts_with("in-test ") => {
                    Some(label.to_owned())
                }
                _ => None,
            }
        })
        .collect()
}

fn pairing_problems(
    fixtures: &[String],
    producers: &Producers,
    in_test: &BTreeSet<String>,
) -> Vec<String> {
    let mut problems = Vec::new();
    for name in fixtures {
        match producers.get(name).map(Vec::as_slice) {
            Some([_]) => {}
            Some(locations) => problems.push(format!(
                "golden `{name}` has {} producers: {locations:?}",
                locations.len()
            )),
            None => problems.push(format!("golden `{name}` has no producer")),
        }
    }
    for (name, locations) in producers {
        if in_test.contains(name) {
            if locations.len() != 1 {
                problems.push(format!(
                    "golden `{name}` checked in-test has {} producers: {locations:?}",
                    locations.len()
                ));
            }
            if fixtures.contains(name) {
                problems.push(format!(
                    "golden `{name}` is committed but listed as checked in-test"
                ));
            }
            continue;
        }
        if !fixtures.contains(name) {
            problems.push(format!("{locations:?} names missing golden `{name}`"));
        }
    }
    problems
}

#[test]
fn every_golden_has_exactly_one_producer_and_every_producer_a_golden() {
    let agent = fixtures(".effects.json");
    assert!(!agent.is_empty(), "the corpus must exist");
    let sites = sites();
    let mut problems = sites.failures.clone();
    problems.extend(pairing_problems(&agent, &sites.agent, &checked_in_test()));
    assert!(
        problems.is_empty(),
        "goldens and producers are not paired:\n{}",
        problems.join("\n")
    );
}
