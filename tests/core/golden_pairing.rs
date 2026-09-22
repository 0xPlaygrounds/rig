//! Each runtime's golden logs have exactly one producer in that runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::visit::{self, Visit};

use rig_test_support::matrix_registry;

#[path = "golden_pairing/tests.rs"]
mod tests;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures(world: bool, suffix: &str) -> Vec<String> {
    let mut directory = root().join("crates/rig-cassette/fixtures/effects");
    if world {
        directory.push("world");
    }
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
    world: Producers,
    ignored_world: Vec<String>,
    failures: Vec<String>,
    file: String,
    native: bool,
    ignored: bool,
    test: Option<String>,
    world_calls: usize,
    produces_log: bool,
    matcher_only: bool,
    log_functions: BTreeSet<String>,
}

impl Sites {
    fn check_helper_name(&mut self, name: &str) {
        if (self.native && name == "golden_effects")
            || (!self.native && name == "world_golden_effects")
        {
            self.fail("golden helper belongs to the other runtime");
        }
    }

    fn fail(&mut self, message: impl std::fmt::Display) {
        self.failures.push(format!("{}: {message}", self.file));
    }

    fn register(&mut self, name: &str, world: bool, ignored: bool) {
        if world != self.native {
            self.fail("golden helper belongs to the other runtime");
        }
        if ignored {
            if world {
                self.ignored_world.push(name.to_owned());
            }
            return;
        }
        let location = self.test.as_ref().map_or_else(
            || self.file.clone(),
            |test| format!("{}::{test}", self.file),
        );
        let producers = if world {
            &mut self.world
        } else {
            &mut self.agent
        };
        producers.entry(name.to_owned()).or_default().push(location);
        if world {
            self.world_calls += 1;
        }
    }

    fn native_rows(&mut self, rows: Vec<matrix_registry::Row>) {
        for row in rows {
            self.register(&row.golden.value(), true, row.ignored);
        }
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
        self.world_calls = 0;
        self.produces_log = false;
        self.matcher_only = false;
        visit::visit_item_fn(self, node);
        if self.native && is_test && !self.ignored && !self.matcher_only && self.world_calls != 1 {
            self.fail(format!(
                "{} produces a log but has {} world goldens",
                node.sig.ident, self.world_calls
            ));
        }
        self.test = None;
        self.ignored = false;
    }

    fn visit_use_name(&mut self, node: &'ast syn::UseName) {
        self.check_helper_name(&node.ident.to_string());
        visit::visit_use_name(self, node);
    }

    fn visit_use_rename(&mut self, node: &'ast syn::UseRename) {
        self.check_helper_name(&node.ident.to_string());
        visit::visit_use_rename(self, node);
    }

    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if let Some(segment) = node.path.segments.last() {
            self.check_helper_name(&segment.ident.to_string());
        }
        visit::visit_expr_path(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if node.method == "effect_log" {
            self.produces_log = true;
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        let Some(name) = node.path.segments.last() else {
            return;
        };
        match name.ident.to_string().as_str() {
            "golden_matrix" => {
                match syn::parse2::<matrix_registry::GoldenMatrix>(node.tokens.clone()) {
                    Ok(matrix) => {
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
                            self.register(&row.golden.value(), false, row.ignored);
                        }
                    }
                    Err(error) => self.fail(error),
                }
            }
            "native_matrix" => {
                match syn::parse2::<matrix_registry::NativeMatrix>(node.tokens.clone()) {
                    Ok(matrix) => self.native_rows(matrix.rows),
                    Err(error) => self.fail(error),
                }
            }
            "resume_matrix" => {
                match syn::parse2::<matrix_registry::ResumeMatrix>(node.tokens.clone()) {
                    Ok(matrix) => self.native_rows(matrix.rows),
                    Err(error) => self.fail(error),
                }
            }
            "case_matrix" => {
                match syn::parse2::<matrix_registry::CaseMatrix>(node.tokens.clone()) {
                    Ok(matrix) => {
                        if self.native
                            && matches!(
                                matrix.family.to_string().as_str(),
                                "wire_matrix_case" | "ecs_faults_case" | "ecs_termination_case"
                            )
                            && matrix.world_goldens.len() != matrix.registrations
                        {
                            self.fail("every native wire case requires a literal world golden");
                        }
                        for (name, ignored) in matrix.world_goldens {
                            self.register(&name.value(), true, ignored);
                        }
                    }
                    Err(error) => self.fail(error),
                }
            }
            _ => {}
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
            let function = parts.last().map(String::as_str);
            if matches!(
                function,
                Some("assert_large_request_rejected" | "assert_stream_request_rejected")
            ) {
                self.matcher_only = true;
            }
            if matches!(function, Some("run_world" | "run_scripted"))
                || function.is_some_and(|name| self.log_functions.contains(name))
            {
                self.produces_log = true;
            }
            if matches!(function, Some("golden_effects" | "world_golden_effects")) {
                let world = function == Some("world_golden_effects");
                if parts.iter().rev().nth(1).map(String::as_str) != Some("goldens") {
                    self.fail("qualify the golden helper with goldens");
                }
                if let Some(syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(name),
                    ..
                })) = call.args.first()
                {
                    self.register(&name.value(), world, self.ignored);
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
                sites.native = path
                    .file_stem()
                    .is_some_and(|stem| stem.to_string_lossy().starts_with("ecs_"));
                sites.log_functions = syntax
                    .items
                    .iter()
                    .filter_map(|item| {
                        let syn::Item::Fn(function) = item else {
                            return None;
                        };
                        let syn::ReturnType::Type(_, output) = &function.sig.output else {
                            return None;
                        };
                        let syn::Type::Path(path) = output.as_ref() else {
                            return None;
                        };
                        path.path
                            .segments
                            .last()
                            .is_some_and(|segment| segment.ident == "EffectLog")
                            .then(|| function.sig.ident.to_string())
                    })
                    .collect();
                sites.visit_file(&syntax);
            }
        }
    }
    sites
}

fn pairing_problems(fixtures: &[String], producers: &Producers, corpus: &str) -> Vec<String> {
    let mut problems = Vec::new();
    for name in fixtures {
        match producers.get(name).map(Vec::as_slice) {
            Some([_]) => {}
            Some(locations) => problems.push(format!(
                "{corpus} golden `{name}` has {} producers: {locations:?}",
                locations.len()
            )),
            None => problems.push(format!("{corpus} golden `{name}` has no producer")),
        }
    }
    for (name, locations) in producers {
        if !fixtures.contains(name) {
            problems.push(format!(
                "{locations:?} names missing {corpus} golden `{name}`"
            ));
        }
    }
    problems
}

#[test]
fn every_golden_has_exactly_one_producer_and_every_producer_a_golden() {
    let agent = fixtures(false, ".effects.json");
    let world = fixtures(true, ".effects.json");
    let programs = fixtures(true, ".programs.json");
    assert_eq!(
        world, programs,
        "every world golden has exactly one configuration sidecar"
    );
    assert!(
        !agent.is_empty() && !world.is_empty(),
        "both corpora must exist"
    );
    let sites = sites();
    let mut problems = sites.failures.clone();
    problems.extend(pairing_problems(&agent, &sites.agent, "agent"));
    problems.extend(pairing_problems(&world, &sites.world, "world"));
    for ignored in sites.ignored_world {
        if world.contains(&ignored) {
            problems.push(format!(
                "ignored native row names committed world golden `{ignored}`"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "goldens and producers are not paired:\n{}",
        problems.join("\n")
    );
}
