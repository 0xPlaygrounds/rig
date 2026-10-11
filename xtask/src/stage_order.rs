//! `check-stage-order`: every corpus crate that adds systems to the agent
//! loop's stages checks, in a test of its own, that they are ordered
//! against the default plugins' and against each other: a test file of
//! the crate calls `rig_defaults_test_support::assert_stages_ordered`.
//!
//! A crate adds stage systems when a file it ships calls
//! `.add_systems(AgentPass, …)`. Test-only files are told apart as
//! `check-test-layout` tells them; source is parsed with `syn`, so a
//! comment or a string naming either call does not count.

use std::path::Path;

use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprMethodCall};

/// Run the check over every `corpus/*/src` tree under `workspace`.
pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let corpus = workspace.join("corpus");
    let entries = std::fs::read_dir(&corpus)
        .map_err(|error| format!("could not read {}: {error}", corpus.display()))?;
    let mut crates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("could not read a corpus entry: {error}"))?;
        let src = entry.path().join("src");
        if src.is_dir() {
            crates.push((entry.file_name().to_string_lossy().into_owned(), src));
        }
    }
    if crates.len() < 60 {
        return Err(format!(
            "found only {} corpus crates; refusing to pass vacuously",
            crates.len()
        ));
    }
    crates.sort();
    let mut unchecked_crates = Vec::new();
    for (name, src) in &crates {
        let mut parsed = Vec::new();
        for path in crate::support::files_under(src, Some("rs"))? {
            let source = std::fs::read_to_string(&path)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            let file = syn::parse_file(&source)
                .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
            parsed.push((path, file));
        }
        let test_only = crate::test_layout::test_only_files(&parsed);
        if unchecked(
            parsed
                .iter()
                .map(|(path, file)| (file, test_only.contains(path))),
        ) {
            unchecked_crates.push(format!("  corpus/{name}"));
        }
    }
    if unchecked_crates.is_empty() {
        println!(
            "ok: every corpus crate with stage systems checks their order ({} crates)",
            crates.len()
        );
        return Ok(());
    }
    Err(format!(
        "these corpus crates add agent-loop systems, and no test of theirs calls \
         `rig_defaults_test_support::assert_stages_ordered`:\n{}",
        unchecked_crates.join("\n")
    ))
}

/// Whether a crate of these files, each with whether it is test-only, adds
/// stage systems in a file it ships and calls `assert_stages_ordered` in
/// none of its tests.
fn unchecked<'a>(files: impl IntoIterator<Item = (&'a syn::File, bool)>) -> bool {
    let mut shipped = Calls::default();
    let mut tests = Calls::default();
    for (file, test_only) in files {
        let calls = if test_only { &mut tests } else { &mut shipped };
        calls.visit_file(file);
    }
    shipped.add_stage_systems && !tests.check_stage_order
}

/// The two calls the check looks for, seen in some files.
#[derive(Default)]
struct Calls {
    /// `.add_systems(AgentPass, …)`.
    add_stage_systems: bool,
    /// `assert_stages_ordered(…)`.
    check_stage_order: bool,
}

impl<'ast> Visit<'ast> for Calls {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if call.method == "add_systems" && call.args.first().is_some_and(names("AgentPass")) {
            self.add_stage_systems = true;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if names("assert_stages_ordered")(&call.func) {
            self.check_stage_order = true;
        }
        visit::visit_expr_call(self, call);
    }
}

/// Whether an expression is a path ending in `name`.
fn names(name: &str) -> impl Fn(&Expr) -> bool + '_ {
    move |expr| {
        matches!(expr, Expr::Path(path)
            if path.path.segments.last().is_some_and(|segment| segment.ident == name))
    }
}

#[cfg(test)]
mod tests;
