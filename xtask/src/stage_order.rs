//! `check-stage-order`: every corpus plugin that adds systems to the agent
//! loop's stages checks, in a test of its own, that they are ordered
//! against the default plugins' and against each other: a test file of
//! the plugin calls `rig_defaults_test_support::assert_stages_ordered`, or
//! `assert_stages_ordered_replacing` for a plugin that replaces a default.
//!
//! A corpus crate holds one plugin per module: `src/<plugin>.rs` and the
//! files under `src/<plugin>/`. A plugin adds stage systems when a file it
//! ships calls
//! `.add_systems(AgentPass, …)`. Test-only files are told apart as
//! `check-test-layout` tells them; source is parsed with `syn`, so a
//! comment or a string naming either call does not count.

use std::path::{Path, PathBuf};

use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprMethodCall};

/// Run the check over every plugin module of every `corpus/*/src` tree
/// under `workspace`.
pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let plugins = plugins(&workspace.join("corpus"))?;
    if plugins.len() < 60 {
        return Err(format!(
            "found only {} corpus plugins; refusing to pass vacuously",
            plugins.len()
        ));
    }
    let mut unchecked_plugins = Vec::new();
    for (name, files) in &plugins {
        let mut parsed = Vec::new();
        for path in files {
            let source = std::fs::read_to_string(path)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            let file = syn::parse_file(&source)
                .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
            parsed.push((path.clone(), file));
        }
        let test_only = crate::test_layout::test_only_files(&parsed);
        if unchecked(
            parsed
                .iter()
                .map(|(path, file)| (file, test_only.contains(path))),
        ) {
            unchecked_plugins.push(format!("  {name}"));
        }
    }
    if unchecked_plugins.is_empty() {
        println!(
            "ok: every corpus plugin with stage systems checks their order ({} plugins)",
            plugins.len()
        );
        return Ok(());
    }
    Err(format!(
        "these corpus plugins add agent-loop systems, and no test of theirs calls \
         `rig_defaults_test_support::assert_stages_ordered`:\n{}",
        unchecked_plugins.join("\n")
    ))
}

/// Every plugin module of the corpus crates under `corpus`, sorted, named
/// `corpus/<crate>/src/<plugin>`, with its files: `src/<plugin>.rs` and
/// every `.rs` file under `src/<plugin>/`.
fn plugins(corpus: &Path) -> Result<Vec<(String, Vec<PathBuf>)>, String> {
    let read = |dir: &Path| {
        std::fs::read_dir(dir).map_err(|error| format!("could not read {}: {error}", dir.display()))
    };
    let mut plugins = Vec::new();
    for entry in read(corpus)? {
        let entry = entry.map_err(|error| format!("could not read a corpus entry: {error}"))?;
        let src = entry.path().join("src");
        if !src.is_dir() {
            continue;
        }
        for module in read(&src)? {
            let module = module.map_err(|error| format!("could not read a module: {error}"))?;
            let path = module.path();
            let Some(stem) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .filter(|_| path.extension().is_some_and(|ext| ext == "rs"))
            else {
                continue;
            };
            if stem == "lib" {
                continue;
            }
            let mut files = vec![path.clone()];
            let folder = src.join(stem);
            if folder.is_dir() {
                files.extend(crate::support::files_under(&folder, Some("rs"))?);
            }
            let name = format!("corpus/{}/src/{stem}", entry.file_name().to_string_lossy());
            plugins.push((name, files));
        }
    }
    plugins.sort();
    Ok(plugins)
}

/// Whether a plugin of these files, each with whether it is test-only, adds
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
    /// `assert_stages_ordered(…)` or `assert_stages_ordered_replacing(…)`.
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
        if names("assert_stages_ordered")(&call.func)
            || names("assert_stages_ordered_replacing")(&call.func)
        {
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
