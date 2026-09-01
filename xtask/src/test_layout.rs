//! `check-test-layout`: every test module is a sibling file, never an inline
//! block.
//!
//! The rule: a `mod` item gated by a test-only `#[cfg(...)]` (`cfg(test)`, or an
//! `all(...)` with `test` among its operands) must be a declaration,
//! `mod tests;`, whose body lives in the file rustc resolves it to. An inline
//! body, `mod tests { ... }`, is rejected. Inline bodies are what turned
//! `agent/runner.rs` into an 11,000-line file with 800 lines of code (#2433
//! moved 214 of them out); this task keeps them out.
//!
//! Files reachable only through a test-gated declaration are test code
//! themselves and may nest modules however they like. Test-onlyness follows
//! declarations transitively: anything a test-only file declares is test-only.
//!
//! Source is parsed with `syn`, so attributes inside strings or comments cannot
//! trip the check and a `cfg` predicate is inspected structurally rather than by
//! text. A file `syn` cannot parse is an error, not a pass.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use syn::{Item, ItemMod, Meta, Token, punctuated::Punctuated};

/// Run the check over every `crates/*/src` tree under `workspace`.
pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let files = source_files(&workspace.join("crates"))?;
    if files.len() < 100 {
        return Err(format!(
            "the source walk found only {} files under crates/; refusing to pass vacuously",
            files.len()
        ));
    }

    let mut parsed = Vec::with_capacity(files.len());
    for path in &files {
        let source = std::fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let file = syn::parse_file(&source)
            .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
        parsed.push((path.clone(), file));
    }

    let test_only = test_only_files(&parsed);

    let mut offenders = Vec::new();
    for (path, file) in &parsed {
        if test_only.contains(path) {
            continue;
        }
        collect_inline_test_modules(path, &file.items, &mut offenders);
    }

    if offenders.is_empty() {
        println!(
            "ok: no inline test modules in {} files ({} test-only files skipped)",
            parsed.len(),
            test_only.len()
        );
        return Ok(());
    }

    let mut message = String::from(
        "inline test modules found; move each body to its sibling file and declare it \
         with `mod <name>;`:\n",
    );
    for offender in &offenders {
        message.push_str("  ");
        message.push_str(offender);
        message.push('\n');
    }
    Err(message)
}

/// Every `.rs` file under `crates/*/src`, sorted for stable output.
fn source_files(crates: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(crates)
        .map_err(|error| format!("could not read {}: {error}", crates.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("could not read crate entry: {error}"))?;
        let src = entry.path().join("src");
        if src.is_dir() {
            walk(&src, &mut out)?;
        }
    }
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("could not read {}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("could not read entry in {}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The set of files reachable only through a test-gated `mod x;` declaration,
/// closed under "declared by a test-only file".
fn test_only_files(parsed: &[(PathBuf, syn::File)]) -> BTreeSet<PathBuf> {
    let mut test_only = BTreeSet::new();
    loop {
        let before = test_only.len();
        for (path, file) in parsed {
            let parent_is_test_only = test_only.contains(path);
            for item in &file.items {
                let Item::Mod(module) = item else { continue };
                if module.content.is_some() {
                    continue;
                }
                if parent_is_test_only || is_test_gated(module) {
                    for child in child_candidates(path, &module.ident.to_string()) {
                        if child.is_file() {
                            test_only.insert(child);
                        }
                    }
                }
            }
        }
        if test_only.len() == before {
            return test_only;
        }
    }
}

/// Where rustc looks for `mod name;` declared in `parent`: `<dir>/name.rs` or
/// `<dir>/name/mod.rs`, where `<dir>` is the parent's own directory for
/// `mod.rs`/`lib.rs`/`main.rs` and `<parent stem>/` otherwise. `#[path]`
/// overrides are not followed; such a file is simply scanned as shipped code.
fn child_candidates(parent: &Path, name: &str) -> Vec<PathBuf> {
    let file_name = parent.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let dir = if matches!(file_name, "mod.rs" | "lib.rs" | "main.rs") {
        parent.parent().map(Path::to_path_buf)
    } else {
        Some(parent.with_extension(""))
    };
    let Some(dir) = dir else { return Vec::new() };
    vec![
        dir.join(format!("{name}.rs")),
        dir.join(name).join("mod.rs"),
    ]
}

/// Record every test-gated module with an inline body, recursing into inline
/// non-test modules (an inline `mod foo { #[cfg(test)] mod tests { } }` is
/// still an inline test module).
fn collect_inline_test_modules(path: &Path, items: &[Item], offenders: &mut Vec<String>) {
    for item in items {
        let Item::Mod(module) = item else { continue };
        let Some((_, content)) = &module.content else {
            continue;
        };
        if is_test_gated(module) {
            let line = module.mod_token.span.start().line;
            offenders.push(format!(
                "{}:{line}: `mod {}` has an inline body",
                path.display(),
                module.ident
            ));
        } else {
            collect_inline_test_modules(path, content, offenders);
        }
    }
}

/// Whether any `#[cfg(...)]` on the module can only hold under `cfg(test)`.
fn is_test_gated(module: &ItemMod) -> bool {
    module.attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .parse_args::<Meta>()
                .is_ok_and(|predicate| is_test_only_predicate(&predicate))
    })
}

/// `test` is test-only. `all(..)` is test-only when any operand is.
/// `any(..)` is test-only only when every operand is. `not(..)` and every
/// other predicate (`feature = ".."`, `target_*`) can hold in a shipped build.
fn is_test_only_predicate(meta: &Meta) -> bool {
    match meta {
        Meta::Path(path) => path.is_ident("test"),
        Meta::List(list) => {
            let operands = list
                .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .unwrap_or_default();
            if list.path.is_ident("all") {
                operands.iter().any(is_test_only_predicate)
            } else if list.path.is_ident("any") {
                !operands.is_empty() && operands.iter().all(is_test_only_predicate)
            } else {
                false
            }
        }
        Meta::NameValue(_) => false,
    }
}

#[cfg(test)]
mod tests;
