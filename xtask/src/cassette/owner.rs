//! Resolve a cassette fixture to the test that records it.
//!
//! A fixture's owner is the test whose body passes the scenario literal to a
//! cassette wrapper (`with_*cassette*(`), directly or through a
//! `CassetteSpec::new(…)` chain, or the matrix row (`name: ("scenario", …)`)
//! that names it. A nested helper function (a tool's `call`) is never the
//! owner: the nearest `#[test]`/`#[tokio::test]` function is. A test that
//! guards itself with `skip_when_recording` is hand-derived and cannot
//! record, and a file that only reads the literal back is a consumer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprLit, ItemFn, ItemMod, Lit};

/// What recording a fixture takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Owner {
    /// The test that records it: its nextest name within the provider binary.
    Test(String),
    /// The test only replays a hand-derived fixture.
    HandDerived(String),
    /// Something names the scenario, but no wrapper call records it.
    ConsumerOnly,
    /// Nothing under the provider's tests names it.
    Unknown,
}

/// Every producer of `provider`'s fixture `scenario` (`dir/name`, no
/// `.yaml`) under `<root>/crates/rig-cassette/tests/providers/<provider>`, in
/// file order; a shared fixture has several. A single
/// [`Owner::ConsumerOnly`] or [`Owner::Unknown`] when there is none.
pub(crate) fn owners(root: &Path, provider: &str, scenario: &str) -> Result<Vec<Owner>, String> {
    let base = root.join("crates/rig-cassette/tests/providers");
    let dir = base.join(provider);
    let mut files = Vec::new();
    rust_files(&dir, &mut files)?;
    files.sort();
    let modules = module_map(&dir.join("mod.rs"), provider);
    let mut named = false;
    let mut found = Vec::new();
    for file in files {
        let source = std::fs::read_to_string(&file)
            .map_err(|error| format!("{}: {error}", file.display()))?;
        if !source.contains(&format!("\"{scenario}\"")) {
            continue;
        }
        named = true;
        let module = modules
            .get(&file)
            .cloned()
            .unwrap_or_else(|| module_path(&base, &file));
        for producer in owners_in_source(&source, scenario, &module)? {
            if !found.contains(&producer) {
                found.push(producer);
            }
        }
    }
    if found.is_empty() {
        found.push(if named {
            Owner::ConsumerOnly
        } else {
            Owner::Unknown
        });
    }
    Ok(found)
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            rust_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The module path of every file reachable from `root_file` (module
/// `root_module`) through `mod` declarations, following `#[path]`
/// attributes the way rustc does.
pub(crate) fn module_map(root_file: &Path, root_module: &str) -> BTreeMap<PathBuf, String> {
    let mut map = BTreeMap::new();
    let mut pending = vec![(root_file.to_path_buf(), root_module.to_owned())];
    while let Some((file, module)) = pending.pop() {
        let Ok(source) = std::fs::read_to_string(&file) else {
            continue;
        };
        let Ok(syntax) = syn::parse_file(&source) else {
            continue;
        };
        let parent = file.parent().unwrap_or(Path::new("")).to_path_buf();
        let is_mod_rs = file.file_name().is_some_and(|name| name == "mod.rs");
        let child_dir = if is_mod_rs {
            parent.clone()
        } else {
            parent.join(file.file_stem().unwrap_or_default())
        };
        for item in &syntax.items {
            let syn::Item::Mod(declaration) = item else {
                continue;
            };
            if declaration.content.is_some() {
                continue;
            }
            let name = declaration.ident.to_string();
            let explicit = declaration.attrs.iter().find_map(|attr| {
                let syn::Meta::NameValue(pair) = &attr.meta else {
                    return None;
                };
                if !pair.path.is_ident("path") {
                    return None;
                }
                match &pair.value {
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(path),
                        ..
                    }) => Some(parent.join(path.value())),
                    _ => None,
                }
            });
            let child = explicit.unwrap_or_else(|| {
                let flat = child_dir.join(format!("{name}.rs"));
                if flat.exists() {
                    flat
                } else {
                    child_dir.join(&name).join("mod.rs")
                }
            });
            let child_module = format!("{module}::{name}");
            if !map.contains_key(&child) {
                pending.push((child.clone(), child_module.clone()));
            }
            map.insert(child, child_module);
        }
        map.entry(file).or_insert(module);
    }
    map
}

/// `openai/cassette/foo.rs` under the providers directory is module
/// `openai::cassette::foo`; a `mod.rs` names its directory. The fallback
/// for a file no `mod` declaration reaches.
fn module_path(base: &Path, file: &Path) -> String {
    let relative = file.strip_prefix(base).unwrap_or(file).with_extension("");
    let mut parts: Vec<String> = relative
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.last().is_some_and(|last| last == "mod") {
        parts.pop();
    }
    parts.join("::")
}

/// Every owner within one source file, in source order.
pub(crate) fn owners_in_source(
    source: &str,
    scenario: &str,
    module: &str,
) -> Result<Vec<Owner>, String> {
    let file = syn::parse_file(source).map_err(|error| error.to_string())?;
    let mut finder = Finder {
        scenario,
        modules: vec![module.to_owned()],
        tests: Vec::new(),
        found: Vec::new(),
    };
    finder.visit_file(&file);
    Ok(finder.found)
}

struct Finder<'a> {
    scenario: &'a str,
    modules: Vec<String>,
    /// Enclosing test functions: name and whether it skips recording.
    tests: Vec<(String, bool)>,
    found: Vec<Owner>,
}

fn is_test_fn(node: &ItemFn) -> bool {
    node.attrs.iter().any(|attr| {
        let path = attr.path();
        path.is_ident("test")
            || path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "test")
    })
}

fn mentions(tokens: &TokenStream, ident: &str) -> bool {
    tokens.clone().into_iter().any(|tree| match tree {
        TokenTree::Ident(found) => found == ident,
        TokenTree::Group(group) => mentions(&group.stream(), ident),
        _ => false,
    })
}

fn names_scenario(expr: &Expr, scenario: &str) -> bool {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(literal),
            ..
        }) => literal.value() == scenario,
        Expr::Call(call) => call
            .args
            .first()
            .is_some_and(|arg| names_scenario(arg, scenario)),
        Expr::MethodCall(call) => names_scenario(&call.receiver, scenario),
        Expr::Paren(paren) => names_scenario(&paren.expr, scenario),
        _ => false,
    }
}

impl Finder<'_> {
    fn qualified(&self, name: &str) -> String {
        format!("{}::{name}", self.modules.join("::"))
    }

    fn record(&mut self) {
        if let Some((name, hand_derived)) = self.tests.last() {
            let name = self.qualified(name);
            self.push(if *hand_derived {
                Owner::HandDerived(name)
            } else {
                Owner::Test(name)
            });
        }
    }

    fn push(&mut self, owner: Owner) {
        if !self.found.contains(&owner) {
            self.found.push(owner);
        }
    }
}

impl<'ast> Visit<'ast> for Finder<'_> {
    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        if node.content.is_some() {
            self.modules.push(node.ident.to_string());
            visit::visit_item_mod(self, node);
            self.modules.pop();
        }
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let test = is_test_fn(node);
        if test {
            let body = quote::quote!(#node);
            self.tests.push((
                node.sig.ident.to_string(),
                mentions(&body, "skip_when_recording"),
            ));
        }
        visit::visit_item_fn(self, node);
        if test {
            self.tests.pop();
        }
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        let wrapper = match node.func.as_ref() {
            Expr::Path(path) => path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            _ => None,
        };
        if wrapper.is_some_and(|name| name.starts_with("with_") && name.contains("cassette"))
            && node
                .args
                .first()
                .is_some_and(|arg| names_scenario(arg, self.scenario))
        {
            self.record();
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        // Matrix rows: `name: ("scenario", …)`; the row name is the test.
        {
            let trees: Vec<TokenTree> = node.tokens.clone().into_iter().collect();
            for window in trees.windows(3) {
                if let [
                    TokenTree::Ident(name),
                    TokenTree::Punct(colon),
                    TokenTree::Group(row),
                ] = window
                    && colon.as_char() == ':'
                    && row.delimiter() == proc_macro2::Delimiter::Parenthesis
                    && matches!(row.stream().into_iter().next(),
                        Some(TokenTree::Literal(literal)) if literal.to_string() == format!("\"{}\"", self.scenario))
                {
                    let owner = Owner::Test(self.qualified(&name.to_string()));
                    self.push(owner);
                }
            }
        }
        visit::visit_macro(self, node);
    }
}

#[cfg(test)]
mod tests;
