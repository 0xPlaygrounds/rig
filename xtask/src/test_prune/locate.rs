//! Where a test lives. nextest names a test by its binary id and its module
//! path within the target; this resolves the target's root file from
//! `cargo metadata`, follows the `mod` declarations (inline, `#[path]` and
//! file modules) to the file and function, and reads the function's facts.
//! A test it cannot place, such as one a macro generates, is not the prune's.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use syn::{Item, ItemFn};

use super::facts::{self, Facts};
use crate::support::output;

/// A placed test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Located {
    /// The source file, relative to the workspace root.
    pub(crate) file: String,
    /// The file's module path within its target, empty at the root.
    pub(crate) module: String,
    pub(crate) facts: Facts,
}

/// Every target's root file by nextest binary id, relative to the workspace
/// root.
pub(crate) fn roots(root: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    let metadata: Value = serde_json::from_str(&output(
        root,
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?)
    .map_err(|error| format!("cargo metadata: {error}"))?;
    let workspace = root
        .canonicalize()
        .map_err(|error| format!("{}: {error}", root.display()))?;
    let mut roots = BTreeMap::new();
    let packages = metadata.get("packages").and_then(Value::as_array);
    for package in packages.into_iter().flatten() {
        let Some(name) = package.get("name").and_then(Value::as_str) else {
            continue;
        };
        let targets = package.get("targets").and_then(Value::as_array);
        for target in targets.into_iter().flatten() {
            let field = |key: &str| target.get(key).and_then(Value::as_str);
            let (Some(target_name), Some(source)) = (field("name"), field("src_path")) else {
                continue;
            };
            let kinds: Vec<&str> = target
                .get("kind")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let Some(id) = binary_id(name, target_name, &kinds) else {
                continue;
            };
            if let Ok(relative) = Path::new(source).strip_prefix(&workspace) {
                roots.insert(id, relative.to_path_buf());
            }
        }
    }
    Ok(roots)
}

/// nextest's binary id for a target of `package`.
pub(crate) fn binary_id(package: &str, target: &str, kinds: &[&str]) -> Option<String> {
    let library = ["lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"];
    if kinds.iter().any(|kind| library.contains(kind)) {
        return Some(package.to_owned());
    }
    let prefix = match kinds.first().copied()? {
        "test" => "",
        "bin" => "bin/",
        "example" => "example/",
        "bench" => "bench/",
        _ => return None,
    };
    Some(format!("{package}::{prefix}{target}"))
}

/// Places tests, parsing each file once.
pub(crate) struct Locator<'a> {
    root: &'a Path,
    files: BTreeMap<PathBuf, Option<syn::File>>,
}

/// One step of the module walk inside a file.
enum Step {
    Missing,
    Found(Box<ItemFn>, bool),
    /// An external module: how many path segments the file consumed, the
    /// inline modules before it, its `#[path]`, and whether the walk passed
    /// a wasm-only gate.
    Descend {
        consumed: usize,
        inline: Vec<String>,
        path: Option<String>,
        wasm: bool,
    },
}

impl<'a> Locator<'a> {
    pub(crate) fn new(root: &'a Path) -> Self {
        Self {
            root,
            files: BTreeMap::new(),
        }
    }

    fn parsed(&mut self, file: &Path) -> Option<&syn::File> {
        let root = self.root;
        self.files
            .entry(file.to_path_buf())
            .or_insert_with(|| {
                std::fs::read_to_string(root.join(file))
                    .ok()
                    .and_then(|source| syn::parse_file(&source).ok())
            })
            .as_ref()
    }

    /// Place `test` (its path within the target) under the target's root
    /// file `start`, keeping only a function `cassette prune`'s editor can
    /// take out: one with a `#[test]`-style attribute.
    pub(crate) fn locate(&mut self, binary: &str, start: &Path, test: &str) -> Option<Located> {
        let segments: Vec<&str> = test.split("::").collect();
        let (name, modules) = segments.split_last()?;
        let mut file = start.to_path_buf();
        let mut consumed = 0;
        let mut mod_rs = true;
        let mut wasm = false;
        loop {
            let rest = modules.get(consumed..)?;
            let step = walk(self.parsed(&file)?, rest, name);
            match step {
                Step::Missing => return None,
                Step::Found(item, gated) => {
                    let label = format!("{binary} {test}");
                    return Some(Located {
                        file: file.to_string_lossy().replace('\\', "/"),
                        module: modules.get(..consumed)?.join("::"),
                        facts: facts::of(&label, &item, wasm || gated),
                    });
                }
                Step::Descend {
                    consumed: taken,
                    inline,
                    path,
                    wasm: gated,
                } => {
                    wasm |= gated;
                    let parent = file.parent()?.to_path_buf();
                    let dir = if mod_rs {
                        parent.clone()
                    } else {
                        parent.join(file.file_stem()?)
                    };
                    let base = inline.iter().fold(dir, |dir, module| dir.join(module));
                    let segment = modules.get(consumed + taken - 1)?;
                    let next = match &path {
                        Some(path) if inline.is_empty() => parent.join(path),
                        Some(path) => base.join(path),
                        None => {
                            let flat = base.join(format!("{segment}.rs"));
                            if self.root.join(&flat).is_file() {
                                flat
                            } else {
                                base.join(segment).join("mod.rs")
                            }
                        }
                    };
                    // A file loaded by `#[path]` nests its modules as a
                    // `mod.rs` does.
                    mod_rs =
                        path.is_some() || next.file_name().is_some_and(|name| name == "mod.rs");
                    file = next;
                    consumed += taken;
                }
            }
        }
    }
}

/// Walk `modules` down from a file's items to `name`.
fn walk(file: &syn::File, modules: &[&str], name: &str) -> Step {
    let mut items: &[Item] = &file.items;
    let mut wasm = facts::gates_wasm(&file.attrs);
    let mut inline = Vec::new();
    for (index, segment) in modules.iter().enumerate() {
        let Some(module) = items.iter().find_map(|item| match item {
            Item::Mod(module) if module.ident == segment => Some(module),
            _ => None,
        }) else {
            return Step::Missing;
        };
        wasm |= facts::gates_wasm(&module.attrs);
        match &module.content {
            Some((_, content)) => {
                items = content;
                inline.push((*segment).to_owned());
            }
            None => {
                return Step::Descend {
                    consumed: index + 1,
                    inline,
                    path: path_attr(&module.attrs),
                    wasm,
                };
            }
        }
    }
    let found = items.iter().find_map(|item| match item {
        Item::Fn(function) if function.sig.ident == name && is_test_fn(function) => Some(function),
        _ => None,
    });
    match found {
        Some(function) => Step::Found(
            Box::new(function.clone()),
            wasm || facts::gates_wasm(&function.attrs),
        ),
        None => Step::Missing,
    }
}

/// The value of a `#[path = "..."]` attribute.
fn path_attr(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|attr| match &attr.meta {
        syn::Meta::NameValue(pair) if pair.path.is_ident("path") => match &pair.value {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(value),
                ..
            }) => Some(value.value()),
            _ => None,
        },
        _ => None,
    })
}

/// A function with an attribute whose last path segment is `test`, as the
/// editor recognizes one.
fn is_test_fn(function: &ItemFn) -> bool {
    function.attrs.iter().any(|attr| {
        attr.path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
    })
}
