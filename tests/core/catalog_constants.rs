//! Every public model constant has a catalog entry.
//!
//! The constants are public `&str` items (`const`, `static` and associated
//! `const`) in rig-core's provider modules and the companion provider crates,
//! and the catalog is data, so no type links them. This guard walks the
//! sources, resolves each constant's value (a literal, or another constant's
//! path through the module tree and its `use` items), and looks it up in
//! `Catalog::builtin()` under the vendor that serves the module's models. A
//! value it cannot read or resolve fails the guard. A constant that names no
//! model (a base URL, an environment variable, a version) is told apart by
//! its name.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rig::catalog::Catalog;
use rig::providers::registry::ProviderId;
use syn::visit::Visit;

/// The crates whose constants name models, and the vendor each crate's
/// models are filed under (`None`: by the provider module's directory).
const CRATES: [(&str, Option<&str>); 5] = [
    ("crates/rig-core/src/providers", None),
    ("crates/rig-bedrock/src", Some("aws_bedrock")),
    ("crates/rig-gemini-grpc/src", Some("gemini-grpc")),
    ("crates/rig-vertexai/src", Some("vertexai")),
    ("crates/rig-candle/src", Some("candle")),
];

/// rig-core's provider modules, by the first path segment under
/// `providers/`, and the vendor each serves. A module missing here fails
/// the guard once it holds a model constant.
const MODULES: [(&str, &str); 26] = [
    ("anthropic", "anthropic"),
    ("azure", "azure.openai"),
    ("chatgpt", "chatgpt"),
    ("cohere", "cohere"),
    ("copilot", "copilot"),
    ("deepseek", "deepseek"),
    ("doubleword", "doubleword"),
    ("gemini", "gcp.gemini"),
    ("groq", "groq"),
    ("huggingface", "huggingface"),
    ("hyperbolic", "hyperbolic"),
    ("llamacpp", "llamacpp"),
    ("minimax", "minimax"),
    ("mira", "mira"),
    ("mistral", "mistral"),
    ("moonshot", "moonshot"),
    ("ollama", "ollama"),
    ("openai", "openai"),
    ("openrouter", "openrouter"),
    ("perplexity", "perplexity"),
    ("together", "together"),
    ("venice", "venice"),
    ("voyageai", "voyageai"),
    ("xai", "xai"),
    ("xiaomimimo", "xiaomimimo"),
    ("zai", "zai"),
];

/// Whether a constant named `name` holds something other than a model id.
fn names_no_model(name: &str) -> bool {
    name.ends_with("_URL")
        || name.ends_with("_ENV")
        || name.starts_with("ANTHROPIC_VERSION")
        || matches!(
            name,
            "PROVIDER_NAME" | "AZURE_DEFAULT_API_VERSION" | "DEFAULT_LOCATION"
        )
}

/// One public `&str` constant or static, at module, impl or trait level.
struct Constant {
    file: String,
    /// The module that holds it, from the crate root.
    module: Vec<String>,
    name: String,
    value: Value,
}

/// What a constant is set to.
enum Value {
    Literal(String),
    /// Another constant, by the path the source writes.
    Alias(Vec<String>),
}

/// The items of one source file that the guard reads.
struct Constants {
    file: String,
    /// The module the visitor is in, from the crate root.
    module: Vec<String>,
    found: Vec<Constant>,
    /// What each module's `use` items bind: module, name, then the path.
    uses: BTreeMap<(Vec<String>, String), Vec<String>>,
    /// Each module's glob imports, by the path before the `*`.
    globs: BTreeMap<Vec<String>, Vec<Vec<String>>>,
}

impl Constants {
    /// Record the public `&str` item `name: ty = expr`, failing on a value
    /// the guard cannot read rather than skipping it.
    fn record(&mut self, public: bool, name: &syn::Ident, ty: &syn::Type, expr: &syn::Expr) {
        let is_str = matches!(ty, syn::Type::Reference(reference)
            if matches!(&*reference.elem, syn::Type::Path(path) if path.path.is_ident("str")));
        if !public || !is_str {
            return;
        }
        let value = match expr {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(text),
                ..
            }) => Value::Literal(text.value()),
            syn::Expr::Path(path) if path.qself.is_none() => Value::Alias(
                path.path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect(),
            ),
            _ => panic!(
                "{}: the value of {name} is neither a string literal nor a constant's path; \
                 write it as one so this guard can check it",
                self.file
            ),
        };
        self.found.push(Constant {
            file: self.file.clone(),
            module: self.module.clone(),
            name: name.to_string(),
            value,
        });
    }

    fn record_use(&mut self, prefix: &mut Vec<String>, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.record_use(prefix, &path.tree);
                prefix.pop();
            }
            syn::UseTree::Name(name) => {
                let mut target = prefix.clone();
                target.push(name.ident.to_string());
                self.uses
                    .insert((self.module.clone(), name.ident.to_string()), target);
            }
            syn::UseTree::Rename(rename) => {
                let mut target = prefix.clone();
                target.push(rename.ident.to_string());
                self.uses
                    .insert((self.module.clone(), rename.rename.to_string()), target);
            }
            syn::UseTree::Group(group) => {
                for tree in &group.items {
                    self.record_use(prefix, tree);
                }
            }
            syn::UseTree::Glob(_) => self
                .globs
                .entry(self.module.clone())
                .or_default()
                .push(prefix.clone()),
        }
    }
}

fn is_public(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

impl<'ast> Visit<'ast> for Constants {
    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        self.record(is_public(&item.vis), &item.ident, &item.ty, &item.expr);
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        self.record(is_public(&item.vis), &item.ident, &item.ty, &item.expr);
    }

    fn visit_impl_item_const(&mut self, item: &'ast syn::ImplItemConst) {
        self.record(is_public(&item.vis), &item.ident, &item.ty, &item.expr);
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.record_use(&mut Vec::new(), &item.tree);
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        let test_only = module.attrs.iter().any(|attr| {
            attr.path().is_ident("cfg")
                && attr
                    .parse_args::<syn::Meta>()
                    .is_ok_and(|meta| meta.path().is_ident("test"))
        });
        if !test_only {
            self.module.push(module.ident.to_string());
            syn::visit::visit_item_mod(self, module);
            self.module.pop();
        }
    }

    // A function body's constants are local, never public API.
    fn visit_item_fn(&mut self, _: &'ast syn::ItemFn) {}
    fn visit_impl_item_fn(&mut self, _: &'ast syn::ImplItemFn) {}
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
            .flatten();
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs")
                && path.file_name().is_some_and(|name| name != "tests.rs")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// The module a source file holds, from its path under the crate's `src`.
fn file_module(relative: &str) -> Vec<String> {
    let (_, path) = relative
        .split_once("/src/")
        .unwrap_or_else(|| panic!("{relative}: not under a crate's src"));
    let path = path.strip_suffix(".rs").unwrap_or(path);
    let mut module: Vec<String> = path.split('/').map(str::to_owned).collect();
    if module
        .last()
        .is_some_and(|last| ["mod", "lib", "main"].contains(&last.as_str()))
    {
        module.pop();
    }
    module
}

/// The crate's constants and `use` bindings, keyed by module and name.
struct Crate {
    values: BTreeMap<(Vec<String>, String), Value>,
    uses: BTreeMap<(Vec<String>, String), Vec<String>>,
    globs: BTreeMap<Vec<String>, Vec<Vec<String>>>,
}

impl Crate {
    /// The literal the path `path`, written in `module`, names: through
    /// `crate::`, `self::`, `super::`, child modules and `use` bindings.
    /// `None` when the path leaves the crate or names no constant.
    fn resolve(&self, module: &[String], path: &[String], depth: usize) -> Option<String> {
        if depth > 32 {
            return None;
        }
        match path {
            [first, rest @ ..] if first == "crate" => self.resolve(&[], rest, depth + 1),
            [first, rest @ ..] if first == "self" => self.resolve(module, rest, depth + 1),
            [first, rest @ ..] if first == "super" => {
                self.resolve(module.split_last()?.1, rest, depth + 1)
            }
            [name] => match self.values.get(&(module.to_vec(), name.clone())) {
                Some(Value::Literal(text)) => Some(text.clone()),
                Some(Value::Alias(alias)) => self.resolve(module, alias, depth + 1),
                None => match self.uses.get(&(module.to_vec(), name.clone())) {
                    Some(target) => self.resolve(module, target, depth + 1),
                    None => self.globs.get(module)?.iter().find_map(|glob| {
                        let path = [glob.as_slice(), std::slice::from_ref(name)].concat();
                        self.resolve(module, &path, depth + 1)
                    }),
                },
            },
            [first, rest @ ..] => match self.uses.get(&(module.to_vec(), first.clone())) {
                Some(target) => {
                    self.resolve(module, &[target.as_slice(), rest].concat(), depth + 1)
                }
                None => {
                    let child = [module, std::slice::from_ref(first)].concat();
                    self.resolve(&child, rest, depth + 1)
                }
            },
            [] => None,
        }
    }
}

/// Every model constant of the crate under `dir`, with the vendor it is
/// filed under.
fn model_constants(root: &Path, dir: &str, vendor: Option<&str>) -> Vec<(Constant, String)> {
    let base = root.join(dir);
    let mut constants = Vec::new();
    let mut uses = BTreeMap::new();
    let mut globs: BTreeMap<Vec<String>, Vec<Vec<String>>> = BTreeMap::new();
    for path in rust_files(&base) {
        let source = std::fs::read_to_string(&path).expect("a readable source file");
        let file =
            syn::parse_file(&source).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let mut visitor = Constants {
            module: file_module(&relative),
            file: relative,
            found: Vec::new(),
            uses: BTreeMap::new(),
            globs: BTreeMap::new(),
        };
        visitor.visit_file(&file);
        constants.extend(visitor.found);
        uses.extend(visitor.uses);
        for (module, paths) in visitor.globs {
            globs.entry(module).or_default().extend(paths);
        }
    }
    let known = Crate {
        values: constants
            .iter()
            .map(|constant| {
                let value = match &constant.value {
                    Value::Literal(text) => Value::Literal(text.clone()),
                    Value::Alias(path) => Value::Alias(path.clone()),
                };
                ((constant.module.clone(), constant.name.clone()), value)
            })
            .collect(),
        uses,
        globs,
    };
    let mut models = Vec::new();
    for constant in constants {
        if names_no_model(&constant.name) {
            continue;
        }
        let value = match &constant.value {
            Value::Literal(text) => text.clone(),
            Value::Alias(path) => known.resolve(&constant.module, path, 0).unwrap_or_else(|| {
                panic!(
                    "{}: {} names `{}`, which this guard cannot resolve to a string \
                         constant of the same crate",
                    constant.file,
                    constant.name,
                    path.join("::")
                )
            }),
        };
        let vendor = match vendor {
            Some(vendor) => vendor.to_owned(),
            None => {
                let module = constant
                    .file
                    .strip_prefix(&format!("{dir}/"))
                    .and_then(|rest| rest.split(['/', '.']).next())
                    .unwrap_or_default();
                MODULES
                    .iter()
                    .find_map(|(name, vendor)| (*name == module).then_some(*vendor))
                    .unwrap_or_else(|| {
                        panic!(
                            "{}: {} is a model constant of a provider module this guard does not \
                             map to a vendor; add `{module}` to MODULES",
                            constant.file, constant.name
                        )
                    })
                    .to_owned()
            }
        };
        models.push((
            Constant {
                value: Value::Literal(value),
                ..constant
            },
            vendor,
        ));
    }
    models
}

#[test]
fn every_public_model_constant_has_a_catalog_entry() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let catalog = Catalog::builtin();
    let mut checked = 0;
    let mut missing = Vec::new();
    for (dir, vendor) in CRATES {
        for (constant, vendor) in model_constants(root, dir, vendor) {
            let Value::Literal(model) = &constant.value else {
                continue;
            };
            let provider = ProviderId::catalog(&vendor)
                .unwrap_or_else(|| panic!("`{vendor}` is no vendor the catalog knows"));
            checked += 1;
            if catalog.get(provider, model).is_none() {
                missing.push(format!(
                    "{}: {} ({model:?}) has no catalog entry under `{vendor}`",
                    constant.file, constant.name
                ));
            }
        }
    }
    assert!(checked > 300, "the walk found the constants: {checked}");
    assert!(
        missing.is_empty(),
        "{} model constants have no catalog entry; add each to \
         xtask/src/catalog/review.json and run `cargo xtask catalog sync`:\n{}",
        missing.len(),
        missing.join("\n")
    );
}
