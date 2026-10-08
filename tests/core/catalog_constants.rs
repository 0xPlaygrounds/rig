//! Every public model constant has a catalog entry.
//!
//! The constants are public string items (`const`, `static`, associated
//! `const` in an inherent impl, any `const` in a trait impl, a `pub` trait's
//! default `const`, and a `pub use` of any of these, named or glob, followed
//! through the target module's own `pub use` items) in rig-core's provider
//! modules and the companion provider crates, and the catalog is data, so no
//! type links them. An item is a string constant when its type is spelled as
//! a reference to `str` (by any path) or to a `type` alias of one, or when
//! its value is a string literal or a path that resolves to a string
//! constant, whatever its type is spelled as. This guard walks the sources,
//! resolves each constant's value (a literal, or another constant's path
//! through the module tree, its `use` items and `rig_core::`), and looks it
//! up in `Catalog::builtin()` under the vendor that serves the re-exporting
//! or defining module. A value it cannot read or resolve fails the guard, and
//! so does an item-level macro call it has not reviewed, whose expansion it
//! cannot see. A constant that names no model (a base URL, an environment
//! variable, a version) is told apart by its name.
//!
//! Known limit: a model id inside a constant of another type (an
//! `Option<&str>`, an array, a struct) is not read.

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

/// The item-level macros this guard has read, whose expansions define no
/// public string constant that names a model: the two vendor-constructor
/// macros (functions only), rig-gemini-grpc's `rest_messages!` (each
/// message's protobuf `NAME`) and `tonic::include_proto!` (the generated
/// protobuf types, whose only strings are service and message names). Any
/// other macro call at item level in a scanned module fails the guard.
const KNOWN_MACROS: [&str; 4] = [
    "openai_vendor",
    "anthropic_vendor",
    "rest_messages",
    "include_proto",
];

/// Whether a constant named `name` in `module` holds something other than a
/// model id. In an `extension` module, `PROVIDER` is a `ProviderExtension`'s
/// provider key and a `_BETA` constant is a beta header value; elsewhere
/// those names are read like any other.
fn names_no_model(module: &[String], name: &str) -> bool {
    let extension = module.last().is_some_and(|last| last == "extension");
    name.ends_with("_URL")
        || name.ends_with("_ENV")
        || name.starts_with("ANTHROPIC_VERSION")
        || matches!(
            name,
            "PROVIDER_NAME" | "AZURE_DEFAULT_API_VERSION" | "DEFAULT_LOCATION"
        )
        || (extension && (name == "PROVIDER" || name.ends_with("_BETA")))
}

/// One public `&str` constant or static, at module, impl or trait level.
struct Constant {
    file: String,
    /// The module that holds it, from the crate root.
    module: Vec<String>,
    name: String,
    value: Value,
    /// How its type is spelled.
    ty: Spelled,
}

/// What a constant is set to.
#[derive(Clone)]
enum Value {
    Literal(String),
    /// Another constant, by the path the source writes.
    Alias(Vec<String>),
    /// Anything else, such as `concat!(..)` or a parenthesised literal.
    Unreadable,
}

/// How a constant's type is spelled.
#[derive(Clone, PartialEq)]
enum Spelled {
    /// A reference to `str`, by any path (`&str`, `&'static str`,
    /// `&core::primitive::str`).
    Str,
    /// A plain path, which may be a `type` alias of `&str`, by its last
    /// segment.
    Named(String),
    /// Anything else.
    Other,
}

/// How `ty` is spelled.
fn spelled(ty: &syn::Type) -> Spelled {
    match ty {
        syn::Type::Paren(inner) => spelled(&inner.elem),
        syn::Type::Group(inner) => spelled(&inner.elem),
        syn::Type::Reference(reference) => match &*reference.elem {
            syn::Type::Path(path)
                if path.qself.is_none()
                    && path
                        .path
                        .segments
                        .last()
                        .is_some_and(|segment| segment.ident == "str") =>
            {
                Spelled::Str
            }
            _ => Spelled::Other,
        },
        syn::Type::Path(path) if path.qself.is_none() => {
            path.path.segments.last().map_or(Spelled::Other, |segment| {
                Spelled::Named(segment.ident.to_string())
            })
        }
        _ => Spelled::Other,
    }
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
    /// Each `pub use` binding and `pub use` glob, as written.
    reexports: Vec<Reexport>,
    /// The names of the `type` aliases of `&str` the files declare.
    str_aliases: std::collections::BTreeSet<String>,
    /// Whether the visitor is in a trait impl, whose consts are public with
    /// the trait.
    in_trait_impl: bool,
    /// Whether the visitor is in a `pub` trait, whose default consts are
    /// public.
    in_pub_trait: bool,
}

/// A `pub use` in `module`: one binding `name` of `target`, or (`name`
/// `None`) a glob of the module `target`.
#[derive(Clone)]
struct Reexport {
    file: String,
    module: Vec<String>,
    name: Option<String>,
    target: Vec<String>,
}

impl Constants {
    /// Record the public item `name: ty = expr`. Whether it is a string
    /// constant is decided once every file is read (see [`Spelled`]); a value
    /// the guard cannot read fails it then, rather than being skipped.
    fn record(&mut self, public: bool, name: &syn::Ident, ty: &syn::Type, expr: &syn::Expr) {
        if !public {
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
            syn::Expr::Lit(_) => return,
            _ => Value::Unreadable,
        };
        self.found.push(Constant {
            file: self.file.clone(),
            module: self.module.clone(),
            name: name.to_string(),
            value,
            ty: spelled(ty),
        });
    }

    /// Fail on the item-level macro call `mac` unless it is one this guard
    /// has read: its expansion could define a constant the walk cannot see.
    fn check_macro(&self, mac: &syn::Macro) {
        let name = mac
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_default();
        if name != "macro_rules" && !KNOWN_MACROS.contains(&name.as_str()) {
            panic!(
                "{}: the item-level macro call `{name}!` in `{}` may define a public string \
                 constant this guard cannot see; write the constants out, or read the macro \
                 and add it to KNOWN_MACROS",
                self.file,
                self.module.join("::")
            );
        }
    }

    /// Record what `tree` binds, and, for a `pub use`, what it re-exports.
    fn record_use(&mut self, public: bool, prefix: &mut Vec<String>, tree: &syn::UseTree) {
        let (name, target) = match tree {
            syn::UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.record_use(public, prefix, &path.tree);
                prefix.pop();
                return;
            }
            syn::UseTree::Group(group) => {
                for tree in &group.items {
                    self.record_use(public, prefix, tree);
                }
                return;
            }
            syn::UseTree::Glob(_) => {
                self.globs
                    .entry(self.module.clone())
                    .or_default()
                    .push(prefix.clone());
                if public {
                    self.reexports.push(Reexport {
                        file: self.file.clone(),
                        module: self.module.clone(),
                        name: None,
                        target: prefix.clone(),
                    });
                }
                return;
            }
            syn::UseTree::Name(name) => (name.ident.to_string(), name.ident.to_string()),
            syn::UseTree::Rename(rename) => (rename.rename.to_string(), rename.ident.to_string()),
        };
        let mut path = prefix.clone();
        path.push(target);
        if public {
            self.reexports.push(Reexport {
                file: self.file.clone(),
                module: self.module.clone(),
                name: Some(name.clone()),
                target: path.clone(),
            });
        }
        self.uses.insert((self.module.clone(), name), path);
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

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let outer = std::mem::replace(&mut self.in_trait_impl, item.trait_.is_some());
        syn::visit::visit_item_impl(self, item);
        self.in_trait_impl = outer;
    }

    fn visit_impl_item_const(&mut self, item: &'ast syn::ImplItemConst) {
        let public = is_public(&item.vis) || self.in_trait_impl;
        self.record(public, &item.ident, &item.ty, &item.expr);
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        let outer = std::mem::replace(&mut self.in_pub_trait, is_public(&item.vis));
        syn::visit::visit_item_trait(self, item);
        self.in_pub_trait = outer;
    }

    fn visit_trait_item_const(&mut self, item: &'ast syn::TraitItemConst) {
        if let Some((_, expr)) = &item.default {
            self.record(self.in_pub_trait, &item.ident, &item.ty, expr);
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.record_use(is_public(&item.vis), &mut Vec::new(), &item.tree);
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        if spelled(&item.ty) == Spelled::Str {
            self.str_aliases.insert(item.ident.to_string());
        }
    }

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        self.check_macro(&item.mac);
    }

    fn visit_impl_item_macro(&mut self, item: &'ast syn::ImplItemMacro) {
        self.check_macro(&item.mac);
    }

    fn visit_trait_item_macro(&mut self, item: &'ast syn::TraitItemMacro) {
        self.check_macro(&item.mac);
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
    fn visit_trait_item_fn(&mut self, _: &'ast syn::TraitItemFn) {}
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

/// The crate's constants and `use` bindings, keyed by module and name, and
/// rig-core's when the crate is a companion crate that names `rig_core::`.
struct Crate<'core> {
    values: BTreeMap<(Vec<String>, String), Value>,
    uses: BTreeMap<(Vec<String>, String), Vec<String>>,
    globs: BTreeMap<Vec<String>, Vec<Vec<String>>>,
    /// Each module's `pub use` items: the name bound (`None` for a glob)
    /// and the path.
    reexports: BTreeMap<Vec<String>, Vec<(Option<String>, Vec<String>)>>,
    core: Option<&'core Crate<'core>>,
}

impl Crate<'_> {
    /// The literal the path `path`, written in `module`, names: through
    /// `crate::`, `self::`, `super::`, `rig_core::`, child modules and `use`
    /// bindings. `None` when the path leaves the crates read or names no
    /// constant.
    fn resolve(&self, module: &[String], path: &[String], depth: usize) -> Option<String> {
        if depth > 32 {
            return None;
        }
        match path {
            [first, rest @ ..] if first == "crate" => self.resolve(&[], rest, depth + 1),
            [first, rest @ ..] if first == "rig_core" => self.core?.resolve(&[], rest, depth + 1),
            [first, rest @ ..] if first == "self" => self.resolve(module, rest, depth + 1),
            [first, rest @ ..] if first == "super" => {
                self.resolve(module.split_last()?.1, rest, depth + 1)
            }
            [name] => match self.values.get(&(module.to_vec(), name.clone())) {
                Some(Value::Literal(text)) => Some(text.clone()),
                Some(Value::Alias(alias)) => self.resolve(module, alias, depth + 1),
                Some(Value::Unreadable) => None,
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

    /// The crate and module the module path `path`, written in `module`,
    /// names, read as [`Crate::resolve`] reads a constant's path.
    fn module<'a>(
        &'a self,
        module: &[String],
        path: &[String],
        depth: usize,
    ) -> Option<(&'a Crate<'a>, Vec<String>)> {
        if depth > 32 {
            return None;
        }
        match path {
            [] => Some((self, module.to_vec())),
            [first, rest @ ..] if first == "crate" => self.module(&[], rest, depth + 1),
            [first, rest @ ..] if first == "rig_core" => self.core?.module(&[], rest, depth + 1),
            [first, rest @ ..] if first == "self" => self.module(module, rest, depth + 1),
            [first, rest @ ..] if first == "super" => {
                self.module(module.split_last()?.1, rest, depth + 1)
            }
            [first, rest @ ..] => match self.uses.get(&(module.to_vec(), first.clone())) {
                Some(target) => self.module(module, &[target.as_slice(), rest].concat(), depth + 1),
                None => {
                    let child = [module, std::slice::from_ref(first)].concat();
                    self.module(&child, rest, depth + 1)
                }
            },
        }
    }

    /// The string constants `module` makes public, by name and value: those
    /// it defines, and those its own `pub use` items re-export, followed
    /// through each glob's target module in turn.
    fn exported(&self, module: &[String], depth: usize) -> Vec<(String, String)> {
        if depth > 32 {
            return Vec::new();
        }
        let mut found: Vec<(String, String)> = self
            .values
            .keys()
            .filter(|(owner, _)| owner == module)
            .filter_map(|(_, name)| {
                let value = self.resolve(module, std::slice::from_ref(name), 0)?;
                Some((name.clone(), value))
            })
            .collect();
        for (name, target) in self.reexports.get(module).into_iter().flatten() {
            match name {
                Some(name) => {
                    if let Some(value) = self.resolve(module, target, 0) {
                        found.push((name.clone(), value));
                    }
                }
                None => {
                    if let Some((owner, inner)) = self.module(module, target, 0) {
                        found.extend(owner.exported(&inner, depth + 1));
                    }
                }
            }
        }
        found
    }
}

/// The source files under `dir`, read: their constants and `use` bindings.
struct Read {
    constants: Vec<Constant>,
    uses: BTreeMap<(Vec<String>, String), Vec<String>>,
    globs: BTreeMap<Vec<String>, Vec<Vec<String>>>,
    reexports: Vec<Reexport>,
    str_aliases: std::collections::BTreeSet<String>,
}

fn read_crate(root: &Path, dir: &str) -> Read {
    let mut read = Read {
        constants: Vec::new(),
        uses: BTreeMap::new(),
        globs: BTreeMap::new(),
        reexports: Vec::new(),
        str_aliases: std::collections::BTreeSet::new(),
    };
    for path in rust_files(&root.join(dir)) {
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
            reexports: Vec::new(),
            str_aliases: std::collections::BTreeSet::new(),
            in_trait_impl: false,
            in_pub_trait: false,
        };
        visitor.visit_file(&file);
        read.constants.extend(visitor.found);
        read.uses.extend(visitor.uses);
        read.reexports.extend(visitor.reexports);
        read.str_aliases.extend(visitor.str_aliases);
        for (module, paths) in visitor.globs {
            read.globs.entry(module).or_default().extend(paths);
        }
    }
    read
}

impl Read {
    fn known<'core>(&self, core: Option<&'core Crate<'core>>) -> Crate<'core> {
        Crate {
            values: self
                .constants
                .iter()
                .map(|constant| {
                    let key = (constant.module.clone(), constant.name.clone());
                    (key, constant.value.clone())
                })
                .collect(),
            uses: self.uses.clone(),
            globs: self.globs.clone(),
            reexports: self.reexports.iter().fold(
                BTreeMap::new(),
                |mut reexports: BTreeMap<_, Vec<_>>, reexport| {
                    reexports
                        .entry(reexport.module.clone())
                        .or_default()
                        .push((reexport.name.clone(), reexport.target.clone()));
                    reexports
                },
            ),
            core,
        }
    }
}

/// Every model constant of the crate `read` from `dir`, with the vendor it
/// is filed under: the constants it defines, and those its `pub use` items
/// re-export, filed under the re-exporting module's vendor.
fn model_constants(
    read: &Read,
    known: &Crate<'_>,
    dir: &str,
    vendor: Option<&str>,
    str_aliases: &std::collections::BTreeSet<String>,
) -> Vec<(Constant, String)> {
    let mut found: Vec<(String, Vec<String>, String, String)> = Vec::new();
    for constant in &read.constants {
        // Spelled as a string: its value must be readable and resolve.
        // Otherwise it is a string constant only if its value is a string
        // literal or resolves to one.
        let string = match &constant.ty {
            Spelled::Str => true,
            Spelled::Named(name) => str_aliases.contains(name),
            Spelled::Other => false,
        };
        let value = match &constant.value {
            Value::Literal(text) => text.clone(),
            Value::Alias(path) => match known.resolve(&constant.module, path, 0) {
                Some(value) => value,
                None if !string => continue,
                None => panic!(
                    "{}: {} names `{}`, which this guard cannot resolve to a string \
                     constant of the same crate",
                    constant.file,
                    constant.name,
                    path.join("::")
                ),
            },
            Value::Unreadable if !string => continue,
            Value::Unreadable => panic!(
                "{}: the value of {} is neither a string literal nor a constant's path; \
                 write it as one so this guard can check it",
                constant.file, constant.name
            ),
        };
        found.push((
            constant.file.clone(),
            constant.module.clone(),
            constant.name.clone(),
            value,
        ));
    }
    // A `pub use` of something other than a string constant (a type, a
    // function, a module) re-exports no model.
    for reexport in &read.reexports {
        match &reexport.name {
            Some(name) => {
                if let Some(value) = known.resolve(&reexport.module, &reexport.target, 0) {
                    found.push((
                        reexport.file.clone(),
                        reexport.module.clone(),
                        name.clone(),
                        value,
                    ));
                }
            }
            None => {
                let Some((owner, module)) = known.module(&reexport.module, &reexport.target, 0)
                else {
                    continue;
                };
                for (name, value) in owner.exported(&module, 0) {
                    found.push((reexport.file.clone(), reexport.module.clone(), name, value));
                }
            }
        }
    }
    let mut models = Vec::new();
    for (file, module, name, value) in found {
        if names_no_model(&module, &name) {
            continue;
        }
        let vendor = match vendor {
            Some(vendor) => vendor.to_owned(),
            None => {
                let first = file
                    .strip_prefix(&format!("{dir}/"))
                    .and_then(|rest| rest.split(['/', '.']).next())
                    .unwrap_or_default();
                MODULES
                    .iter()
                    .find_map(|(module, vendor)| (*module == first).then_some(*vendor))
                    .unwrap_or_else(|| {
                        panic!(
                            "{file}: {name} is a model constant of a provider module this guard \
                             does not map to a vendor; add `{first}` to MODULES"
                        )
                    })
                    .to_owned()
            }
        };
        models.push((
            Constant {
                file,
                module,
                name,
                value: Value::Literal(value),
                ty: Spelled::Str,
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
    let mut missing = std::collections::BTreeSet::new();
    let (core_dir, _) = CRATES[0];
    let core_read = read_crate(root, core_dir);
    let core = core_read.known(None);
    for (dir, vendor) in CRATES {
        let read = match dir == core_dir {
            true => None,
            false => Some(read_crate(root, dir)),
        };
        let (read, known) = match &read {
            Some(read) => (read, read.known(Some(&core))),
            None => (&core_read, core_read.known(None)),
        };
        let str_aliases = core_read
            .str_aliases
            .union(&read.str_aliases)
            .cloned()
            .collect();
        for (constant, vendor) in model_constants(read, &known, dir, vendor, &str_aliases) {
            let Value::Literal(model) = &constant.value else {
                continue;
            };
            let provider = ProviderId::catalog(&vendor)
                .unwrap_or_else(|| panic!("`{vendor}` is no vendor the catalog knows"));
            checked += 1;
            if catalog.get_exact(provider, model).is_none() {
                missing.insert(format!(
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
        missing.into_iter().collect::<Vec<_>>().join("\n")
    );
}
