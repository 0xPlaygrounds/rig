//! Every public model constant has a catalog entry.
//!
//! The constants are `pub const NAME: &str` items in rig-core's provider
//! modules and the companion provider crates, and the catalog is data, so no
//! type links them. This guard walks the sources, resolves each constant's
//! value, and looks it up in `Catalog::builtin()` under the vendor that
//! serves the module's models. A constant that names no model (a base URL,
//! an environment variable, a version) is told apart by its name.

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

/// One `pub const NAME: &str` item.
struct Constant {
    file: String,
    name: String,
    value: Value,
}

/// What a constant is set to.
enum Value {
    Literal(String),
    /// Another constant, by its last path segment.
    Alias(String),
}

#[derive(Default)]
struct Constants {
    file: String,
    found: Vec<Constant>,
}

impl<'ast> Visit<'ast> for Constants {
    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        let public = matches!(item.vis, syn::Visibility::Public(_));
        let is_str = matches!(&*item.ty, syn::Type::Reference(reference)
            if matches!(&*reference.elem, syn::Type::Path(path) if path.path.is_ident("str")));
        if !public || !is_str {
            return;
        }
        let value = match &*item.expr {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(text),
                ..
            }) => Value::Literal(text.value()),
            syn::Expr::Path(path) => match path.path.segments.last() {
                Some(segment) => Value::Alias(segment.ident.to_string()),
                None => return,
            },
            _ => return,
        };
        self.found.push(Constant {
            file: self.file.clone(),
            name: item.ident.to_string(),
            value,
        });
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        let test_only = module.attrs.iter().any(|attr| {
            attr.path().is_ident("cfg")
                && attr
                    .parse_args::<syn::Meta>()
                    .is_ok_and(|meta| meta.path().is_ident("test"))
        });
        if !test_only {
            syn::visit::visit_item_mod(self, module);
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

/// Every model constant of the crate under `dir`, with the vendor it is
/// filed under.
fn model_constants(root: &Path, dir: &str, vendor: Option<&str>) -> Vec<(Constant, String)> {
    let base = root.join(dir);
    let mut constants = Vec::new();
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
            file: relative,
            found: Vec::new(),
        };
        visitor.visit_file(&file);
        constants.extend(visitor.found);
    }
    let values: BTreeMap<String, String> = constants
        .iter()
        .filter_map(|constant| match &constant.value {
            Value::Literal(text) => Some((constant.name.clone(), text.clone())),
            Value::Alias(_) => None,
        })
        .collect();
    let mut models = Vec::new();
    for constant in constants {
        if names_no_model(&constant.name) {
            continue;
        }
        let value = match &constant.value {
            Value::Literal(text) => text.clone(),
            Value::Alias(name) => values
                .get(name)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: {} aliases an unknown constant",
                        constant.file, constant.name
                    )
                })
                .clone(),
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
