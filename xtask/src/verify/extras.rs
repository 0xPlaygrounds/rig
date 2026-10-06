//! The `extras-off-decode-path` guard, run by `verify --check source-guards`.
//!
//! A provider's typed options and reply extras live in its `extension`
//! module, which no other provider code may name, so no decoder can come to
//! need a typed view to decode a reply. Under rig-core's `providers` tree and
//! the companion provider crates' `src`, outside `extension` modules, a
//! non-test file may not name a path through a module named `extension`
//! (in a `use`, an expression, a type, a pattern or a macro's tokens), nor
//! `ReplyExtras` or `ProviderExtension`. Elsewhere in rig-core and the
//! facade, no `pub use` or `pub type` re-exports an extension item. Files are
//! parsed with `syn`, and `use` trees (grouped, renamed, `self`, `super`,
//! glob) and `type` aliases resolve names before they are matched.
//!
//! The same files may not write a citation past the completion fold: no
//! `Citation` or `Span` struct expression, no call of `Text::with_citations`,
//! `Text::span` or the fold's `citation::attach`, and no method call
//! `.with_citations(..)` or `.span(x)`, whose receiver the guard cannot type.
//! Decoders hand citations to `Out::cite`, which resolves their spans.

use std::collections::BTreeMap;
use std::path::Path;

use syn::spanned::Spanned;
use syn::visit::{self, Visit};

use super::guards::{
    Imports, WIRE_ROOTS, is_cfg_test, is_test_file, item_attrs, read_through, token_paths,
};

/// The module every provider's typed views live in.
const EXTENSION: &str = "extension";

/// The names only extension modules may use.
const EXTENSION_TRAITS: &[&str] = &["ReplyExtras", "ProviderExtension"];

/// The types whose values only the completion fold and `Text` build.
const CITATION_TYPES: &[&str] = &["Citation", "Span"];

/// The `Text` methods that attach a citation or make a span.
const CITATION_METHODS: &[&str] = &["with_citations", "span"];

/// Where an extension item may not be re-exported, besides the decode
/// roots: the facade and the rest of rig-core.
const REEXPORT_ROOTS: &[&str] = &["src", "crates/rig-core/src"];

/// Every finding of the guard over the workspace at `root`.
pub(super) fn check(root: &Path) -> Result<Vec<String>, String> {
    let mut findings = Vec::new();
    for (roots, decode) in [(WIRE_ROOTS, true), (REEXPORT_ROOTS, false)] {
        for wire_root in roots {
            let dir = root.join(wire_root);
            if !dir.is_dir() {
                continue;
            }
            for path in crate::support::files_under(&dir, Some("rs"))? {
                let relative = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                if is_test_file(&relative)
                    || in_extension_module(&relative)
                    || (!decode && in_decode_root(&relative))
                {
                    continue;
                }
                let source = std::fs::read_to_string(&path)
                    .map_err(|error| format!("{relative}: {error}"))?;
                findings.extend(offenders(&relative, &source, decode)?);
            }
        }
    }
    Ok(findings)
}

/// Whether `relative` is a file of an `extension` module.
pub(super) fn in_extension_module(relative: &str) -> bool {
    relative.ends_with("/extension.rs") || relative.contains("/extension/")
}

fn in_decode_root(relative: &str) -> bool {
    WIRE_ROOTS
        .iter()
        .any(|root| relative.starts_with(&format!("{root}/")))
}

/// The guard's findings in `source`, the file at `file`: every rule when
/// `decode` (a file on the decode path), else only the re-export rule.
pub(super) fn offenders(file: &str, source: &str, decode: bool) -> Result<Vec<String>, String> {
    let parsed = syn::parse_file(source).map_err(|error| format!("{file}: {error}"))?;
    let mut imports = Imports::default();
    imports.visit_file(&parsed);
    let mut guard = Guard {
        file,
        decode,
        imports: imports.map,
        findings: Vec::new(),
    };
    guard.visit_file(&parsed);
    Ok(guard.findings)
}

struct Guard<'a> {
    file: &'a str,
    decode: bool,
    imports: BTreeMap<String, Vec<String>>,
    findings: Vec<String>,
}

impl Guard<'_> {
    fn report(&mut self, span: proc_macro2::Span, what: &str) {
        let line = span.start().line;
        self.findings.push(format!(
            "extras-off-decode-path: {}:{line}: {what}",
            self.file
        ));
    }

    /// What `segments` name that the decode path may not, read through the
    /// file's imports. `module_only` counts `extension` only where a segment
    /// follows it, as a module: a lone `extension` in an expression is a
    /// local value.
    fn forbidden(&self, segments: Vec<String>, module_only: bool) -> Option<String> {
        let resolved = read_through(&self.imports, segments.clone(), 0);
        [segments, resolved].into_iter().find_map(|path| {
            let modules = if module_only {
                path.len().saturating_sub(1)
            } else {
                path.len()
            };
            if path
                .iter()
                .take(modules)
                .any(|segment| segment == EXTENSION)
            {
                Some(format!("names {}", path.join("::")))
            } else {
                path.iter()
                    .find(|segment| EXTENSION_TRAITS.contains(&segment.as_str()))
                    .map(|name| format!("names {name}"))
            }
        })
    }

    /// What a resolved call path names that writes a citation past the fold.
    fn citation_call(&self, segments: Vec<String>) -> Option<String> {
        let resolved = read_through(&self.imports, segments.clone(), 0);
        [segments, resolved]
            .into_iter()
            .find_map(|path| match path.as_slice() {
                [.., owner, method]
                    if (owner == "Text" && CITATION_METHODS.contains(&method.as_str()))
                        || (owner == "citation" && method == "attach") =>
                {
                    Some(format!(
                        "calls {owner}::{method}, which writes a citation past the fold"
                    ))
                }
                _ => None,
            })
    }

    fn check_segments(
        &mut self,
        segments: Vec<String>,
        module_only: bool,
        span: proc_macro2::Span,
    ) {
        if let Some(what) = self.forbidden(segments, module_only) {
            self.report(span, &what);
        }
    }
}

/// Every path a `use` tree imports, as segments.
fn use_paths(tree: &syn::UseTree, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            use_paths(&path.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = prefix.clone();
            full.push(name.ident.to_string());
            out.push(full);
        }
        syn::UseTree::Rename(rename) => {
            let mut full = prefix.clone();
            full.push(rename.ident.to_string());
            out.push(full);
        }
        syn::UseTree::Glob(_) => out.push(prefix.clone()),
        syn::UseTree::Group(group) => {
            for tree in &group.items {
                use_paths(tree, prefix, out);
            }
        }
    }
}

fn is_pub(visibility: &syn::Visibility) -> bool {
    !matches!(visibility, syn::Visibility::Inherited)
}

fn segments(path: &syn::Path) -> Vec<String> {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect()
}

impl<'ast> Visit<'ast> for Guard<'_> {
    fn visit_expr_struct(&mut self, expr: &'ast syn::ExprStruct) {
        if self.decode {
            let resolved = read_through(&self.imports, segments(&expr.path), 0);
            if let Some(name) = resolved
                .last()
                .filter(|name| CITATION_TYPES.contains(&name.as_str()))
            {
                self.report(
                    expr.span(),
                    &format!("builds a {name} past the fold; hand a WireCitation to Out::cite"),
                );
            }
        }
        visit::visit_expr_struct(self, expr);
    }

    // A call's callee is a path expression, and so is the function passed
    // as a value (`.map(Text::span)`).
    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        if self.decode
            && let Some(what) = self.citation_call(segments(&expr.path))
        {
            self.report(expr.span(), &what);
        }
        visit::visit_expr_path(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let method = expr.method.to_string();
        let writes = method == "with_citations" || (method == "span" && expr.args.len() == 1);
        if self.decode && writes {
            self.report(
                expr.span(),
                &format!(".{method}(..) writes a citation past the fold"),
            );
        }
        visit::visit_expr_method_call(self, expr);
    }

    fn visit_item(&mut self, item: &'ast syn::Item) {
        if !is_cfg_test(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }

    // An inline `extension` module is an extension module.
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.ident != EXTENSION {
            visit::visit_item_mod(self, item);
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        let mut paths = Vec::new();
        use_paths(&item.tree, &mut Vec::new(), &mut paths);
        for path in paths {
            let through_extension = path.iter().any(|segment| segment == EXTENSION);
            if self.decode {
                self.check_segments(path, false, item.span());
            } else if through_extension && is_pub(&item.vis) {
                self.report(
                    item.span(),
                    &format!("re-exports {}, an extension item", path.join("::")),
                );
            }
        }
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        if !self.decode
            && is_pub(&item.vis)
            && let syn::Type::Path(ty) = &*item.ty
        {
            let segments: Vec<String> = ty
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            if segments.iter().any(|segment| segment == EXTENSION) {
                self.report(
                    item.span(),
                    &format!("re-exports {}, an extension item", segments.join("::")),
                );
            }
        }
        visit::visit_item_type(self, item);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if self.decode {
            let segments = path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            self.check_segments(segments, true, path.span());
        }
        visit::visit_path(self, path);
    }

    // A macro's tokens are not parsed: every `a::b::C` run in them is
    // matched as a path, which covers a `macro_rules!` body. A lone
    // `extension` counts there, since `extension::$item` splits at `$`.
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if self.decode {
            let mut paths = Vec::new();
            token_paths(mac.tokens.clone(), &mut Vec::new(), &mut paths);
            for (segments, span) in paths {
                self.check_segments(segments, false, span);
            }
        }
        visit::visit_macro(self, mac);
    }
}
