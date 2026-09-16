//! `check-wires`: inside `crates/rig-core/src/providers/`, everything is a
//! wire.
//!
//! A wire is data plus an encoder and a decoder. That is not a style
//! preference — it is what makes a provider testable from bytes, storable in
//! a scene, and portable to a host that owns its own I/O. The moment one
//! provider grows an `async fn`, a transport type parameter, or its own
//! consumer-trait impl, the axis it adds multiplies against every other
//! provider again, which is the shape this model exists to end.
//!
//! The rules, each with the failure it prevents:
//!
//! | rule | what it prevents |
//! |---|---|
//! | no `.await`, `async fn`, or `async` block | a provider owning its transport |
//! | no `Arc`, `Box<dyn`, or `impl Future` in an `impl Wire` block | a wire that is not data |
//! | no `struct`/`enum` with an `H` or `T` parameter | the transport parameter returning |
//! | no consumer-trait impl | a second way to be a model |
//!
//! The one exception is `openai/responses_api/websocket.rs`: a session — one
//! connection, many turns, warmup — is not a request/response exchange, so it
//! keeps its own API and decodes every message through the shared driver.
//!
//! Source is parsed with `syn`, so an `.await` in a doc comment or a string
//! cannot trip the check, and a file `syn` cannot parse is an error rather
//! than a pass.

use std::path::{Path, PathBuf};

use syn::visit::{self, Visit};
use syn::{Expr, File, ImplItemFn, ItemEnum, ItemFn, ItemImpl, ItemStruct, Type};

/// The files allowed to own something a pure `encode` structurally cannot
/// be, each with the reason it is not a wire:
///
/// - the Responses websocket is a **connection**: one socket, many turns,
///   warmup — a session rather than a request/response exchange;
/// - Gemini's cached content is a **resource lifecycle**: its replies are
///   cache documents, not assistant turns, and it manages the inputs a wire
///   later references.
///
/// Credential exchange is the third, matched by path in
/// [`is_credential_exchange`] because every provider has one.
///
/// These surfaces are exempt from the transport rule as well as the
/// `async` ones: a session holds the socket it is a session over, which is
/// exactly what distinguishes it from a wire.
const SESSION_EXCEPTIONS: &[&str] = &[
    "openai/responses_api/websocket.rs",
    "gemini/cached_content.rs",
];

/// Whether `relative` is one of the named non-wire surfaces.
fn is_session(relative: &str) -> bool {
    SESSION_EXCEPTIONS.contains(&relative)
}

/// Credential exchange is a conversation, not a request/response exchange:
/// a device flow polls, an OAuth refresh round-trips, and a token cache is
/// shared. None of that can live in a pure `encode`, and none of it is a
/// wire — it *produces* the `Secret` a wire holds. These modules are
/// therefore exempt from the no-`async` rules, and only from those.
fn is_credential_exchange(relative: &str) -> bool {
    relative.contains("/auth/") || relative.ends_with("/auth.rs")
}

/// The traits a consumer calls a model through. Exactly one implementation of
/// each ships, and it is `driver::Bound` — a provider contributes a wire.
const CONSUMER_TRAITS: &[&str] = &[
    "CompletionModel",
    "EmbeddingModel",
    "ImageEmbeddingModel",
    "TranscriptionModel",
    "ImageGenerationModel",
    "AudioGenerationModel",
    "RerankModel",
    "ModelLister",
];

/// Run the check over `crates/rig-core/src/providers/`.
pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let providers = workspace.join("crates/rig-core/src/providers");
    if !providers.is_dir() {
        return Err(format!("{} is not a directory", providers.display()));
    }
    let mut files = Vec::new();
    walk(&providers, &mut files)?;
    files.sort();

    let mut offenders = Vec::new();
    for path in &files {
        let relative = path
            .strip_prefix(&providers)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        // Test code may do whatever it likes: it is not shipped, and a test
        // that drives a wire through a fake socket has to await something.
        if relative.ends_with("tests.rs") || relative.contains("/tests/") {
            continue;
        }
        let source = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let parsed: File =
            syn::parse_file(&source).map_err(|error| format!("{}: {error}", path.display()))?;
        let mut visitor = Wires {
            file: relative.clone(),
            session: is_session(&relative) || is_credential_exchange(&relative),
            offenders: Vec::new(),
        };
        visitor.visit_file(&parsed);
        offenders.extend(visitor.offenders);
    }

    if offenders.is_empty() {
        return Ok(());
    }
    Err(format!(
        "inside crates/rig-core/src/providers/ everything is a wire:\n{}\n\
         a provider is data plus an encoder and a decoder; transport, futures \
         and consumer-trait impls belong to rig_core::driver",
        offenders.join("\n")
    ))
}

/// Every `.rs` file under `dir`, recursively.
fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

struct Wires {
    file: String,
    /// Whether this file may own a connection or a conversation: the
    /// websocket session, or a credential exchange.
    session: bool,
    offenders: Vec<String>,
}

impl Wires {
    fn report(&mut self, what: &str) {
        self.offenders.push(format!("  {}: {what}", self.file));
    }
}

/// A type's source text, for the substring checks the rules are stated in.
fn quote_type(ty: &Type) -> String {
    use quote::ToTokens as _;
    ty.to_token_stream().to_string().replace(' ', "")
}

impl<'ast> Visit<'ast> for Wires {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if !self.session {
            match expr {
                Expr::Await(_) => self.report("`.await` — the driver owns the I/O"),
                Expr::Async(_) => self.report("an `async` block — the driver owns the I/O"),
                _ => {}
            }
        }
        visit::visit_expr(self, expr);
    }

    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        if !self.session && item.sig.asyncness.is_some() {
            self.report(&format!(
                "`async fn {}` — a wire's encode and decode are pure",
                item.sig.ident
            ));
        }
        visit::visit_item_fn(self, item);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast ImplItemFn) {
        if !self.session && item.sig.asyncness.is_some() {
            self.report(&format!(
                "`async fn {}` — a wire's encode and decode are pure",
                item.sig.ident
            ));
        }
        visit::visit_impl_item_fn(self, item);
    }

    fn visit_item_struct(&mut self, item: &'ast ItemStruct) {
        for parameter in item.generics.type_params() {
            if parameter.ident == "H" && !self.session {
                self.report(&format!(
                    "`struct {}<{}>` — a wire holds no transport",
                    item.ident, parameter.ident
                ));
            }
        }
        visit::visit_item_struct(self, item);
    }

    fn visit_item_enum(&mut self, item: &'ast ItemEnum) {
        for parameter in item.generics.type_params() {
            if parameter.ident == "H" && !self.session {
                self.report(&format!(
                    "`enum {}<{}>` — a wire holds no transport",
                    item.ident, parameter.ident
                ));
            }
        }
        visit::visit_item_enum(self, item);
    }

    fn visit_item_impl(&mut self, item: &'ast ItemImpl) {
        let implemented = item
            .trait_
            .as_ref()
            .and_then(|(_, path, _)| path.segments.last())
            .map(|segment| segment.ident.to_string());

        if let Some(name) = &implemented {
            if CONSUMER_TRAITS.contains(&name.as_str()) {
                self.report(&format!(
                    "`impl {name} for {}` — the one implementation is `driver::Bound`",
                    quote_type(&item.self_ty)
                ));
            }
            // A wire is data: nothing in its own impl may be shared,
            // erased, or deferred.
            if name == "Wire" {
                for forbidden in ["Arc<", "Box<dyn", "implFuture"] {
                    if item
                        .items
                        .iter()
                        .any(|inner| impl_item_mentions(inner, forbidden))
                    {
                        self.report(&format!(
                            "`impl Wire for {}` mentions `{}` — a wire is data",
                            quote_type(&item.self_ty),
                            forbidden.replace("implFuture", "impl Future")
                        ));
                    }
                }
            }
        }
        visit::visit_item_impl(self, item);
    }
}

/// Whether an impl item's source text mentions `needle` (whitespace removed,
/// so `Box<dyn Foo>` matches `Box<dyn`).
fn impl_item_mentions(item: &syn::ImplItem, needle: &str) -> bool {
    use quote::ToTokens as _;
    item.to_token_stream()
        .to_string()
        .replace(' ', "")
        .contains(needle)
}

#[cfg(test)]
mod tests;
