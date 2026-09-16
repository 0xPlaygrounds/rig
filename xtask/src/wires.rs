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
//! | no `struct`/`enum` with an `H` parameter, or a `T` parameter outside [`DATA_GENERICS`] | the transport parameter returning |
//! | no consumer-trait impl | a second way to be a model |
//!
//! The one exception is `openai/responses_api/websocket.rs`: a session — one
//! connection, many turns, warmup — is not a request/response exchange, so it
//! keeps its own API and decodes every message through the shared driver.
//! Credential exchange (`/auth/`) is exempt from the `async` rules only.
//!
//! Source is parsed with `syn`, so an `.await` in a doc comment or a string
//! cannot trip the check, and a file `syn` cannot parse is an error rather
//! than a pass. `syn` does not look inside a macro invocation's tokens,
//! though, so the `.await` rule runs as a second pass over the file's raw
//! token stream after the AST pass: the one place the AST cannot see is
//! exactly where an `async_stream::stream! { … .await … }` hides.
//!
//! `*tests.rs` files and `tests/` directories are skipped: a test that drives
//! a wire through a fake socket has to await something, and test code is not
//! shipped.

use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use syn::visit::{self, Visit};
use syn::{Expr, File, ImplItemFn, ItemEnum, ItemFn, ItemImpl, ItemStruct, Type};

/// The files allowed to own something a pure `encode` structurally cannot
/// be, each with the reason it is not a wire:
///
/// - the Responses websocket is a **connection**: one socket, many turns,
///   warmup — a session rather than a request/response exchange.
///
/// Credential exchange is the other, matched by path in
/// [`is_credential_exchange`] because every provider has one.
///
/// A session is exempt from the transport rule as well as the `async`
/// ones: it holds the socket it is a session over, which is exactly what
/// distinguishes it from a wire.
const SESSION_EXCEPTIONS: &[&str] = &["openai/responses_api/websocket.rs"];

/// The `T`-generic types that are parametric *data* rather than a held
/// transport, each with why. A `struct Foo<T>` outside this list is
/// rejected: `T` is the letter a transport parameter reaches for once `H`
/// is forbidden.
const DATA_GENERICS: &[&str] = &[
    // The classifier's verdict, generic over the wire's own event type.
    "WireEvent",
    // The typed-transport triage, generic over an SDK's event type.
    "TypedEvent",
    // Cohere's embed reply, generic over the answer shape of the two embed routes.
    "EmbedReply",
];

/// Whether a `T`-generic type is one of the allowlisted data shapes.
fn is_data_generic(ident: &str) -> bool {
    DATA_GENERICS.contains(&ident)
}

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
        let mut visitor = Wires::new(&relative);
        visitor.visit_file(&parsed);
        visitor.scan_awaits(&source);
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
    /// Whether this file may hold a conversation — `async` and `.await` —
    /// because it is a session or a credential exchange.
    conversation: bool,
    /// Whether this file may hold the socket it is a session over: the
    /// transport parameter is allowed too. A credential exchange is not a
    /// session; it produces the `Secret` a wire holds.
    session: bool,
    offenders: Vec<String>,
}

impl Wires {
    fn new(relative: &str) -> Self {
        let session = is_session(relative);
        Self {
            file: relative.to_owned(),
            conversation: session || is_credential_exchange(relative),
            session,
            offenders: Vec::new(),
        }
    }

    fn report(&mut self, what: &str) {
        self.offenders.push(format!("  {}: {what}", self.file));
    }

    /// The transport-parameter rule: no `H`, and no `T` outside
    /// [`DATA_GENERICS`].
    fn check_type_params(&mut self, kind: &str, ident: &syn::Ident, generics: &syn::Generics) {
        if self.session {
            return;
        }
        for parameter in generics.type_params() {
            let offending = parameter.ident == "H"
                || (parameter.ident == "T" && !is_data_generic(&ident.to_string()));
            if offending {
                self.report(&format!(
                    "`{kind} {ident}<{}>` — a wire holds no transport (parametric data is \
                     allowlisted in `DATA_GENERICS`)",
                    parameter.ident
                ));
            }
        }
    }

    /// Every `.await` in `source`, by line, macro bodies included: `syn`
    /// keeps a macro invocation's body as opaque tokens, so the AST pass
    /// never sees `stream! { … .await … }`. Token-level rather than textual:
    /// a comment is not a token and a string literal is one, so neither can
    /// name an `.await` here.
    fn scan_awaits(&mut self, source: &str) {
        if self.conversation {
            return;
        }
        let tokens: TokenStream = match source.parse() {
            Ok(tokens) => tokens,
            // `syn` already parsed this source, so its tokens lex; report
            // rather than pass if that ever stops being true.
            Err(error) => {
                self.report(&format!(
                    "could not tokenize for the `.await` scan: {error}"
                ));
                return;
            }
        };
        let mut lines = Vec::new();
        await_lines(tokens, &mut lines);
        for line in lines {
            self.report(&format!("line {line}: `.await` — the driver owns the I/O"));
        }
    }
}

/// Collect the line of every `.` immediately followed by `await`,
/// descending into every delimited group (a macro body is one).
fn await_lines(tokens: TokenStream, lines: &mut Vec<usize>) {
    let mut after_dot = false;
    for token in tokens {
        match &token {
            TokenTree::Group(group) => await_lines(group.stream(), lines),
            TokenTree::Ident(ident) if after_dot && ident == "await" => {
                lines.push(ident.span().start().line);
            }
            _ => {}
        }
        after_dot = matches!(&token, TokenTree::Punct(punct) if punct.as_char() == '.');
    }
}

/// A type's source text, for the substring checks the rules are stated in.
fn quote_type(ty: &Type) -> String {
    use quote::ToTokens as _;
    ty.to_token_stream().to_string().replace(' ', "")
}

impl<'ast> Visit<'ast> for Wires {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if !self.conversation && matches!(expr, Expr::Async(_)) {
            self.report("an `async` block — the driver owns the I/O");
        }
        visit::visit_expr(self, expr);
    }

    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        if !self.conversation && item.sig.asyncness.is_some() {
            self.report(&format!(
                "`async fn {}` — a wire's encode and decode are pure",
                item.sig.ident
            ));
        }
        visit::visit_item_fn(self, item);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast ImplItemFn) {
        if !self.conversation && item.sig.asyncness.is_some() {
            self.report(&format!(
                "`async fn {}` — a wire's encode and decode are pure",
                item.sig.ident
            ));
        }
        visit::visit_impl_item_fn(self, item);
    }

    fn visit_item_struct(&mut self, item: &'ast ItemStruct) {
        self.check_type_params("struct", &item.ident, &item.generics);
        visit::visit_item_struct(self, item);
    }

    fn visit_item_enum(&mut self, item: &'ast ItemEnum) {
        self.check_type_params("enum", &item.ident, &item.generics);
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
