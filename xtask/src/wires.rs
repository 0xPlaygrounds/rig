//! `check-wires`: inside `crates/rig-core/src/providers/`, everything is a
//! wire.
//!
//! A wire is data plus an encoder and a decoder, which keeps providers testable
//! from bytes and independent of the host's I/O.
//!
//! The rules, each with the failure it prevents:
//!
//! | rule | what it prevents |
//! |---|---|
//! | no `.await`, `async fn`, or `async` block | a provider owning its transport |
//! | no `Arc`, `Box<dyn`, or `impl Future` in an `impl Wire` block | a wire that is not data |
//! | no `struct`/`enum` parameter bounded by `HttpClientExt` or defaulted to `BoxedHttpClient` | the transport parameter returning |
//! | no consumer-trait impl | a second way to be a model |
//!
//! `openai/responses_api/websocket.rs` is exempt because a session spans many
//! turns over one connection rather than a single exchange. The credential
//! exchanges named in [`CREDENTIAL_EXCHANGES`] are exempt from the `async` rules
//! only.
//!
//! Source is parsed with `syn`, so awaits in comments or strings cannot trip the
//! check and an unparsable file is an error. Because `syn` does not descend into
//! macro invocations, the `.await` rule runs a second pass over the raw token
//! stream. Test files and `tests/` directories are skipped.

use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::visit::{self, Visit};
use syn::{Expr, File, ImplItemFn, ItemEnum, ItemFn, ItemImpl, ItemStruct};

/// Files holding a session rather than a request/response exchange, currently
/// only the Responses websocket.
///
/// A session is exempt from the transport rule as well as the `async` ones
/// because it owns the socket it runs over.
const SESSION_EXCEPTIONS: &[&str] = &["openai/responses_api/websocket.rs"];

/// Files performing credential exchanges, which poll device flows, refresh
/// tokens, and share caches, and so produce the secrets wires carry.
///
/// Exempt from the `async` rules only, since they own no socket. Listed per file
/// rather than matched by path so adding one is reviewable; the tests assert the
/// list's length.
const CREDENTIAL_EXCHANGES: &[&str] = &[
    // GitHub Copilot: device flow, API-key refresh, on-disk token cache.
    "copilot/auth/mod.rs",
    "copilot/auth/native.rs",
    "copilot/auth/wasm.rs",
    // ChatGPT: the same shape over OpenAI's own device flow.
    "chatgpt/auth/mod.rs",
    "chatgpt/auth/native.rs",
    "chatgpt/auth/wasm.rs",
    // The request/send helpers both exchanges round-trip through.
    "internal/auth.rs",
];

/// The traits a consumer calls a model through. `driver::Bound` is the only
/// implementation of each; providers contribute wires instead.
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
    /// Whether `async` and `.await` are allowed, as in a session or a
    /// credential exchange.
    conversation: bool,
    /// Whether a transport parameter is allowed, which holds for sessions only.
    session: bool,
    offenders: Vec<String>,
}

impl Wires {
    fn new(relative: &str) -> Self {
        let session = SESSION_EXCEPTIONS.contains(&relative);
        Self {
            file: relative.to_owned(),
            conversation: session || CREDENTIAL_EXCHANGES.contains(&relative),
            session,
            offenders: Vec::new(),
        }
    }

    fn report(&mut self, what: &str) {
        self.offenders.push(format!("  {}: {what}", self.file));
    }

    /// Rejects transport parameters, recognized by their bounds rather than
    /// their names: a parameter bounded by `HttpClientExt`, whether in its own
    /// bound list or the item's `where` clause, or defaulted to
    /// `BoxedHttpClient`. Parameters without such a bound are ordinary data.
    fn check_type_params(&mut self, kind: &str, ident: &syn::Ident, generics: &syn::Generics) {
        if self.session {
            return;
        }
        for parameter in generics.type_params() {
            if !is_transport_parameter(parameter, generics) {
                continue;
            }
            self.report(&format!(
                "`{kind} {ident}<{}>` — a wire holds no transport, and that parameter is \
                 bounded by `HttpClientExt` or defaulted to `BoxedHttpClient`",
                parameter.ident
            ));
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

/// A syntax node's source text, whitespace removed, so the rules can be
/// stated as substring checks (`Box<dyn Foo>` matches `Box<dyn`).
fn source_text(node: &impl ToTokens) -> String {
    node.to_token_stream().to_string().replace(' ', "")
}

/// Whether `parameter` is the socket rather than parametric data: bounded by
/// `HttpClientExt` in its own bounds or in `generics`' `where` clause, or
/// defaulted to `BoxedHttpClient`.
fn is_transport_parameter(parameter: &syn::TypeParam, generics: &syn::Generics) -> bool {
    // Every bound stated for this parameter, wherever it was stated: the
    // inline list and each `where` predicate naming it are one bound set.
    let mut bounds = source_text(&parameter.bounds);
    for predicate in generics
        .where_clause
        .iter()
        .flat_map(|clause| &clause.predicates)
    {
        if let syn::WherePredicate::Type(bounded) = predicate
            && matches!(&bounded.bounded_ty, syn::Type::Path(named)
                if named.qself.is_none() && named.path.is_ident(&parameter.ident))
        {
            bounds.push_str(&source_text(&bounded.bounds));
        }
    }
    bounds.contains("HttpClientExt")
        || parameter
            .default
            .as_ref()
            .is_some_and(|default| source_text(default).contains("BoxedHttpClient"))
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
                    source_text(&item.self_ty)
                ));
            }
            // A wire is data: nothing in its own impl may be shared,
            // erased, or deferred.
            if name == "Wire" {
                for forbidden in ["Arc<", "Box<dyn", "implFuture"] {
                    if item
                        .items
                        .iter()
                        .any(|inner| source_text(inner).contains(forbidden))
                    {
                        self.report(&format!(
                            "`impl Wire for {}` mentions `{}` — a wire is data",
                            source_text(&item.self_ty),
                            forbidden.replace("implFuture", "impl Future")
                        ));
                    }
                }
            }
        }
        visit::visit_item_impl(self, item);
    }
}

#[cfg(test)]
mod tests;
