//! The typed-options source guards, run by `verify --check source-guards`.
//!
//! `options-mapping` keeps every completion wire answering for every option:
//! an `OptionFields` pattern names each field (no `..`, no `_` binding), an
//! `OptionMap` literal has no `..base`, and no wire file builds or rewrites a
//! request's options. `options-precedence` keeps the merge in
//! `request_params`: no wire file reads `additional_params` as a field or
//! through a pattern, builds or opens request bytes but from
//! `FinalBody::into_body` (or sends none: `Body::empty()`, a WebSocket
//! handshake's `NoBody`), or calls `.body_mut()`. Both run over the
//! completion-wire files listed in [`COMPLETION_WIRE_FILES`], and a file
//! holding a completion `Wire` or a `ReplayTarget` impl that the list misses
//! is a finding too. Files are parsed with `syn`, and `use` trees (grouped,
//! renamed, `self`, glob) and `type` aliases resolve names before they are
//! matched. `#[cfg(test)]` items and test files are skipped.

use std::collections::BTreeMap;
use std::path::Path;

use quote::ToTokens;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

/// Every file that encodes a completion request, with its helpers and the
/// companion crates' request builders. Paths are from the workspace root.
pub(crate) const COMPLETION_WIRE_FILES: &[&str] = &[
    "crates/rig-core/src/providers/anthropic/completion.rs",
    "crates/rig-core/src/providers/anthropic/options.rs",
    "crates/rig-core/src/providers/anthropic/wire.rs",
    "crates/rig-core/src/providers/cohere/chat.rs",
    "crates/rig-core/src/providers/cohere/wire.rs",
    "crates/rig-core/src/providers/copilot/wire.rs",
    "crates/rig-core/src/providers/gemini/completion.rs",
    "crates/rig-core/src/providers/gemini/interactions_api/mod.rs",
    "crates/rig-core/src/providers/gemini/options.rs",
    "crates/rig-core/src/providers/ollama/chat.rs",
    "crates/rig-core/src/providers/openai/options.rs",
    "crates/rig-core/src/providers/openai/responses_api/mod.rs",
    "crates/rig-core/src/providers/openai/responses_api/websocket.rs",
    "crates/rig-core/src/providers/openai/responses_api/wire.rs",
    "crates/rig-core/src/providers/openai/wire/chat.rs",
    "crates/rig-core/src/providers/openai/wire/route.rs",
    "crates/rig-bedrock/src/completion.rs",
    "crates/rig-bedrock/src/options.rs",
    "crates/rig-bedrock/src/request.rs",
    "crates/rig-candle/src/generation.rs",
    "crates/rig-candle/src/model.rs",
    "crates/rig-candle/src/protocol.rs",
    "crates/rig-gemini-grpc/src/completion.rs",
    "crates/rig-vertexai/src/completion.rs",
];

/// The `additional_params` field reads the precedence guard allows, by file
/// and receiver: the WebSocket session writes the chain's
/// `previous_response_id` into the raw layer as a caller would, and a
/// document or video part's own `additional_params` is another field.
const ADDITIONAL_PARAMS_ALLOWED: &[(&str, &str)] = &[
    (
        "crates/rig-core/src/providers/openai/responses_api/websocket.rs",
        "completion_request",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion.rs",
        "document",
    ),
    (
        "crates/rig-core/src/providers/gemini/completion.rs",
        "video",
    ),
];

/// Where a file holding a completion wire may live: rig-core's providers and
/// the companion provider crates.
const WIRE_ROOTS: &[&str] = &[
    "crates/rig-core/src/providers",
    "crates/rig-bedrock/src",
    "crates/rig-candle/src",
    "crates/rig-gemini-grpc/src",
    "crates/rig-vertexai/src",
];

/// Run both guards over the workspace at `root`.
pub(crate) fn check(root: &Path) -> Result<(), String> {
    let mut findings = Vec::new();
    for file in COMPLETION_WIRE_FILES {
        let source =
            std::fs::read_to_string(root.join(file)).map_err(|error| format!("{file}: {error}"))?;
        findings.extend(offenders(file, &source)?);
    }
    for wire_root in WIRE_ROOTS {
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
            if is_test_file(&relative) || COMPLETION_WIRE_FILES.contains(&relative.as_str()) {
                continue;
            }
            let source =
                std::fs::read_to_string(&path).map_err(|error| format!("{relative}: {error}"))?;
            if holds_a_completion_wire(&source)? {
                findings.push(format!(
                    "options-mapping: {relative}: holds a completion `Wire` or a `ReplayTarget` \
                     impl but is not in xtask's COMPLETION_WIRE_FILES"
                ));
            }
        }
    }
    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn is_test_file(relative: &str) -> bool {
    relative.ends_with("tests.rs") || relative.contains("/tests/")
}

/// Whether `source` implements `Wire` with `type Op = …Completion`, or
/// `ReplayTarget`, outside `#[cfg(test)]` items.
pub(super) fn holds_a_completion_wire(source: &str) -> Result<bool, String> {
    let file = syn::parse_file(source).map_err(|error| error.to_string())?;
    let mut finder = WireFinder { found: false };
    finder.visit_file(&file);
    Ok(finder.found)
}

struct WireFinder {
    found: bool,
}

impl<'ast> Visit<'ast> for WireFinder {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if !is_cfg_test(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let Some((_, trait_path, _)) = &item.trait_ else {
            return;
        };
        match last(trait_path).as_deref() {
            Some("ReplayTarget") => self.found = true,
            Some("Wire") => {
                let completion = item.items.iter().any(|item| {
                    matches!(item, syn::ImplItem::Type(ty)
                        if ty.ident == "Op"
                            && matches!(&ty.ty, syn::Type::Path(path)
                                if last(&path.path).as_deref() == Some("Completion")))
                });
                self.found |= completion;
            }
            _ => {}
        }
    }
}

/// Every finding of both guards in `source`, the file at `file`.
pub(super) fn offenders(file: &str, source: &str) -> Result<Vec<String>, String> {
    let parsed = syn::parse_file(source).map_err(|error| format!("{file}: {error}"))?;
    let mut imports = Imports::default();
    imports.visit_file(&parsed);
    let mut guard = Guard {
        file,
        imports: imports.map,
        globs: imports.globs,
        findings: Vec::new(),
        function: Vec::new(),
    };
    guard.visit_file(&parsed);
    Ok(guard.findings)
}

/// Local names to the full paths `use` items and `type` aliases give them,
/// and the prefixes of glob imports.
#[derive(Default)]
struct Imports {
    map: BTreeMap<String, Vec<String>>,
    globs: Vec<Vec<String>>,
}

impl Imports {
    fn add(&mut self, prefix: &mut Vec<String>, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.add(prefix, &path.tree);
                prefix.pop();
            }
            syn::UseTree::Name(name) if name.ident == "self" => {
                if let Some(local) = prefix.last() {
                    self.map.insert(local.clone(), prefix.clone());
                }
            }
            syn::UseTree::Name(name) => {
                let mut full = prefix.clone();
                full.push(name.ident.to_string());
                self.map.insert(name.ident.to_string(), full);
            }
            syn::UseTree::Rename(rename) => {
                let mut full = prefix.clone();
                if rename.ident != "self" {
                    full.push(rename.ident.to_string());
                }
                self.map.insert(rename.rename.to_string(), full);
            }
            syn::UseTree::Glob(_) => self.globs.push(prefix.clone()),
            syn::UseTree::Group(group) => {
                for tree in &group.items {
                    self.add(prefix, tree);
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for Imports {
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.add(&mut Vec::new(), &item.tree);
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        if let syn::Type::Path(ty) = &*item.ty {
            let full = ty
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            self.map.insert(item.ident.to_string(), full);
        }
        visit::visit_item_type(self, item);
    }
}

/// The attributes of `item`, for the `#[cfg(test)]` check.
fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Struct(item) => &item.attrs,
        syn::Item::Enum(item) => &item.attrs,
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Trait(item) => &item.attrs,
        syn::Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .to_token_stream()
                .to_string()
                .replace(' ', "")
                .contains("cfg(test)")
    })
}

/// The last segment of `path`.
fn last(path: &syn::Path) -> Option<String> {
    path.segments
        .last()
        .map(|segment| segment.ident.to_string())
}

/// What one function in the guarded file has done so far.
#[derive(Default)]
struct Function {
    /// Whether it destructures `OptionFields`.
    destructures_fields: bool,
    /// The lines it calls `options::param` on.
    param_calls: Vec<usize>,
    /// Its parameters typed `CompletionRequest` or a reference to one.
    request_bindings: Vec<String>,
}

struct Guard<'a> {
    file: &'a str,
    imports: BTreeMap<String, Vec<String>>,
    globs: Vec<Vec<String>>,
    findings: Vec<String>,
    function: Vec<Function>,
}

impl Guard<'_> {
    fn report(&mut self, guard: &str, span: proc_macro2::Span, what: &str) {
        let line = span.start().line;
        self.findings
            .push(format!("{guard}: {}:{line}: {what}", self.file));
    }

    /// `path`'s segments with its first one read through the file's imports.
    fn resolve(&self, path: &syn::Path) -> Vec<String> {
        self.resolve_segments(
            path.segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect(),
        )
    }

    fn resolve_segments(&self, segments: Vec<String>) -> Vec<String> {
        read_through(&self.imports, segments, 0)
    }

    /// Every full path `segments` may name: read through the imports, and,
    /// when its first segment is no import, under each glob import's prefix.
    fn candidates(&self, segments: Vec<String>) -> Vec<Vec<String>> {
        let resolved = self.resolve_segments(segments.clone());
        let mut candidates = vec![resolved];
        if let Some(first) = segments.first()
            && !self.imports.contains_key(first)
        {
            for glob in &self.globs {
                let prefix = self.resolve_segments(glob.clone());
                candidates.push(prefix.into_iter().chain(segments.iter().cloned()).collect());
            }
        }
        candidates
    }

    /// Whether `path` resolves to a path ending in `tail`.
    fn ends_with(&self, path: &syn::Path, tail: &[&str]) -> bool {
        let segments = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        self.segments_end_with(segments, tail)
    }

    fn segments_end_with(&self, segments: Vec<String>, tail: &[&str]) -> bool {
        self.candidates(segments).iter().any(|resolved| {
            resolved
                .len()
                .checked_sub(tail.len())
                .and_then(|start| resolved.get(start..))
                .is_some_and(|end| {
                    end.iter()
                        .zip(tail)
                        .all(|(segment, expected)| segment == expected)
                })
        })
    }

    /// Whether `segments` name `Body::Bytes` or `Body::Multipart`.
    fn names_a_body_constructor(&self, segments: Vec<String>) -> bool {
        self.segments_end_with(segments.clone(), &["Body", "Bytes"])
            || self.segments_end_with(segments, &["Body", "Multipart"])
    }

    /// Reports a field `additional_params` read through `receiver`, unless
    /// the allowlist names it for this file.
    fn check_additional_params(&mut self, receiver: &str, span: proc_macro2::Span) {
        let allowed = ADDITIONAL_PARAMS_ALLOWED
            .iter()
            .any(|(file, base)| *file == self.file && receiver == *base);
        if !allowed {
            self.report(
                "options-precedence",
                span,
                "reads additional_params outside request_params",
            );
        }
    }

    fn enter_function(&mut self, inputs: impl IntoIterator<Item = (String, String)>) {
        let request_bindings = inputs
            .into_iter()
            .filter(|(_, ty)| {
                ty.trim_start_matches('&')
                    .trim_start_matches("mut")
                    .rsplit("::")
                    .next()
                    .is_some_and(|name| name == "CompletionRequest")
            })
            .map(|(name, _)| name)
            .collect();
        self.function.push(Function {
            request_bindings,
            ..Function::default()
        });
    }

    fn leave_function(&mut self) {
        if let Some(function) = self.function.pop()
            && function.destructures_fields
        {
            for line in function.param_calls {
                self.findings.push(format!(
                    "options-mapping: {}:{line}: `options::param` in a function that maps \
                     `OptionFields`: a mapping cannot defer to a raw key",
                    self.file
                ));
            }
        }
    }

    fn check_fields_pattern(&mut self, pattern: &syn::PatStruct) {
        let resolved = self.resolve(&pattern.path);
        if resolved.last().map(String::as_str) != Some("OptionFields") {
            return;
        }
        if let Some(function) = self.function.last_mut() {
            function.destructures_fields = true;
        }
        if pattern.rest.is_some() {
            self.report(
                "options-mapping",
                pattern.span(),
                "an OptionFields pattern uses '..'",
            );
        }
        for field in &pattern.fields {
            match &*field.pat {
                syn::Pat::Wild(_) => self.report(
                    "options-mapping",
                    field.span(),
                    "an OptionFields pattern binds a field to '_'",
                ),
                syn::Pat::Ident(ident) if ident.ident.to_string().starts_with('_') => self.report(
                    "options-mapping",
                    field.span(),
                    "an OptionFields pattern binds a field to a '_'-prefixed name",
                ),
                _ => {}
            }
        }
    }
}

/// `pattern`'s binding name and its type's text, for a typed argument.
fn typed_input(input: &syn::FnArg) -> Option<(String, String)> {
    let syn::FnArg::Typed(typed) = input else {
        return None;
    };
    let syn::Pat::Ident(ident) = &*typed.pat else {
        return None;
    };
    Some((
        ident.ident.to_string(),
        typed.ty.to_token_stream().to_string().replace(' ', ""),
    ))
}

impl<'ast> Visit<'ast> for Guard<'_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if !is_cfg_test(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if is_cfg_test(&item.attrs) {
            return;
        }
        self.enter_function(item.sig.inputs.iter().filter_map(typed_input));
        visit::visit_impl_item_fn(self, item);
        self.leave_function();
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.enter_function(item.sig.inputs.iter().filter_map(typed_input));
        visit::visit_item_fn(self, item);
        self.leave_function();
    }

    fn visit_pat_struct(&mut self, pattern: &'ast syn::PatStruct) {
        self.check_fields_pattern(pattern);
        // A pattern has no receiver: the struct's type name, in snake case,
        // stands for it in the allowlist.
        let receiver = last(&pattern.path)
            .map(|name| snake_case(&name))
            .unwrap_or_default();
        for field in &pattern.fields {
            if matches!(&field.member, syn::Member::Named(name) if name == "additional_params") {
                self.check_additional_params(&receiver, field.span());
            }
        }
        visit::visit_pat_struct(self, pattern);
    }

    fn visit_expr_struct(&mut self, expr: &'ast syn::ExprStruct) {
        let resolved = self.resolve(&expr.path);
        match resolved.last().map(String::as_str) {
            Some("OptionMap") if expr.rest.is_some() => self.report(
                "options-mapping",
                expr.span(),
                "an OptionMap literal takes its fields from '..'",
            ),
            Some("CompletionRequest") => self.report(
                "options-mapping",
                expr.span(),
                "builds a CompletionRequest, whose options the wire would choose",
            ),
            _ => {}
        }
        visit::visit_expr_struct(self, expr);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(func) = &*call.func {
            if self.ends_with(&func.path, &["CompletionRequest", "new"]) {
                self.report(
                    "options-mapping",
                    call.span(),
                    "builds a CompletionRequest, whose options the wire would choose",
                );
            }
            if self.ends_with(&func.path, &["options", "param"]) {
                let line = call.span().start().line;
                if let Some(function) = self.function.last_mut() {
                    function.param_calls.push(line);
                }
            }
            let serializes = ["to_value", "to_vec", "to_string"]
                .iter()
                .any(|function| self.ends_with(&func.path, &["serde_json", function]));
            if serializes && let Some(argument) = call.args.first() {
                let argument = match argument {
                    syn::Expr::Reference(reference) => &*reference.expr,
                    other => other,
                };
                if let syn::Expr::Path(path) = argument
                    && let Some(name) = path.path.get_ident()
                    && self.function.last().is_some_and(|function| {
                        function.request_bindings.contains(&name.to_string())
                    })
                {
                    self.report(
                        "options-precedence",
                        call.span(),
                        "serializes a CompletionRequest into the request body",
                    );
                }
            }
        }
        visit::visit_expr_call(self, call);
    }

    // Every path, in an expression or a pattern: matching `Body::Bytes`
    // reaches the bytes as surely as building one.
    fn visit_path(&mut self, path: &'ast syn::Path) {
        let segments = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        if self.names_a_body_constructor(segments) {
            self.report(
                "options-precedence",
                path.span(),
                "builds or opens a request body without FinalBody::into_body",
            );
        }
        visit::visit_path(self, path);
    }

    // A macro's tokens are not parsed: every `a::b::C` run in them is
    // matched as a path.
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        let mut paths = Vec::new();
        token_paths(mac.tokens.clone(), &mut Vec::new(), &mut paths);
        for (segments, span) in paths {
            if self.names_a_body_constructor(segments) {
                self.report(
                    "options-precedence",
                    span,
                    "builds or opens a request body without FinalBody::into_body",
                );
            }
        }
        visit::visit_macro(self, mac);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = call.method.to_string();
        match method.as_str() {
            "body" if call.args.len() == 1 => {
                let allowed = match call.args.first() {
                    Some(syn::Expr::MethodCall(inner)) => {
                        inner.method == "into_body" && inner.args.is_empty()
                    }
                    Some(syn::Expr::Call(inner)) => matches!(&*inner.func,
                        syn::Expr::Path(path) if self.ends_with(&path.path, &["Body", "empty"])),
                    // The WebSocket handshake is a `GET` with no body.
                    Some(syn::Expr::Path(path)) => self.ends_with(&path.path, &["NoBody"]),
                    _ => false,
                };
                if !allowed {
                    self.report(
                        "options-precedence",
                        call.span(),
                        "builds a request body without FinalBody::into_body",
                    );
                }
            }
            "body_mut" => self.report(
                "options-precedence",
                call.span(),
                "writes to a built request's body",
            ),
            "options" if call.args.len() == 1 => self.report(
                "options-mapping",
                call.span(),
                "sets a request's options, which the wire must take as given",
            ),
            "deserialize" => {
                let to_json = call.turbofish.as_ref().is_some_and(|turbofish| {
                    turbofish.args.iter().any(|argument| {
                        matches!(argument, syn::GenericArgument::Type(syn::Type::Path(ty))
                            if matches!(last(&ty.path).as_deref(), Some("Value" | "Map")))
                    })
                });
                if to_json {
                    self.report(
                        "options-precedence",
                        call.span(),
                        "copies a FinalBody into mutable JSON",
                    );
                }
            }
            _ => {}
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        if let syn::Expr::Field(field) = &*assign.left
            && matches!(&field.member, syn::Member::Named(name) if name == "options")
        {
            self.report(
                "options-mapping",
                assign.span(),
                "assigns a request's options, which the wire must take as given",
            );
        }
        visit::visit_expr_assign(self, assign);
    }

    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if let syn::Member::Named(name) = &field.member {
            if name == "additional_params" {
                let receiver = field.base.to_token_stream().to_string().replace(' ', "");
                self.check_additional_params(&receiver, field.span());
            }
            if name == "options" {
                self.report(
                    "options-mapping",
                    field.span(),
                    "reads a request's options outside OptionFields",
                );
            }
        }
        visit::visit_expr_field(self, field);
    }
}

/// `name` in snake case: `CompletionRequest` is `completion_request`.
fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (index, character) in name.chars().enumerate() {
        if character.is_uppercase() {
            if index > 0 {
                out.push('_');
            }
            out.extend(character.to_lowercase());
        } else {
            out.push(character);
        }
    }
    out
}

/// Every run of `ident :: ident ...` in `tokens`, with the span of its last
/// segment, into `paths`.
fn token_paths(
    tokens: proc_macro2::TokenStream,
    current: &mut Vec<(String, proc_macro2::Span)>,
    paths: &mut Vec<(Vec<String>, proc_macro2::Span)>,
) {
    fn flush(
        current: &mut Vec<(String, proc_macro2::Span)>,
        paths: &mut Vec<(Vec<String>, proc_macro2::Span)>,
    ) {
        if let Some((_, span)) = current.last() {
            let span = *span;
            paths.push((current.drain(..).map(|(name, _)| name).collect(), span));
        }
    }
    let mut after_colons = false;
    for token in tokens {
        match token {
            proc_macro2::TokenTree::Ident(ident) => {
                if !after_colons {
                    flush(current, paths);
                }
                current.push((ident.to_string(), ident.span()));
                after_colons = false;
            }
            proc_macro2::TokenTree::Punct(punct) if punct.as_char() == ':' => {
                after_colons = !current.is_empty();
            }
            proc_macro2::TokenTree::Group(group) => {
                flush(current, paths);
                after_colons = false;
                token_paths(group.stream(), &mut Vec::new(), paths);
            }
            _ => {
                flush(current, paths);
                after_colons = false;
            }
        }
    }
    flush(current, paths);
}

/// `segments` with its first one read through `imports`, repeatedly, so an
/// alias of an import resolves too.
fn read_through(
    imports: &BTreeMap<String, Vec<String>>,
    segments: Vec<String>,
    depth: usize,
) -> Vec<String> {
    match segments.split_first() {
        Some((first, rest)) if depth < 8 => match imports.get(first) {
            Some(full) if full.len() > 1 || full.first() != Some(first) => {
                let next = full.iter().chain(rest).cloned().collect();
                read_through(imports, next, depth + 1)
            }
            _ => segments,
        },
        _ => segments,
    }
}
