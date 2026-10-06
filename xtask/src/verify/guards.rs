//! The typed-options source guards, run by `verify --check source-guards`.
//!
//! `options-mapping` keeps every completion wire answering for every option:
//! an `OptionFields` pattern names each field (no `..`, no `_` binding), an
//! `OptionMap` literal has no `..base`, and no wire file builds or rewrites a
//! request's options. `options-precedence` keeps the merge in
//! `request_params`: no wire file reads `additional_params` as a field or
//! builds request bytes but from `FinalBody::into_body` (or sends none:
//! `Body::empty()`, a WebSocket handshake's `NoBody`). Both run over the
//! completion-wire files listed in [`COMPLETION_WIRE_FILES`], and a file
//! holding a completion `Wire` or a `ReplayTarget` impl that the list misses
//! is a finding too. Files are parsed with `syn`, and `use` trees (grouped,
//! renamed, `self`) resolve names before they are matched. `#[cfg(test)]`
//! items and test files are skipped.

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
        findings: Vec::new(),
        function: Vec::new(),
    };
    guard.visit_file(&parsed);
    Ok(guard.findings)
}

/// Local names to the full paths `use` items give them.
#[derive(Default)]
struct Imports {
    map: BTreeMap<String, Vec<String>>,
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
            syn::UseTree::Glob(_) => {}
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
        let segments: Vec<String> = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        match segments.split_first() {
            Some((first, rest)) => match self.imports.get(first) {
                Some(full) => full.iter().chain(rest).cloned().collect(),
                None => segments,
            },
            None => segments,
        }
    }

    /// Whether `path` resolves to a path ending in `tail`.
    fn ends_with(&self, path: &syn::Path, tail: &[&str]) -> bool {
        let resolved = self.resolve(path);
        resolved
            .len()
            .checked_sub(tail.len())
            .and_then(|start| resolved.get(start..))
            .is_some_and(|end| {
                end.iter()
                    .zip(tail)
                    .all(|(segment, expected)| segment == expected)
            })
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

    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        if self.ends_with(&expr.path, &["Body", "Bytes"])
            || self.ends_with(&expr.path, &["Body", "Multipart"])
        {
            self.report(
                "options-precedence",
                expr.span(),
                "builds a request body without FinalBody::into_body",
            );
        }
        visit::visit_expr_path(self, expr);
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
                let allowed = ADDITIONAL_PARAMS_ALLOWED
                    .iter()
                    .any(|(file, base)| *file == self.file && receiver == *base);
                if !allowed {
                    self.report(
                        "options-precedence",
                        field.span(),
                        "reads additional_params outside request_params",
                    );
                }
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
