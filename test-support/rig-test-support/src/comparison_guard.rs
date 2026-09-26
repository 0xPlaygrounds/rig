//! Find cassette-test comparisons that fail on a recording pass.
//!
//! The recorder normalizes volatile keys such as `created`, so a recording
//! pass compares a live value with a fixture that never holds it, while
//! replay compares the fixture with itself and passes. A comparison written
//! exactly therefore stays green in CI and fails only when someone records.
//! [`crate::support::assert_matches_recorded_document`] compares such keys by
//! type in a recording pass; this guard finds the comparisons that bypass it.

use proc_macro2::{TokenStream, TokenTree};
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{Expr, ExprForLoop, ExprIf, ExprLit, ExprMatch, ItemFn, Lit, Macro, Pat, Token};

use crate::scenario_registry::ScenarioError;

/// Comparisons in `source` that treat a volatile key exactly outside replay:
///
/// - a loop over a document's `.keys()`, or over a literal key list naming a
///   volatile key, comparing `get(key)` values without calling
///   `is_volatile_json_key`;
/// - a `for (key, value)` loop over a recorded JSON object (`.as_object()` of a
///   `recorded_*` document or a binding to one), or one that consults such a
///   binding, comparing each value with another document's entry at `key` the
///   same way;
/// - an `assert_eq!`/`assert_ne!` (or an `assert!` with `==`/`!=`) that
///   indexes or `get`s a key in `volatile`;
/// - one that compares a whole recorded document (a call to a `recorded_*`
///   function, or a `let` bound to one and not rebound since, not narrowed to
///   one key) exactly with a value read out of a live document by index or
///   `.get` (`observation.raw["extras"]`).
///
/// A `CassetteMode::Replay` arm, a branch taken only in replay, and the
/// `else` of a branch that rules replay out are exempt. A `matches!` or
/// `if let` pattern selects replay only when it names `Replay` and not
/// `Record`, and every or-pattern in it, at any depth, has `Replay` in each
/// alternative. A `for (key, value)` loop over a binding of a recorded
/// document's `.as_object()` (through `let`, `let … else`, `if let` or a
/// let-chain) is checked too, and any name a `let` pattern binds shadows an
/// earlier recorded binding. Keys match in any ASCII case, and a comparison
/// with `None` (a presence check) is exempt.
/// Each finding names its function or method.
pub fn exact_volatile_comparisons(
    source: &str,
    volatile: &[&str],
) -> Result<Vec<String>, ScenarioError> {
    let syntax = syn::parse_file(source)?;
    let mut visitor = GuardVisitor {
        volatile,
        function: String::new(),
        replay_depth: 0,
        recorded_bindings: Vec::new(),
        recorded_objects: Vec::new(),
        findings: Vec::new(),
    };
    visitor.visit_file(&syntax);
    Ok(visitor.findings)
}

struct GuardVisitor<'a> {
    volatile: &'a [&'a str],
    function: String,
    replay_depth: usize,
    /// Local names bound to a recorded document in the current function.
    recorded_bindings: Vec<String>,
    /// Local names bound to a recorded document's JSON object
    /// (`let object = recorded.as_object().unwrap()`).
    recorded_objects: Vec<String>,
    findings: Vec<String>,
}

fn mentions(tokens: &TokenStream, ident: &str) -> bool {
    tokens.clone().into_iter().any(|tree| match tree {
        TokenTree::Ident(found) => found == ident,
        TokenTree::Group(group) => mentions(&group.stream(), ident),
        _ => false,
    })
}

/// Whether `tokens` hold `get(<name>)` or `[<name>]` for the identifier `name`.
fn accesses_ident(tokens: &TokenStream, name: &str) -> usize {
    let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
    let mut count = 0;
    for (index, tree) in trees.iter().enumerate() {
        if let TokenTree::Group(group) = tree {
            let inner: Vec<TokenTree> = group.stream().into_iter().collect();
            let names_it = matches!(inner.as_slice(), [TokenTree::Ident(ident)] if ident == name);
            let after_get =
                index > 0 && matches!(&trees[index - 1], TokenTree::Ident(ident) if ident == "get");
            let bracket = group.delimiter() == proc_macro2::Delimiter::Bracket;
            if names_it && (after_get || bracket) {
                count += 1;
            }
            count += accesses_ident(&group.stream(), name);
        }
    }
    count
}

/// Whether `tokens` index (`["created"]`) or `get` (`.get("created")`) one
/// of `volatile` as a string literal.
fn accesses_volatile(tokens: &TokenStream, volatile: &[&str]) -> bool {
    let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
    trees.iter().enumerate().any(|(index, tree)| {
        let TokenTree::Group(group) = tree else {
            return false;
        };
        let inner: Vec<TokenTree> = group.stream().into_iter().collect();
        let literal_key = matches!(inner.as_slice(), [TokenTree::Literal(literal)]
        if volatile.iter().any(|key| {
            literal.to_string().eq_ignore_ascii_case(&format!("\"{key}\""))
        }));
        let after_get =
            index > 0 && matches!(&trees[index - 1], TokenTree::Ident(ident) if ident == "get");
        let bracket = group.delimiter() == proc_macro2::Delimiter::Bracket;
        (literal_key && (after_get || bracket)) || accesses_volatile(&group.stream(), volatile)
    })
}

/// Whether `expr` is a whole recorded document: a call to a function named
/// `recorded_*`, or a local in `bindings` bound to one, possibly borrowed or
/// cloned.
fn is_recorded_document(expr: &Expr, bindings: &[String]) -> bool {
    match expr {
        Expr::Call(call) => matches!(call.func.as_ref(), Expr::Path(path)
        if path.path.segments.last().is_some_and(|segment| {
            segment.ident.to_string().starts_with("recorded_")
        })),
        Expr::Path(path) => path
            .path
            .get_ident()
            .is_some_and(|ident| bindings.iter().any(|name| ident == name)),
        Expr::Reference(reference) => is_recorded_document(&reference.expr, bindings),
        Expr::Paren(paren) => is_recorded_document(&paren.expr, bindings),
        Expr::MethodCall(call) if call.method == "clone" => {
            is_recorded_document(&call.receiver, bindings)
        }
        _ => false,
    }
}

/// Whether `tokens` call `name(…)`, not merely mention it.
fn calls(tokens: &TokenStream, name: &str) -> bool {
    let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
    trees.iter().enumerate().any(|(index, tree)| match tree {
        TokenTree::Ident(ident) if ident == name => matches!(
            trees.get(index + 1),
            Some(TokenTree::Group(group)) if group.delimiter() == proc_macro2::Delimiter::Parenthesis
        ),
        TokenTree::Group(group) => calls(&group.stream(), name),
        _ => false,
    })
}

/// Whether an `if` condition selects replay: `== …Replay`, `matches!(…,
/// …Replay)` or `let …Replay = …`, alone or in a conjunction. A negated or
/// `!=` condition selects the recording pass.
fn is_replay_condition(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(binary) => match binary.op {
            syn::BinOp::Eq(_) => {
                let (left, right) = (&binary.left, &binary.right);
                mentions(&quote::quote!(#left), "Replay")
                    || mentions(&quote::quote!(#right), "Replay")
            }
            syn::BinOp::And(_) => {
                is_replay_condition(&binary.left) || is_replay_condition(&binary.right)
            }
            _ => false,
        },
        Expr::Macro(mac) => {
            mac.mac.path.is_ident("matches")
                && matches_pattern(&mac.mac).is_some_and(|pattern| is_replay_pattern(&pattern))
        }
        Expr::Let(binding) => is_replay_pattern(&binding.pat),
        Expr::Paren(paren) => is_replay_condition(&paren.expr),
        _ => false,
    }
}

/// The pattern of a `matches!(expression, pattern)` call, ignoring any
/// `if` guard after it.
fn matches_pattern(mac: &Macro) -> Option<Pat> {
    mac.parse_body_with(|input: syn::parse::ParseStream<'_>| {
        input.parse::<Expr>()?;
        input.parse::<Token![,]>()?;
        let pattern = Pat::parse_multi_with_leading_vert(input)?;
        input.parse::<TokenStream>()?;
        Ok(pattern)
    })
    .ok()
}

/// Every name a pattern binds: `recorded` in `let (recorded, _) = …`.
fn bound_names(pattern: &Pat, names: &mut Vec<String>) {
    match pattern {
        Pat::Ident(ident) => names.push(ident.ident.to_string()),
        Pat::Type(typed) => bound_names(&typed.pat, names),
        Pat::Tuple(tuple) => tuple.elems.iter().for_each(|elem| bound_names(elem, names)),
        Pat::TupleStruct(tuple) => tuple.elems.iter().for_each(|elem| bound_names(elem, names)),
        Pat::Struct(fields) => fields
            .fields
            .iter()
            .for_each(|field| bound_names(&field.pat, names)),
        Pat::Reference(reference) => bound_names(&reference.pat, names),
        Pat::Slice(slice) => slice.elems.iter().for_each(|elem| bound_names(elem, names)),
        _ => {}
    }
}

/// Whether `expr` is a recorded document's JSON object: a method chain
/// through `as_object` (and `unwrap`, `expect`, `clone`, `cloned`, `?`) on a
/// recorded document.
fn is_recorded_object(expr: &Expr, bindings: &[String]) -> bool {
    fn walk(expr: &Expr, bindings: &[String], seen_object: bool) -> bool {
        match expr {
            Expr::MethodCall(call) => {
                let method = call.method.to_string();
                match method.as_str() {
                    "as_object" => walk(&call.receiver, bindings, true),
                    "unwrap" | "expect" | "clone" | "cloned" => {
                        walk(&call.receiver, bindings, seen_object)
                    }
                    _ => false,
                }
            }
            Expr::Reference(reference) => walk(&reference.expr, bindings, seen_object),
            Expr::Paren(paren) => walk(&paren.expr, bindings, seen_object),
            Expr::Try(attempt) => walk(&attempt.expr, bindings, seen_object),
            expr => seen_object && is_recorded_document(expr, bindings),
        }
    }
    walk(expr, bindings, false)
}

/// Whether a pattern selects replay only: it names `Replay`, never `Record`,
/// and every or-pattern in it, at any depth, has `Replay` in each
/// alternative. A pattern over one mode value matches both modes only
/// through an or-pattern (`Replay | Record`, `Some(Replay | _)`); one over
/// several (`(Record, Replay)`) can name both without one, so a pattern
/// naming `Record` is never replay-only.
fn is_replay_pattern(pattern: &Pat) -> bool {
    struct EveryAlternative(bool);
    impl<'ast> Visit<'ast> for EveryAlternative {
        fn visit_pat_or(&mut self, node: &'ast syn::PatOr) {
            if !node
                .cases
                .iter()
                .all(|case| mentions(&quote::quote!(#case), "Replay"))
            {
                self.0 = false;
            }
            visit::visit_pat_or(self, node);
        }
    }
    let tokens = quote::quote!(#pattern);
    let mut every = EveryAlternative(true);
    every.visit_pat(pattern);
    every.0 && mentions(&tokens, "Replay") && !mentions(&tokens, "Record")
}

/// The one name a pattern binds when it binds exactly one, directly or
/// through `Some(..)`/`Ok(..)`: `object` in `let Some(object) = … else`.
fn single_binding(pattern: &Pat) -> Option<String> {
    match pattern {
        Pat::Ident(ident) => Some(ident.ident.to_string()),
        Pat::Type(typed) => single_binding(&typed.pat),
        Pat::Reference(reference) => single_binding(&reference.pat),
        Pat::TupleStruct(tuple)
            if tuple.elems.len() == 1
                && tuple
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "Some" || segment.ident == "Ok") =>
        {
            tuple.elems.first().and_then(single_binding)
        }
        _ => None,
    }
}

/// Whether `mac`'s first two arguments include `None`: a presence check,
/// which holds in both modes.
fn checks_presence(mac: &Macro) -> bool {
    mac.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
        .is_ok_and(|args| {
            args.iter()
                .take(2)
                .any(|arg| matches!(arg, Expr::Path(path) if path.path.is_ident("None")))
        })
}

/// The document a method chain reads from: `recorded.as_object().unwrap()`
/// is `recorded`.
fn root_of(expr: &Expr) -> &Expr {
    match expr {
        Expr::MethodCall(call) => root_of(&call.receiver),
        Expr::Reference(reference) => root_of(&reference.expr),
        Expr::Paren(paren) => root_of(&paren.expr),
        Expr::Try(attempt) => root_of(&attempt.expr),
        expr => expr,
    }
}

/// Whether an `if` condition rules replay out, so its `else` runs in replay:
/// `!= …Replay`, `!matches!(…, …Replay)`, `!(… == …Replay)`, alone or in a
/// disjunction.
fn is_negated_replay_condition(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(binary) => match binary.op {
            syn::BinOp::Ne(_) => {
                let (left, right) = (&binary.left, &binary.right);
                mentions(&quote::quote!(#left), "Replay")
                    || mentions(&quote::quote!(#right), "Replay")
            }
            syn::BinOp::Or(_) => {
                is_negated_replay_condition(&binary.left)
                    || is_negated_replay_condition(&binary.right)
            }
            _ => false,
        },
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Not(_)) => {
            is_replay_condition(&unary.expr)
        }
        Expr::Paren(paren) => is_negated_replay_condition(&paren.expr),
        _ => false,
    }
}

/// Whether `expr` iterates a literal list of keys naming one of `volatile`:
/// `["created", "id"]`, borrowed or through `.iter()`.
fn iterates_volatile_literals(expr: &Expr, volatile: &[&str]) -> bool {
    match expr {
        Expr::Array(array) => array.elems.iter().any(|element| {
            matches!(element, Expr::Lit(ExprLit { lit: Lit::Str(key), .. })
                if volatile.iter().any(|volatile| volatile.eq_ignore_ascii_case(&key.value())))
        }),
        Expr::Reference(reference) => iterates_volatile_literals(&reference.expr, volatile),
        Expr::MethodCall(call) if call.method == "iter" || call.method == "into_iter" => {
            iterates_volatile_literals(&call.receiver, volatile)
        }
        _ => false,
    }
}

/// Whether `mac` is an `assert!` over an `==` or `!=` comparison.
fn is_assert_comparison(mac: &Macro) -> bool {
    mac.path.is_ident("assert")
        && mac
            .parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
            .is_ok_and(|args| {
                matches!(args.first(), Some(Expr::Binary(binary))
                    if matches!(binary.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_)))
            })
}

/// Whether `expr` reads a value out of a JSON document: an index or a
/// `.get(…)`, possibly borrowed or cloned. A struct field (`observation.text`)
/// is a decoded value, not a document.
fn is_document_access(expr: &Expr) -> bool {
    match expr {
        Expr::Index(_) => true,
        Expr::MethodCall(call) if call.method == "get" => true,
        Expr::MethodCall(call) if call.method == "clone" => is_document_access(&call.receiver),
        Expr::Reference(reference) => is_document_access(&reference.expr),
        Expr::Paren(paren) => is_document_access(&paren.expr),
        _ => false,
    }
}

fn compares_recorded_document(mac: &Macro, bindings: &[String]) -> bool {
    mac.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
        .is_ok_and(|args| {
            let mut operands = args.iter().take(2);
            match (operands.next(), operands.next()) {
                (Some(left), Some(right)) => {
                    (is_recorded_document(left, bindings) && is_document_access(right))
                        || (is_recorded_document(right, bindings) && is_document_access(left))
                }
                _ => false,
            }
        })
}

fn is_assert_eq(mac: &Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "assert_eq" || segment.ident == "assert_ne")
}

impl GuardVisitor<'_> {
    /// Remove every name `pattern` binds from the recorded bindings.
    fn shadow(&mut self, pattern: &Pat) {
        let mut names = Vec::new();
        bound_names(pattern, &mut names);
        self.recorded_bindings
            .retain(|bound| !names.contains(bound));
        self.recorded_objects.retain(|bound| !names.contains(bound));
    }

    /// Bind `pattern` to `init`: judge `init` against the bindings as they
    /// stand, shadow every name the pattern binds, then record its one name
    /// when `init` is a recorded document or object. The order matters for
    /// `let body = body.as_object()…`.
    fn bind(&mut self, pattern: &Pat, init: &Expr) {
        let recorded = is_recorded_document(init, &self.recorded_bindings);
        let object = is_recorded_object(init, &self.recorded_bindings);
        self.shadow(pattern);
        if let Some(name) = single_binding(pattern) {
            if recorded {
                self.recorded_bindings.push(name.clone());
            }
            if object {
                self.recorded_objects.push(name);
            }
        }
    }

    /// Apply every `let` of an `if` condition, in order: a bare `if let` or
    /// each `let` of a let-chain (`if let Some(x) = … && ready`).
    fn bind_condition(&mut self, cond: &Expr) {
        match cond {
            Expr::Let(binding) => self.bind(&binding.pat, &binding.expr),
            Expr::Binary(binary) if matches!(binary.op, syn::BinOp::And(_)) => {
                // Left first: a later `let` may read a name an earlier one bound.
                self.bind_condition(&binary.left);
                self.bind_condition(&binary.right);
            }
            _ => {}
        }
    }

    /// Whether `expr` is a local bound to a recorded document's JSON object.
    fn names_recorded_object(&self, expr: &Expr) -> bool {
        matches!(root_of(expr), Expr::Path(path)
        if path.path.get_ident().is_some_and(|ident| {
            self.recorded_objects.iter().any(|name| ident == name)
        }))
    }
}

impl<'ast> Visit<'ast> for GuardVisitor<'_> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let outer = std::mem::replace(&mut self.function, node.sig.ident.to_string());
        let bindings = std::mem::take(&mut self.recorded_bindings);
        let objects = std::mem::take(&mut self.recorded_objects);
        visit::visit_item_fn(self, node);
        self.recorded_bindings = bindings;
        self.recorded_objects = objects;
        self.function = outer;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let outer = std::mem::replace(&mut self.function, node.sig.ident.to_string());
        let bindings = std::mem::take(&mut self.recorded_bindings);
        let objects = std::mem::take(&mut self.recorded_objects);
        visit::visit_impl_item_fn(self, node);
        self.recorded_bindings = bindings;
        self.recorded_objects = objects;
        self.function = outer;
    }

    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let Some(init) = &node.init {
            self.bind(&node.pat, &init.expr);
        } else {
            self.shadow(&node.pat);
        }
        visit::visit_local(self, node);
    }

    fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
        let scrutinee = &node.expr;
        let on_mode = mentions(&quote::quote!(#scrutinee), "CassetteMode")
            || node.arms.iter().any(|arm| {
                let pattern = &arm.pat;
                mentions(&quote::quote!(#pattern), "CassetteMode")
            });
        self.visit_expr(&node.expr);
        for arm in &node.arms {
            let replay = on_mode && is_replay_pattern(&arm.pat);
            self.replay_depth += usize::from(replay);
            visit::visit_arm(self, arm);
            self.replay_depth -= usize::from(replay);
        }
    }

    fn visit_expr_if(&mut self, node: &'ast ExprIf) {
        let replay = is_replay_condition(&node.cond);
        self.visit_expr(&node.cond);
        // `if let Some(object) = recorded.as_object() { … }` binds a recorded
        // object (or document) for the `then` block only.
        let (bindings, objects) = (
            self.recorded_bindings.clone(),
            self.recorded_objects.clone(),
        );
        self.bind_condition(&node.cond);
        self.replay_depth += usize::from(replay);
        self.visit_block(&node.then_branch);
        self.replay_depth -= usize::from(replay);
        self.recorded_bindings = bindings;
        self.recorded_objects = objects;
        if let Some((_, otherwise)) = &node.else_branch {
            // The `else` of a condition that rules replay out runs in replay.
            let replay = is_negated_replay_condition(&node.cond);
            self.replay_depth += usize::from(replay);
            self.visit_expr(otherwise);
            self.replay_depth -= usize::from(replay);
        }
    }

    fn visit_expr_for_loop(&mut self, node: &'ast ExprForLoop) {
        let iterated = &node.expr;
        let iterates_keys = mentions(&quote::quote!(#iterated), "keys")
            || iterates_volatile_literals(&node.expr, self.volatile);
        if iterates_keys
            && self.replay_depth == 0
            && let Pat::Ident(pat) = node.pat.as_ref()
        {
            let block = &node.body;
            let body = quote::quote!(#block);
            let name = pat.ident.to_string();
            let compares = accesses_ident(&body, &name) >= 2
                && (mentions(&body, "assert_eq") || mentions(&body, "assert_ne"));
            if compares && !calls(&body, "is_volatile_json_key") {
                self.findings.push(format!(
                    "{}: compares a document key by key without treating volatile keys; use \
                     assert_matches_recorded_document",
                    self.function
                ));
            }
        }
        // `for (key, value) in recorded.as_object()…`: each recorded value
        // compared with a live document's entry at `key`, or the reverse.
        if self.replay_depth == 0
            && let Pat::Tuple(tuple) = node.pat.as_ref()
            && let [Pat::Ident(key), Pat::Ident(value)] = tuple.elems.iter().collect::<Vec<_>>()[..]
            && (mentions(&quote::quote!(#iterated), "as_object")
                || self.names_recorded_object(&node.expr))
        {
            let block = &node.body;
            let body = quote::quote!(#block);
            let involves_recorded =
                is_recorded_document(root_of(&node.expr), &self.recorded_bindings)
                    || is_recorded_object(&node.expr, &self.recorded_bindings)
                    || self.names_recorded_object(&node.expr)
                    || self
                        .recorded_bindings
                        .iter()
                        .any(|name| mentions(&body, name));
            let compares = involves_recorded
                && accesses_ident(&body, &key.ident.to_string()) >= 1
                && mentions(&body, &value.ident.to_string())
                && (mentions(&body, "assert_eq") || mentions(&body, "assert_ne"));
            if compares && !calls(&body, "is_volatile_json_key") {
                self.findings.push(format!(
                    "{}: compares a document entry by entry without treating volatile keys; \
                     use assert_matches_recorded_document",
                    self.function
                ));
            }
        }
        visit::visit_expr_for_loop(self, node);
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        if (is_assert_eq(node) || is_assert_comparison(node))
            && self.replay_depth == 0
            && !checks_presence(node)
            && accesses_volatile(&node.tokens, self.volatile)
        {
            self.findings.push(format!(
                "{}: compares a volatile key exactly outside replay; use \
                 assert_wire_value_matches or assert_matches_recorded_document",
                self.function
            ));
        }
        if is_assert_eq(node)
            && self.replay_depth == 0
            && compares_recorded_document(node, &self.recorded_bindings)
        {
            self.findings.push(format!(
                "{}: compares a whole recorded document exactly outside replay; use \
                 assert_matches_recorded_document",
                self.function
            ));
        }
        visit::visit_macro(self, node);
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        visit::visit_expr(self, node);
    }
}

#[cfg(test)]
mod tests;
