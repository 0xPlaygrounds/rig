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

/// Comparisons in `source` that treat a volatile key exactly outside replay.
/// An exact comparison is an `assert_eq!`/`assert_ne!`, or an `assert!` over
/// `==` or `!=`; one with `None` on a side (a presence check) is exempt.
/// Flagged: a loop over a document's `.keys()`, or over a literal key list
/// naming a volatile key, comparing `get(key)` values; a `for (key, value)`
/// loop over a JSON object comparing entries at `key`, when the object or
/// the other side is a recorded document; a comparison that indexes or
/// `get`s a key in `volatile`; and one of a whole recorded document (a call
/// to a `recorded_*` function, or a binding to one, seen through
/// `.as_object()`, `unwrap`, `expect` or `?`) with a value read out of a live
/// document by index or `.get` (`observation.raw["extras"]`). A loop that
/// calls `is_volatile_json_key` is exempt.
///
/// A `CassetteMode::Replay` arm or branch, and the `else` of a condition
/// that rules replay out, run in replay. A pattern selects replay only when
/// it never names `Record` and cannot match without `Replay`. A name bound by
/// any pattern (`let`, `if let`, `while let`, a loop, a match arm, a closure
/// parameter) is no longer a recorded binding within that pattern's scope.
/// Keys match in any ASCII case. Each finding names its function or method.
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
        Expr::MethodCall(call)
            if ["clone", "as_object", "unwrap", "expect"]
                .iter()
                .any(|method| call.method == method) =>
        {
            is_recorded_document(&call.receiver, bindings)
        }
        Expr::Try(attempt) => is_recorded_document(&attempt.expr, bindings),
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

/// Whether an `if` condition rules replay out, so its `else` runs only in
/// replay: `!= …Replay`, or the negation of a replay condition, alone or in
/// a disjunction.
fn rules_replay_out(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(binary) => match binary.op {
            syn::BinOp::Ne(_) => {
                let (left, right) = (&binary.left, &binary.right);
                mentions(&quote::quote!(#left), "Replay")
                    || mentions(&quote::quote!(#right), "Replay")
            }
            syn::BinOp::Or(_) => rules_replay_out(&binary.left) || rules_replay_out(&binary.right),
            _ => false,
        },
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Not(_)) => {
            is_replay_condition(&unary.expr)
        }
        Expr::Paren(paren) => rules_replay_out(&paren.expr),
        _ => false,
    }
}

/// The pattern of `matches!(expression, pattern)`, ignoring any guard.
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

/// Whether a pattern selects replay only: it never names `Record` (a
/// pattern over several modes may name both), and it cannot match a value
/// without matching `Replay` in it.
fn is_replay_pattern(pattern: &Pat) -> bool {
    !mentions(&quote::quote!(#pattern), "Record") && requires_replay(pattern)
}

/// Whether `pattern` matches only values holding `Replay`: a `Replay` path,
/// an or-pattern each of whose alternatives does, or a tuple, struct or
/// reference one of whose parts does.
fn requires_replay(pattern: &Pat) -> bool {
    match pattern {
        Pat::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "Replay"),
        Pat::Ident(ident) => ident.ident == "Replay",
        Pat::Or(alternatives) => alternatives.cases.iter().all(requires_replay),
        Pat::Tuple(tuple) => tuple.elems.iter().any(requires_replay),
        Pat::TupleStruct(tuple) => tuple.elems.iter().any(requires_replay),
        Pat::Struct(fields) => fields
            .fields
            .iter()
            .any(|field| requires_replay(&field.pat)),
        Pat::Reference(reference) => requires_replay(&reference.pat),
        Pat::Paren(paren) => requires_replay(&paren.pat),
        _ => false,
    }
}

/// Every name `pattern` binds.
fn bound_names(pattern: &Pat) -> Vec<String> {
    struct Names(Vec<String>);
    impl<'ast> Visit<'ast> for Names {
        fn visit_pat_ident(&mut self, node: &'ast syn::PatIdent) {
            self.0.push(node.ident.to_string());
            visit::visit_pat_ident(self, node);
        }
    }
    let mut names = Names(Vec::new());
    names.visit_pat(pattern);
    names.0
}

/// The two sides `mac` compares exactly: the first two arguments of an
/// `assert_eq!`/`assert_ne!`, or the operands of an `assert!` over `==` or
/// `!=`.
fn compared_operands(mac: &Macro) -> Option<(Expr, Expr)> {
    let args = mac
        .parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
        .ok()?;
    let mut args = args.into_iter();
    if is_assert_eq(mac) {
        return args.next().zip(args.next());
    }
    match args.next() {
        Some(Expr::Binary(binary))
            if mac.path.is_ident("assert")
                && matches!(binary.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_)) =>
        {
            Some((*binary.left, *binary.right))
        }
        _ => None,
    }
}

/// Whether one side of `mac`'s comparison is `None`: a presence check,
/// which holds in both modes.
fn checks_presence(mac: &Macro) -> bool {
    compared_operands(mac).is_some_and(|(left, right)| {
        [left, right]
            .iter()
            .any(|side| matches!(side, Expr::Path(path) if path.path.is_ident("None")))
    })
}

/// Whether `block` holds an exact comparison anywhere in it.
fn compares_exactly(block: &syn::Block) -> bool {
    struct Finds(bool);
    impl<'ast> Visit<'ast> for Finds {
        fn visit_macro(&mut self, node: &'ast Macro) {
            self.0 |= compared_operands(node).is_some();
        }
    }
    let mut finds = Finds(false);
    finds.visit_block(block);
    finds.0
}

/// `expr` without a trailing `.iter()` or `.into_iter()`.
fn without_iter(expr: &Expr) -> &Expr {
    match expr {
        Expr::MethodCall(call) if call.method == "iter" || call.method == "into_iter" => {
            without_iter(&call.receiver)
        }
        expr => expr,
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
    compared_operands(mac).is_some_and(|(left, right)| {
        (is_recorded_document(&left, bindings) && is_document_access(&right))
            || (is_recorded_document(&right, bindings) && is_document_access(&left))
    })
}

fn is_assert_eq(mac: &Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "assert_eq" || segment.ident == "assert_ne")
}

impl GuardVisitor<'_> {
    /// Bind `pattern` to `init`: every name it binds shadows an earlier
    /// recorded binding, and its one name, if it binds one, holds a recorded
    /// document when `init` is one.
    fn bind(&mut self, pattern: &Pat, init: Option<&Expr>) {
        let recorded = init.is_some_and(|init| is_recorded_document(init, &self.recorded_bindings));
        let names = bound_names(pattern);
        self.recorded_bindings
            .retain(|bound| !names.contains(bound));
        if recorded && let [name] = names.as_slice() {
            self.recorded_bindings.push(name.clone());
        }
    }

    /// Bind each `let` of an `if` condition, left to right.
    fn bind_condition(&mut self, cond: &Expr) {
        match cond {
            Expr::Let(binding) => self.bind(&binding.pat, Some(&binding.expr)),
            Expr::Binary(binary) if matches!(binary.op, syn::BinOp::And(_)) => {
                self.bind_condition(&binary.left);
                self.bind_condition(&binary.right);
            }
            Expr::Paren(paren) => self.bind_condition(&paren.expr),
            _ => {}
        }
    }
}

impl<'ast> Visit<'ast> for GuardVisitor<'_> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let outer = std::mem::replace(&mut self.function, node.sig.ident.to_string());
        let bindings = std::mem::take(&mut self.recorded_bindings);
        visit::visit_item_fn(self, node);
        self.recorded_bindings = bindings;
        self.function = outer;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let outer = std::mem::replace(&mut self.function, node.sig.ident.to_string());
        let bindings = std::mem::take(&mut self.recorded_bindings);
        visit::visit_impl_item_fn(self, node);
        self.recorded_bindings = bindings;
        self.function = outer;
    }

    fn visit_local(&mut self, node: &'ast syn::Local) {
        // The initializer reads the bindings from before this `let`.
        if let Some(init) = &node.init {
            self.visit_local_init(init);
        }
        self.bind(&node.pat, node.init.as_ref().map(|init| init.expr.as_ref()));
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
            let bindings = self.recorded_bindings.clone();
            self.bind(&arm.pat, None);
            self.replay_depth += usize::from(replay);
            visit::visit_arm(self, arm);
            self.replay_depth -= usize::from(replay);
            self.recorded_bindings = bindings;
        }
    }

    /// A `let` inside a block binds until the block ends.
    fn visit_block(&mut self, node: &'ast syn::Block) {
        let bindings = self.recorded_bindings.clone();
        visit::visit_block(self, node);
        self.recorded_bindings = bindings;
    }

    fn visit_expr_if(&mut self, node: &'ast ExprIf) {
        let replay = is_replay_condition(&node.cond);
        self.visit_expr(&node.cond);
        // An `if let` binds for the `then` block only.
        let bindings = self.recorded_bindings.clone();
        self.bind_condition(&node.cond);
        self.replay_depth += usize::from(replay);
        self.visit_block(&node.then_branch);
        self.replay_depth -= usize::from(replay);
        self.recorded_bindings = bindings;
        if let Some((_, otherwise)) = &node.else_branch {
            let replay = rules_replay_out(&node.cond);
            self.replay_depth += usize::from(replay);
            self.visit_expr(otherwise);
            self.replay_depth -= usize::from(replay);
        }
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.visit_expr(&node.cond);
        let bindings = self.recorded_bindings.clone();
        self.bind_condition(&node.cond);
        self.visit_block(&node.body);
        self.recorded_bindings = bindings;
    }

    fn visit_expr_closure(&mut self, node: &'ast syn::ExprClosure) {
        let bindings = self.recorded_bindings.clone();
        for input in &node.inputs {
            self.bind(input, None);
        }
        self.visit_expr(&node.body);
        self.recorded_bindings = bindings;
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
            let compares = accesses_ident(&body, &name) >= 2 && compares_exactly(&node.body);
            if compares && !calls(&body, "is_volatile_json_key") {
                self.findings.push(format!(
                    "{}: compares a document key by key without treating volatile keys; use \
                     assert_matches_recorded_document",
                    self.function
                ));
            }
        }
        // The loop's pattern binds for its body only.
        self.visit_expr(&node.expr);
        let bindings = self.recorded_bindings.clone();
        let iterates_recorded = is_recorded_document(without_iter(&node.expr), &bindings);
        self.bind(&node.pat, None);
        if self.replay_depth == 0
            && let Pat::Tuple(tuple) = node.pat.as_ref()
            && let [Pat::Ident(key), Pat::Ident(value)] = tuple.elems.iter().collect::<Vec<_>>()[..]
        {
            let block = &node.body;
            let body = quote::quote!(#block);
            let recorded = iterates_recorded
                || (mentions(&quote::quote!(#iterated), "as_object")
                    && self
                        .recorded_bindings
                        .iter()
                        .any(|name| mentions(&body, name)));
            let compares = accesses_ident(&body, &key.ident.to_string()) >= 1
                && mentions(&body, &value.ident.to_string())
                && compares_exactly(&node.body);
            if recorded && compares && !calls(&body, "is_volatile_json_key") {
                self.findings.push(format!(
                    "{}: compares a document entry by entry without treating volatile keys; \
                     use assert_matches_recorded_document",
                    self.function
                ));
            }
        }
        self.visit_block(&node.body);
        self.recorded_bindings = bindings;
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        if compared_operands(node).is_some()
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
        if self.replay_depth == 0 && compares_recorded_document(node, &self.recorded_bindings) {
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
