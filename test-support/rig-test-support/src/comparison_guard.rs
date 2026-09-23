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
/// a loop over a document's `.keys()`, or over a literal key list naming a
/// volatile key, comparing `get(key)` values without calling
/// `is_volatile_json_key`; an `assert_eq!`/`assert_ne!` (or an `assert!`
/// with `==`/`!=`) that indexes or `get`s a key in `volatile` outside a
/// `CassetteMode::Replay` arm or branch; and one that compares a whole
/// recorded document (a call to a `recorded_*` function, or a `let` bound to
/// one, not narrowed to one key) exactly with a value read out of a live
/// document by index or `.get` (`observation.raw["extras"]`) outside replay.
/// Keys match in any ASCII case, and a comparison with `None` (a presence
/// check) is exempt. Each finding names its function or method.
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
        Expr::Macro(mac) => mac.mac.path.is_ident("matches") && mentions(&mac.mac.tokens, "Replay"),
        Expr::Let(binding) => {
            let pattern = &binding.pat;
            mentions(&quote::quote!(#pattern), "Replay")
        }
        Expr::Paren(paren) => is_replay_condition(&paren.expr),
        _ => false,
    }
}

/// Whether a match arm's pattern selects replay only: `Replay`, or an
/// or-pattern every alternative of which is `Replay`.
fn is_replay_pattern(pattern: &Pat) -> bool {
    match pattern {
        Pat::Or(alternatives) => alternatives.cases.iter().all(is_replay_pattern),
        pattern => mentions(&quote::quote!(#pattern), "Replay"),
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
        if let Some(init) = &node.init
            && is_recorded_document(&init.expr, &self.recorded_bindings)
            && let Pat::Ident(pattern) = &node.pat
        {
            self.recorded_bindings.push(pattern.ident.to_string());
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
        self.replay_depth += usize::from(replay);
        self.visit_block(&node.then_branch);
        self.replay_depth -= usize::from(replay);
        if let Some((_, otherwise)) = &node.else_branch {
            self.visit_expr(otherwise);
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
