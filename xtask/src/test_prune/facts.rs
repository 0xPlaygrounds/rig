//! What the prune reads from a test function's source: its size, whether it
//! is table-driven, its assertions, and whether it asserts a contract that
//! line coverage and the mutation sample cannot see.
//!
//! The contract rule is read from tokens, so it over-approximates: a test it
//! names as a contract is kept, never the reverse. A contract test may still
//! go when each assertion that makes it one appears, token for token, in an
//! earlier kept contract test.

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens as _;
use syn::spanned::Spanned as _;
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, ExprForLoop, ItemFn};

/// Why a test is a contract test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Reason {
    /// Runs on wasm, whose coverage no native run measures.
    Wasm,
    /// A compile-fail or trybuild case.
    CompileFail,
    /// Asserts a public API shape.
    ApiShape,
    /// A security, redaction or scrub test.
    Security,
    /// A serde round trip of a stored or wire format.
    StoredFormat,
    /// Asserts an error message.
    ErrorMessage,
    /// Asserts rendered text: a `to_string` or `format!` against a literal.
    RenderedText,
    /// Named by another file: a findings registry, a contract table or a
    /// doc comment cites it as the test of what it states.
    Cited,
}

impl Reason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Wasm => "wasm",
            Self::CompileFail => "compile-fail",
            Self::ApiShape => "api-shape",
            Self::Security => "security",
            Self::StoredFormat => "stored-format",
            Self::ErrorMessage => "error-message",
            Self::RenderedText => "rendered-text",
            Self::Cited => "cited",
        }
    }
}

/// A contract test: its reason, and the assertions whose identical copy
/// elsewhere would let it go (none: it never goes). Only an error-message or
/// rendered-text contract is its assertions; any other is the whole test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Contract {
    pub(crate) reason: Reason,
    pub(crate) assertions: Vec<String>,
}

/// The facts of one test function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Facts {
    /// Source lines, attributes included.
    pub(crate) lines: usize,
    /// Loops over a table of cases.
    pub(crate) table: bool,
    pub(crate) contract: Option<Contract>,
    /// Every assertion, token-normalized.
    pub(crate) assertions: BTreeSet<String>,
}

/// Name fragments that mark a contract test, by reason.
const NAMED: &[(Reason, &[&str])] = &[
    (
        Reason::ApiShape,
        &[
            "api_surface",
            "public_api",
            "reexport",
            "re_export",
            "facade",
        ],
    ),
    (
        Reason::Security,
        &[
            "scrub",
            "redact",
            "secret",
            "credential",
            "sanitiz",
            "password",
            "api_key",
            "apikey",
            "leak",
            "security",
            "authoriz",
            "authenticat",
        ],
    ),
    (
        Reason::StoredFormat,
        &[
            "round_trip",
            "roundtrip",
            "serde",
            "serializ",
            "deserializ",
            "persist",
            "stored",
            "golden",
        ],
    ),
];

/// Body identifiers that mark a security test.
const SECURITY_IDENTS: &[&str] = &["scrub", "redact", "sanitiz"];

/// Identifiers that render a value as text.
const RENDERS: &[&str] = &[
    "to_string",
    "format",
    "display",
    "contains",
    "starts_with",
    "ends_with",
];

const SERIALIZE: &[&str] = &[
    "to_string",
    "to_string_pretty",
    "to_value",
    "to_vec",
    "to_vec_pretty",
    "to_writer",
];
const DESERIALIZE: &[&str] = &["from_str", "from_value", "from_slice", "from_reader"];

/// Literal fragments that name a committed fixture or effect log.
/// A recorded cassette counts too: the cassette prune keeps a fixture a test
/// outside it names, so deleting that test would change what it keeps.
const FIXTURE_PATHS: &[&str] = &["fixtures/", ".effects.json", ".programs.json", ".yaml"];

/// The facts of `item`, a test named `name` (binary id and test path).
/// `wasm_gated` says an enclosing module or file only builds for wasm.
pub(crate) fn of(name: &str, item: &ItemFn, wasm_gated: bool) -> Facts {
    let start = item
        .attrs
        .iter()
        .map(|attr| attr.span().start().line)
        .chain([item.sig.span().start().line])
        .min()
        .unwrap_or(0);
    let lines = item.block.span().end().line.saturating_sub(start) + 1;
    let mut tables = Tables::default();
    tables.visit_item_fn(item);
    let mut scan = Scan::default();
    scan.walk(item.block.to_token_stream());
    let assertions: BTreeSet<String> = scan
        .assertions
        .iter()
        .map(|assertion| assertion.text.clone())
        .collect();
    let contract = contract(name, item, wasm_gated, tables.found, &scan);
    Facts {
        lines,
        table: tables.found,
        contract,
        assertions,
    }
}

fn contract(
    name: &str,
    item: &ItemFn,
    wasm_gated: bool,
    table: bool,
    scan: &Scan,
) -> Option<Contract> {
    let never = |reason| {
        Some(Contract {
            reason,
            assertions: Vec::new(),
        })
    };
    if wasm_gated || item.attrs.iter().any(names_wasm_test) || scan.has("wasm_bindgen_test") {
        return never(Reason::Wasm);
    }
    if scan.has("trybuild") || scan.has("compile_fail") {
        return never(Reason::CompileFail);
    }
    let lowered = name.to_lowercase();
    for (reason, fragments) in NAMED {
        if fragments.iter().any(|fragment| lowered.contains(fragment)) {
            return never(*reason);
        }
    }
    // Nothing fails at run time, so what it checks is that its paths and
    // types resolve.
    if scan.assertions.is_empty() && !scan.checks {
        return never(Reason::ApiShape);
    }
    if SECURITY_IDENTS
        .iter()
        .any(|fragment| scan.idents.iter().any(|ident| ident.contains(fragment)))
    {
        return never(Reason::Security);
    }
    // A golden or a recorded fixture is a stored format the test pins.
    let pins_a_file = scan.idents.iter().any(|ident| ident.contains("golden"))
        || scan.literals.iter().any(|literal| {
            FIXTURE_PATHS
                .iter()
                .any(|fragment| literal.contains(fragment))
        });
    if pins_a_file {
        return never(Reason::StoredFormat);
    }
    let serializes = SERIALIZE.iter().any(|call| scan.serde.contains(*call));
    let deserializes = DESERIALIZE.iter().any(|call| scan.serde.contains(*call));
    if serializes && deserializes {
        return never(Reason::StoredFormat);
    }
    let errs = scan.idents.iter().any(|ident| ident.contains("err"));
    let renders = RENDERS.iter().any(|render| scan.has(render));
    // In a table-driven test the literals are the table's, so an assertion
    // that matches text against a case stands for one that holds a literal.
    let from_table =
        |assertion: &&Assertion| table && assertion.matches_text && !scan.literals.is_empty();
    let literal: Vec<String> = scan
        .assertions
        .iter()
        .filter(|assertion| assertion.literal || from_table(assertion))
        .map(|assertion| assertion.text.clone())
        .collect();
    if errs && renders && !literal.is_empty() {
        return Some(Contract {
            reason: Reason::ErrorMessage,
            assertions: literal,
        });
    }
    let rendered: Vec<String> = scan
        .assertions
        .iter()
        .filter(|assertion| (assertion.literal || from_table(assertion)) && assertion.renders)
        .map(|assertion| assertion.text.clone())
        .collect();
    if !rendered.is_empty() {
        return Some(Contract {
            reason: Reason::RenderedText,
            assertions: rendered,
        });
    }
    None
}

/// Whether an attribute names `wasm_bindgen_test`, directly or in a
/// `cfg_attr`.
fn names_wasm_test(attr: &Attribute) -> bool {
    let mut scan = Scan::default();
    scan.walk(attr.meta.to_token_stream());
    scan.has("wasm_bindgen_test")
}

/// Whether `attrs` gate an item to wasm: a `cfg` with a wasm target
/// predicate outside any `not(...)`.
pub(crate) fn gates_wasm(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && match &attr.meta {
                syn::Meta::List(list) => positive_wasm(list.tokens.clone()),
                _ => false,
            }
    })
}

fn positive_wasm(tokens: TokenStream) -> bool {
    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut index = 0;
    while let Some(tree) = trees.get(index) {
        match tree {
            TokenTree::Ident(ident) if ident == "not" => {
                // Skip the negated group.
                index += 2;
                continue;
            }
            TokenTree::Ident(ident) if ident == "target_family" || ident == "target_arch" => {
                if let Some(TokenTree::Literal(value)) = trees.get(index + 2)
                    && value.to_string().contains("wasm")
                {
                    return true;
                }
            }
            TokenTree::Group(group) if positive_wasm(group.stream()) => return true,
            _ => {}
        }
        index += 1;
    }
    false
}

/// One assertion macro: its normalized text, and whether its asserted
/// arguments (not its panic message) hold a string literal and render text.
struct Assertion {
    text: String,
    literal: bool,
    renders: bool,
    /// Its asserted arguments render or match text.
    matches_text: bool,
}

/// Identifiers that fail a test at run time besides an assertion.
const CHECKS: &[&str] = &[
    "unwrap",
    "expect",
    "unwrap_err",
    "expect_err",
    "panic",
    "unreachable",
];

/// Identifiers that render a value as text inside an assertion.
const RENDERED: &[&str] = &["to_string", "format", "display"];

/// What a token walk of the body finds.
#[derive(Default)]
struct Scan {
    /// Every identifier, lowercased.
    idents: BTreeSet<String>,
    /// The functions called as `serde_json::f` or `serde_yaml::f`.
    serde: BTreeSet<String>,
    assertions: Vec<Assertion>,
    /// A `?` or a call that panics on failure.
    checks: bool,
    /// Every string literal's source text.
    literals: Vec<String>,
}

impl Scan {
    fn has(&self, ident: &str) -> bool {
        self.idents.contains(ident)
    }

    fn walk(&mut self, tokens: TokenStream) {
        let trees: Vec<TokenTree> = tokens.into_iter().collect();
        for (index, tree) in trees.iter().enumerate() {
            match tree {
                TokenTree::Ident(ident) => {
                    let text = ident.to_string();
                    if (text == "serde_json" || text == "serde_yaml")
                        && let (
                            Some(TokenTree::Punct(first)),
                            Some(TokenTree::Punct(second)),
                            Some(TokenTree::Ident(call)),
                        ) = (
                            trees.get(index + 1),
                            trees.get(index + 2),
                            trees.get(index + 3),
                        )
                        && first.as_char() == ':'
                        && second.as_char() == ':'
                    {
                        self.serde.insert(call.to_string());
                    }
                    if is_assertion(&text)
                        && let (Some(TokenTree::Punct(bang)), Some(TokenTree::Group(args))) =
                            (trees.get(index + 1), trees.get(index + 2))
                        && bang.as_char() == '!'
                    {
                        let asserted = asserted(&text, args.stream());
                        self.assertions.push(Assertion {
                            text: format!("{text}!({})", args.stream()),
                            literal: asserted.iter().any(holds_string),
                            renders: asserted.iter().any(|tree| holds_ident(tree, RENDERED)),
                            matches_text: asserted.iter().any(|tree| holds_ident(tree, RENDERS)),
                        });
                    }
                    if CHECKS.contains(&text.as_str()) {
                        self.checks = true;
                    }
                    self.idents.insert(text.to_lowercase());
                }
                TokenTree::Group(group) => self.walk(group.stream()),
                TokenTree::Punct(punct) => {
                    if punct.as_char() == '?' {
                        self.checks = true;
                    }
                }
                TokenTree::Literal(literal) => self.literals.push(literal.to_string()),
            }
        }
    }
}

fn is_assertion(name: &str) -> bool {
    ["assert", "prop_assert", "debug_assert"]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

/// The asserted arguments of `name!(args)`: the first argument, or the
/// first two of an `_eq`, `_ne` or `_matches` form; the rest is the panic
/// message.
fn asserted(name: &str, args: TokenStream) -> Vec<TokenTree> {
    let count = if name.ends_with("_eq") || name.ends_with("_ne") || name.ends_with("_matches") {
        2
    } else {
        1
    };
    let mut argument = 0;
    let mut out = Vec::new();
    for tree in args {
        if matches!(&tree, TokenTree::Punct(punct) if punct.as_char() == ',') {
            argument += 1;
            if argument >= count {
                break;
            }
            continue;
        }
        out.push(tree);
    }
    out
}

fn holds_ident(tree: &TokenTree, names: &[&str]) -> bool {
    match tree {
        TokenTree::Ident(ident) => names.contains(&ident.to_string().to_lowercase().as_str()),
        TokenTree::Group(group) => group
            .stream()
            .into_iter()
            .any(|tree| holds_ident(&tree, names)),
        TokenTree::Literal(_) | TokenTree::Punct(_) => false,
    }
}

fn holds_string(tree: &TokenTree) -> bool {
    match tree {
        TokenTree::Literal(literal) => {
            let text = literal.to_string();
            text.starts_with('"') || text.starts_with("r\"") || text.starts_with("r#")
        }
        TokenTree::Group(group) => group.stream().into_iter().any(|tree| holds_string(&tree)),
        TokenTree::Ident(_) | TokenTree::Punct(_) => false,
    }
}

/// Finds a `for` loop over a table of cases.
#[derive(Default)]
struct Tables {
    found: bool,
}

impl<'ast> Visit<'ast> for Tables {
    fn visit_expr_for_loop(&mut self, node: &'ast ExprForLoop) {
        if is_table(&node.expr) {
            self.found = true;
        }
        visit::visit_expr_for_loop(self, node);
    }
}

/// Whether a loop's iterable is a table: an array or `vec!` literal, or a
/// binding named for cases, rows or a table, through references and
/// iterator adapters.
fn is_table(expr: &Expr) -> bool {
    match expr {
        Expr::Array(_) => true,
        Expr::Reference(reference) => is_table(&reference.expr),
        Expr::Paren(paren) => is_table(&paren.expr),
        Expr::Macro(mac) => mac.mac.path.is_ident("vec"),
        Expr::MethodCall(call) => {
            matches!(
                call.method.to_string().as_str(),
                "iter" | "into_iter" | "enumerate" | "copied" | "cloned" | "zip"
            ) && is_table(&call.receiver)
        }
        Expr::Path(path) => path.path.segments.last().is_some_and(|segment| {
            let name = segment.ident.to_string().to_lowercase();
            ["case", "table", "row"]
                .iter()
                .any(|fragment| name.contains(fragment))
        }),
        _ => false,
    }
}
