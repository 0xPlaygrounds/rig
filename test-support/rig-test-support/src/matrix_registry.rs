//! Parse the same explicit rows that `golden_matrix!` turns into tests.
//! Macro definitions and arbitrary Rust macros are not expanded or registered.

use std::collections::BTreeSet;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Ident, LitStr, Path, Token, parenthesized};

#[cfg(test)]
#[path = "matrix_registry/tests.rs"]
mod tests;

/// Golden-producing or native-parity matrix declarations.
pub struct GoldenMatrix {
    /// The wire-specific cassette wrapper called by each generated test.
    pub wrapper: Path,
    /// Original golden producer or native parity helper.
    pub oracle: Path,
    /// Test rows in declaration order.
    pub rows: Vec<Row>,
}

/// One generated test and its fixed fixture references.
pub struct Row {
    /// Cassette path relative to the provider directory.
    pub scenario: LitStr,
    /// Golden name within the matrix's runtime corpus, without an extension.
    pub golden: LitStr,
    /// Whether this row requires no recording because it is ignored.
    pub ignored: bool,
}

fn field(input: ParseStream<'_>, name: &str) -> syn::Result<Path> {
    let key: Ident = input.parse()?;
    if key != name {
        return Err(syn::Error::new(key.span(), format!("expected {name}")));
    }
    input.parse::<Token![:]>()?;
    input.parse()
}

fn registration(input: ParseStream<'_>, names: &mut BTreeSet<String>) -> syn::Result<bool> {
    let attrs = input.call(Attribute::parse_outer)?;
    let name: Ident = input.parse()?;
    if !names.insert(name.to_string()) {
        return Err(syn::Error::new(name.span(), "duplicate matrix test name"));
    }
    let test = attrs.iter().any(|attr| {
        let path: Vec<_> = attr
            .path()
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        path == ["tokio", "test"]
    });
    if !test {
        return Err(syn::Error::new(
            name.span(),
            "matrix row requires #[tokio::test]",
        ));
    }
    // Conditional ignores need explicit branches in the source. Treating
    // a cfg_attr as an unconditional ignore would hide recorded rows.
    if attrs.iter().any(|attr| attr.path().is_ident("cfg_attr")) {
        return Err(syn::Error::new(
            name.span(),
            "cfg_attr is unsupported in matrix rows",
        ));
    }
    let ignored = attrs.iter().any(|attr| attr.path().is_ident("ignore"));
    Ok(ignored)
}

impl Parse for GoldenMatrix {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let wrapper = field(input, "wrapper")?;
        input.parse::<Token![,]>()?;
        field(input, "wire")?;
        input.parse::<Token![,]>()?;
        field(input, "run")?;
        input.parse::<Token![,]>()?;
        let oracle = field(input, "oracle")?;
        input.parse::<Token![;]>()?;
        let mut rows = Vec::new();
        let mut names = BTreeSet::new();
        while !input.is_empty() {
            let ignored = registration(input, &mut names)?;
            input.parse::<Token![:]>()?;
            let args;
            parenthesized!(args in input);
            let scenario = args.parse()?;
            args.parse::<Token![,]>()?;
            args.parse::<Path>()?;
            args.parse::<Token![,]>()?;
            let golden = args.parse()?;
            if !args.is_empty() {
                return Err(args.error("expected scenario, cell path, and golden literal"));
            }
            input.parse::<Token![;]>()?;
            rows.push(Row {
                scenario,
                golden,
                ignored,
            });
        }
        if rows.is_empty() {
            return Err(input.error("matrix must register at least one test"));
        }
        Ok(Self {
            wrapper,
            oracle,
            rows,
        })
    }
}

/// Native cells and their world golden names.
pub struct NativeMatrix {
    /// The wire-specific cassette wrapper called by each generated test.
    pub wrapper: Path,
    /// Native fixture references and ignore status.
    pub rows: Vec<Row>,
}

impl Parse for NativeMatrix {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let wrapper = field(input, "wrapper")?;
        input.parse::<Token![,]>()?;
        field(input, "wire")?;
        input.parse::<Token![,]>()?;
        field(input, "run")?;
        input.parse::<Token![;]>()?;
        let mut rows = Vec::new();
        let mut names = BTreeSet::new();
        while !input.is_empty() {
            let ignored = registration(input, &mut names)?;
            input.parse::<Token![:]>()?;
            let args;
            parenthesized!(args in input);
            let scenario: LitStr = args.parse()?;
            args.parse::<Token![,]>()?;
            args.parse::<Path>()?;
            args.parse::<Token![,]>()?;
            let golden: LitStr = args.parse()?;
            if !args.is_empty() {
                return Err(args.error("expected scenario, cell path, and world golden literal"));
            }
            input.parse::<Token![;]>()?;
            rows.push(Row {
                scenario,
                golden,
                ignored,
            });
        }
        if rows.is_empty() {
            return Err(input.error("matrix must register at least one test"));
        }
        Ok(Self { wrapper, rows })
    }
}

/// Resume variants of native rows: the cut point beside the cell.
pub struct ResumeMatrix {
    /// The wire-specific cassette wrapper called by each generated test.
    pub wrapper: Path,
    /// Native fixture references and ignore status.
    pub rows: Vec<Row>,
}

impl Parse for ResumeMatrix {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let wrapper = field(input, "wrapper")?;
        input.parse::<Token![,]>()?;
        field(input, "wire")?;
        input.parse::<Token![,]>()?;
        field(input, "run")?;
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            field(input, "after")?;
        }
        input.parse::<Token![;]>()?;
        let mut rows = Vec::new();
        let mut names = BTreeSet::new();
        while !input.is_empty() {
            let ignored = registration(input, &mut names)?;
            input.parse::<Token![:]>()?;
            let args;
            parenthesized!(args in input);
            let scenario: LitStr = args.parse()?;
            args.parse::<Token![,]>()?;
            args.parse::<Path>()?;
            args.parse::<Token![,]>()?;
            args.parse::<syn::Expr>()?;
            args.parse::<Token![,]>()?;
            let golden: LitStr = args.parse()?;
            if !args.is_empty() {
                return Err(
                    args.error("expected scenario, cell, resume point, and world golden literal")
                );
            }
            input.parse::<Token![;]>()?;
            rows.push(Row {
                scenario,
                golden,
                ignored,
            });
        }
        if rows.is_empty() {
            return Err(input.error("matrix must register at least one test"));
        }
        Ok(Self { wrapper, rows })
    }
}

/// A family-specific body receives the scenario from the same registered row
/// that supplies its generated function name and attributes.
pub struct CaseMatrix {
    /// Macro family that executes the registered rows.
    pub family: Ident,
    /// Number of test registrations, including ignored rows.
    pub registrations: usize,
    /// The cassette wrapper, or none for rows using only local scripted transports.
    pub wrapper: Option<Path>,
    /// Scenario literals and whether each row is ignored.
    pub rows: Vec<(LitStr, bool)>,
    /// Explicit world golden names and ignore status.
    pub world_goldens: Vec<(LitStr, bool)>,
}

impl Parse for CaseMatrix {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let first: Ident = input.fork().parse()?;
        let wrapper = if first == "wrapper" {
            let wrapper = field(input, "wrapper")?;
            input.parse::<Token![,]>()?;
            Some(wrapper)
        } else {
            None
        };
        let key: Ident = input.parse()?;
        if key != "family" {
            return Err(syn::Error::new(key.span(), "expected family"));
        }
        input.parse::<Token![:]>()?;
        let family = input.parse::<Ident>()?;
        input.parse::<Token![;]>()?;
        let mut rows = Vec::new();
        let mut world_goldens = Vec::new();
        let mut names = BTreeSet::new();
        while !input.is_empty() {
            let ignored = registration(input, &mut names)?;
            input.parse::<Token![:]>()?;
            if wrapper.is_none() {
                input.parse::<Ident>()?;
                if input.peek(Token![=>]) {
                    input.parse::<Token![=>]>()?;
                    world_goldens.push((input.parse::<LitStr>()?, ignored));
                }
                input.parse::<Token![;]>()?;
                continue;
            }
            let args;
            parenthesized!(args in input);
            let scenario: LitStr = args.parse()?;
            args.parse::<Token![,]>()?;
            args.parse::<Ident>()?;
            if args.peek(Token![,]) {
                args.parse::<Token![,]>()?;
                let argument = args.parse::<syn::Expr>()?;
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(name),
                    ..
                }) = argument
                {
                    world_goldens.push((name, ignored));
                }
            }
            if !args.is_empty() {
                return Err(args.error("expected scenario and case selector"));
            }
            input.parse::<Token![;]>()?;
            rows.push((scenario, ignored));
        }
        if names.is_empty() {
            return Err(input.error("matrix must register at least one test"));
        }
        Ok(Self {
            family,
            registrations: names.len(),
            wrapper,
            rows,
            world_goldens,
        })
    }
}
