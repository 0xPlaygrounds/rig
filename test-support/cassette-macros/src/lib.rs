//! Keep test bodies ordinary Rust while emitting compiled cassette declarations.

use proc_macro::TokenStream;

/// Declare the scenarios opened by an async repository test. The arguments are
/// `Scenario` expressions, forwarded unchanged to the registration macro.
#[proc_macro_attribute]
pub fn cassette(scenarios: TokenStream, test: TokenStream) -> TokenStream {
    let scenarios = proc_macro2::TokenStream::from(scenarios);
    let test = proc_macro2::TokenStream::from(test);
    quote::quote! {
        ::rig_test_support::recording::declarations::cassette_test! {
            scenarios: [#scenarios];
            #test
        }
    }
    .into()
}
