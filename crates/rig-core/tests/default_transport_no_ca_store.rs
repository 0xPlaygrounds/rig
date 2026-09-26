//! Missing platform certificates fail model calls, never default construction.
//! Linux honors the certificate-file overrides below; macOS and Windows use
//! keychains. Native TLS does not read its store at client-build time, so this
//! regression runs in the rustls-only feature graph.

#![cfg(all(target_os = "linux", feature = "rustls", not(feature = "native-tls")))]

use rig_core::client::DefaultTransport;
use rig_core::completion::{CompletionModel, CompletionRequestBuilder};
use rig_core::error::ProviderError;
use rig_core::providers::openai::OpenAI;

fn empty_the_ca_store() {
    // SAFETY: this binary has one test; no other thread reads the environment
    // before the test constructs its first transport.
    unsafe {
        std::env::set_var("SSL_CERT_FILE", "/nonexistent/rig-no-ca.pem");
        std::env::set_var("SSL_CERT_DIR", "/nonexistent/rig-no-ca");
        std::env::set_var("OPENAI_API_KEY", "test-key");
    }
}

#[test]
fn every_default_transport_reports_a_missing_ca_store_on_every_call() {
    empty_the_ca_store();
    let _default = rig_reqwest::ReqwestClient::default();
    let from_env = OpenAI::from_env()
        .expect("configuration")
        .bound()
        .expect("default handle");
    let literal = OpenAI::new("test-key").bound().expect("default handle");
    for (name, provider) in [("from_env", from_env), ("literal", literal)] {
        let model = provider.completion("gpt-5.2");
        for _ in 0..2 {
            let request = CompletionRequestBuilder::unbound("hello").build();
            let result = futures::executor::block_on(model.completion(request));
            let outcome = match result {
                Err(ProviderError::Http(error)) => error.to_string(),
                Err(other) => format!("wrong variant: {other}"),
                Ok(_) => "sent a request with no CA store".to_owned(),
            };
            assert!(outcome.contains("CA certificates"), "{name}: {outcome}");
        }
    }
}
