use super::*;

#[test]
fn default_calls_share_the_same_transport() {
    assert!(shared().ptr_eq(&shared()));
    let first = crate::ReqwestClient::default();
    let second = crate::ReqwestClient::default();
    assert!(Arc::ptr_eq(&first.0, &second.0));
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn concurrent_callers_share_the_same_transport() {
    let callers: Vec<_> = (0..16).map(|_| std::thread::spawn(shared)).collect();
    let default = shared();
    for caller in callers {
        assert!(default.ptr_eq(&caller.join().expect("caller completed")));
    }
}

#[test]
fn explicit_clients_do_not_replace_the_default() {
    let default = shared();
    let explicit = crate::ReqwestClient::new(reqwest::Client::new()).boxed();
    assert!(!default.ptr_eq(&explicit));
    assert!(default.ptr_eq(&shared()));
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn failed_client_reports_the_build_error_on_every_kind_of_send() {
    use bytes::Bytes;
    use futures::executor::block_on;
    use http_client::{HttpClientExt, MultipartForm, NoBody, Request};

    let error = reqwest::Client::builder()
        .user_agent("\n")
        .build()
        .expect_err("invalid header");
    let client = crate::ReqwestClient(Arc::new(Err(Arc::new(error))), crate::RuntimePolicy::Shared);
    fn failed<T>(result: http_client::Result<T>) {
        use std::error::Error as _;
        let Err(error) = result else {
            panic!("a failed client must not send")
        };
        assert!(
            error
                .to_string()
                .contains("could not build the bundled reqwest transport")
        );
        let mut source = error.source();
        let mut original_found = false;
        while let Some(cause) = source {
            if let Some(original) = cause.downcast_ref::<reqwest::Error>() {
                assert!(original.is_builder());
                original_found = true;
                break;
            }
            source = cause.source();
        }
        assert!(original_found, "original reqwest error survives each send");
        assert_eq!(error.non_success_status(), None);
    }
    failed(client.inner());
    failed(client.clone().into_inner());
    for _ in 0..3 {
        let request = || {
            Request::builder()
                .uri("https://must-not-connect.invalid/")
                .body(NoBody)
                .expect("request")
        };
        failed(block_on(client.send::<_, Bytes>(request())));
        failed(block_on(client.send_streaming(request())));
        let multipart = Request::builder()
            .uri("https://must-not-connect.invalid/")
            .body(MultipartForm::new().text("x", "y"))
            .expect("request");
        failed(block_on(client.send_multipart::<Bytes>(multipart)));
    }
}
