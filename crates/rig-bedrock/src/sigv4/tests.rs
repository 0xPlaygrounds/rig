use super::*;
use aws_credential_types::provider::future::ProvideCredentials as ProvideFuture;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The AWS SigV4 test suite's credentials, region and timestamp
/// (2015-08-30T12:36:00Z).
fn suite_signer() -> Signer {
    Signer {
        credentials: SharedCredentialsProvider::new(Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        )),
        cached: Mutex::new(None),
        region: "us-east-1".to_owned(),
        service: "service".to_owned(),
    }
}

fn suite_time() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_440_938_160)
}

/// The suite's `post-vanilla` case, sent with the credential headers a
/// provider wire sets and an uppercase host: the headers are dropped, the
/// host is sent as it is signed, and the signature is the suite's.
#[tokio::test]
async fn signs_post_vanilla_as_the_aws_test_suite_does() {
    let request = Request::post("https://Example.AmazonAWS.com/")
        .header("x-api-key", "")
        .header("authorization", "Bearer ")
        .body(Bytes::new())
        .expect("request");

    let signed = suite_signer()
        .sign(request, suite_time())
        .await
        .expect("signed");

    let header = |name: &str| {
        signed
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
    };
    assert_eq!(
        header("authorization"),
        Some(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b"
        )
    );
    assert_eq!(header("x-amz-date"), Some("20150830T123600Z"));
    assert_eq!(header("x-api-key"), None);
    assert_eq!(signed.uri(), "https://example.amazonaws.com/");
}

/// Credentials that expire an hour after they are issued, counting how
/// often they are asked for.
#[derive(Debug)]
struct Counting(Arc<AtomicUsize>);

impl ProvideCredentials for Counting {
    fn provide_credentials<'a>(&'a self) -> ProvideFuture<'a>
    where
        Self: 'a,
    {
        self.0.fetch_add(1, Ordering::SeqCst);
        ProvideFuture::ready(Ok(Credentials::new(
            "AKIDEXAMPLE",
            "secret",
            None,
            Some(suite_time() + Duration::from_secs(3600)),
            "test",
        )))
    }
}

/// Signing reuses credentials until they are about to expire, so an
/// assumed role is not refreshed through STS on every request.
#[tokio::test]
async fn credentials_are_fetched_again_only_near_expiry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let signer = Signer {
        credentials: SharedCredentialsProvider::new(Counting(Arc::clone(&calls))),
        ..suite_signer()
    };
    let request = || {
        Request::post("https://example.amazonaws.com/")
            .body(Bytes::new())
            .expect("request")
    };

    for minutes in [0, 10, 50] {
        signer
            .sign(request(), suite_time() + Duration::from_secs(minutes * 60))
            .await
            .expect("signed");
    }
    let before_expiry = calls.load(Ordering::SeqCst);
    signer
        .sign(request(), suite_time() + Duration::from_secs(56 * 60))
        .await
        .expect("signed");

    assert_eq!(before_expiry, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_uri_without_a_host_is_refused() {
    let request = Request::post("/v1/messages")
        .body(Bytes::new())
        .expect("request");

    let error = suite_signer().sign(request, suite_time()).await;

    assert!(matches!(error, Err(AwsError::NoHost(_))));
}
