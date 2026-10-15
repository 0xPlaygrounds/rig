use super::*;
use aws_credential_types::Credentials;
use std::time::Duration;

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
        region: "us-east-1".to_owned(),
        service: "service".to_owned(),
    }
}

fn suite_time() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_440_938_160)
}

/// The suite's `post-vanilla` case, sent with the credential headers a
/// provider wire sets: they are dropped and the signature is the suite's.
#[tokio::test]
async fn signs_post_vanilla_as_the_aws_test_suite_does() {
    let request = Request::post("https://example.amazonaws.com/")
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
}

#[tokio::test]
async fn a_uri_without_a_host_is_refused() {
    let request = Request::post("/v1/messages")
        .body(Bytes::new())
        .expect("request");

    let error = suite_signer().sign(request, suite_time()).await;

    assert!(matches!(error, Err(AwsError::NoHost(_))));
}
