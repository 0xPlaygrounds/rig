use super::*;

#[tokio::test]
async fn explicit_region_applies_alongside_a_profile() {
    let runtime = BedrockRuntime::builder()
        .profile_name("rig-bedrock-test-absent-profile")
        .region("eu-west-1")
        .build();

    let region = runtime.inner().await.config().region().map(Region::as_ref);

    assert_eq!(region, Some("eu-west-1"));
}
