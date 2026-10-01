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

#[tokio::test]
async fn supplied_client_is_used_as_given() {
    let config = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("ap-south-1"))
        .build();
    let runtime = BedrockRuntime::from(aws_sdk_bedrockruntime::Client::from_conf(config));

    let region = runtime.inner().await.config().region().map(Region::as_ref);

    assert_eq!(region, Some("ap-south-1"));
}
