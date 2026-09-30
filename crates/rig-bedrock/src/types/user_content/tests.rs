use crate::types::user_content;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::message::UserContent;

#[test]
fn user_content_to_aws_content_block() {
    let uc = UserContent::Text("txt".into());
    let aws_content_blocks: Result<Vec<aws_bedrock::ContentBlock>, _> = user_content::to_aws(uc);
    assert!(aws_content_blocks.is_ok());
    let aws_content_blocks = aws_content_blocks.unwrap();
    assert_eq!(
        aws_content_blocks,
        vec![aws_bedrock::ContentBlock::Text("txt".into())]
    );
}
