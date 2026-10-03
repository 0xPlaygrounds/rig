use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::{UserContent, Video, VideoMediaType};

use super::source::{self, Source};
use super::{document, image, tool};
use crate::completion::Family;

/// What stands in for content left empty once blank text is skipped:
/// Converse rejects blank text and empty content.
pub(crate) const EMPTY_TEXT: &str = "<empty>";

/// The Converse content blocks for one piece of user content sent to a
/// model of `family`. Blank text has none.
pub(crate) fn to_aws(
    content: UserContent,
    family: Family,
) -> Result<Vec<aws_bedrock::ContentBlock>, ProviderError> {
    match content {
        UserContent::Text(text) if text.text.trim().is_empty() => Ok(Vec::new()),
        UserContent::Text(text) => Ok(vec![aws_bedrock::ContentBlock::Text(text.text)]),
        UserContent::ToolResult(tool_result) => {
            let mut content = Vec::new();
            for part in tool_result.content {
                content.extend(tool::to_aws(part)?);
            }
            if content.is_empty() {
                content.push(aws_bedrock::ToolResultContentBlock::Text(
                    EMPTY_TEXT.to_owned(),
                ));
            }
            // Converse reads a result with no status as a success, and
            // documents the field for Nova and Claude only.
            let status = (tool_result.is_error && family != Family::Other)
                .then_some(aws_bedrock::ToolResultStatus::Error);
            let builder = aws_bedrock::ToolResultBlock::builder()
                .tool_use_id(tool_result.call.wire().into_owned())
                .set_content(Some(content))
                .set_status(status)
                .build()
                .map_err(|e| ProviderError::Provider(e.to_string()))?;
            Ok(vec![aws_bedrock::ContentBlock::ToolResult(builder)])
        }
        UserContent::Image(image) => {
            let image = image::to_aws(image)?;
            Ok(vec![aws_bedrock::ContentBlock::Image(image)])
        }
        UserContent::Document(document) => {
            let doc = document::to_aws(document)?;
            // Converse requires accompanying prompt text for document blocks.
            Ok(vec![
                aws_bedrock::ContentBlock::Text("Use provided document".to_string()),
                aws_bedrock::ContentBlock::Document(doc),
            ])
        }
        UserContent::Video(video) => {
            Ok(vec![aws_bedrock::ContentBlock::Video(self::video(video)?)])
        }
        UserContent::Audio(_) => Err(ProviderError::request("Converse takes no audio")),
    }
}

/// The Converse video block for `video`, in a format Converse lists.
pub(crate) fn video(video: Video) -> Result<aws_bedrock::VideoBlock, ProviderError> {
    let format = match video.media_type {
        Some(VideoMediaType::MP4) => aws_bedrock::VideoFormat::Mp4,
        Some(VideoMediaType::MPEG) => aws_bedrock::VideoFormat::Mpeg,
        Some(VideoMediaType::MOV) => aws_bedrock::VideoFormat::Mov,
        Some(VideoMediaType::WEBM) => aws_bedrock::VideoFormat::Webm,
        Some(VideoMediaType::AVI) | None => {
            return Err(ProviderError::request(
                "Converse takes no video in this format",
            ));
        }
    };
    let source = match source::of(video.data)? {
        Source::Bytes(blob) => aws_bedrock::VideoSource::Bytes(blob),
        Source::Stored(location) => aws_bedrock::VideoSource::S3Location(location),
    };
    aws_bedrock::VideoBlock::builder()
        .format(format)
        .source(source)
        .build()
        .map_err(ProviderError::request)
}

#[cfg(test)]
mod tests;
