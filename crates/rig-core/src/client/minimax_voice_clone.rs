//! The verbs of a MiniMax [`VoiceClones`] model. Each is one call through
//! the driver.

use crate::driver::{Model, Transport};
use crate::error::ProviderError;
use crate::providers::minimax::voice_clone::{
    CloneAudio, ClonedVoice, UploadedFile, VoiceClone, VoiceCloneRequest, VoiceClones,
};

impl<T> Model<VoiceClones, T>
where
    T: Transport<VoiceClones>,
{
    /// Upload `audio` and return the stored file.
    pub async fn upload(&self, audio: CloneAudio) -> Result<UploadedFile, ProviderError> {
        self.call(VoiceCloneRequest::Upload(audio))
            .await?
            .file
            .ok_or_else(|| ProviderError::Response("MiniMax's upload reply named no file".into()))
    }

    /// Clone a sample as a new voice.
    pub async fn clone_voice(&self, clone: VoiceClone) -> Result<ClonedVoice, ProviderError> {
        let voice_id = clone.created_voice_id().to_owned();
        let reply = self.call(VoiceCloneRequest::Clone(clone)).await?;
        Ok(ClonedVoice {
            voice_id,
            demo_audio: reply.demo_audio.filter(|url| !url.is_empty()),
            extra_info: reply.extra_info,
        })
    }
}
