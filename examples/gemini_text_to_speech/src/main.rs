//! Gemini text-to-speech: one line in one voice, then a two-host podcast
//! intro spoken in two parts as headerless PCM, joined and wrapped in a WAV
//! header built from the sample rate the replies report.
//!
//! ```sh
//! cargo run -p gemini_text_to_speech
//! ```

use anyhow::{Context, Result};
use rig::audio_generation::AudioGenerationRequestBuilder;
use rig::providers::gemini::Gemini;
use rig::providers::gemini::audio_generation::{
    AudioFormat, GEMINI_3_8_FLASH_LITE_TTS, GEMINI_3_8_FLASH_TTS, Script, Turn, format_of,
};

/// The podcast's hosts and their voices.
const HOSTS: [(&str, &str); 2] = [("Joe", "Puck"), ("Jane", "Kore")];

#[tokio::main]
async fn main() -> Result<()> {
    let gemini = Gemini::from_env()?;
    let dir = std::env::temp_dir();

    // One voice: the reply is a WAV file.
    let hello = gemini
        .audio_generation(GEMINI_3_8_FLASH_LITE_TTS)
        .call(AudioGenerationRequestBuilder::new("Have a wonderful day!", "Kore").build())
        .await?;
    let path = dir.join("rig-gemini-hello.wav");
    std::fs::write(&path, &hello.audio)?;
    println!("Wrote {}", path.display());

    // Two hosts, two requests, each turn styled: headerless PCM joins by
    // concatenation.
    let parts = [
        vec![
            Turn::new("Welcome back to the show!")
                .speaker("Joe")
                .style("upbeat"),
            Turn::new("Today we talk about bananas.")
                .speaker("Jane")
                .style("warm"),
        ],
        vec![
            Turn::new("<laugh> Of course we do.").speaker("Joe"),
            Turn::new("Let's get started.").speaker("Jane"),
        ],
    ];
    let model = gemini.audio_generation(GEMINI_3_8_FLASH_TTS);
    let mut pcm = Vec::new();
    let mut format = None;
    for turns in parts {
        let script = turns
            .into_iter()
            .fold(Script::speakers(HOSTS), Script::turn)
            .format(AudioFormat::pcm(24_000));
        let speech = model.call(script.build()).await?;
        format = format_of(&speech);
        pcm.extend_from_slice(&speech.audio);
    }
    let format = format.context("the replies state their format")?;
    let sample_rate = u32::try_from(format.sample_rate.unwrap_or(24_000))?;
    let channels = u16::try_from(format.channels.unwrap_or(1))?;
    let path = dir.join("rig-gemini-podcast.wav");
    std::fs::write(&path, wav(&pcm, sample_rate, channels)?)?;
    println!("Wrote {} ({sample_rate} Hz)", path.display());

    Ok(())
}

/// `pcm`, 16-bit little-endian samples, as a WAV file.
fn wav(pcm: &[u8], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    let data = u32::try_from(pcm.len())?;
    let block = channels * 2;
    let mut file = Vec::with_capacity(pcm.len() + 44);
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(36 + data).to_le_bytes());
    file.extend_from_slice(b"WAVEfmt ");
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes());
    file.extend_from_slice(&channels.to_le_bytes());
    file.extend_from_slice(&sample_rate.to_le_bytes());
    file.extend_from_slice(&(sample_rate * u32::from(block)).to_le_bytes());
    file.extend_from_slice(&block.to_le_bytes());
    file.extend_from_slice(&16u16.to_le_bytes());
    file.extend_from_slice(b"data");
    file.extend_from_slice(&data.to_le_bytes());
    file.extend_from_slice(pcm);
    Ok(file)
}
