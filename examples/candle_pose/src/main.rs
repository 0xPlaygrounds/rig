use anyhow::Context;
use futures::StreamExt;
use rig::candle::pose::{BodyPart, CandlePoseModel, ImageFrame, PoseRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_dir = match std::env::var_os("MODEL_DIR") {
        Some(directory) => std::path::PathBuf::from(directory),
        None => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model"),
    };
    // Every image named on the command line is one frame of a video.
    let mut images: Vec<std::path::PathBuf> = std::env::args_os().skip(1).map(Into::into).collect();
    if images.is_empty() {
        images.push(model_dir.join("bike.jpg"));
    }
    let frames = images
        .iter()
        .map(|path| {
            let image = image::open(path)
                .with_context(|| format!("decoding {}", path.display()))?
                .to_rgb8();
            let (width, height) = image.dimensions();
            Ok(ImageFrame::from_rgb8(width, height, image.into_raw())?)
        })
        .collect::<anyhow::Result<PoseRequest>>()?;

    let weights = std::fs::read(model_dir.join("yolov8n-pose.safetensors"))
        .context("run ./examples/candle_pose/download_model.sh first")?;
    let model = CandlePoseModel::from_safetensors_async(weights)
        .await?
        .pose();

    // Each frame's poses arrive as soon as that frame is estimated.
    let mut stream = model.stream(frames)?;
    while let Some(frame) = stream.next().await {
        let frame = frame?;
        println!(
            "frame {} of {}: {} people in {} ms",
            frame.frame + 1,
            frame.frames,
            frame.people.len(),
            frame.elapsed_ms
        );
        for person in &frame.people {
            let visible = BodyPart::ALL
                .into_iter()
                .filter(|part| person.keypoint(*part).confidence > 0.5)
                .count();
            let nose = person.keypoint(BodyPart::Nose);
            println!(
                "  {:.0}% at ({:.0}, {:.0})-({:.0}, {:.0}), {visible}/17 keypoints visible, nose at ({:.0}, {:.0})",
                person.confidence * 100.,
                person.bbox.x_min,
                person.bbox.y_min,
                person.bbox.x_max,
                person.bbox.y_max,
                nose.x,
                nose.y,
            );
        }
    }
    Ok(())
}
