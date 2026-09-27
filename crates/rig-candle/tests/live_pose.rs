//! The real `yolov8n-pose` checkpoint on a photo of a cycling race. Opt-in:
//! run `tests/download_yolov8_pose.sh`, then
//! `cargo test --release -p rig-candle --test live_pose -- --ignored --nocapture`.
#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::path::PathBuf;

use futures::StreamExt;
use rig_candle::pose::{BodyPart, CandlePoseModel, ImageFrame, PersonPose, PoseRequest};

fn model_dir() -> PathBuf {
    std::env::var_os("RIG_CANDLE_POSE_MODEL_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-models/yolov8n-pose"),
        PathBuf::from,
    )
}

fn photo() -> ImageFrame {
    let photo = image::load_from_memory(
        &std::fs::read(model_dir().join("bike.jpg")).expect("run tests/download_yolov8_pose.sh"),
    )
    .expect("the photo decodes")
    .to_rgb8();
    let (width, height) = photo.dimensions();
    ImageFrame::from_rgb8(width, height, photo.into_raw()).expect("an RGB frame")
}

/// A confident keypoint of `part`, when the model saw it.
fn seen(person: &PersonPose, part: BodyPart) -> Option<(f32, f32)> {
    let keypoint = person.keypoint(part);
    (keypoint.confidence > 0.5).then_some((keypoint.x, keypoint.y))
}

#[tokio::test]
#[ignore = "needs the downloaded yolov8n-pose checkpoint"]
async fn the_riders_are_found_with_upright_skeletons() {
    let weights = std::fs::read(model_dir().join("yolov8n-pose.safetensors"))
        .expect("run tests/download_yolov8_pose.sh");
    let model = CandlePoseModel::from_safetensors_async(weights)
        .await
        .expect("the checkpoint loads")
        .pose();
    let photo = photo();
    let (width, height) = (photo.width() as f32, photo.height() as f32);

    // The same photo twice, as two frames of a video.
    let mut stream = model
        .stream(PoseRequest::from(vec![photo.clone(), photo]))
        .expect("the stream opens");
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        frames.push(frame.expect("a frame's poses"));
    }
    assert_eq!(frames.len(), 2);
    let people = &frames[0].people;
    for person in people {
        println!(
            "{:.2} {:?} nose {:?}",
            person.confidence,
            person.bbox,
            seen(person, BodyPart::Nose)
        );
    }
    println!("{} ms per frame", frames[0].elapsed_ms);
    if let Some(path) = std::env::var_os("RIG_CANDLE_POSE_DUMP") {
        std::fs::write(path, serde_json::to_vec(&frames[0]).expect("serializes"))
            .expect("the dump writes");
    }

    // Inference is deterministic across frames.
    assert_eq!(frames[0].people, frames[1].people);

    let riders: Vec<&PersonPose> = people.iter().filter(|p| p.confidence > 0.5).collect();
    assert!(riders.len() >= 4, "found {} confident people", riders.len());
    for rider in riders {
        let bbox = rider.bbox;
        assert!(bbox.x_min < bbox.x_max && bbox.y_min < bbox.y_max);
        assert!(bbox.x_max <= width && bbox.y_max <= height);
        let inside = |(x, y): (f32, f32)| {
            x >= bbox.x_min - 8.
                && x <= bbox.x_max + 8.
                && y >= bbox.y_min - 8.
                && y <= bbox.y_max + 8.
        };
        for part in BodyPart::ALL {
            if let Some(point) = seen(rider, part) {
                assert!(inside(point), "{part:?} at {point:?} outside {bbox:?}");
            }
        }
        // A rider leans forward, but the head stays above the hips and the
        // shoulders above the ankles.
        let lowest = |parts: [BodyPart; 2]| {
            parts
                .iter()
                .filter_map(|part| seen(rider, *part))
                .map(|(_, y)| y)
                .reduce(f32::max)
        };
        let highest = |parts: [BodyPart; 2]| {
            parts
                .iter()
                .filter_map(|part| seen(rider, *part))
                .map(|(_, y)| y)
                .reduce(f32::min)
        };
        let eyes = lowest([BodyPart::LeftEye, BodyPart::RightEye]);
        let shoulders = lowest([BodyPart::LeftShoulder, BodyPart::RightShoulder]);
        let hips = highest([BodyPart::LeftHip, BodyPart::RightHip]);
        let ankles = highest([BodyPart::LeftAnkle, BodyPart::RightAnkle]);
        if let (Some(eyes), Some(hips)) = (eyes, hips) {
            assert!(eyes < hips, "eyes {eyes} below hips {hips}");
        }
        if let (Some(shoulders), Some(ankles)) = (shoulders, ankles) {
            assert!(
                shoulders < ankles,
                "shoulders {shoulders} below ankles {ankles}"
            );
        }
    }
}
