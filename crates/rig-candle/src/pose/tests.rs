use candle_core::{DType, Device, Tensor};
#[cfg(not(target_family = "wasm"))]
use candle_nn::{VarBuilder, VarMap};
#[cfg(not(target_family = "wasm"))]
use futures::StreamExt;
#[cfg(not(target_family = "wasm"))]
use rig_core::DynModel;

use super::*;

#[cfg(not(target_family = "wasm"))]
/// A nano checkpoint with random weights: the real layout, so it loads and
/// runs, without predictions worth checking.
fn random_nano_checkpoint() -> Vec<u8> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    YoloV8Pose::load(vb, Multiples::N, 17).expect("the nano layout builds");
    let tensors: Vec<(String, Tensor)> = varmap
        .data()
        .lock()
        .expect("the var map is not poisoned")
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect();
    safetensors::serialize(tensors, None).expect("the checkpoint serializes")
}

fn frame(width: u32, height: u32) -> ImageFrame {
    let pixels = (0..width * height * 3)
        .map(|byte| (byte % 251) as u8)
        .collect();
    ImageFrame::from_rgb8(width, height, pixels).expect("a valid frame")
}

fn keypoints(x: f32, y: f32) -> [f32; KEYPOINT_VALUES] {
    let mut values = [0.; KEYPOINT_VALUES];
    for (index, keypoint) in values.chunks_exact_mut(3).enumerate() {
        keypoint[0] = x + index as f32;
        keypoint[1] = y;
        keypoint[2] = 0.9;
    }
    values
}

/// Predictions `(56, anchors)` from rows of `[cx, cy, w, h, confidence]`
/// and keypoints.
fn predictions(anchors: &[([f32; 5], [f32; KEYPOINT_VALUES])]) -> Tensor {
    let rows: Vec<f32> = anchors
        .iter()
        .flat_map(|(detection, keypoints)| detection.iter().chain(keypoints).copied())
        .collect();
    Tensor::from_vec(rows, (anchors.len(), PREDICTION_ROWS), &Device::Cpu)
        .and_then(|anchors| anchors.t())
        .expect("a prediction tensor")
}

#[cfg(not(target_family = "wasm"))]
fn without_timing(mut frames: Vec<FramePoses>) -> Vec<FramePoses> {
    for frame in &mut frames {
        frame.elapsed_ms = 0;
    }
    frames
}

#[test]
fn a_frame_keeps_its_aspect_ratio_at_the_input_size() {
    assert_eq!(input_dimensions(800, 556, 640), (640, 416));
    assert_eq!(input_dimensions(556, 800, 640), (416, 640));
    assert_eq!(input_dimensions(640, 640, 640), (640, 640));
    assert_eq!(input_dimensions(4000, 10, 640), (640, 32));
    assert_eq!(input_dimensions(30, 20, 64), (64, 32));
}

#[test]
fn a_frame_must_hold_three_bytes_per_pixel() {
    assert!(ImageFrame::from_rgb8(2, 2, vec![0; 12]).is_ok());
    for (width, height, bytes) in [(2, 2, 11), (0, 2, 0), (2, 0, 0), (MAX_FRAME_SIDE + 1, 1, 0)] {
        assert!(
            matches!(
                ImageFrame::from_rgb8(width, height, vec![0; bytes]),
                Err(CandleError::InvalidImage(_))
            ),
            "{width}x{height} with {bytes} bytes"
        );
    }
}

#[test]
fn settings_out_of_range_are_refused_before_loading() {
    let refused = |builder: CandlePoseModelBuilder| {
        matches!(builder.build(), Err(CandleError::InvalidPoseSetting(_)))
    };
    assert!(refused(CandlePoseModel::builder(vec![]).input_size(100)));
    assert!(refused(CandlePoseModel::builder(vec![]).input_size(4096)));
    assert!(refused(
        CandlePoseModel::builder(vec![]).confidence_threshold(1.5)
    ));
    assert!(refused(
        CandlePoseModel::builder(vec![]).iou_threshold(f32::NAN)
    ));
    assert!(matches!(
        CandlePoseModel::builder(vec![])
            .max_concurrent_requests(0)
            .build(),
        Err(CandleError::InvalidConcurrencyLimit)
    ));
    #[cfg(not(target_family = "wasm"))]
    assert!(refused(
        CandlePoseModel::builder(vec![]).max_concurrent_requests(usize::MAX)
    ));
}

#[test]
fn a_checkpoint_that_is_not_a_pose_checkpoint_is_refused() {
    assert!(matches!(
        detect_size(&[]),
        Err(CandleError::EmptyBuffer { .. })
    ));
    assert!(matches!(
        detect_size(b"not safetensors"),
        Err(CandleError::InvalidCheckpoint(_))
    ));

    let tensor = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &Device::Cpu).expect("zeros");
    let checkpoint = |stem: usize, keypoints: usize, classes: usize| {
        safetensors::serialize(
            [
                (STEM_TENSOR, tensor(&[stem, 3, 3, 3])),
                (KEYPOINT_TENSOR, tensor(&[keypoints, 51, 1, 1])),
                (CLASS_TENSOR, tensor(&[classes, 64, 1, 1])),
            ],
            None,
        )
        .expect("serializes")
    };
    assert_eq!(
        detect_size(&checkpoint(32, 51, 1)).expect("a small pose checkpoint"),
        PoseModelSize::Small
    );
    assert!(matches!(
        detect_size(&checkpoint(24, 51, 1)),
        Err(CandleError::InvalidCheckpoint(_))
    ));
    // A detection checkpoint has 80 classes; a hand checkpoint 21 keypoints.
    assert!(matches!(
        detect_size(&checkpoint(16, 51, 80)),
        Err(CandleError::InvalidCheckpoint(_))
    ));
    assert!(matches!(
        detect_size(&checkpoint(16, 63, 1)),
        Err(CandleError::InvalidCheckpoint(_))
    ));
    let without_head =
        safetensors::serialize([(STEM_TENSOR, tensor(&[16, 3, 3, 3]))], None).expect("serializes");
    assert!(matches!(
        detect_size(&without_head),
        Err(CandleError::MissingTensor(name)) if name == KEYPOINT_TENSOR
    ));
}

#[test]
fn overlapping_people_are_suppressed_and_scaled_to_the_frame() {
    let predictions = predictions(&[
        ([100., 100., 40., 80., 0.9], keypoints(90., 70.)),
        // The same person, less confident: suppressed.
        ([102., 101., 40., 80., 0.8], keypoints(0., 0.)),
        // Another person, partly outside the input.
        ([10., 150., 40., 40., 0.6], keypoints(5., 140.)),
        // Below the threshold.
        ([300., 100., 40., 80., 0.2], keypoints(0., 0.)),
        ([300., 100., 40., 80., f32::NAN], keypoints(0., 0.)),
    ]);
    let people = decode_people(
        &predictions,
        &PoseSettings::default(),
        (2., 0.5),
        (400., 200.),
    )
    .expect("the predictions decode");

    assert_eq!(people.len(), 2, "{people:?}");
    let first = &people[0];
    assert_eq!(first.confidence, 0.9);
    assert_eq!(
        first.bbox,
        BoundingBox {
            x_min: 160.,
            y_min: 30.,
            x_max: 240.,
            y_max: 70.,
        }
    );
    assert_eq!(
        first.keypoint(BodyPart::Nose),
        Keypoint {
            x: 180.,
            y: 35.,
            confidence: 0.9,
        }
    );
    assert_eq!(first.keypoint(BodyPart::RightAnkle).x, (90. + 16.) * 2.);
    // Clamped at the frame's left edge.
    assert_eq!(people[1].bbox.x_min, 0.);
}

#[test]
fn every_body_part_selects_its_own_keypoint() {
    let keypoints: [Keypoint; 17] = std::array::from_fn(|index| Keypoint {
        x: index as f32,
        ..Keypoint::default()
    });
    let person = PersonPose {
        bbox: BoundingBox::default(),
        confidence: 1.,
        keypoints,
    };
    for (index, part) in BodyPart::ALL.into_iter().enumerate() {
        assert_eq!(person.keypoint(part).x, index as f32, "{part:?}");
    }
}

#[test]
fn a_reply_that_skips_or_stops_before_a_frame_is_refused() {
    let request = PoseRequest::from(vec![frame(2, 2), frame(2, 2)]);
    let event = |index| FramePoses {
        frame: index,
        frames: 2,
        width: 2,
        height: 2,
        people: Vec::new(),
        elapsed_ms: 0,
    };
    let fold = || PoseFold {
        expected: request.frames.len(),
        track: PoseTrack::default(),
    };
    let reply = || Reply {
        provider: "candle".into(),
        raw: serde_json::Value::Null,
        provider_request_id: None,
    };

    let mut skipped = fold();
    assert!(skipped.absorb(&event(1)).is_err());

    let mut truncated = fold();
    truncated.absorb(&event(0)).expect("the first frame");
    assert!(truncated.finish((), reply()).is_err());

    let mut whole = fold();
    whole.absorb(&event(0)).expect("the first frame");
    whole.absorb(&event(1)).expect("the second frame");
    assert_eq!(
        whole.finish((), reply()).expect("every frame").frames,
        vec![event(0), event(1)]
    );
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test]
async fn a_loaded_checkpoint_streams_one_event_per_frame_in_order() {
    let candle = CandlePoseModel::builder(random_nano_checkpoint())
        .input_size(64)
        .confidence_threshold(0.)
        .build_async()
        .await
        .expect("the checkpoint loads");
    assert_eq!(candle.size(), PoseModelSize::Nano);
    let model = candle.pose();
    assert_eq!(model.name(), "candle");
    assert_eq!(model.id(), Some("yolov8n-pose"));

    let video = PoseRequest::from(vec![frame(48, 32), frame(20, 40), frame(64, 64)]);
    let mut stream = model.stream(video.clone()).expect("the stream opens");
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        if let rig_core::streaming::Item::Event(event) = item.expect("a frame's poses") {
            events.push(event);
        }
    }
    assert_eq!(
        events
            .iter()
            .map(|event| (event.frame, event.frames, event.width, event.height))
            .collect::<Vec<_>>(),
        [(0, 3, 48, 32), (1, 3, 20, 40), (2, 3, 64, 64)]
    );
    for person in events.iter().flat_map(|event| &event.people) {
        assert!(person.bbox.x_min >= 0. && person.bbox.x_max <= 64.);
    }

    // A call is the same stream, drained; the erased model runs it too.
    let erased: DynModel<PoseEstimation> = model.erase();
    let track = erased.call(video).await.expect("the call folds");
    assert_eq!(without_timing(track.frames), without_timing(events));
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test]
async fn a_request_without_frames_is_refused_before_anything_runs() {
    let model = CandlePoseModel::builder(random_nano_checkpoint())
        .input_size(64)
        .build()
        .expect("the checkpoint loads")
        .pose();
    // The refusal is the stream's only item.
    let mut stream = model.stream(PoseRequest::default()).expect("the stream opens");
    assert!(matches!(stream.next().await, Some(Err(_))));
    assert!(stream.next().await.is_none());
    assert!(model.call(PoseRequest::default()).await.is_err());
}
