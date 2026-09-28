//! Local human pose estimation with YOLOv8 pose checkpoints. A
//! [`CandlePoseModel`] loads one caller-supplied safetensors checkpoint and
//! serves the [`PoseEstimation`] operation: a request of RGB frames yields
//! one [`FramePoses`] event per frame, in order, and a call folds them into
//! a [`PoseTrack`]. Coordinates are pixels of the frame as supplied.
//!
//! ```no_run
//! use rig_candle::pose::{CandlePoseModel, ImageFrame};
//!
//! # async fn example(weights: Vec<u8>, width: u32, height: u32, rgb: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
//! let model = CandlePoseModel::from_safetensors(weights)?.pose();
//! let track = model.call(ImageFrame::from_rgb8(width, height, rgb)?).await?;
//! for person in &track.frames[0].people {
//!     println!("{:?}", person.keypoint(rig_candle::pose::BodyPart::Nose));
//! }
//! # Ok(())
//! # }
//! ```

mod network;

use std::fmt;
use std::sync::Arc;

use candle_core::{DType, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{Bbox, non_maximum_suppression};
use rig_core::Model;
use rig_core::driver::{Exchange, Local, Opened, Opening, Step, Transport};
use rig_core::error::ProviderError;
use rig_core::wasm_compat::WasmBoxedStream;
use rig_core::wire::{Call, Fold, Free, Operation, Reply};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::CandleError;
#[cfg(not(target_family = "wasm"))]
use crate::runtime::{CancelOnDrop, ReceiverStream, acquire_concurrency};
use crate::runtime::{CancellationSignal, RuntimeDevice, check_cancellation};
use crate::types::PROVIDER_NAME;
use network::{Multiples, YoloV8Pose};

/// Estimating the poses of the people in a sequence of frames.
///
/// Each frame is one event, in frame order, and the runtime ends the reply
/// once every frame was estimated. A reply that ends before every frame was
/// answered fails the call rather than returning a shorter track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoseEstimation;

impl Operation for PoseEstimation {
    type Request = PoseRequest;
    type Event = FramePoses;
    type End = ();
    type Response = PoseTrack;
    type Fold = PoseFold;
    type Emit = Free;

    fn fold(request: &PoseRequest, _call: &mut Call<'_>) -> PoseFold {
        PoseFold {
            expected: request.frames.len(),
            track: PoseTrack::default(),
        }
    }
}

/// The frames to estimate poses in, in order: one image, or a video's frames.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PoseRequest {
    /// The frames, in the order their events are yielded.
    pub frames: Vec<ImageFrame>,
}

impl From<ImageFrame> for PoseRequest {
    fn from(frame: ImageFrame) -> Self {
        Self {
            frames: vec![frame],
        }
    }
}

impl From<Vec<ImageFrame>> for PoseRequest {
    fn from(frames: Vec<ImageFrame>) -> Self {
        Self { frames }
    }
}

impl FromIterator<ImageFrame> for PoseRequest {
    fn from_iter<I: IntoIterator<Item = ImageFrame>>(frames: I) -> Self {
        Self {
            frames: frames.into_iter().collect(),
        }
    }
}

/// The longest side a frame may have, in pixels.
pub const MAX_FRAME_SIDE: u32 = 16_384;

/// One decoded image: 8-bit RGB pixels, row by row, without padding.
#[derive(Clone, PartialEq, Eq)]
pub struct ImageFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl ImageFrame {
    /// A frame of `width` by `height` RGB pixels. Fails when a side is zero
    /// or longer than [`MAX_FRAME_SIDE`], or when `pixels` does not hold
    /// exactly three bytes per pixel.
    pub fn from_rgb8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, CandleError> {
        if width == 0 || height == 0 || width > MAX_FRAME_SIDE || height > MAX_FRAME_SIDE {
            return Err(CandleError::InvalidImage(format!(
                "a {width}x{height} frame is outside 1..={MAX_FRAME_SIDE} pixels per side"
            )));
        }
        let expected = u64::from(width) * u64::from(height) * 3;
        if pixels.len() as u64 != expected {
            return Err(CandleError::InvalidImage(format!(
                "a {width}x{height} RGB frame holds {expected} bytes, not {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// The frame's width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The frame's height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The frame's RGB bytes.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

impl fmt::Debug for ImageFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// The people found in one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FramePoses {
    /// The frame's index in the request.
    pub frame: usize,
    /// How many frames the request holds.
    pub frames: usize,
    /// The frame's width in pixels.
    pub width: u32,
    /// The frame's height in pixels.
    pub height: u32,
    /// The people found, most confident first.
    pub people: Vec<PersonPose>,
    /// Time spent on this frame, in milliseconds.
    pub elapsed_ms: u64,
}

/// One person: a box, a confidence and the 17 COCO keypoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonPose {
    /// The person's box, clamped to the frame.
    pub bbox: BoundingBox,
    /// How confident the model is that the box holds a person, from 0 to 1.
    pub confidence: f32,
    /// The keypoints in [`BodyPart::ALL`] order.
    pub keypoints: [Keypoint; 17],
}

impl PersonPose {
    /// The keypoint of `part`.
    pub fn keypoint(&self, part: BodyPart) -> Keypoint {
        part.select(&self.keypoints)
    }
}

/// An axis-aligned box in frame pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Left edge.
    pub x_min: f32,
    /// Top edge.
    pub y_min: f32,
    /// Right edge.
    pub x_max: f32,
    /// Bottom edge.
    pub y_max: f32,
}

/// One body keypoint in frame pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    /// Horizontal position.
    pub x: f32,
    /// Vertical position.
    pub y: f32,
    /// How confident the model is that the keypoint is visible, from 0 to 1.
    /// A keypoint the model cannot see still has a guessed position.
    pub confidence: f32,
}

/// The 17 COCO body keypoints, in the order a model predicts them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyPart {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
}

impl BodyPart {
    /// Every keypoint, in prediction order.
    pub const ALL: [Self; 17] = [
        Self::Nose,
        Self::LeftEye,
        Self::RightEye,
        Self::LeftEar,
        Self::RightEar,
        Self::LeftShoulder,
        Self::RightShoulder,
        Self::LeftElbow,
        Self::RightElbow,
        Self::LeftWrist,
        Self::RightWrist,
        Self::LeftHip,
        Self::RightHip,
        Self::LeftKnee,
        Self::RightKnee,
        Self::LeftAnkle,
        Self::RightAnkle,
    ];

    /// The limbs that join keypoints, for drawing a skeleton.
    pub const SKELETON: [(Self, Self); 16] = [
        (Self::Nose, Self::LeftEye),
        (Self::Nose, Self::RightEye),
        (Self::LeftEye, Self::LeftEar),
        (Self::RightEye, Self::RightEar),
        (Self::LeftShoulder, Self::RightShoulder),
        (Self::LeftShoulder, Self::LeftHip),
        (Self::RightShoulder, Self::RightHip),
        (Self::LeftHip, Self::RightHip),
        (Self::LeftShoulder, Self::LeftElbow),
        (Self::RightShoulder, Self::RightElbow),
        (Self::LeftElbow, Self::LeftWrist),
        (Self::RightElbow, Self::RightWrist),
        (Self::LeftHip, Self::LeftKnee),
        (Self::RightHip, Self::RightKnee),
        (Self::LeftKnee, Self::LeftAnkle),
        (Self::RightKnee, Self::RightAnkle),
    ];

    fn select(self, keypoints: &[Keypoint; 17]) -> Keypoint {
        let [
            nose,
            left_eye,
            right_eye,
            left_ear,
            right_ear,
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
            left_wrist,
            right_wrist,
            left_hip,
            right_hip,
            left_knee,
            right_knee,
            left_ankle,
            right_ankle,
        ] = *keypoints;
        match self {
            Self::Nose => nose,
            Self::LeftEye => left_eye,
            Self::RightEye => right_eye,
            Self::LeftEar => left_ear,
            Self::RightEar => right_ear,
            Self::LeftShoulder => left_shoulder,
            Self::RightShoulder => right_shoulder,
            Self::LeftElbow => left_elbow,
            Self::RightElbow => right_elbow,
            Self::LeftWrist => left_wrist,
            Self::RightWrist => right_wrist,
            Self::LeftHip => left_hip,
            Self::RightHip => right_hip,
            Self::LeftKnee => left_knee,
            Self::RightKnee => right_knee,
            Self::LeftAnkle => left_ankle,
            Self::RightAnkle => right_ankle,
        }
    }
}

/// Every frame's poses, in frame order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PoseTrack {
    /// One entry per requested frame.
    pub frames: Vec<FramePoses>,
}

/// The fold from a pose reply's events to its [`PoseTrack`]. It refuses an
/// event out of frame order, and a reply that answered fewer frames than
/// the request held.
#[derive(Debug)]
pub struct PoseFold {
    expected: usize,
    track: PoseTrack,
}

impl Fold<PoseEstimation> for PoseFold {
    fn absorb(&mut self, event: &FramePoses) -> Result<(), ProviderError> {
        let next = self.track.frames.len();
        if event.frame != next {
            return Err(ProviderError::Response(format!(
                "pose estimation answered frame {} where frame {next} was due",
                event.frame
            )));
        }
        self.track.frames.push(event.clone());
        Ok(())
    }

    fn finish(self, _end: (), _reply: Reply) -> Result<PoseTrack, ProviderError> {
        let answered = self.track.frames.len();
        if answered != self.expected {
            return Err(ProviderError::Response(format!(
                "pose estimation ended after {answered} of {} frames",
                self.expected
            )));
        }
        Ok(self.track)
    }
}

/// The size of a loaded YOLOv8 pose checkpoint, detected from its tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoseModelSize {
    /// `yolov8n-pose`.
    Nano,
    /// `yolov8s-pose`.
    Small,
    /// `yolov8m-pose`.
    Medium,
    /// `yolov8l-pose`.
    Large,
    /// `yolov8x-pose`.
    ExtraLarge,
}

impl PoseModelSize {
    /// The checkpoint's conventional name, which the model reports as its id.
    pub fn model_id(self) -> &'static str {
        match self {
            Self::Nano => "yolov8n-pose",
            Self::Small => "yolov8s-pose",
            Self::Medium => "yolov8m-pose",
            Self::Large => "yolov8l-pose",
            Self::ExtraLarge => "yolov8x-pose",
        }
    }

    fn multiples(self) -> Multiples {
        match self {
            Self::Nano => Multiples::N,
            Self::Small => Multiples::S,
            Self::Medium => Multiples::M,
            Self::Large => Multiples::L,
            Self::ExtraLarge => Multiples::X,
        }
    }

    /// The size whose stem has `channels` output channels.
    fn from_stem(channels: usize) -> Option<Self> {
        Some(match channels {
            16 => Self::Nano,
            32 => Self::Small,
            48 => Self::Medium,
            64 => Self::Large,
            80 => Self::ExtraLarge,
            _ => return None,
        })
    }
}

/// The tensor whose output channels identify the checkpoint size.
const STEM_TENSOR: &str = "net.b1.0.conv.weight";
/// The keypoint head's last convolution: 17 keypoints times three values.
const KEYPOINT_TENSOR: &str = "head.cv4.0.2.weight";
/// The class head's last convolution: one class, a person.
const CLASS_TENSOR: &str = "head.cv3.0.2.weight";
const KEYPOINT_VALUES: usize = 17 * 3;
/// Box, one class score, then the keypoint values.
const PREDICTION_ROWS: usize = 4 + 1 + KEYPOINT_VALUES;
/// The network's coarsest stride; input sides are multiples of it.
const STRIDE: u32 = 32;
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1;
#[cfg(not(target_family = "wasm"))]
const STREAM_CHANNEL_CAPACITY: usize = 8;

/// How frames are resized and detections filtered.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PoseSettings {
    input_size: u32,
    confidence_threshold: f32,
    iou_threshold: f32,
}

impl Default for PoseSettings {
    fn default() -> Self {
        Self {
            input_size: 640,
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
        }
    }
}

impl PoseSettings {
    fn validate(&self) -> Result<(), CandleError> {
        if self.input_size < STRIDE
            || self.input_size > 2048
            || !self.input_size.is_multiple_of(STRIDE)
        {
            return Err(CandleError::InvalidPoseSetting(format!(
                "input_size {} must be a multiple of {STRIDE} from {STRIDE} to 2048",
                self.input_size
            )));
        }
        for (name, value) in [
            ("confidence_threshold", self.confidence_threshold),
            ("iou_threshold", self.iou_threshold),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(CandleError::InvalidPoseSetting(format!(
                    "{name} {value} must be from 0 to 1"
                )));
            }
        }
        Ok(())
    }
}

/// A cloneable CPU pose model sharing one loaded YOLOv8 pose checkpoint.
#[derive(Clone)]
pub struct CandlePoseModel {
    state: Arc<LoadedPose>,
}

struct LoadedPose {
    network: YoloV8Pose,
    size: PoseModelSize,
    settings: PoseSettings,
    runtime: RuntimeDevice,
    #[cfg(not(target_family = "wasm"))]
    concurrency: Arc<tokio::sync::Semaphore>,
}

/// Builder for a [`CandlePoseModel`]: the checkpoint and the defaults its
/// requests run with.
pub struct CandlePoseModelBuilder {
    weights: Vec<u8>,
    settings: PoseSettings,
    max_concurrent_requests: usize,
}

impl fmt::Debug for CandlePoseModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CandlePoseModel")
            .field("size", &self.state.size)
            .field("settings", &self.state.settings)
            .finish_non_exhaustive()
    }
}

impl CandlePoseModel {
    /// The pose model this runtime serves: a local [`PoseEstimation`] wire
    /// named `candle`, addressing the checkpoint's [`PoseModelSize::model_id`].
    pub fn pose(&self) -> Model<Local<PoseEstimation>, Self> {
        Model::new(
            Local::new(PROVIDER_NAME).with_id(self.state.size.model_id()),
            self.clone(),
        )
    }

    /// Loads one YOLOv8 pose safetensors checkpoint with default settings.
    pub fn from_safetensors(weights: Vec<u8>) -> Result<Self, CandleError> {
        Self::builder(weights).build()
    }

    /// Loads a checkpoint on Tokio's blocking pool.
    #[cfg(not(target_family = "wasm"))]
    pub async fn from_safetensors_async(weights: Vec<u8>) -> Result<Self, CandleError> {
        Self::builder(weights).build_async().await
    }

    /// Starts a builder over one YOLOv8 pose safetensors checkpoint.
    pub fn builder(weights: Vec<u8>) -> CandlePoseModelBuilder {
        CandlePoseModelBuilder {
            weights,
            settings: PoseSettings::default(),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }

    /// The loaded checkpoint's size.
    pub fn size(&self) -> PoseModelSize {
        self.state.size
    }
}

impl CandlePoseModelBuilder {
    /// Sets the longest side, in pixels, a frame is resized to before
    /// inference. It is a multiple of 32 from 32 to 2048; the default is 640.
    /// Smaller is faster and misses smaller people.
    pub fn input_size(mut self, input_size: u32) -> Self {
        self.settings.input_size = input_size;
        self
    }

    /// Sets the confidence a person needs to be reported, from 0 to 1. The
    /// default is 0.25.
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.settings.confidence_threshold = threshold;
        self
    }

    /// Sets the overlap, from 0 to 1, above which the less confident of two
    /// boxes is dropped as the same person. The default is 0.45.
    pub fn iou_threshold(mut self, threshold: f32) -> Self {
        self.settings.iou_threshold = threshold;
        self
    }

    /// Sets the number of native requests admitted concurrently. The default
    /// is one; WASM inference is synchronous and ignores it.
    pub fn max_concurrent_requests(mut self, max_concurrent_requests: usize) -> Self {
        self.max_concurrent_requests = max_concurrent_requests;
        self
    }

    /// Validates the settings and the checkpoint, and loads its tensors onto
    /// the CPU.
    pub fn build(self) -> Result<CandlePoseModel, CandleError> {
        self.settings.validate()?;
        if self.max_concurrent_requests == 0 {
            return Err(CandleError::InvalidConcurrencyLimit);
        }
        #[cfg(not(target_family = "wasm"))]
        if self.max_concurrent_requests > tokio::sync::Semaphore::MAX_PERMITS {
            return Err(CandleError::InvalidPoseSetting(format!(
                "max_concurrent_requests {} exceeds {}",
                self.max_concurrent_requests,
                tokio::sync::Semaphore::MAX_PERMITS
            )));
        }
        let size = detect_size(&self.weights)?;
        let runtime = RuntimeDevice::cpu();
        let vb = VarBuilder::from_slice_safetensors(&self.weights, DType::F32, runtime.device())
            .map_err(|error| CandleError::InvalidCheckpoint(error.to_string()))?;
        let network = YoloV8Pose::load(vb, size.multiples(), 17)
            .map_err(|error| CandleError::ModelLoading(error.to_string()))?;
        Ok(CandlePoseModel {
            state: Arc::new(LoadedPose {
                network,
                size,
                settings: self.settings,
                runtime,
                #[cfg(not(target_family = "wasm"))]
                concurrency: Arc::new(tokio::sync::Semaphore::new(self.max_concurrent_requests)),
            }),
        })
    }

    /// [`Self::build`] on Tokio's blocking pool. Dropping the future does
    /// not stop a load that has started.
    #[cfg(not(target_family = "wasm"))]
    pub async fn build_async(self) -> Result<CandlePoseModel, CandleError> {
        tokio::task::spawn_blocking(move || self.build())
            .await
            .map_err(|error| CandleError::BlockingTaskJoin(error.to_string()))?
    }
}

/// The checkpoint's size, after checking it is a one-class, 17-keypoint
/// YOLOv8 pose checkpoint.
fn detect_size(weights: &[u8]) -> Result<PoseModelSize, CandleError> {
    if weights.is_empty() {
        return Err(CandleError::EmptyBuffer {
            artifact: "pose weights",
        });
    }
    let tensors = SafeTensors::deserialize(weights)
        .map_err(|error| CandleError::InvalidCheckpoint(error.to_string()))?;
    let leading = |name: &str| -> Result<usize, CandleError> {
        let tensor = tensors
            .tensor(name)
            .map_err(|_| CandleError::MissingTensor(name.to_owned()))?;
        tensor
            .shape()
            .first()
            .copied()
            .ok_or_else(|| CandleError::InvalidCheckpoint(format!("`{name}` is a scalar")))
    };
    let stem = leading(STEM_TENSOR)?;
    let size = PoseModelSize::from_stem(stem).ok_or_else(|| {
        CandleError::InvalidCheckpoint(format!(
            "`{STEM_TENSOR}` has {stem} channels, which no YOLOv8 size has"
        ))
    })?;
    let keypoints = leading(KEYPOINT_TENSOR)?;
    let classes = leading(CLASS_TENSOR)?;
    if keypoints != KEYPOINT_VALUES || classes != 1 {
        return Err(CandleError::InvalidCheckpoint(format!(
            "a YOLOv8 pose checkpoint predicts one class and {KEYPOINT_VALUES} keypoint values; \
             this one predicts {classes} and {keypoints}"
        )));
    }
    Ok(size)
}

/// One item of a pose reply: a frame's event, or the failure that ends it.
type PoseItem = Result<Step<PoseEstimation>, ProviderError>;

impl Transport<Local<PoseEstimation>> for CandlePoseModel {
    /// Refuses a request without frames; each frame is answered as it is
    /// estimated. Unary and streamed calls run the same inference.
    fn send(&self, request: PoseRequest, _exchange: Exchange) -> Opening<Step<PoseEstimation>> {
        if request.frames.is_empty() {
            return Opening::failed(
                CandleError::InvalidImage("a pose request needs at least one frame".into()).into(),
            );
        }
        #[cfg(not(target_family = "wasm"))]
        if self.state.concurrency.is_closed() {
            return Opening::failed(CandleError::ConcurrencyControllerClosed.into());
        }
        let model = self.clone();
        Opening::new(async move {
            Ok(match model.open(request).await {
                Ok(items) => Opened::new(items),
                Err(error) => Opened::failed(error),
            })
        })
    }
}

impl CandlePoseModel {
    /// Estimate every frame on a blocking worker, delivering each frame's
    /// event through a bounded channel. Dropping the stream stops the worker
    /// before its next frame.
    #[cfg(not(target_family = "wasm"))]
    async fn open(
        &self,
        request: PoseRequest,
    ) -> Result<WasmBoxedStream<'static, PoseItem>, ProviderError> {
        let cancellation = CancellationSignal::default();
        let mut cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let permit = acquire_concurrency(Arc::clone(&self.state.concurrency)).await?;
        let state = Arc::clone(&self.state);
        let (sender, receiver) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let producer = sender.clone();
        let producer_cancellation = cancellation.clone();
        let task = tokio::task::spawn_blocking(move || {
            let result = state.runtime.device().with_context(|| {
                state.estimate_frames(request, &producer_cancellation, |event| {
                    producer
                        .blocking_send(Ok(Step::Event(event)))
                        .map_err(|_| CandleError::StreamingChannelClosed)
                })
            });
            let _ = producer.blocking_send(match result {
                Ok(()) => Ok(Step::End(())),
                Err(error) => Err(error.into()),
            });
            drop(permit);
        });
        tokio::spawn(async move {
            if let Err(error) = task.await {
                let error = CandleError::BlockingTaskJoin(error.to_string());
                let _ = sender.send(Err(error.into())).await;
            }
        });
        cancel_on_drop.disarm();
        Ok(Box::pin(ReceiverStream::new(receiver, cancellation)))
    }

    /// Estimate every frame synchronously; the reply is complete when it
    /// opens.
    #[cfg(target_family = "wasm")]
    async fn open(
        &self,
        request: PoseRequest,
    ) -> Result<WasmBoxedStream<'static, PoseItem>, ProviderError> {
        let mut items = Vec::new();
        if let Err(error) = self
            .state
            .estimate_frames(request, &CancellationSignal, |event| {
                items.push(Ok(Step::Event(event)));
                Ok(())
            })
        {
            items.push(Err(error.into()));
        } else {
            items.push(Ok(Step::End(())));
        }
        Ok(Box::pin(futures::stream::iter(items)))
    }
}

impl LoadedPose {
    /// Estimate each frame in order and hand its event to `emit`, checking
    /// for cancellation before each frame.
    fn estimate_frames(
        &self,
        request: PoseRequest,
        cancellation: &CancellationSignal,
        mut emit: impl FnMut(FramePoses) -> Result<(), CandleError>,
    ) -> Result<(), CandleError> {
        let frames = request.frames.len();
        for (frame, image) in request.frames.into_iter().enumerate() {
            check_cancellation(cancellation)?;
            let started = web_time::Instant::now();
            let people = self
                .estimate(&image)
                .map_err(|error| CandleError::Inference(error.to_string()))?;
            emit(FramePoses {
                frame,
                frames,
                width: image.width,
                height: image.height,
                people,
                elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            })?;
        }
        Ok(())
    }

    fn estimate(&self, image: &ImageFrame) -> candle_core::Result<Vec<PersonPose>> {
        let (input_width, input_height) =
            input_dimensions(image.width, image.height, self.settings.input_size);
        let device = self.runtime.device();
        let pixels = Tensor::from_slice(
            &image.pixels,
            (image.height as usize, image.width as usize, 3),
            device,
        )?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(1. / 255., 0.)?
        .unsqueeze(0)?
        .contiguous()?;
        let pixels = if (input_width, input_height) == (image.width, image.height) {
            pixels
        } else {
            pixels.upsample_bilinear2d(input_height as usize, input_width as usize, false)?
        };
        let predictions = self.network.forward(&pixels)?.i(0)?;
        let scale = (
            image.width as f32 / input_width as f32,
            image.height as f32 / input_height as f32,
        );
        decode_people(
            &predictions,
            &self.settings,
            scale,
            (image.width as f32, image.height as f32),
        )
    }
}

/// The network input for a `width` by `height` frame: the longest side is
/// `input_size`, and the other keeps the aspect ratio, rounded down to a
/// multiple of the stride.
fn input_dimensions(width: u32, height: u32, input_size: u32) -> (u32, u32) {
    let scaled = |short: u32, long: u32| {
        let side = u64::from(short) * u64::from(input_size) / u64::from(long.max(1));
        let side = u32::try_from(side).unwrap_or(input_size);
        (side / STRIDE * STRIDE).max(STRIDE)
    };
    if width >= height {
        (input_size, scaled(height, width))
    } else {
        (scaled(width, height), input_size)
    }
}

/// The people in one frame's `(56, anchors)` predictions: confident anchors,
/// overlapping boxes suppressed, scaled from network input to frame pixels.
fn decode_people(
    predictions: &Tensor,
    settings: &PoseSettings,
    (scale_x, scale_y): (f32, f32),
    (width, height): (f32, f32),
) -> candle_core::Result<Vec<PersonPose>> {
    let (rows, _anchors) = predictions.dims2()?;
    if rows != PREDICTION_ROWS {
        candle_core::bail!("expected {PREDICTION_ROWS} prediction rows, got {rows}");
    }
    let anchors = predictions.t()?.contiguous()?.to_vec2::<f32>()?;
    let mut candidates = Vec::new();
    for anchor in anchors {
        let [cx, cy, w, h, confidence, keypoints @ ..] = anchor.as_slice() else {
            continue;
        };
        if !confidence.is_finite() || *confidence <= settings.confidence_threshold {
            continue;
        }
        let keypoints: Vec<Keypoint> = keypoints
            .chunks_exact(3)
            .filter_map(|values| match values {
                [x, y, confidence] => Some(Keypoint {
                    x: x * scale_x,
                    y: y * scale_y,
                    confidence: *confidence,
                }),
                _ => None,
            })
            .collect();
        let Ok(keypoints) = <[Keypoint; 17]>::try_from(keypoints) else {
            continue;
        };
        candidates.push(Bbox {
            xmin: cx - w / 2.,
            ymin: cy - h / 2.,
            xmax: cx + w / 2.,
            ymax: cy + h / 2.,
            confidence: *confidence,
            data: keypoints,
        });
    }
    let mut candidates = [candidates];
    non_maximum_suppression(&mut candidates, settings.iou_threshold);
    let [people] = candidates;
    Ok(people
        .into_iter()
        .map(|candidate| PersonPose {
            bbox: BoundingBox {
                x_min: (candidate.xmin * scale_x).clamp(0., width),
                y_min: (candidate.ymin * scale_y).clamp(0., height),
                x_max: (candidate.xmax * scale_x).clamp(0., width),
                y_max: (candidate.ymax * scale_y).clamp(0., height),
            },
            confidence: candidate.confidence,
            keypoints: candidate.data,
        })
        .collect())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
mod tests;
