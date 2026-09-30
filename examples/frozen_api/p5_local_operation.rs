use futures::{StreamExt, stream};
use rig::driver::{Exchange, Local, Model, Opened, Opening, Step, Transport};
use rig::error::ProviderError;
use rig::streaming::Item;
use rig::wire::{Call, Fold, Free, Operation, Reply};

/// Pose estimation over a sequence of video frames.
pub struct PoseEstimation;

#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl Frame {
    pub fn black(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgb: vec![0; (width * height * 3) as usize],
        }
    }
}

/// At least one frame: [`PoseEstimation`] rejects a request without one.
pub struct PoseRequest {
    pub frames: Vec<Frame>,
}

#[derive(Clone, Debug)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// One event per frame.
#[derive(Clone, Debug)]
pub struct FramePoses {
    pub frame: usize,
    pub people: Vec<Vec<Keypoint>>,
}

/// The folded response.
#[derive(Debug, Default)]
pub struct PoseTrack {
    pub frames: Vec<FramePoses>,
}

impl Operation for PoseEstimation {
    type Request = PoseRequest;
    type Event = FramePoses;
    // The runtime's end of the reply carries nothing, but it must be sent:
    // a runtime that stops early leaves the call truncated, not short.
    type End = ();
    type Response = PoseTrack;
    type Fold = PoseTrack;
    // Events are plain data this crate builds itself.
    type Emit = Free;

    fn fold(_request: &PoseRequest, _call: &mut Call<'_>) -> PoseTrack {
        PoseTrack::default()
    }

    // Checked before the request reaches the runtime.
    fn validate(request: &PoseRequest) -> Result<(), ProviderError> {
        if request.frames.is_empty() {
            return Err(ProviderError::request(
                "a pose request needs at least one frame",
            ));
        }
        Ok(())
    }
}

impl Fold<PoseEstimation> for PoseTrack {
    fn absorb(&mut self, poses: &FramePoses) -> Result<(), ProviderError> {
        self.frames.push(poses.clone());
        Ok(())
    }

    fn finish(self, _end: (), _reply: Reply) -> Result<PoseTrack, ProviderError> {
        Ok(self)
    }
}

/// The in-process runtime (a candle model in `rig-candle`).
#[derive(Clone)]
pub struct PoseRuntime {
    pub threshold: f32,
}

impl PoseRuntime {
    fn estimate(&self, index: usize, frame: &Frame) -> FramePoses {
        // Stand-in for the forward pass: one centred keypoint per frame.
        let centre = Keypoint {
            x: frame.width as f32 / 2.0,
            y: frame.height as f32 / 2.0,
            score: self.threshold,
        };
        FramePoses {
            frame: index,
            people: vec![vec![centre]],
        }
    }
}

impl Transport<Local<PoseEstimation>> for PoseRuntime {
    fn send(&self, request: PoseRequest, _exchange: Exchange) -> Opening<Step<PoseEstimation>> {
        let runtime = self.clone();
        let poses = stream::iter(request.frames.into_iter().enumerate())
            .map(move |(index, frame)| Ok(Step::Event(runtime.estimate(index, &frame))));
        let end = stream::once(async { Ok::<_, ProviderError>(Step::End(())) });
        Opening::ready(Opened::new(poses.chain(end)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::new(
        Local::<PoseEstimation>::new("candle-pose"),
        PoseRuntime { threshold: 0.5 },
    );
    let frames = vec![Frame::black(640, 480), Frame::black(640, 480)];

    let mut stream = model.stream(PoseRequest { frames })?;
    while let Some(item) = stream.next().await {
        if let Item::Event(poses) = item? {
            println!("frame {}: {} people", poses.frame, poses.people.len());
        }
    }
    let track = stream.finish().await?;
    println!("{} frames tracked", track.frames.len());
    Ok(())
}
