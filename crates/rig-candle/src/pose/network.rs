//! The YOLOv8 pose network: a DarkNet backbone, a feature-pyramid neck and a
//! pose head that predicts one box, one confidence and the COCO keypoints
//! per anchor. The layout and tensor names follow Candle's YOLOv8 example
//! (MIT or Apache-2.0), so its safetensors checkpoints load unchanged.

use candle_core::{D, DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, batch_norm, conv2d, conv2d_no_bias};

/// Depth, width and ratio multiples of one YOLOv8 size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Multiples {
    depth: f64,
    width: f64,
    ratio: f64,
}

impl Multiples {
    pub(crate) const N: Self = Self::new(0.33, 0.25, 2.0);
    pub(crate) const S: Self = Self::new(0.33, 0.50, 2.0);
    pub(crate) const M: Self = Self::new(0.67, 0.75, 1.5);
    pub(crate) const L: Self = Self::new(1.00, 1.00, 1.0);
    pub(crate) const X: Self = Self::new(1.00, 1.25, 1.0);

    const fn new(depth: f64, width: f64, ratio: f64) -> Self {
        Self {
            depth,
            width,
            ratio,
        }
    }

    /// Channels scaled by the width multiple.
    fn channels(&self, base: f64) -> usize {
        (base * self.width) as usize
    }

    /// Repeats scaled by the depth multiple.
    fn repeats(&self, base: f64) -> usize {
        (base * self.depth).round() as usize
    }

    /// The channels of the three feature maps the head reads.
    fn filters(&self) -> (usize, usize, usize) {
        (
            self.channels(256.),
            self.channels(512.),
            self.channels(512. * self.ratio),
        )
    }
}

/// A convolution with its batch norm folded in, then SiLU.
#[derive(Debug)]
struct ConvBlock {
    conv: Conv2d,
}

impl ConvBlock {
    fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize, stride: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: k / 2,
            stride,
            ..Default::default()
        };
        let bn = batch_norm(c2, 1e-3, vb.pp("bn"))?;
        let conv = conv2d_no_bias(c1, c2, k, cfg, vb.pp("conv"))?.absorb_bn(&bn)?;
        Ok(Self { conv })
    }
}

impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(&self.conv.forward(xs)?)
    }
}

#[derive(Debug)]
struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    residual: bool,
}

impl Bottleneck {
    fn load(vb: VarBuilder, c1: usize, c2: usize, shortcut: bool) -> Result<Self> {
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c1, c2, 3, 1)?,
            cv2: ConvBlock::load(vb.pp("cv2"), c2, c2, 3, 1)?,
            residual: c1 == c2 && shortcut,
        })
    }
}

impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.cv2.forward(&self.cv1.forward(xs)?)?;
        if self.residual { xs + ys } else { Ok(ys) }
    }
}

/// Cross-stage partial block with two convolutions.
#[derive(Debug)]
struct C2f {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottleneck: Vec<Bottleneck>,
}

impl C2f {
    fn load(vb: VarBuilder, c1: usize, c2: usize, n: usize, shortcut: bool) -> Result<Self> {
        let c = c2 / 2;
        let bottleneck = (0..n)
            .map(|index| Bottleneck::load(vb.pp(format!("bottleneck.{index}")), c, c, shortcut))
            .collect::<Result<_>>()?;
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c1, 2 * c, 1, 1)?,
            cv2: ConvBlock::load(vb.pp("cv2"), (2 + n) * c, c2, 1, 1)?,
            bottleneck,
        })
    }
}

impl Module for C2f {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut ys = self.cv1.forward(xs)?.chunk(2, 1)?;
        let mut last = ys
            .last()
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("C2f split produced no chunk".into()))?;
        for bottleneck in &self.bottleneck {
            last = bottleneck.forward(&last)?;
            ys.push(last.clone());
        }
        self.cv2.forward(&Tensor::cat(&ys, 1)?)
    }
}

/// Spatial pyramid pooling, fast: three chained max pools.
#[derive(Debug)]
struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
}

impl Sppf {
    fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize) -> Result<Self> {
        let c = c1 / 2;
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c1, c, 1, 1)?,
            cv2: ConvBlock::load(vb.pp("cv2"), c * 4, c2, 1, 1)?,
            k,
        })
    }

    fn pool(&self, xs: &Tensor) -> Result<Tensor> {
        let pad = self.k / 2;
        xs.pad_with_zeros(2, pad, pad)?
            .pad_with_zeros(3, pad, pad)?
            .max_pool2d_with_stride(self.k, 1)
    }
}

impl Module for Sppf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.cv1.forward(xs)?;
        let xs2 = self.pool(&xs)?;
        let xs3 = self.pool(&xs2)?;
        let xs4 = self.pool(&xs3)?;
        self.cv2.forward(&Tensor::cat(&[&xs, &xs2, &xs3, &xs4], 1)?)
    }
}

/// Distribution focal loss decoding: the expected box side from 16 bins.
#[derive(Debug)]
struct Dfl {
    conv: Conv2d,
    bins: usize,
}

impl Dfl {
    fn load(vb: VarBuilder, bins: usize) -> Result<Self> {
        let conv = conv2d_no_bias(bins, 1, 1, Default::default(), vb.pp("conv"))?;
        Ok(Self { conv, bins })
    }
}

impl Module for Dfl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, _channels, anchors) = xs.dims3()?;
        let xs = xs
            .reshape((batch, 4, self.bins, anchors))?
            .transpose(2, 1)?;
        let xs = candle_nn::ops::softmax(&xs, 1)?;
        self.conv.forward(&xs)?.reshape((batch, 4, anchors))
    }
}

#[derive(Debug)]
struct DarkNet {
    b1: [ConvBlock; 2],
    b2_0: C2f,
    b2_1: ConvBlock,
    b2_2: C2f,
    b3_0: ConvBlock,
    b3_1: C2f,
    b4_0: ConvBlock,
    b4_1: C2f,
    b5: Sppf,
}

impl DarkNet {
    fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let c = |base| m.channels(base);
        let top = c(512. * m.ratio);
        Ok(Self {
            b1: [
                ConvBlock::load(vb.pp("b1.0"), 3, c(64.), 3, 2)?,
                ConvBlock::load(vb.pp("b1.1"), c(64.), c(128.), 3, 2)?,
            ],
            b2_0: C2f::load(vb.pp("b2.0"), c(128.), c(128.), m.repeats(3.), true)?,
            b2_1: ConvBlock::load(vb.pp("b2.1"), c(128.), c(256.), 3, 2)?,
            b2_2: C2f::load(vb.pp("b2.2"), c(256.), c(256.), m.repeats(6.), true)?,
            b3_0: ConvBlock::load(vb.pp("b3.0"), c(256.), c(512.), 3, 2)?,
            b3_1: C2f::load(vb.pp("b3.1"), c(512.), c(512.), m.repeats(6.), true)?,
            b4_0: ConvBlock::load(vb.pp("b4.0"), c(512.), top, 3, 2)?,
            b4_1: C2f::load(vb.pp("b4.1"), top, top, m.repeats(3.), true)?,
            b5: Sppf::load(vb.pp("b5.0"), top, top, 5)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let [b1_0, b1_1] = &self.b1;
        let x1 = b1_1.forward(&b1_0.forward(xs)?)?;
        let x2 = self
            .b2_2
            .forward(&self.b2_1.forward(&self.b2_0.forward(&x1)?)?)?;
        let x3 = self.b3_1.forward(&self.b3_0.forward(&x2)?)?;
        let x4 = self.b4_1.forward(&self.b4_0.forward(&x3)?)?;
        Ok((x2, x3, self.b5.forward(&x4)?))
    }
}

#[derive(Debug)]
struct Neck {
    n1: C2f,
    n2: C2f,
    n3: ConvBlock,
    n4: C2f,
    n5: ConvBlock,
    n6: C2f,
}

impl Neck {
    fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let c = |base| m.channels(base);
        let n = m.repeats(3.);
        Ok(Self {
            n1: C2f::load(vb.pp("n1"), c(512. * (1. + m.ratio)), c(512.), n, false)?,
            n2: C2f::load(vb.pp("n2"), c(768.), c(256.), n, false)?,
            n3: ConvBlock::load(vb.pp("n3"), c(256.), c(256.), 3, 2)?,
            n4: C2f::load(vb.pp("n4"), c(768.), c(512.), n, false)?,
            n5: ConvBlock::load(vb.pp("n5"), c(512.), c(512.), 3, 2)?,
            n6: C2f::load(
                vb.pp("n6"),
                c(512. * (1. + m.ratio)),
                c(512. * m.ratio),
                n,
                false,
            )?,
        })
    }

    fn forward(&self, p3: &Tensor, p4: &Tensor, p5: &Tensor) -> Result<[Tensor; 3]> {
        let x = self.n1.forward(&Tensor::cat(&[&upsample(p5)?, p4], 1)?)?;
        let head_1 = self.n2.forward(&Tensor::cat(&[&upsample(&x)?, p3], 1)?)?;
        let head_2 = self
            .n4
            .forward(&Tensor::cat(&[&self.n3.forward(&head_1)?, &x], 1)?)?;
        let head_3 = self
            .n6
            .forward(&Tensor::cat(&[&self.n5.forward(&head_2)?, p5], 1)?)?;
        Ok([head_1, head_2, head_3])
    }
}

/// Nearest-neighbour upsampling by two.
fn upsample(xs: &Tensor) -> Result<Tensor> {
    let (_batch, _channels, h, w) = xs.dims4()?;
    xs.upsample_nearest2d(2 * h, 2 * w)
}

/// Two convolution blocks and a plain 1x1 convolution: one branch of a head
/// at one scale.
#[derive(Debug)]
struct Branch(ConvBlock, ConvBlock, Conv2d);

impl Branch {
    fn load(vb: VarBuilder, filter: usize, hidden: usize, out: usize) -> Result<Self> {
        Ok(Self(
            ConvBlock::load(vb.pp("0"), filter, hidden, 3, 1)?,
            ConvBlock::load(vb.pp("1"), hidden, hidden, 3, 1)?,
            conv2d(hidden, out, 1, Default::default(), vb.pp("2"))?,
        ))
    }

    /// One branch per feature map, named `<name>.0` to `<name>.2`.
    fn load_scales(
        vb: &VarBuilder,
        name: &str,
        filters: (usize, usize, usize),
        hidden: usize,
        out: usize,
    ) -> Result<[Self; 3]> {
        Ok([
            Self::load(vb.pp(format!("{name}.0")), filters.0, hidden, out)?,
            Self::load(vb.pp(format!("{name}.1")), filters.1, hidden, out)?,
            Self::load(vb.pp(format!("{name}.2")), filters.2, hidden, out)?,
        ])
    }
}

impl Module for Branch {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.2.forward(&self.1.forward(&self.0.forward(xs)?)?)
    }
}

/// Strides of the three feature maps.
const STRIDES: [usize; 3] = [8, 16, 32];

/// Box distribution bins per side.
const BINS: usize = 16;

/// Predicts a box, a class score and the keypoints per anchor, in input
/// pixels.
#[derive(Debug)]
struct PoseHead {
    dfl: Dfl,
    boxes: [Branch; 3],
    classes: [Branch; 3],
    keypoints: [Branch; 3],
    classes_count: usize,
    kpt: (usize, usize),
}

impl PoseHead {
    fn load(
        vb: VarBuilder,
        classes_count: usize,
        kpt: (usize, usize),
        filters: (usize, usize, usize),
    ) -> Result<Self> {
        let nk = kpt.0 * kpt.1;
        Ok(Self {
            dfl: Dfl::load(vb.pp("dfl"), BINS)?,
            boxes: Branch::load_scales(
                &vb,
                "cv2",
                filters,
                usize::max(filters.0 / 4, BINS * 4),
                4 * BINS,
            )?,
            classes: Branch::load_scales(
                &vb,
                "cv3",
                filters,
                usize::max(filters.0, classes_count),
                classes_count,
            )?,
            keypoints: Branch::load_scales(&vb, "cv4", filters, usize::max(filters.0 / 4, nk), nk)?,
            classes_count,
            kpt,
        })
    }

    /// `(batch, 4 + classes + keypoints * 3, anchors)`: box centre and size,
    /// class scores, then each keypoint's x, y and visibility.
    fn forward(&self, features: &[Tensor; 3]) -> Result<Tensor> {
        let per_anchor = 4 * BINS + self.classes_count;
        let mut detections = Vec::with_capacity(3);
        let mut keypoints = Vec::with_capacity(3);
        let mut anchors = Vec::with_capacity(3);
        let mut strides = Vec::with_capacity(3);
        for ((((xs, boxes), classes), kpts), stride) in features
            .iter()
            .zip(&self.boxes)
            .zip(&self.classes)
            .zip(&self.keypoints)
            .zip(STRIDES)
        {
            let (batch, _, h, w) = xs.dims4()?;
            let detection = Tensor::cat(&[boxes.forward(xs)?, classes.forward(xs)?], 1)?;
            detections.push(detection.reshape((batch, per_anchor, h * w))?);
            keypoints.push(
                kpts.forward(xs)?
                    .reshape((batch, self.kpt.0 * self.kpt.1, h * w))?,
            );
            let (points, scale) = anchor_grid(xs, stride)?;
            anchors.push(points);
            strides.push(scale);
        }
        // Anchor centres `(1, 2, anchors)` and their strides `(1, anchors)`.
        let anchors = Tensor::cat(&anchors, 0)?.t()?.unsqueeze(0)?;
        let strides = Tensor::cat(&strides, 0)?.unsqueeze(0)?;

        let detection = Tensor::cat(&detections, 2)?;
        let distances = self.dfl.forward(&detection.i((.., ..4 * BINS))?)?;
        let boxes = distance_to_box(&distances, &anchors)?.broadcast_mul(&strides.unsqueeze(1)?)?;
        let scores = candle_nn::ops::sigmoid(&detection.i((.., 4 * BINS..))?)?;

        let keypoints = Tensor::cat(&keypoints, D::Minus1)?;
        let (batch, _, count) = keypoints.dims3()?;
        let keypoints = keypoints.reshape((batch, self.kpt.0, self.kpt.1, count))?;
        let positions =
            ((keypoints.i((.., .., 0..2))? * 2.)?.broadcast_add(&anchors.unsqueeze(1)?)? - 0.5)?
                .broadcast_mul(&strides.unsqueeze(1)?.unsqueeze(1)?)?;
        let visibility = candle_nn::ops::sigmoid(&keypoints.i((.., .., 2..3))?)?;
        let keypoints = Tensor::cat(&[positions, visibility], 2)?.flatten(1, 2)?;
        Tensor::cat(&[boxes, scores, keypoints], 1)
    }
}

/// The cell centres of one feature map, `(h * w, 2)`, and their stride,
/// `(h * w)`.
fn anchor_grid(xs: &Tensor, stride: usize) -> Result<(Tensor, Tensor)> {
    let device = xs.device();
    let (_, _, h, w) = xs.dims4()?;
    let sx = (Tensor::arange(0, w as u32, device)?.to_dtype(DType::F32)? + 0.5)?;
    let sy = (Tensor::arange(0, h as u32, device)?.to_dtype(DType::F32)? + 0.5)?;
    let sx = sx.reshape((1, w))?.repeat((h, 1))?.flatten_all()?;
    let sy = sy.reshape((h, 1))?.repeat((1, w))?.flatten_all()?;
    let points = Tensor::stack(&[&sx, &sy], D::Minus1)?;
    let scale = (Tensor::ones(h * w, DType::F32, device)? * stride as f64)?;
    Ok((points, scale))
}

/// Left-top and right-bottom distances from each anchor to its box centre
/// and size.
fn distance_to_box(distance: &Tensor, anchors: &Tensor) -> Result<Tensor> {
    let lt = distance.i((.., 0..2))?;
    let rb = distance.i((.., 2..4))?;
    let x1y1 = anchors.broadcast_sub(&lt)?;
    let x2y2 = anchors.broadcast_add(&rb)?;
    let centre = ((&x1y1 + &x2y2)? * 0.5)?;
    let size = (&x2y2 - &x1y1)?;
    Tensor::cat(&[centre, size], 1)
}

/// The whole pose network.
#[derive(Debug)]
pub(crate) struct YoloV8Pose {
    net: DarkNet,
    fpn: Neck,
    head: PoseHead,
}

impl YoloV8Pose {
    /// Load a `(keypoints, 3)` pose network of size `m` with one class.
    pub(crate) fn load(vb: VarBuilder, m: Multiples, keypoints: usize) -> Result<Self> {
        Ok(Self {
            net: DarkNet::load(vb.pp("net"), m)?,
            fpn: Neck::load(vb.pp("fpn"), m)?,
            head: PoseHead::load(vb.pp("head"), 1, (keypoints, 3), m.filters())?,
        })
    }
}

impl Module for YoloV8Pose {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (p3, p4, p5) = self.net.forward(xs)?;
        self.head.forward(&self.fpn.forward(&p3, &p4, &p5)?)
    }
}
