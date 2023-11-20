use candle_core::{Device, DType, Tensor, Result};
use candle_nn::{ops, VarBuilder, Linear, BatchNorm, Conv2d};
use image::io::Reader as ImageReader;
trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
struct MnistModel {
    fc1: Linear,
    fc2: Linear,
    bn1: BatchNorm,
    bn2: BatchNorm,
    conv1: Conv2d,
    conv2: Conv2d,
}

impl Model for MnistModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let fc1: Linear = candle_nn::linear(320, 50, vb.push_prefix("fc1"))?;
        let fc2 = candle_nn::linear(50, 10, vb.push_prefix("fc2"))?;
        let bn1 = candle_nn::batch_norm(10, 1e-5, vb.push_prefix("bn1"))?;
        let bn2 = candle_nn::batch_norm(20, 1e-5, vb.push_prefix("bn2"))?;
        let conv1 = candle_nn::conv2d(1, 10, 5, Default::default(), vb.push_prefix("conv1"))?;
        let conv2 = candle_nn::conv2d(10, 20, 5, Default::default(), vb.push_prefix("conv2"))?;

        Ok(Self{fc1, fc2, bn1, bn2, conv1, conv2})
    }
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;

        let x = xs.reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .relu()?
            .apply(&self.bn1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .relu()?
            .apply(&self.bn2)?
            .max_pool2d(2)?
            .reshape((b_sz, 320))?
            .apply(&self.fc1)?
            .relu()?
            .apply(&self.fc2)?;

            ops::log_softmax(&x, 1)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let model_path = "./models/mnist_cnn.safetensors";
    let vb= unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    // let net = safetensors::load(model_path.clone(), &device)?;
    // for (key, val) in net {
    //     println!("{}: {:}", key, val);
    // }

    let model = MnistModel::new(vb)?;

    let xs = ImageReader::open("./data/1.png")?.decode()?.into_bytes().to_vec();
    let input = Tensor::from_vec(xs.clone(), (1, 784), &device)?.to_dtype(DType::F32)?;
    let logits = model.forward(&input)?;

    println!("This digit is: {}", logits.argmax(1)?.to_vec1::<u32>()?[0]);

    Ok(())
}
