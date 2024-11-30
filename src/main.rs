use image::{imageops::FilterType, ImageReader};
use ndarray::{Array, Axis, Ix4};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    tensor::ArrayExtensions,
    value::Value,
};
use std::{collections::HashMap, error::Error, io::Cursor};

static MNIST_MODEL: &[u8] = include_bytes!("../models/mnist-12.onnx");

fn main() -> Result<(), Box<dyn Error>> {
    let result = run_inference(MNIST_MODEL, include_bytes!("../images/test5.jpg"))?;
    println!("Tensor values: {:?}", result);

    let max_index = result
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();
    println!("Predicted digit: {}", max_index);

    Ok(())
}

/**
 * ONNXモデルを読み込み、セッションを作成する
 */
fn create_session(model_data: &[u8]) -> Result<Session, Box<dyn Error>> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_memory(model_data)?;
    Ok(session)
}

/**
 * 画像データを前処理する
 */
fn preprocess_image(img_data: &[u8]) -> Result<Array<f32, Ix4>, Box<dyn Error>> {
    let img = ImageReader::new(Cursor::new(img_data))
        .with_guessed_format()?
        .decode()?
        .to_luma8();

    // 28×28にリサイズ
    let img = image::imageops::resize(&img, 28, 28, FilterType::Nearest);

    let img_array = Array::from_shape_vec(
        (1, 1, 28, 28),
        img.iter().map(|&p| p as f32 / 255.0).collect(),
    )?;

    Ok(img_array)
}

/**
 * 推論を実行する
 */
fn run_inference(model_data: &[u8], img_array: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    let model = create_session(model_data)?;
    let img_array = preprocess_image(img_array)?;

    let input_tensor: Value<ort::value::TensorValueType<f32>> = Value::from_array(img_array)?;
    let mut inputs = HashMap::new();
    inputs.insert("Input3", input_tensor);

    let outputs = model.run(inputs)?;

    let probabilities: Vec<f32> = outputs[0]
        .try_extract_tensor()?
        .softmax(Axis(1))
        .iter()
        .copied()
        .collect();

    Ok(probabilities)
}
