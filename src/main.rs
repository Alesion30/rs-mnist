use image::{imageops::FilterType, ImageReader};
use ndarray::{Array, ArrayD, Ix4};
use ort::{session::{builder::GraphOptimizationLevel, Session}, value::Value};
use std::{collections::HashMap, error::Error, io::Cursor};

static MNIST_MODEL: &[u8] = include_bytes!("../models/mnist-12.onnx");

fn main() -> Result<(), Box<dyn Error>> {
    let model = create_session(MNIST_MODEL)?;
    let img_array = preprocess_image(include_bytes!("../images/test5.jpg"))?;

    run_inference(&model, img_array)?;

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
fn run_inference(model: &Session, img_array: Array<f32, Ix4>) -> Result<(), Box<dyn Error>> {
    let input_tensor: Value<ort::value::TensorValueType<f32>> = Value::from_array(img_array)?;
    let mut inputs = HashMap::new();
    inputs.insert("Input3", input_tensor);

    let outputs = model.run(inputs)?;

    // 結果を表示
    for (name, tensor) in outputs.iter() {
        println!("Output {}: {:?}", name, tensor);

        let array_view: ndarray::ArrayViewD<f32> = tensor.try_extract_tensor()?;
        let array: ArrayD<f32> = array_view.to_owned();
        println!("Tensor values: {:?}", array);

        // 最も高い値を持つインデックスを見つける
        let max_index = array
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        println!("Predicted digit: {}", max_index);
    }

    Ok(())
}
