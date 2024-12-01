use image::{ImageBuffer, Luma, Pixel};
use ort::{ArrayExtensions, Session};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use ndarray::Array4;
use serde_wasm_bindgen::to_value;

static MODEL_BYTES: &[u8] = include_bytes!("../models/mnist.ort");

#[wasm_bindgen]
pub fn classify_image(image_bytes: &[u8]) -> Result<JsValue, JsValue> {
    let session_builder = match Session::builder() {
        Ok(builder) => builder,
        Err(e) => return Err(JsValue::from_str(&format!("Could not create session builder: {:?}", e))),
    };

    let session = match session_builder.commit_from_memory_directly(MODEL_BYTES) {
        Ok(s) => s,
        Err(e) => return Err(JsValue::from_str(&format!("Could not read model from memory: {:?}", e))),
    };

    let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = match image::load_from_memory(image_bytes) {
        Ok(img) => img.to_luma8(),
        Err(e) => return Err(JsValue::from_str(&format!("Could not load image from memory: {:?}", e))),
    };

    let array = Array4::from_shape_fn((1, 1, 28, 28), |(_, _, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();
        return (channels[0] as f32) / 255.0
    });

    let inputs = match ort::inputs![array] {
        Ok(i) => i,
        Err(e) => return Err(JsValue::from_str(&format!("Error creating inputs: {:?}", e))),
    };

    let outputs = match session.run(inputs) {
        Ok(o) => o,
        Err(e) => return Err(JsValue::from_str(&format!("Error during inference: {:?}", e))),
    };

    let probabilities: Vec<f32> = match outputs[0].try_extract_tensor() {
        Ok(tensor) => tensor.softmax(ndarray::Axis(1)).iter().copied().collect(),
        Err(e) => return Err(JsValue::from_str(&format!("Error extracting tensor: {:?}", e))),
    };

    // 確率をパーセンテージ形式で小数点第10位までフォーマット
    let formatted_probabilities: Vec<String> = probabilities.iter().map(|&x| format!("{:.10}%", x * 100.0)).collect();

    Ok(to_value(&formatted_probabilities).map_err(|e| JsValue::from_str(&format!("Error serializing output: {:?}", e)))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::console_log;
    use wasm_bindgen_test::wasm_bindgen_test;
    use serde_wasm_bindgen::from_value;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn run_test() {
        use tracing::Level;
        use tracing_subscriber::fmt;
        use tracing_subscriber_wasm::MakeConsoleWriter;

        #[cfg(target_arch = "wasm32")]
        ort::wasm::initialize();

        fmt()
            .with_ansi(false)
            .with_max_level(Level::DEBUG)
            .with_writer(MakeConsoleWriter::default().map_trace_level_to(Level::DEBUG))
            .without_time()
            .init();

        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        let image_bytes: &[u8] = include_bytes!("../images/test5.jpg");
        let result = classify_image(image_bytes).unwrap();

        let formatted_probabilities: Vec<String> = from_value(result).unwrap();
        console_log!("Probabilities: {:?}", formatted_probabilities);
    }
}
