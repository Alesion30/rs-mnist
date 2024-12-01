use image::{ImageBuffer, Luma, Pixel};
use ort::{ArrayExtensions, Session};
use wasm_bindgen::prelude::*;

static IMAGE_BYTES: &[u8] = include_bytes!("../images/test5.jpg");
static MODEL_BYTES: &[u8] = include_bytes!("../models/mnist-12.onnx");

pub fn upsample_inner() -> ort::Result<()> {
    let session = Session::builder()?
        .commit_from_memory_directly(MODEL_BYTES)
        .expect("Could not read model from memory");

    let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
        image::load_from_memory(IMAGE_BYTES).unwrap().to_luma8();
    let array = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();
        (channels[c] as f32) / 255.0
    });

    let outputs = session.run(ort::inputs![array]?)?;

    let mut probabilities: Vec<(usize, f32)> = outputs[0]
        .try_extract_tensor()?
        .softmax(ndarray::Axis(1))
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    assert_eq!(
        probabilities[0].0, 5,
        "Expecting class '5' (got {})",
        probabilities[0].0
    );

    Ok(())
}

macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

#[wasm_bindgen]
pub fn upsample() {
    if let Err(e) = upsample_inner() {
        console_log!("Error occurred while performing inference: {e:?}");
    }
}

#[cfg(test)]
#[wasm_bindgen_test::wasm_bindgen_test]
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

    upsample();
}
