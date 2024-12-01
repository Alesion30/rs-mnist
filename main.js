import init, { classify_image } from "./pkg/rs_mnist.js";

document.addEventListener("DOMContentLoaded", async () => {

  const img = document.getElementById("test5");
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, img.width, img.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const uint8Array = new Uint8Array(imageData.data);

  console.log("Uint8Array of image data:", uint8Array);

  console.log("Initializing WebAssembly module...");
  await init("./pkg/rs_mnist_bg.wasm");

  canvas.toBlob((blob) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const arrayBuffer = reader.result;
      const uint8Array = new Uint8Array(arrayBuffer);
      classify_image(uint8Array);
    };
    reader.readAsArrayBuffer(blob);
  }, "image/jpeg");

  // try {
  //   console.log("Initializing WebAssembly module...");
  //   await init("./pkg/rs_mnist_bg.wasm");
  //   const result = await classify_image(uint8Array);
  //   console.log("Result:", result);
  // } catch (e) {
  //   console.error(e);
  // }
});
