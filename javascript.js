let globalCanvas = undefined;
let globalContext = undefined;
let globalSession = undefined;

let drawCounter = 0;

function handleDOMContentLoaded() {
  const canvas = document.getElementsByTagName('canvas').item(0);
  const ctx = canvas.getContext('2d');
  const clearButton = document.getElementById('button-clear');
  const barRectsHTMLCollection = document.getElementsByClassName(
    'bar__rect-fill'
  );
  const barRects = [...barRectsHTMLCollection];
  let isPainting = false;

  globalCanvas = canvas;
  globalContext = ctx;

  function drawCanvasPlaceholder() {
    const canvas = document.getElementsByTagName('canvas').item(0);
    const ctx = canvas.getContext('2d');
    const placeholderImage = new Image(280, 280);
    placeholderImage.src = './assets/images/canvas-placeholder.png';
    ctx.drawImage(placeholderImage, 0, 0);
  }

  function removeCanvasPlaceholder() {}

  function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function resetBars() {
    barRects.forEach((rect) => {
      rect.style.height = 0;
    });
  }

  function startPosition(e) {
    isPainting = true;
    draw(e);
  }

  function finishedPosition() {
    isPainting = false;
    ctx.beginPath();
  }

  function draw(e) {
    // Prevent scrolling on touch devices while drawing
    e.preventDefault();

    if (!isPainting) {
      return;
    }

    drawCounter += 1;

    const [x, y] = getCoordinates(e);

    // Get coordinates translated to canvas-local coordinates
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);

    makeInference();
  }

  function getCoordinates(e) {
    const { left, top } = event.target.getBoundingClientRect();
    let { clientX, clientY } = e;

    // for touch event
    if (e.touches && e.touches.length) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }

    const leftRelativeToDocument = window.scrollX + left;
    const topRelativeToDocument = window.scrollY + top;

    const x = clientX - leftRelativeToDocument;
    const y = clientY - topRelativeToDocument;
    return [x, y];
  }

  function handleClearClick() {
    resetCanvas();
    resetBars();
  }

  function softmax(arr) {
    const C = Math.max(...arr);
    const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
    return arr.map((value, index) => {
      return Math.exp(value - C) / d;
    });
  }

  function makeInference() {
    const input = preprocess(globalContext);
    const inputs = [input];
    let result;

    globalSession.run(inputs).then((output) => {
      const data = output.get('Plus214_Output_0')['data'];
      const probabilities = softmax(data);

      barRects.forEach((rect, index) => {
        rect.style.height = `${probabilities[index] * 100}%`;
      });
    });

    return result;
  }

  resetCanvas();
  drawCanvasPlaceholder();
  ctx.lineWidth = 25;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';

  canvas.addEventListener('mousedown', startPosition);
  canvas.addEventListener('mouseup', finishedPosition);
  canvas.addEventListener('mouseleave', finishedPosition);
  canvas.addEventListener('mousemove', draw);

  canvas.addEventListener('touchstart', startPosition);
  canvas.addEventListener('touchend', finishedPosition);
  canvas.addEventListener('touchmove', draw);

  clearButton.addEventListener('click', handleClearClick);

  // create a session
  const session = new onnx.InferenceSession();
  globalSession = session;
  session.loadModel('./model.onnx');
}

function addCanvasEventListeners(canvas, ctx) {
  canvas.addEventListener('mousedown', handleMouseDown);
}

function preprocess(ctx) {
  const ctxScaled = document
    .getElementById('input-canvas-scaled')
    .getContext('2d');
  ctxScaled.save();
  ctxScaled.scale(28 / ctx.canvas.width, 28 / ctx.canvas.height);
  ctxScaled.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctxScaled.drawImage(document.getElementById('input-canvas'), 0, 0);
  const imageDataScaled = ctxScaled.getImageData(
    0,
    0,
    ctxScaled.canvas.width,
    ctxScaled.canvas.height
  );
  ctxScaled.restore();
  // process image data for model input
  const { data } = imageDataScaled;
  const input = new Float32Array(784);
  for (let i = 0, len = data.length; i < len; i += 4) {
    // 0, represents black, 255 represents white
    // We want 0 for white AKA background, and 1 for black AKA foreground
    // data[i] represents the first channel, R, so Red
    input[i / 4] = 1 - data[i] / 255;
  }

  const tensor = new Tensor(input, 'float32', [1, 1, 28, 28]);
  return tensor;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', handleDOMContentLoaded);
} else {
  handleDOMContentLoaded();
}
