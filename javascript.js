let globalCanvas = undefined;
let globalContext = undefined;
let globalSession = undefined;
let answerText = undefined;

let drawCounter = 0;

function handleDOMContentLoaded() {
  const canvas = document.getElementsByTagName('canvas').item(0);
  const ctx = canvas.getContext('2d');
  const clearButton = document.getElementById('button-clear');
  // const saveButton = document.getElementById('button-save');
  let isPainting = false;

  globalCanvas = canvas;
  globalContext = ctx;
  answerText = document.getElementById('answer-text');

  function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    answerText.textContent = '🤔';
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
  }

  // function handleSaveClick() {
  //   const image = new Image(canvas.width, canvas.height);
  //   image.src = canvas.toDataURL();
  //   document.body.append(image);
  // }

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
      const highestProbability = Math.max(...probabilities);
      const maxIndex = probabilities.indexOf(highestProbability);
      answerText.textContent = String(maxIndex);
    });

    return result;
  }

  resetCanvas();
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
  // saveButton.addEventListener('click', handleSaveClick);

  // create a session
  const session = new onnx.InferenceSession();
  globalSession = session;
  session
    .loadModel('./mnist/model.onnx')
    .then(() => {
      // console.log('model loaded');
      // const array = [...Array(784).keys()];
      // const inputTensor = new Tensor(imageArray, 'float32', [1, 1, 28, 28]);
      // const inputs = [inputTensor];
      // return session.run(inputs);
    })
    .then((output) => {
      // const data = output.get('Plus214_Output_0')['data'];
      // console.log(`Likelihood: ${data}`);
    });
  // load the ONNX model file
  // myOnnxSession.loadModel('./rf_mnist.onnx').then(() => {
  //   console.log('model loaded');
  // // generate model input
  // const inferenceInputs = getInputs();
  // // execute the model
  // myOnnxSession.run(inferenceInputs).then((output) => {
  //   // consume the output
  //   const outputTensor = output.values().next().value;
  //   console.log(`model output tensor: ${outputTensor.data}.`);
  // });
  // });
}

function addCanvasEventListeners(canvas, ctx) {
  canvas.addEventListener('mousedown', handleMouseDown);
}

function preprocess(ctx) {
  // center crop
  // const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  // const imageDataCenterCrop = centerCrop(imageData);
  // const ctxCenterCrop = document
  //   .getElementById('input-canvas-centercrop')
  //   .getContext('2d');
  // ctxCenterCrop.canvas.width = imageDataCenterCrop.width;
  // ctxCenterCrop.canvas.height = imageDataCenterCrop.height;
  // ctxCenterCrop.putImageData(imageDataCenterCrop, 0, 0);
  // scaled to 28 x 28
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

function centerCrop(imageData) {
  const { data, width, height } = imageData;
  let [xmin, ymin] = [width, height];
  let [xmax, ymax] = [-1, -1];
  for (let i = 0; i < width; i++) {
    for (let j = 0; j < height; j++) {
      const idx = i + j * width;
      if (data[4 * idx + 3] > 0) {
        if (i < xmin) {
          xmin = i;
        }
        if (i > xmax) {
          xmax = i;
        }
        if (j < ymin) {
          ymin = j;
        }
        if (j > ymax) {
          ymax = j;
        }
      }
    }
  }

  // add a little padding
  xmin -= 20;
  xmax += 20;
  ymin -= 20;
  ymax += 20;

  // make bounding box square
  let [widthNew, heightNew] = [xmax - xmin + 1, ymax - ymin + 1];
  if (widthNew < heightNew) {
    // new width < new height
    const halfBefore = Math.floor((heightNew - widthNew) / 2);
    const halfAfter = heightNew - widthNew - halfBefore;
    xmax += halfAfter;
    xmin -= halfBefore;
  } else if (widthNew > heightNew) {
    // new width > new height
    const halfBefore = Math.floor((widthNew - heightNew) / 2);
    const halfAfter = widthNew - heightNew - halfBefore;
    ymax += halfAfter;
    ymin -= halfBefore;
  }

  widthNew = xmax - xmin + 1;
  heightNew = ymax - ymin + 1;
  const dataNew = new Uint8ClampedArray(widthNew * heightNew * 4);
  for (let i = xmin; i <= xmax; i++) {
    for (let j = ymin; j <= ymax; j++) {
      if (i >= 0 && i < width && j >= 0 && j < height) {
        const idx = i + j * width;
        const idxNew = i - xmin + (j - ymin) * widthNew;
        dataNew[4 * idxNew + 3] = data[4 * idx + 3];
      }
    }
  }

  return new ImageData(dataNew, widthNew, heightNew);
}

const imageArray = [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.054901958,
  0.19215685,
  0.16862744,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.035294116,
  0.47450978,
  0.73725486,
  0.7921569,
  0.75686276,
  0.78431374,
  0.6627451,
  0.09019607,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.17254901,
  0.77254903,
  0.6431372,
  0.20392156,
  0.14117646,
  0.0,
  0.035294116,
  0.52156866,
  0.7882353,
  0.082352936,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.37647057,
  0.8509804,
  0.3843137,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.517647,
  0.7294117,
  0.02352941,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0039215684,
  0.58823526,
  0.76862746,
  0.18431371,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.70980394,
  0.3960784,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.52549016,
  0.7176471,
  0.039215684,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.20392156,
  0.7921569,
  0.04705882,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.30196077,
  0.8156863,
  0.06666666,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.7607843,
  0.18431371,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0745098,
  0.827451,
  0.23921567,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.73333335,
  0.19999999,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.60392153,
  0.5568627,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.011764705,
  0.77254903,
  0.17254901,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.25490195,
  0.8039216,
  0.05098039,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.21568626,
  0.76862746,
  0.02352941,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.58431375,
  0.44313723,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.5764706,
  0.47450978,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.007843137,
  0.745098,
  0.2117647,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.05098039,
  0.8039216,
  0.13333333,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.058823526,
  0.78431374,
  0.07843137,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.33725488,
  0.69411767,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.04705882,
  0.8039216,
  0.11372548,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.6823529,
  0.34509802,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.5803921,
  0.54117644,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.12156862,
  0.80784315,
  0.058823526,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.09803921,
  0.84313726,
  0.24705881,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.50196075,
  0.5764706,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.26274508,
  0.85490197,
  0.23137254,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.12549019,
  0.84313726,
  0.1372549,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.2862745,
  0.8509804,
  0.44313723,
  0.043137252,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.16862744,
  0.8235294,
  0.42745095,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.15294117,
  0.6627451,
  0.78039217,
  0.5333333,
  0.34117645,
  0.25098038,
  0.2745098,
  0.517647,
  0.8352941,
  0.37647057,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.19215685,
  0.49019605,
  0.64705884,
  0.7058823,
  0.69411767,
  0.54117644,
  0.14509803,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
];

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', handleDOMContentLoaded);
} else {
  handleDOMContentLoaded();
}
