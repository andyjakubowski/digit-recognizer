function handleDOMContentLoaded() {
  const canvas = document.getElementsByTagName('canvas').item(0);
  const ctx = canvas.getContext('2d');
  const clearButton = document.getElementById('button-clear');
  const saveButton = document.getElementById('button-save');
  let isPainting = false;

  function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
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
    if (!isPainting) {
      return;
    }

    // Get coordinates translated to canvas-local coordinates
    ctx.lineTo(e.clientX, e.clientY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX, e.clientY);
  }

  function handleClearClick() {
    resetCanvas();
  }

  function handleSaveClick() {
    const image = new Image(canvas.width, canvas.height);
    image.src = canvas.toDataURL();
    document.body.append(image);
  }

  resetCanvas();
  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';

  canvas.addEventListener('mousedown', startPosition);
  canvas.addEventListener('mouseup', finishedPosition);
  canvas.addEventListener('mousemove', draw);

  clearButton.addEventListener('click', handleClearClick);
  saveButton.addEventListener('click', handleSaveClick);
}

function addCanvasEventListeners(canvas, ctx) {
  canvas.addEventListener('mousedown', handleMouseDown);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', handleDOMContentLoaded);
} else {
  handleDOMContentLoaded();
}
