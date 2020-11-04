function handleDOMContentLoaded() {
  const canvas = document.getElementsByTagName('canvas').item(0);
  const context = canvas.getContext('2d');
  let isPainting = false;

  function startPosition(e) {
    isPainting = true;
    draw(e);
  }

  function finishedPosition() {
    isPainting = false;
    context.beginPath();
  }

  function draw(e) {
    if (!isPainting) {
      return;
    }

    // Get coordinates translated to canvas-local coordinates
    context.lineTo(e.clientX, e.clientY);
    context.stroke();
    context.beginPath();
    context.moveTo(e.clientX, e.clientY);
  }

  context.lineWidth = 10;
  context.lineCap = 'round';
  context.strokeStyle = 'black';

  canvas.addEventListener('mousedown', startPosition);
  canvas.addEventListener('mouseup', finishedPosition);
  canvas.addEventListener('mousemove', draw);
}

function addCanvasEventListeners(canvas, context) {
  canvas.addEventListener('mousedown', handleMouseDown);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', handleDOMContentLoaded);
} else {
  handleDOMContentLoaded();
}
