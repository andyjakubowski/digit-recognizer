# Digit Recognizer

A convolutional neural network pre-trained for handwritten digit recognition, running in the browser via [ONNX.js](https://github.com/microsoft/onnxjs).

![GIF of the Digit Recognizer demo](dist/assets/images/project-intro.gif)

## Try it out

[https://andyjakubowski.github.io/digit-recognizer/](https://andyjakubowski.github.io/digit-recognizer/)

## Folder structure

[`dist`](dist) is the demo website that runs the model.

[`notebooks`](notebooks) holds my own [scikit-learn](https://scikit-learn.org) model that I initially trained, but couldn’t deploy. It also includes the pre-trained model I downloaded from the [ONNX Model Zoo](https://github.com/onnx/models).

## Twitter thread

The goal of this project was to deploy a working machine learning model in five days. I described the details, and my learnings, in this [Twitter thread](https://twitter.com/jakubowskiandy/status/1326233668000641024).

![Image of my Twitter thread describing project learnings](dist/assets/images/twitter-thread.png)

## License

Licensed under the [MIT License](LICENSE). The pre-trained [MNIST model](https://github.com/onnx/models/tree/master/vision/classification/mnist) taken from the [ONNX Model Zoo](https://github.com/onnx/models) is licensed under the MIT License, too.
