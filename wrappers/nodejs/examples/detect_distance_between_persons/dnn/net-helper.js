const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');

const loadNet = () => {
  const modelPath = path.resolve(__dirname, './model');

  const prototxt = path.resolve(modelPath, 'MobileNetSSD_deploy.prototxt');
  const modelFile = path.resolve(modelPath, 'MobileNetSSD_deploy.caffemodel');

  if (!fs.existsSync(prototxt) || !fs.existsSync(modelFile)) {
    console.log('could not find MobileNet-SSD object detection network');
    console.log('download the prototxt from: https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/f5d072ccc7e3dcddaa830e9805da4bf1000b2836/MobileNetSSD_deploy.prototxt');
    console.log('download the model from: https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc');
    throw new Error('exiting');
  }
  return cv.readNetFromCaffe(prototxt, modelFile);
};

module.exports = {
  loadNet,
};
