const rs2 = require('../../index.js');
const cv = require('opencv4nodejs');
const { PNG } = require('pngjs');

/**
 * convert video frame to buffer
 * @param {Frame} frame
 * @return {Buffer}
 */
const frameToBuffer = frame => {
  const opt = {
    width: frame.width,
    height: frame.height,
    inputColorType: 2,
  };

  const png = new PNG(opt);
  png.data = frame.getData();
  return PNG.sync.write(png, opt);
};

/**
 * Convert rs2.frame to cv.Mat
 * @param {Frame} frame
 * @return {Mat}
 */
const frameToMat = frame => {
  const weight = frame.width;
  const height = frame.height;

  const frameFormat = frame.format;
  let matFrame;

  switch (frameFormat) {
    case rs2.format.FORMAT_BGR8:
      matFrame = new cv.Mat(Buffer.from(frame.getData()), weight, height, cv.CV_8UC3);
      break;
    case rs2.format.FORMAT_RGB8:
      const RGBMat = new cv.Mat(Buffer.from(frame.getData()), weight, height, cv.CV_8UC3);
      matFrame = RGBMat.cvtColor(cv.COLOR_RGB2BGR);
      break;
    case rs2.format.FORMAT_Z16:
      matFrame = new cv.Mat(Buffer.from(frame.getData()), weight, height, cv.CV_16UC1);
      break;
    case rs2.format.FORMAT_Y8:
      matFrame = new cv.Mat(Buffer.from(frame.getData()), weight, height, cv.CV_8UC1);
      break;
    case rs2.format.FORMAT_DISPARITY32:
      matFrame = new cv.Mat(Buffer.from(frame.getData()), weight, height, cv.CV_32FC1);
      break;
    default:
      throw new Error("Frame format is not supported");

  }

  return matFrame;
};

/**
 * Converts depth frame to a matrix of doubles with distances in meters
 * @param depthFrame {Frame}
 * @return {number}
 */
const depthFrameToMeters = depthFrame => {
  const matFrame = frameToMat(depthFrame);
  const depthMat = matFrame.convertTo(cv.CV_64F);
  return depthMat * depthFrame.getUnits();
};

module.exports = {
  frameToMat,
  depthFrameToMeters,
  frameToBuffer,
};