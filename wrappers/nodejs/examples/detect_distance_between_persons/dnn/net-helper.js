const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');

const PERSON_CODE = 15; // code from classNames
const confidenceThreshold = 0.8;

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

const classifyPersons = (net, imageMat, scaleFactor, inWidth, inHeight, meanVal) => {
  // get height and width from the image
  const imgHeight = imageMat.rows;
  const imgWidth = imageMat.cols;

  // Covert Mat object to blob
  const inputBlob = cv.blobFromImage(
    imageMat,
    scaleFactor,
    new cv.Size(inWidth, inHeight),
    new cv.Vec3(meanVal, meanVal, meanVal),
    false
  );

  net.setInput(inputBlob);
  const outputBlob = net.forward();

  const numRows = outputBlob.sizes.slice(2, 3);
  const persons = [];
  for (let y = 0; y < numRows; y += 1) {
    const confidence = outputBlob.at([0, 0, y, 2]);
    const classId = outputBlob.at([0, 0, y, 1]);
    console.log(`classId: ${classId}, confidence: ${confidence}`);

    if(classId === PERSON_CODE && confidence > confidenceThreshold ) {
      const xLeftBottom = imgWidth * outputBlob.at([0, 0, y, 3]);
      const yLeftBottom = imgHeight * outputBlob.at([0, 0, y, 4]);
      const xRightTop = imgWidth * outputBlob.at([0, 0, y, 5]);
      const yRightTop = imgHeight * outputBlob.at([0, 0, y, 6]);

      // count center of person area
      const centerX = xLeftBottom + (xRightTop - xLeftBottom) / 2;
      const centerY = yLeftBottom + (yRightTop - yLeftBottom) / 2;

      console.log(`person found: ${confidence} coordinates: ${centerX},${centerY}`);
      persons.push({
        x: centerX,
        y: centerY
      });
    }
  }

  return persons;
};

module.exports = {
  loadNet,
  classifyPersons,
};
