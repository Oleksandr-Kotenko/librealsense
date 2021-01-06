#!/usr/bin/env node

// Copyright (c) 2017 Intel Corporation. All rights reserved.
// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

'use strict';

const rs2 = require('../../index.js');
const {GLFWWindow, glfw} = require('../glfw-window.js');
const cv = require('opencv4nodejs');
const { PNG } = require('pngjs');

// A GLFW Window to display the captured image
const win = new GLFWWindow(1280, 720, 'Node.js Detect Example');

// The main work pipeline of camera
const pipeline = new rs2.Pipeline();
const colorizer = new rs2.Colorizer();

// create classifier
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

/**
 * get image from frame and pass it into classificator.detectMultiScale
 * @param {Frame} frame - video frame from framset
 */
function getFaceRectFromFrame(frame) {
  const opt = {
    width: frame.width,
    height: frame.height,
    inputColorType: 2,
  };

  const png = new PNG(opt);
  png.data = frame.getData();
  const buf = PNG.sync.write(png, opt);
  
  const image = cv.imdecode(buf);
  if(image.empty) {
    return null;
  }
  const faceRect = classifier.detectMultiScale(image.bgrToGray());
  return faceRect;
}

/**
 * Get video frames and get recognition results
 * @param {FrameSet} frameset
 */
function getDetectedFaces(frameset) {
  for(let i = 0; i < frameset.size; i++) {
    let frame = frameset.at(i);
    if (frame instanceof rs2.VideoFrame) {
      if (frame instanceof rs2.DepthFrame) {
        frame = colorizer.colorize(frame);
      }
      
      const faces = getFaceRectFromFrame(frame);
      console.log(faces);
    }
  }
}

// Start the camera
pipeline.start();

while (! win.shouldWindowClose()) {
  
  const frameset = pipeline.waitForFrames();
  getDetectedFaces(frameset);
  // Paint the images onto the window
  win.beginPaint();
  const color = frameset.colorFrame;
  glfw.draw2x2Streams(win.window, 1, color.data, 'rgb8', color.width, color.height);
  win.endPaint();
}

pipeline.stop();
pipeline.destroy();
win.destroy();
rs2.cleanup();
