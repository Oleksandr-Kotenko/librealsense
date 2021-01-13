#!/usr/bin/env node

// Copyright (c) 2017 Intel Corporation. All rights reserved.
// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

'use strict';

const rs2 = require('../../index.js');
const {GLFWWindow, glfw} = require('../glfw-window.js');
const cv = require('opencv4nodejs');
const { loadNet, classifyPersons } = require('./dnn/net-helper');
const { frameToMat, depthFrameToMeters } = require('./common');

const IN_WIDTH = 300;
const IN_HEIGHT = 300;
const WHRatio = IN_WIDTH / IN_HEIGHT;
const IN_SCALE_FACTOR = 0.007843;
const MEAN_VAL = 127.5;


// A GLFW Window to display the captured image
const win = new GLFWWindow(1280, 720, 'Node.js Detect Example');

// The main work pipeline of camera
const pipeline = new rs2.Pipeline();
const colorizer = new rs2.Colorizer();

const net = loadNet();

// Start the camera
const config = pipeline.start();
const profile = config.getStream('color');

while (! win.shouldWindowClose()) {

  const frameset = pipeline.waitForFrames();
  const colorFrame = frameset.colorFrame;
  const depthFrame = frameset.depthFrame;
  const colorMat = frameToMat(colorFrame);

  classifyPersons(net, colorMat, IN_SCALE_FACTOR, IN_WIDTH, IN_HEIGHT, MEAN_VAL);
  // Paint the images onto the window
  win.beginPaint();

  glfw.draw2x2Streams(win.window, 1, colorFrame.data, 'rgb8', colorFrame.width, colorFrame.height);
  win.endPaint();
}

pipeline.stop();
pipeline.destroy();
win.destroy();
rs2.cleanup();