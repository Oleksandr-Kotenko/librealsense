#!/usr/bin/env node

// Copyright (c) 2017 Intel Corporation. All rights reserved.
// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

'use strict';

const rs2 = require('../../index.js');
const {GLFWWindow, glfw} = require('../glfw-window.js');
const cv = require('opencv4nodejs');
const { loadNet } = require('./dnn/net-helper');
const { frameToMat, depthFrameToMeters } = require('./common');

// A GLFW Window to display the captured image
const win = new GLFWWindow(1280, 720, 'Node.js Detect Example');

// The main work pipeline of camera
const pipeline = new rs2.Pipeline();
const colorizer = new rs2.Colorizer();

const net = loadNet();

// Start the camera
const config = pipeline.start();
const profile = config.getStream('RS2_STREAM_COLOR');

while (! win.shouldWindowClose()) {

  const frameset = pipeline.waitForFrames();
  const colorFrame = frameset.colorFrame;
  const depthFrame = frameset.depthFrame;
  const matFrame = frameToMat(frameset);
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