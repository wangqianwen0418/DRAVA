import React, { useState, useEffect, useRef } from 'react';
import { Card, Slider } from 'antd';
import { TResultRow } from 'types';
import { getMax, getMin } from 'helpers';
import { datasetConfig } from 'config';
import { getItemURL } from 'dataService';

interface Props {
  isDataLoading: boolean;
  samples: TResultRow[];
  height: number;
  width: number;
  dataset: string;
}

const ImageContext = (props: Props) => {
  const { isDataLoading, samples, height, width, dataset } = props;

  const rootStyle = getComputedStyle(document.documentElement),
    cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

  const canvasRef = useRef(null);
  const imgSamples = samples.map(s => {
    const { id, filtered } = s;
    var x: number = 0,
      y: number = 0;
    if (dataset == 'IDC') {
      x = parseInt(id.match(/(_x)\d+/)![0].replace('_x', ''));
      y = parseInt(id.match(/(_y)\d+/)![0].replace('_y', ''));
    } else {
      x = s['x'];
      y = s['y'];
      if (x == undefined || y == undefined) {
        console.warn(
          `item.x or item.y is undefined for item ${id}.\n Please make sure each item has an x and a y property when using the context view`
        );
      }
    }

    const url = getItemURL(dataset, id);
    return { x, y, url, filtered };
  });

  const canvasLeft = getMin(imgSamples.map(s => +s.x)),
    canvasRight = getMax(imgSamples.map(s => +s.x)),
    canvasWidth = canvasRight - canvasLeft,
    canvasTop = getMin(imgSamples.map(s => +s.y)),
    canvasBottom = getMax(imgSamples.map(s => +s.y)),
    canvasHeight = canvasBottom - canvasTop;

  const imgSize = datasetConfig[dataset].imgSize || 50;

  // draw images
  useEffect(() => {
    const canvas: any = canvasRef.current;
    if (canvas == null) return;
    const ctx = canvas.getContext('2d');
    // const scale = Math.min(width / canvasWidth, (height - cardHeadHeight) / canvasHeight);
    // ctx.scale(scale, scale);

    imgSamples.forEach(sample => {
      const image = new Image(imgSize, imgSize); // Using optional size for image
      image.src = sample.url;
      image.onload = () => {
        // console.info(sample.x, sample.y, canvasTop, canvasLeft);
        ctx.drawImage(image, sample.x - canvasLeft, sample.y - canvasTop, imgSize, imgSize);
      };
    });
    return () => {
      ctx.clearRect(0, 0, width, height - cardHeadHeight);
    };
  }, [samples.length, width, height]);

  // draw image mask
  useEffect(() => {
    const canvas: any = canvasRef.current;
    if (canvas == null) return;
    const ctx = canvas.getContext('2d');
    imgSamples
      .filter(d => d.filtered)
      .forEach(sample => {
        ctx.beginPath();
        ctx.globalAlpha = 0.1;
        ctx.fillStyle = 'white';
        ctx.fillRect(sample.x - canvasLeft, sample.y - canvasTop, imgSize, imgSize);
        ctx.globalAlpha = 1.0;
      });
  }, [samples]);

  return (
    <Card
      title={`Context View`}
      size="small"
      bodyStyle={{ overflow: 'scroll', height: height - cardHeadHeight, padding: '0px' }}
      loading={isDataLoading}
    >
      <canvas id="imageContext" ref={canvasRef} width={canvasWidth} height={canvasHeight}></canvas>
    </Card>
  );
};

export default ImageContext;
