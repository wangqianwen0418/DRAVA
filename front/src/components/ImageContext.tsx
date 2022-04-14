import React, { useState, useEffect, useRef } from 'react';
import { Card, Slider } from 'antd';
import { TResultRow } from 'types';
import { BASE_URL } from 'Const';
import { getMax } from 'helpers';
import { image } from 'd3-fetch';

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
    const x = parseInt(id.match(/(_x)\d+/)![0].replace('_x', ''));
    const y = parseInt(id.match(/(_y)\d+/)![0].replace('_y', ''));
    const url = `${BASE_URL}/api/get_${dataset}_sample?id=${id}`;
    return { x, y, url, filtered };
  });

  const canvasOverallWidth = getMax(imgSamples.map(s => s.x)),
    canvasOverallHeight = getMax(imgSamples.map(s => s.y)),
    canvasHeight = height - cardHeadHeight;

  // draw images
  useEffect(() => {
    const canvas: any = canvasRef.current;
    if (canvas == null) return;
    const ctx = canvas.getContext('2d');
    const scale = width / canvasOverallWidth;
    ctx.scale(scale, height / canvasOverallHeight);

    imgSamples.forEach(sample => {
      const image = new Image(50, 50); // Using optional size for image
      image.src = sample.url;
      image.onload = () => {
        ctx.drawImage(image, sample.x - 50, sample.y - 50, 50, 50);
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
        ctx.fillRect(sample.x - 50, sample.y - 50, 50, 50);
        ctx.globalAlpha = 1.0;
      });
  }, [samples]);

  return (
    <Card
      title={`Context View`}
      size="small"
      bodyStyle={{ overflow: 'scroll', height: height - cardHeadHeight }}
      loading={isDataLoading}
    >
      <canvas id="imageContext" ref={canvasRef} width={width} height={canvasHeight}></canvas>
    </Card>
  );
};

export default ImageContext;
