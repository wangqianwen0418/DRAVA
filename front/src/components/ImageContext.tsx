import React, { useState, useEffect, useRef } from 'react';
import { Card } from 'antd';
import { TResultRow } from 'types';
import { BASE_URL } from 'Const';
import { ImageResource } from 'pixi.js';
import { getMax } from 'helpers';

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
    const id = s.id;
    const x = parseInt(id.match(/(?<=_x)\d+/)![0]);
    const y = parseInt(id.match(/(?<=_y)\d+/)![0]);
    const url = `${BASE_URL}/api/get_${dataset}_sample?id=${id}`;
    return { x, y, url };
  });

  const canvasWidth = getMax(imgSamples.map(s => s.x)),
    canvasHeight = getMax(imgSamples.map(s => s.y));

  useEffect(() => {
    const canvas: any = canvasRef.current;
    if (canvas == null) return;
    const ctx = canvas.getContext('2d');

    imgSamples.forEach(sample => {
      const image = new Image(50, 50); // Using optional size for image
      image.src = sample.url;
      image.onload = () => {
        ctx.drawImage(image, sample.x - 50, sample.y - 50, 50, 50);
      };
    });

    return () => {
      ctx.clearRect(0, 0, width, canvasHeight);
    };
  }, [samples]);

  return (
    <Card
      title={`Image Context`}
      size="small"
      bodyStyle={{ overflow: 'scroll', height: height - cardHeadHeight }}
      loading={isDataLoading}
    >
      <canvas id="imageContext" ref={canvasRef} width={canvasWidth} height={canvasHeight}></canvas>
    </Card>
  );
};

export default ImageContext;
