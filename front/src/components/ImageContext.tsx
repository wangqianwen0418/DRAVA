import React, { useState, useEffect, useRef } from 'react';
import { Card, Slider } from 'antd';
import { TResultRow } from 'types';
import { BASE_URL } from 'Const';
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

  const [zoom, changeZoom] = useState(1);

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

  const canvasOverallWidth = getMax(imgSamples.map(s => s.x)),
    canvasOverallHeight = getMax(imgSamples.map(s => s.y)),
    canvasHeight = height - cardHeadHeight;

  // draw images
  useEffect(() => {
    const canvas: any = canvasRef.current;
    if (canvas == null) return;
    const ctx = canvas.getContext('2d');
    const scale = (zoom * width) / canvasOverallWidth;
    ctx.scale(scale, scale);

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
  }, [samples]);

  //   // update canvas zoom
  //   useEffect(() => {
  //     const canvas: any = canvasRef.current;
  //     if (canvas == null) return;
  //     const ctx = canvas.getContext('2d');
  //     console.info(zoom, ctx);
  //     ctx.scale(zoom, zoom);
  //   }, [zoom]);

  return (
    <Card
      title={`Context View`}
      size="small"
      bodyStyle={{ overflow: 'scroll', height: height - cardHeadHeight }}
      loading={isDataLoading}
      //   extra={
      //     <Slider
      //       min={1}
      //       max={10}
      //       step={0.1}
      //       value={zoom / (width / canvasOverallWidth)}
      //       style={{ width: 100 }}
      //       onChange={v => changeZoom(v * (width / canvasOverallWidth))}
      //     />
      //   }
    >
      <canvas id="imageContext" ref={canvasRef} width={width} height={canvasHeight}></canvas>
    </Card>
  );
};

export default ImageContext;
