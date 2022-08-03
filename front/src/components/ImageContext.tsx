import React, { useEffect, useRef } from 'react';
import { Card } from 'antd';
import { TResultRow } from 'types';
import { datasetConfig } from 'config';
import { getItemURL } from 'dataService';
import * as PIXI from 'pixi.js';

interface Props {
  isDataLoading: boolean;
  samples: TResultRow[];
  height: number;
  width: number;
  dataset: string;
}

const MIN_SCALE = 1;
const MAX_SCALE = 5;

const ImageContext = (props: Props) => {
  const { isDataLoading, samples, height: heightInclHeader, width, dataset } = props;
  
  const pixiRenderer = useRef<PIXI.AbstractRenderer>();
  const pixiRoot = useRef<PIXI.Container | undefined>(new PIXI.Container());
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prevMousePos = useRef<{ [k in 'x' | 'y']: number}>();

  const rootStyle = getComputedStyle(document.documentElement),
    cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));
  const height = heightInclHeader - cardHeadHeight;

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
    const url = `${getItemURL(dataset, id)}&border=0`;
    return { x, y, url, filtered };
  });
  const imgSize = datasetConfig[dataset].imgSize || 50;
  const minX = Math.min(...imgSamples.map(s => +s.x));
  const maxX = Math.max(...imgSamples.map(s => +s.x));
  const minY = Math.min(...imgSamples.map(s => +s.y));
  const maxY = Math.max(...imgSamples.map(s => +s.y));
  const canvasWidth = maxX - minX + imgSize;
  const canvasHeight = maxY - minY + imgSize;

  function animate() {
    if(pixiRoot.current) pixiRenderer.current?.render(pixiRoot.current);
    requestAnimationFrame(animate);
  }

  useEffect(() => {
    if(!canvasRef.current) return;
    
    const options = { width, height, transparent: true, view: canvasRef.current };
    pixiRenderer.current = PIXI.autoDetectRenderer(options);

    const tiles = new PIXI.Container();
    const scale = Math.min(width / canvasWidth, height / canvasHeight);
    imgSamples.forEach(sample => {
      const { url, x, y } = sample;
      const tile = PIXI.Sprite.from(url);
      tile.x = (x - minX) * scale;
      tile.y = (y - minY) * scale;
      tile.width = imgSize * scale;
      tile.height = imgSize * scale;
      tiles.addChild(tile);
    });

    pixiRoot.current?.removeChildren();
    pixiRoot.current?.addChild(tiles);

    animate();
    
    return () => {
      // pixiRoot.current?.removeChildren();
      // pixiRoot.current?.destroy();
      // pixiRenderer.current?.destroy();
      // pixiRoot.current = undefined;
      // pixiRenderer.current = undefined;
    }
  }, [canvasRef.current, samples, width, height]);

  // TODO (Aug-3-2022): Support performant filtering
  // draw image mask
  // useEffect(() => {
  //   if (!canvasRef.current) return;
  //   const ctx = canvasRef.current.getContext('2d')!;
  //   imgSamples
  //     .filter(d => d.filtered)
  //     .forEach(sample => {
  //       ctx.beginPath();
  //       ctx.globalAlpha = 0.1;
  //       ctx.fillStyle = 'white';
  //       ctx.fillRect(sample.x - offsetX, sample.y - offsetY, imgSize, imgSize);
  //       ctx.globalAlpha = 1.0;
  //     });
  // }, [samples]);

  return (
    <Card
      id='imageContainer'
      title={`Context View`}
      size="small"
      bodyStyle={{ height, overflow: 'hidden', position: 'relative' }}
      loading={isDataLoading}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ overflow: 'hidden', touchAction: 'none' }}
        onMouseDown={(e) => {
          prevMousePos.current = { x: e.clientX, y: e.clientY };
        }}
        onMouseMove={(e) => {
          if(prevMousePos.current && pixiRoot.current) {
            const { clientX: x, clientY: y } = e;
            const [deltaX, deltaY] = [x - prevMousePos.current.x, y - prevMousePos.current.y];
            pixiRoot.current.x += deltaX;
            pixiRoot.current.y += deltaY;
            prevMousePos.current = { x, y };
          }
        }}
        onMouseUp={() => { prevMousePos.current = undefined; }}
        onWheel={(e) => {
          if(pixiRoot.current) {
            const { x: parentX, y: parentY } = canvasRef.current!.parentElement!.getBoundingClientRect();
            const [mx, my] = [e.clientX - parentX, e.clientY - parentY];
            const wd = (e.deltaX + e.deltaY) / 2.0;
            const f = 1 / 100;
            const prevScale = pixiRoot.current.scale.x;
            const scale = Math.min(Math.max(prevScale - wd * f, MIN_SCALE), MAX_SCALE);
            const dx = mx - pixiRoot.current.position.x;
            const dy = my - pixiRoot.current.position.y;
            pixiRoot.current.position.x -= dx * (scale / prevScale - 1);
            pixiRoot.current.position.y -= dy * (scale / prevScale - 1);
            pixiRoot.current.scale.x = pixiRoot.current.scale.y = scale;
          }
        }}
      />
    </Card>
  );
};

export default ImageContext;
