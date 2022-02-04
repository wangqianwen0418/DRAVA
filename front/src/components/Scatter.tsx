import { getDimValues, getMax, getMin } from 'helpers';
import * as React from 'react';
import { TResultRow } from 'types';
import { scaleLinear } from 'd3-scale';

const getScatter = (dim: [string, string], height: number, width: number, samples: TResultRow[]) => {
  const [dimX, dimY] = dim;
  const xValues = getDimValues(samples, dimX),
    yValues = getDimValues(samples, dimY);

  const xScale = scaleLinear()
    .domain([getMin(xValues), getMax(xValues)])
    .range([0, width]);
  const yScale = scaleLinear()
    .domain([getMin(yValues), getMax(yValues)])
    .range([0, height]);

  const points = xValues.map((x, idx) => {
    const y = yValues[idx];
    return <circle key={samples[idx].id} cx={xScale(x)} cy={yScale(y)} r="3" fill="steelblue" />;
  });

  return <g className={`scatter_${dimX}_${dimY}`}> {points} </g>;
};

export default getScatter;
