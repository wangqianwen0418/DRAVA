import { getDimValues, getMax, getMin } from 'helpers';
import * as React from 'react';
import { TResultRow } from 'types';
import { scaleLinear } from 'd3-scale';

const getScatter = (
  dim: [string, string],
  height: number,
  width: number,
  samples: TResultRow[],
  dimUserNames: { [oldName: string]: string }
) => {
  const margin = 15;
  const [dimX, dimY] = dim;
  const xValues = getDimValues(samples, dimX),
    yValues = getDimValues(samples, dimY);

  const xScale = scaleLinear()
    .domain([getMin(xValues), getMax(xValues)])
    .range([margin, width - margin]);
  const yScale = scaleLinear()
    .domain([getMin(yValues), getMax(yValues)])
    .range([margin, height - margin]);

  const points = xValues.map((x, idx) => {
    const y = yValues[idx];
    return <circle key={samples[idx].id} cx={xScale(x)} cy={yScale(y)} r="3" fill="steelblue" />;
  });

  return (
    <g className={`scatter_${dimX}_${dimY}`}>
      <line className="x" x1={margin} x2={width - margin} y1={height - margin} y2={height - margin} stroke="black" />
      <text x={width / 2} y={height} textAnchor="middle">
        {dimUserNames[dimX] || dimX}
      </text>
      <line className="x" x1={margin} x2={margin} y1={margin} y2={height - margin} stroke="black" />
      <text textAnchor="middle" transform={`rotate(-90deg) translate(${0}, ${height / 2})`}>
        {dimUserNames[dimY] || dimY}
      </text>
      {points}
    </g>
  );
};

export default getScatter;
