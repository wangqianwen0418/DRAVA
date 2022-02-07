import React, { useState, ReactNode } from 'react';
import { TDistribution } from 'types';
import { Tooltip } from 'antd';
import clsx from 'clsx';
import styles from './getRow.module.css';

export const getRow = (
  row: TDistribution,
  dimName: string,
  stepWidth: number,
  yScale: any,
  barHeight: number,
  barLabelHeight: number,
  gap: number,
  spanWidth: number,
  dataset: string,
  isSelected: (dimName: string, col_idx: number) => boolean,
  setFilters: (dimName: string, col_idx: number) => void
) => {
  return dimName.includes('dim_')
    ? getLatentDim(
        row,
        dimName,
        stepWidth,
        yScale,
        barHeight,
        barLabelHeight,
        gap,
        spanWidth,
        dataset,
        isSelected,
        setFilters
      )
    : getAdditionalDim(
        row,
        dimName,
        stepWidth,
        yScale,
        barHeight,
        barLabelHeight,
        gap,
        spanWidth,
        dataset,
        isSelected,
        setFilters
      );
};

const getLatentDim = (
  row: TDistribution,
  dimName: string,
  stepWidth: number,
  yScale: any,
  barHeight: number,
  barLabelHeight: number,
  gap: number,
  spanWidth: number,
  dataset: string,
  isSelected: (dimName: string, col_idx: number) => boolean,
  setFilters: (dimName: string, col_idx: number) => void
): ReactNode => {
  const imgSize = Math.min(stepWidth, barHeight);
  const dimNum = dimName.split('_')[1];
  return row['histogram'].map((h, col_idx) => {
    const href = `assets/${dataset}_simu/${dimNum}_${Math.floor(col_idx / 2)}.png`;
    const image = (
      <Tooltip title={<img width={64} src={href} />} destroyTooltipOnHide placement="top">
        <g>
          <image
            href={href}
            className={clsx(styles.latentImage, isSelected(dimName, col_idx) && styles.isImageSelected)}
            x={gap / 2}
            y={barHeight + barLabelHeight + gap + gap / 2}
            width={imgSize}
            height={imgSize}
          />
          <rect
            className={clsx(styles.imageBorder, isSelected(dimName, col_idx) && styles.isImageSelected)}
            y={barHeight + barLabelHeight + gap}
            fill="none"
            width={imgSize + gap}
            height={imgSize + gap}
          />
        </g>
      </Tooltip>
    );

    return (
      <g
        key={`bar_${col_idx}`}
        // onClick={() => onSetFilter(dimName, col_idx)}
        onClick={() => setFilters(dimName, col_idx)}
        transform={`translate(${spanWidth + (stepWidth + gap) * col_idx}, 0)`}
      >
        {/* histogram */}
        <text
          x={(stepWidth + gap) * 0.5}
          fontSize={8}
          textAnchor="middle"
          y={barHeight + barLabelHeight - yScale(h) || 0}
        >
          {h > 0 ? h : ''}
        </text>
        <rect
          height={yScale(h) || 0}
          width={stepWidth}
          y={barHeight + barLabelHeight - yScale(h) || 0}
          fill="lightgray"
          className={clsx(isSelected(dimName, col_idx) && styles.isBarSelected)}
        />

        {/* thumbnails and their borders */}
        {col_idx % 2 == 0 ? image : <></>}
      </g>
    );
  });
};

const getAdditionalDim = (
  row: TDistribution,
  dimName: string,
  stepWidth: number,
  yScale: any,
  barHeight: number,
  barLabelHeight: number,
  gap: number,
  spanWidth: number,
  dataset: string,
  isSelected: (dimName: string, col_idx: number) => boolean,
  setFilters: (dimName: string, col_idx: number) => void
): ReactNode => {
  return row['histogram'].map((h: number, col_idx: number) => {
    return (
      <g
        key={`bar_${col_idx}`}
        transform={`translate(${spanWidth + (stepWidth + gap) * col_idx}, 0)`}
        onClick={() => setFilters(dimName, col_idx)}
      >
        {/* histogram */}
        <text x={(stepWidth + gap) * 0.5} y={barHeight + barLabelHeight - yScale(h)!} fontSize={8} textAnchor="middle">
          {h > 0 ? h : ''}
        </text>
        <rect
          height={yScale(h)}
          width={stepWidth}
          y={barHeight + barLabelHeight - yScale(h)!}
          fill="lightgray"
          className={clsx(isSelected(dimName, col_idx) && styles.isBarSelected)}
        />
        <text x={(stepWidth + gap) * 0.5} y={barHeight + barLabelHeight * 2} fontSize={8} textAnchor="middle">
          {col_idx % 4 == 0 ? row['labels'][col_idx] : ''}
        </text>
      </g>
    );
  });
};
