import React, { useState } from 'react';
import { TDistribution } from 'types';
import { Tooltip } from 'antd';
import clsx from 'clsx';
import styles from './DimRow.module.css';

type Props = {
  row: TDistribution;
  dimName: string;
  stepWidth: number;
  yScale: any;
  barHeight: number;
  barLabelHeight: number;
  gap: number;
  dataset: string;
  imageSize?: number;
  isSelected?: (dimName: string, col_idx: number) => boolean;
  setFilters?: (dimName: string, col_idx: number) => void;
};

export const DimRow = (props: Props) => {
  return <g>{props.dimName.includes('dim_') ? getLatentDim(props) : getAdditionalDim(props)}</g>;
};

const getLatentDim = (props: Props) => {
  const {
    row,
    dimName,
    stepWidth,
    yScale,
    barHeight,
    barLabelHeight,
    gap,
    dataset,
    isSelected,
    setFilters,
    imageSize
  } = props;
  const imgSize = imageSize || Math.min(stepWidth, barHeight);
  const dimNum = dimName.split('_')[1];
  return row['histogram'].map((h, col_idx) => {
    const href = `assets/${dataset}_simu/${dimNum}_${Math.floor(col_idx / 2)}.png`;
    const selectFlag = isSelected ? isSelected(dimName, col_idx) : true;
    const image = (
      // <Tooltip title={<img width={64} src={href} />} destroyTooltipOnHide placement="top">
      <g>
        <image
          href={href}
          className={clsx(styles.latentImage, selectFlag && styles.isImageSelected)}
          x={gap / 2}
          y={barHeight + barLabelHeight + gap + gap / 2}
          width={imgSize}
          height={imgSize}
        />
        <rect
          className={clsx(styles.imageBorder, selectFlag && styles.isImageSelected)}
          y={barHeight + barLabelHeight + gap}
          fill="none"
          width={imgSize + gap}
          height={imgSize + gap}
        />
      </g>
      // </Tooltip>
    );

    return (
      <g
        key={`bar_${col_idx}`}
        // onClick={() => onSetFilter(dimName, col_idx)}
        onClick={() => (setFilters ? setFilters(dimName, col_idx) : '')}
        transform={`translate(${(stepWidth + gap) * col_idx}, 0)`}
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
          className={clsx(selectFlag && styles.isBarSelected)}
        />

        {/* thumbnails and their borders */}
        {col_idx % 2 == 0 ? image : <></>}
      </g>
    );
  });
};

const getAdditionalDim = (props: Props) => {
  const { row, dimName, stepWidth, yScale, barHeight, barLabelHeight, gap, isSelected, setFilters } = props;
  return row['histogram'].map((h: number, col_idx: number) => {
    const selectFlag = isSelected ? isSelected(dimName, col_idx) : true;
    return (
      <g
        key={`bar_${col_idx}`}
        transform={`translate(${(stepWidth + gap) * col_idx}, 0)`}
        onClick={() => (setFilters ? setFilters(dimName, col_idx) : '')}
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
          className={clsx(selectFlag && styles.isBarSelected)}
        />
        <text x={(stepWidth + gap) * 0.5} y={barHeight + barLabelHeight * 2} fontSize={8} textAnchor="middle">
          {col_idx % 4 == 0 ? row['labels'][col_idx] : ''}
        </text>
      </g>
    );
  });
};
