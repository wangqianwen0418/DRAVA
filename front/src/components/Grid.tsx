import React from 'react';
import styles from './Grid.module.css';
import clsx from 'clsx';
import { Card } from 'antd';

import { getMax, debounce } from 'helpers';
import { STEP_NUM } from 'Const';

import { scaleLinear, scaleLog } from 'd3-scale';

interface Props {
  filters: number[][];
  dataset: string;
  setFilters: (row: number, col: number) => void;
  hist: number[][];
  height: number;
  width: number;
}
interface States {}

export default class Grid extends React.Component<Props, States> {
  constructor(props: Props) {
    super(props);
  }
  render() {
    const { filters, height, width, dataset } = this.props;
    const { hist } = this.props;

    const onSetFilter = debounce((row_idx: number, col_idx: number) => this.props.setFilters(row_idx, col_idx), 200);

    const rootStyle = getComputedStyle(document.documentElement);
    const cardPadding = parseInt(rootStyle.getPropertyValue('--card-body-padding')),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const spanWidth = 80, // width used for left-side dimension annotation
      barHeight = 30, // height of bar chart
      gap = 3, //horizontal gap between thumbnails
      stepWidth = (width - 2 * cardPadding - spanWidth) / STEP_NUM - gap,
      barLabelHeight = 14,
      rowGap = 10; // vertical gap between rows

    const cardInnerHeight = hist.length * (barHeight + stepWidth + barHeight + rowGap);

    const maxV = getMax(hist.flat());
    const yScale = scaleLog().domain([0.1, maxV]).range([0, barHeight]);
    const isSelected = (row_idx: number, col_idx: number) => filters[row_idx].includes(col_idx);

    const getRow = (row: number[], row_idx: number) => {
      return row.map((h, col_idx) => {
        const image = (
          <g>
            <image
              href={
                dataset == 'sequence'
                  ? `assets/simu/${row_idx}_${Math.floor(col_idx / 2)}.png`
                  : `assets/tad_simu/${row_idx}_${Math.floor(col_idx / 2)}.png`
              }
              className="latentImage"
              x={gap / 2}
              y={barHeight + barLabelHeight + gap / 2}
              width={stepWidth}
              height={stepWidth}
            />
            <rect
              className={clsx(styles.imageBorder, isSelected(row_idx, col_idx) && styles.isImageSelected)}
              y={barHeight + barLabelHeight}
              fill="none"
              width={stepWidth + gap}
              height={stepWidth + gap}
            />
          </g>
        );

        return (
          <g
            key={`bar_${col_idx}`}
            onClick={() => onSetFilter(row_idx, col_idx)}
            transform={`translate(${spanWidth + (stepWidth + gap) * col_idx}, 0)`}
          >
            {/* histogram */}
            <rect
              height={yScale(h)}
              width={stepWidth}
              y={barHeight - yScale(h)!}
              fill="lightgray"
              className={clsx(isSelected(row_idx, col_idx) && styles.isBarSelected)}
            />
            <text x={(stepWidth + gap) * 0.5} y={barHeight + barLabelHeight} fontSize={8} textAnchor="middle">
              {h}
            </text>

            {/* thumbnails and their borders */}
            {col_idx % 2 == 0 ? image : <></>}
          </g>
        );
      });
    };

    return (
      <Card
        title="Pattern Space"
        size="small"
        bodyStyle={{ height: height - cardHeadHeight, width: width, overflowY: 'scroll' }}
      >
        {/* the pcp charts */}
        <svg height={cardInnerHeight} width={width - 2 * cardPadding} className="pcp">
          {hist.map((row, row_idx) => {
            return (
              <g
                key={`row_${row_idx}`}
                transform={`translate(0, ${row_idx * (barHeight + barLabelHeight + stepWidth + rowGap)})`}
              >
                <text
                  className="dim_annotation"
                  y={barHeight + barLabelHeight}
                  onClick={() => onSetFilter(row_idx, -1)}
                >
                  Dim_{row_idx}
                </text>

                {/* get each cell of a row */}
                {getRow(row, row_idx)}
              </g>
            );
          })}
        </svg>
      </Card>
    );
  }
}
