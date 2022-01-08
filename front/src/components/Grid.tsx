import React from 'react';
import styles from './Grid.module.css';
import clsx from 'clsx';
import { Card, Select } from 'antd';

import { getMax, debounce, getSampleHist, generateDistribution } from 'helpers';
import { STEP_NUM } from 'Const';

import { scaleLinear, scaleLog } from 'd3-scale';
import { TResultRow } from 'types';

const { Option } = Select;

interface Props {
  filters: number[][];
  dataset: string;
  setFilters: (row: number, col: number) => void;
  samples: TResultRow[];
  height: number;
  width: number;
}
interface States {
  dims: string[]; // dimensions in the latent space
}

export default class Grid extends React.Component<Props, States> {
  constructor(props: Props) {
    super(props);
    this.state = {
      dims: props.samples[0].z.map((dim, dim_idx) => `dim_${dim_idx}`)
    };
  }
  onChangeDim(dimNames: string[]) {
    this.setState({ dims: dimNames });
  }
  render() {
    const { filters, height, width, dataset, samples } = this.props;
    const hist = getSampleHist(samples.map(sample => sample.z as number[]));

    let { dims } = this.state;
    // TO-DO, maybe resort is not a smart way
    dims = dims.sort((a, b) => {
      if (a.includes('dim') && b.includes('dim')) {
        return parseInt(a.replace('dim_', '')) - parseInt(b.replace('dim_', ''));
      } else return a.includes('dim') ? -1 : 1;
    });

    const onSetFilter = debounce((row_idx: number, col_idx: number) => this.props.setFilters(row_idx, col_idx), 200);

    const rootStyle = getComputedStyle(document.documentElement);
    const cardPadding = parseInt(rootStyle.getPropertyValue('--card-body-padding')),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const spanWidth = 80, // width used for left-side dimension annotation
      barHeight = 30, // height of bar chart
      gap = 3, //horizontal gap between thumbnails
      barLabelHeight = 14,
      rowGap = 10; // vertical gap between rows

    const cardInnerHeight = dims.length * (barHeight + 2 * barHeight + rowGap);

    const maxV = getMax(hist.flat());
    const yScale = scaleLog().domain([0.1, maxV]).range([0, barHeight]);
    const isSelected = (dimNum: number, col_idx: number) => filters[dimNum].includes(col_idx);

    const getRow = (row: number[], row_idx: number) => {
      const stepWidth = (width - 2 * cardPadding - spanWidth) / STEP_NUM - gap;
      const imgSize = Math.min(stepWidth, barHeight);
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
              y={barHeight + barLabelHeight + gap + gap / 2}
              width={imgSize}
              height={imgSize}
            />
            <rect
              className={clsx(styles.imageBorder, isSelected(row_idx, col_idx) && styles.isImageSelected)}
              y={barHeight + barLabelHeight + gap}
              fill="none"
              width={imgSize + gap}
              height={imgSize + gap}
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
            <text
              x={(stepWidth + gap) * 0.5}
              y={barHeight + barLabelHeight - yScale(h)!}
              fontSize={8}
              textAnchor="middle"
            >
              {h > 0 ? h : ''}
            </text>
            <rect
              height={yScale(h)}
              width={stepWidth}
              y={barHeight + barLabelHeight - yScale(h)!}
              fill="lightgray"
              className={clsx(isSelected(row_idx, col_idx) && styles.isBarSelected)}
            />

            {/* thumbnails and their borders */}
            {col_idx % 2 == 0 ? image : <></>}
          </g>
        );
      });
    };

    const getAdditionalRow = (dimName: string) => {
      let row: any = {};
      const binNum = 11;
      if (dimName == 'size') {
        row = generateDistribution(
          samples.map(s => s.end - s.start),
          false,
          binNum
        );
      } else if (dimName == 'level') {
        row = generateDistribution(
          samples.map(s => s['level']),
          true,
          undefined
        );
      } else {
        row = generateDistribution(
          samples.map(s => s[dimName]),
          false,
          binNum
        );
      }
      const stepWidth = (width - 2 * cardPadding - spanWidth) / binNum - gap;
      return row['histogram'].map((h: number, col_idx: number) => {
        return (
          <g key={`bar_${col_idx}`} transform={`translate(${spanWidth + (stepWidth + gap) * col_idx}, 0)`}>
            {/* histogram */}
            <text
              x={(stepWidth + gap) * 0.5}
              y={barHeight + barLabelHeight - yScale(h)!}
              fontSize={8}
              textAnchor="middle"
            >
              {h > 0 ? h : ''}
            </text>
            <rect
              height={yScale(h)}
              width={stepWidth}
              y={barHeight + barLabelHeight - yScale(h)!}
              fill="lightgray"
              //   className={clsx(isSelected(row_idx, col_idx) && styles.isBarSelected)}
            />
            <text x={(stepWidth + gap) * 0.5} y={barHeight + barLabelHeight * 2} fontSize={8} textAnchor="middle">
              {row['labels'][col_idx]}
            </text>
          </g>
        );
      });
    };

    //   axis controller
    const axisController = (
      <Select
        mode="multiple"
        allowClear
        placeholder="Add more dimensions"
        style={{ width: '350px', height: '30px', overflowY: 'scroll' }}
        value={this.state.dims}
        onChange={this.onChangeDim.bind(this)}
      >
        {/* options about the latent dimensions */}
        {hist.map((hist, idx) => (
          <Option key={idx} value={`dim_${idx}`}>
            {' '}
            {`dim_${idx}`}{' '}
          </Option>
        ))}
        {/* options about the user added metrics */}
        {this.props.dataset == 'sequence' ? (
          <>
            <Option value="size">size</Option>
          </>
        ) : (
          <>
            <Option value="level">level</Option>
            <Option value="size">size</Option>
            <Option value="score">score</Option>
            <Option value="ctcf_mean">CTCF mean</Option>
            <Option value="ctcf_left">CTCF left</Option>
            <Option value="ctcf_right">CTCF right</Option>
            <Option value="atac_mean">ATAC mean</Option>
            <Option value="atac_left">ATAC left</Option>
            <Option value="atac_right">ATAC right</Option>
            <Option value="active">active</Option>
            <Option value="express">express</Option>
          </>
        )}
      </Select>
    );

    return (
      <Card
        title="Pattern Space"
        size="small"
        extra={axisController}
        bodyStyle={{ height: height - cardHeadHeight, width: width, overflowY: 'scroll' }}
      >
        {/* the pcp charts */}
        <svg height={cardInnerHeight} width={width - 2 * cardPadding} className="pcp">
          {dims.map((dimName, row_idx) => {
            const dimNum = parseInt(dimName.split('_')[1]);
            return (
              <g key={dimName} transform={`translate(0, ${row_idx * (barHeight * 2 + barLabelHeight + rowGap)})`}>
                <text className="dim_annotation" y={barHeight + barLabelHeight} onClick={() => onSetFilter(dimNum, -1)}>
                  {dimName}
                </text>

                {/* get each cell of a row */}
                {dimName.includes('dim_') ? getRow(hist[dimNum], dimNum) : getAdditionalRow(dimName)}
              </g>
            );
          })}
        </svg>
      </Card>
    );
  }
}
