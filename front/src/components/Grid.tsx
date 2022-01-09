import React, { ReactNode } from 'react';
import styles from './Grid.module.css';
import clsx from 'clsx';
import { Card, Select } from 'antd';

import { getMax, debounce, getSampleHist, generateDistribution } from 'helpers';
import { STEP_NUM } from 'Const';

import { scaleLinear, scaleLog, ScaleLogarithmic } from 'd3-scale';
import { TDistribution, TResultRow, TFilter } from 'types';

const { Option } = Select;

interface Props {
  filters: TFilter;
  dataset: string;
  setFilters: (dimName: string, col: number) => void;
  samples: TResultRow[];
  height: number;
  width: number;
}
interface States {
  dims: string[]; // dimensions in the latent space
}

export default class Grid extends React.Component<Props, States> {
  spanWidth = 80; // width used for left-side dimension annotation
  barHeight = 30; // height of bar chart
  gap = 3; //horizontal gap between thumbnails
  barLabelHeight = 14;
  rowGap = 10; // vertical gap between rows
  constructor(props: Props) {
    super(props);
    this.state = {
      dims: props.samples[0].z.map((dim, dim_idx) => `dim_${dim_idx}`)
    };
  }
  // @compute
  matrixData(): { [dimName: string]: TDistribution } {
    const { dims } = this.state;
    const { samples } = this.props;

    var matrixData: { [k: string]: TDistribution } = {},
      row: TDistribution = { histogram: [], labels: [], groupedSamples: [] };
    dims.forEach((dimName, idx) => {
      if (dimName.includes('dim')) {
        const dimNum = parseInt(dimName.split('_')[1]);
        row = generateDistribution(
          samples.map(sample => sample['z'][dimNum]),
          false,
          STEP_NUM
        );
      } else if (dimName == 'size') {
        row = generateDistribution(
          samples.map(s => s.end - s.start),
          false,
          STEP_NUM
        );
      } else if (dimName == 'level') {
        row = generateDistribution(
          samples.map(s => s['level']),
          true
        );
      } else {
        row = generateDistribution(
          samples.map(s => s[dimName]),
          false,
          STEP_NUM
        );
      }
      matrixData[dimName] = row;
    });
    return matrixData;
  }
  isSelected(dimName: string, col_idx: number) {
    return this.props.filters[dimName].includes(col_idx);
  }
  // @drawing
  getRow(row: TDistribution, dimName: string, stepWidth: number, yScale: any) {
    const imgSize = Math.min(stepWidth, this.barHeight);
    const dimNum = dimName.split('_')[1];
    return row['histogram'].map((h, col_idx) => {
      const image = (
        <g>
          <image
            href={
              this.props.dataset == 'sequence'
                ? `assets/simu/${dimNum}_${Math.floor(col_idx / 2)}.png`
                : `assets/tad_simu/${dimNum}_${Math.floor(col_idx / 2)}.png`
            }
            className="latentImage"
            x={this.gap / 2}
            y={this.barHeight + this.barLabelHeight + this.gap + this.gap / 2}
            width={imgSize}
            height={imgSize}
          />
          <rect
            className={clsx(styles.imageBorder, this.isSelected(dimName, col_idx) && styles.isImageSelected)}
            y={this.barHeight + this.barLabelHeight + this.gap}
            fill="none"
            width={imgSize + this.gap}
            height={imgSize + this.gap}
          />
        </g>
      );

      return (
        <g
          key={`bar_${col_idx}`}
          // onClick={() => this.onSetFilter(row_idx, col_idx)}
          transform={`translate(${this.spanWidth + (stepWidth + this.gap) * col_idx}, 0)`}
        >
          {/* histogram */}
          <text
            x={(stepWidth + this.gap) * 0.5}
            y={this.barHeight + this.barLabelHeight - yScale(h)!}
            fontSize={8}
            textAnchor="middle"
          >
            {h > 0 ? h : ''}
          </text>
          <rect
            height={yScale(h)}
            width={stepWidth}
            y={this.barHeight + this.barLabelHeight - yScale(h)!}
            fill="lightgray"
            className={clsx(this.isSelected(dimName, col_idx) && styles.isBarSelected)}
          />

          {/* thumbnails and their borders */}
          {col_idx % 2 == 0 ? image : <></>}
        </g>
      );
    });
  }
  // @drawing
  getAdditionalRow(row: TDistribution, stepWidth: number, yScale: any) {
    // const stepWidth = (width - 2 * cardPadding - spanWidth) / binNum - gap;
    const gap = this.gap,
      barHeight = this.barHeight,
      barLabelHeight = this.barLabelHeight,
      spanWidth = this.spanWidth;
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
            {col_idx % 4 == 0 ? row['labels'][col_idx] : ''}
          </text>
        </g>
      );
    });
  }
  // @drawing
  getLinks(matrixData: { [k: string]: TDistribution }, stepWidth: number) {
    const { dims } = this.state;
    const linkGroups: ReactNode[] = [];

    for (let i = 0; i < dims.length - 1; i++) {
      const links: ReactNode[] = [];
      const prevRow = matrixData[dims[i]],
        nextRow = matrixData[dims[i + 1]];

      prevRow['groupedSamples'].forEach((prevSampleIds, prevIdx) => {
        nextRow['groupedSamples'].forEach((nextSampleIds, nextIdx) => {
          const insectSampleIds = prevSampleIds.filter(sampleId => nextSampleIds.includes(sampleId)),
            prevX = this.spanWidth + prevIdx * (stepWidth + this.gap) + stepWidth / 2,
            prevY = i * (this.barHeight * 2 + this.barLabelHeight + this.rowGap) + this.barHeight + this.barLabelHeight,
            nextX = this.spanWidth + nextIdx * (stepWidth + this.gap) + stepWidth / 2,
            nextY = prevY + (this.barHeight * 2 + this.barLabelHeight + this.rowGap);

          const link = (
            <line
              x1={prevX}
              y1={prevY}
              x2={nextX}
              y2={nextY}
              stroke="steelblue"
              strokeWidth={1}
              opacity={
                insectSampleIds.length / prevSampleIds.length > 0.1 ||
                insectSampleIds.length / nextSampleIds.length > 0.1
                  ? 0.3
                  : 0
              }
              key={`${prevIdx}_${nextIdx}`}
            />
          );
          links.push(link);
        });
      });
      linkGroups.push(
        <g key={i} className={`row_${i}`}>
          {links}
        </g>
      );
    }
    return <g className="links">{linkGroups}</g>;
  }
  onChangeDim(dimNames: string[]) {
    this.setState({ dims: dimNames });
  }

  render() {
    const { filters, height, width, dataset, samples } = this.props;
    const hist = getSampleHist(samples.map(sample => sample.z as number[]));

    const { dims } = this.state;
    // // TO-DO, maybe resort is not a smart way
    // dims = dims.sort((a, b) => {
    //   if (a.includes('dim') && b.includes('dim')) {
    //     return parseInt(a.replace('dim_', '')) - parseInt(b.replace('dim_', ''));
    //   } else return a.includes('dim') ? -1 : 1;
    // });

    const matrixData = this.matrixData();

    const onSetFilter = debounce((dimName: string, col_idx: number) => this.props.setFilters(dimName, col_idx), 200);

    const rootStyle = getComputedStyle(document.documentElement);
    const cardPadding = parseInt(rootStyle.getPropertyValue('--card-body-padding')),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const cardInnerHeight = dims.length * (this.barHeight + 2 * this.barHeight + this.rowGap);

    const maxV = getMax(hist.map(d => d.histogram).flat());
    const yScale = scaleLog().domain([0.1, maxV]).range([0, this.barHeight]);

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

    const stepWidth = (width - 2 * cardPadding - this.spanWidth) / STEP_NUM - this.gap;

    return (
      <Card
        title="Pattern Space"
        size="small"
        extra={axisController}
        bodyStyle={{ height: height - cardHeadHeight, width: width, overflowY: 'scroll' }}
      >
        {/* the pcp charts */}
        <svg height={cardInnerHeight} width={width - 2 * cardPadding} className="pcp">
          {/* get rows */}
          {dims.map((dimName, row_idx) => {
            const dimNum = parseInt(dimName.split('_')[1]);

            return (
              <g
                key={dimName}
                transform={`translate(0, ${row_idx * (this.barHeight * 2 + this.barLabelHeight + this.rowGap)})`}
              >
                <text
                  className="dim_annotation"
                  y={this.barHeight + this.barLabelHeight}
                  onClick={() => onSetFilter(dimName, -1)}
                >
                  {dimName}
                </text>

                {/* get each cell of a row */}
                {dimName.includes('dim_')
                  ? this.getRow(matrixData[dimName], dimName, stepWidth, yScale)
                  : this.getAdditionalRow(matrixData[dimName], stepWidth, yScale)}
              </g>
            );
          })}
          <g className="links">{this.getLinks(matrixData, stepWidth)}</g>
        </svg>
      </Card>
    );
  }
}
