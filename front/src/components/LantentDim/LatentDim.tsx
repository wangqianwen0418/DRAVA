import React, { ReactNode } from 'react';
import styles from './LatentDim.module.css';
import clsx from 'clsx';
import { Card, Select, Tooltip } from 'antd';

import { getMax, debounce } from 'helpers';
import { STEP_NUM } from 'Const';

import { scaleLinear, scaleLog, ScaleLogarithmic } from 'd3-scale';
import { TDistribution, TFilter, TResultRow } from 'types';

import { Correlations } from './AddCorrelation';
import { ConfigDim } from './ConfigDim';
import { DimRow } from './DimRow';

const { Option } = Select;

interface Props {
  filters: TFilter;
  dataset: string;
  matrixData: { [dimName: string]: TDistribution };
  samples: TResultRow[];
  height: number;
  width: number;
  setFilters: (dimName: string, col: number) => void;
  updateDims: (dimNames: string[]) => void;
  dimUserNames: { [key: string]: string };
  setDimUserNames: (dictName: { [key: string]: string }) => void;
  isDataLoading: boolean;
}
interface States {
  dimSampleIndex: { [dimName: string]: number | undefined }; // the index of samples used to generate simu images for each latent dimension
}

export default class LatentDim extends React.Component<Props, States> {
  spanWidth = 80; // width used for left-side dimension annotation
  barHeight = 30; // height of bar chart
  gap = 3; //horizontal gap between thumbnails
  barLabelHeight = 14;
  rowGap = 10; // vertical gap between rows
  constructor(props: Props) {
    super(props);
    this.state = {
      dimSampleIndex: {}
    };
    this.isSelected = this.isSelected.bind(this);
    this.changeDimSamples = this.changeDimSamples.bind(this);
  }

  /**
   * @compute
   * */
  isSelected(dimName: string, col_idx: number): boolean {
    return this.props.filters[dimName][col_idx];
  }

  // @drawing
  // getLinks(matrixData: { [k: string]: TDistribution }, stepWidth: number) {
  //   const { filters } = this.props,
  //     dims = Object.keys(filters);
  //   const linkGroups: ReactNode[] = [];

  //   for (let i = 0; i < dims.length - 1; i++) {
  //     const links: ReactNode[] = [];
  //     const prevRow = matrixData[dims[i]],
  //       nextRow = matrixData[dims[i + 1]];

  //     prevRow['groupedSamples'].forEach((prevSampleIds, prevIdx) => {
  //       nextRow['groupedSamples'].forEach((nextSampleIds, nextIdx) => {
  //         const isShow =
  //           this.isSelected(dims[i], prevIdx) &&
  //           this.isSelected(dims[i + 1], nextIdx) &&
  //           (!dims[i].includes('dim') || !dims[i + 1].includes('dim'));

  //         const insectSampleIds = prevSampleIds.filter(sampleId => nextSampleIds.includes(sampleId)),
  //           prevX = this.spanWidth + prevIdx * (stepWidth + this.gap) + stepWidth / 2,
  //           prevY = i * (this.barHeight * 2 + this.barLabelHeight + this.rowGap) + this.barHeight + this.barLabelHeight,
  //           nextX = this.spanWidth + nextIdx * (stepWidth + this.gap) + stepWidth / 2,
  //           nextY = prevY + (this.barHeight * 2 + this.barLabelHeight + this.rowGap);

  //         const link = (
  //           <line
  //             x1={prevX}
  //             y1={prevY}
  //             x2={nextX}
  //             y2={nextY}
  //             stroke="steelblue"
  //             strokeWidth={2}
  //             opacity={0.3}
  //             key={`${prevIdx}_${nextIdx}`}
  //           />
  //         );
  //         if (
  //           (insectSampleIds.length / prevSampleIds.length > 0.1 ||
  //             insectSampleIds.length / nextSampleIds.length > 0.1) &&
  //           isShow
  //         ) {
  //           links.push(link);
  //         }
  //       });
  //     });
  //     linkGroups.push(
  //       <g key={i} className={`row_${i}`}>
  //         {links}
  //       </g>
  //     );
  //   }
  //   return <g className="links">{linkGroups}</g>;
  // }

  /***
   *  @call_props_functions
   * */
  onChangeDim(dimNames: string[]) {
    this.props.updateDims(dimNames);
  }

  /***
   *  @call_props_functions
   * */
  onChangeDimNames(dimName: string, newName: string) {
    this.props.setDimUserNames({ [dimName]: newName });
  }

  /**
   * @update_state
   */
  changeDimSamples(dimName: string, newSampleIndex: number) {
    const { dimSampleIndex } = this.state;
    dimSampleIndex[dimName] = newSampleIndex;
    this.setState({ dimSampleIndex });
  }

  render() {
    const { filters, height, width, matrixData, isDataLoading, dimUserNames, samples, dataset } = this.props;

    const dims = Object.keys(filters);

    const rootStyle = getComputedStyle(document.documentElement);
    const cardPadding = parseInt(rootStyle.getPropertyValue('--card-body-padding')),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const cardInnerHeight = dims.length * (this.barHeight + 2 * this.barHeight + this.rowGap);

    const maxV = getMax(
      Object.values(matrixData)
        .map(d => d.histogram)
        .flat()
    );
    const yScale = scaleLog().domain([0.1, maxV]).range([0, this.barHeight]);

    //   axis controller
    const axisController = (
      <Select
        mode="multiple"
        allowClear
        placeholder="Add more dimensions"
        style={{ width: '350px', height: '30px', overflowY: 'scroll' }}
        value={dims}
        onChange={this.onChangeDim.bind(this)}
      >
        {Object.keys(matrixData).map(dimName => (
          <Option key={dimName} value={dimName}>
            {dimUserNames[dimName] || dimName}
          </Option>
        ))}
      </Select>
    );

    const stepWidth = (width - 2 * cardPadding - this.spanWidth) / STEP_NUM - this.gap;

    return (
      <Card
        title="Pattern Space"
        size="small"
        extra={axisController}
        bodyStyle={{ height: height - cardHeadHeight, width: width, overflowY: 'scroll' }}
        loading={isDataLoading}
      >
        {/* the pcp charts */}
        <svg height={cardInnerHeight} width={width - 2 * cardPadding} className="pcp">
          {/* get rows */}
          {dims.map((dimName, row_idx) => {
            const baseSampleIndex = this.state.dimSampleIndex[dimName];
            const row = (
              <DimRow
                row={matrixData[dimName]}
                dimName={dimName}
                stepWidth={stepWidth}
                yScale={yScale}
                barHeight={this.barHeight}
                barLabelHeight={this.barLabelHeight}
                gap={this.gap}
                dataset={this.props.dataset}
                latentZ={baseSampleIndex != undefined ? samples[baseSampleIndex].z : undefined}
                isSelected={this.isSelected}
                setFilters={this.props.setFilters}
              />
            );
            return (
              <g
                key={dimName}
                transform={`translate(0, ${row_idx * (this.barHeight * 2 + this.barLabelHeight + this.rowGap)})`}
              >
                <text
                  y={this.barHeight + this.barLabelHeight}
                  onClick={e => {
                    e.preventDefault();
                    this.props.setFilters(dimName, -1);
                  }}
                >
                  {dimUserNames[dimName] || dimName}
                </text>

                {/* only show configure for latent dim */}
                {dimName.includes('dim_') && (
                  <g
                    className="configure"
                    transform={`translate(0, ${this.barHeight + this.barLabelHeight + this.gap})`}
                  >
                    <ConfigDim
                      row={matrixData[dimName]}
                      dimName={dimName}
                      dimNames={Object.keys(matrixData)}
                      samples={samples}
                      dimUserNames={dimUserNames}
                      dataset={dataset}
                      baseSampleIndex={baseSampleIndex}
                      changeDimSamples={this.changeDimSamples}
                      setDimUserNames={this.props.setDimUserNames}
                    />
                  </g>
                )}

                {/* get each cell of a row */}
                <g transform={`translate(${this.spanWidth}, 0)`}>{row}</g>
              </g>
            );
          })}

          {/* <g className="links">{this.getLinks(matrixData, stepWidth)}</g> */}
          {dataset == 'matrix' ? (
            <g transform={`translate(0, ${dims.length * (this.barHeight * 2 + this.barLabelHeight + this.rowGap)})`}>
              <Correlations
                samples={samples}
                dimNames={Object.keys(matrixData)}
                dimUserNames={dimUserNames}
                width={width}
              />
            </g>
          ) : (
            <></>
          )}
        </svg>
      </Card>
    );
  }
}
