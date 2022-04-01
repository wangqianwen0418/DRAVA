import React, { ReactNode } from 'react';
import styles from './LatentDim.module.css';
import clsx from 'clsx';

import { Button, Card, Select, Tooltip } from 'antd';

import { getMax, debounce } from 'helpers';
import { STEP_NUM } from 'Const';

import { scaleLinear, scaleLog, ScaleLogarithmic } from 'd3-scale';
import { TDistribution, TFilter, TResultRow } from 'types';

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
  dimScores: { [dimName: string]: number };
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
      dimSampleIndex: {},
      dimScores: {}
    };
    this.isSelected = this.isSelected.bind(this);
    this.changeDimSamples = this.changeDimSamples.bind(this);
    this.changeDimScores = this.changeDimScores.bind(this);
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

  /**
   * @update_state
   */
  changeDimScores(dimName: string, score: number) {
    const { dimScores } = this.state;
    dimScores[dimName] = score;
    this.setState({ dimScores });
  }

  // clean dim scores after changing dataset
  componentDidUpdate(prevProps: Props) {
    if (prevProps.dataset != this.props.dataset) {
      this.setState({ dimScores: {} });
    }
  }

  render() {
    const { filters, height, width, matrixData, isDataLoading, dimUserNames, samples, dataset } = this.props;
    const { dimScores } = this.state;

    const dims = Object.keys(filters);
    // sort dims based on dim score
    dims.sort((a, b) => -dimScores[a] + dimScores[b]);

    const rootStyle = getComputedStyle(document.documentElement);
    const cardPadding = parseInt(rootStyle.getPropertyValue('--card-body-padding')),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const cardInnerHeight = dims.length * (this.barHeight + 2 * this.barHeight + this.rowGap);

    const maxV = getMax(
      Object.values(matrixData)
        .map(d => d.histogram)
        .flat()
    );
    const yScaleLog = scaleLog()
      .domain([0.1, maxV])
      .range([this.barHeight / 10, this.barHeight]);
    const yScaleLinear = scaleLinear()
      .domain([0.1, maxV])
      .range([this.barHeight / 10, this.barHeight]);

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
    const maxScore = Math.max(...Object.values(this.state.dimScores)) || 0.0000001;

    return (
      <Card
        title="Concept View"
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
                yScale={dimName.includes('dim') ? yScaleLog : yScaleLinear}
                barHeight={this.barHeight}
                barLabelHeight={this.barLabelHeight}
                gap={this.gap}
                dataset={this.props.dataset}
                latentZ={baseSampleIndex != undefined ? samples[baseSampleIndex].z : undefined}
                isSelected={this.isSelected}
                setFilters={this.props.setFilters}
                changeDimScores={this.changeDimScores}
              />
            );
            const score = this.state.dimScores[dimName] || 0;
            const barWidth = this.spanWidth - 10;
            const scoreBarWidth = (score / maxScore) * barWidth;
            return (
              <g
                key={dimName}
                transform={`translate(0, ${row_idx * (this.barHeight * 2 + this.barLabelHeight + this.rowGap)})`}
              >
                <foreignObject className={styles.inputTextWrapper} width={60} height={30}>
                  <input
                    value={dimUserNames[dimName] || dimName}
                    className={clsx(styles.inputText)}
                    unselectable="on"
                    onChange={e => this.onChangeDimNames(dimName, e.target.value)}
                  />
                </foreignObject>
                {/* dim importance score */}
                {dimName.includes('dim_') ? (
                  <g
                    transform={`translate(0, ${this.barHeight + 2 * this.barLabelHeight})`}
                    className={styles.toggleFilter}
                    // onClick={e => {
                    //   e.preventDefault();
                    //   this.props.setFilters(dimName, -1);
                    // }}
                  >
                    <rect height={this.barLabelHeight} width={barWidth} stroke="lightgray" fill="transparent"></rect>
                    <rect height={this.barLabelHeight} width={scoreBarWidth} stroke="lightgray" fill="lightgray"></rect>
                    {/* <text y={this.barLabelHeight}>{score.toFixed(3)}</text> */}
                  </g>
                ) : (
                  <></>
                )}

                {/* get each cell of a row */}
                <g transform={`translate(${this.spanWidth}, 0)`}>{row}</g>
              </g>
            );
          })}

          {/* <g className="links">{this.getLinks(matrixData, stepWidth)}</g> */}
        </svg>
      </Card>
    );
  }
}
