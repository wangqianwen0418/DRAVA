import React, { ReactNode } from 'react';
import styles from './LatentDim.module.css';
import clsx from 'clsx';

import { Button, Card, Dropdown, Menu, Select } from 'antd';
import { PlusOutlined } from '@ant-design/icons';

import { getMax, debounce } from 'helpers';
import { STEP_NUM } from 'Const';

import { scaleLinear, scaleLog } from 'd3-scale';
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

export default class LatentDims extends React.Component<Props, States> {
  spanWidth = 100; // width used for left-side dimension annotation
  barHeight = 30; // height of bar chart
  gap = 3; //horizontal gap between thumbnails
  barLabelHeight = 14;
  deleteBtnSize = 20;
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

    // Dropdown menu to add a new dimension
    const dimensionMenuItems = Object.keys(matrixData)
      .map(dimName => dimUserNames[dimName] || dimName)
      // show only the dimensions that are not yet added
      .filter(name => dims.indexOf(name) === -1)
      .map(name => {
        return {
          key: name,
          disabled: dims.indexOf(name) !== -1,
          label: <div onClick={() => this.onChangeDim([...dims, name])}>{name}</div>
        };
      });
    const dropdown = (
      <Dropdown
        className={clsx(styles.dimDropDown)}
        overlay={<Menu items={dimensionMenuItems} />}
        placement="bottomRight"
      >
        <Button shape="circle" icon={<PlusOutlined />} />
      </Dropdown>
    );

    const stepWidth = (width - 2 * cardPadding - this.spanWidth) / STEP_NUM - this.gap;
    const maxScore = Math.max(...Object.values(this.state.dimScores)) || 0.0000001;

    return (
      <Card
        title="Concept View"
        size="small"
        bodyStyle={{ height: height - cardHeadHeight, width: width, overflowY: 'scroll' }}
        loading={isDataLoading}
      >
        {dropdown}
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
                <foreignObject
                  transform={`translate(0, ${this.barHeight})`}
                  width={barWidth - this.deleteBtnSize}
                  height={this.barHeight}
                >
                  <input
                    value={dimUserNames[dimName] || dimName}
                    className={clsx(styles.inputText)}
                    style={{ width: barWidth - this.deleteBtnSize - 5 }}
                    unselectable="on"
                    onChange={e => this.onChangeDimNames(dimName, e.target.value)}
                  />
                </foreignObject>
                <g
                  className={styles.deleteBtnGroup}
                  onClick={() => this.onChangeDim(dims.filter((_, i) => row_idx !== i))}
                  transform={`translate(${barWidth - this.deleteBtnSize}, ${this.barHeight})`}
                >
                  <rect className={styles.deleteBtn} width={this.deleteBtnSize} height={this.deleteBtnSize} />
                  <text className={styles.deleteBtnText} x={6} y={this.deleteBtnSize - 6}>
                    X
                  </text>
                </g>
                {/* dim importance score */}
                {dimName.includes('dim_') && (
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
