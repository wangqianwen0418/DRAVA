import { TDistribution, TResultRow, TFilter } from 'types';
import React, { useState } from 'react';
import { Card, Select, Button } from 'antd';
import { RightCircleOutlined, LeftCircleOutlined } from '@ant-design/icons';

import styles from './index.module.css';
import clsx from 'clsx';

import { STEP_NUM } from 'Const';
import { scaleLinear, scaleLog, ScaleLogarithmic } from 'd3-scale';
import { getMax } from 'helpers';
import { DimRow } from 'components/LatentDim/DimRow';
import Piling from './Piling';

import { BASE_URL } from 'Const';

const { Option } = Select;

type Props = {
  // row: TDistribution;
  dataset: string;
  samples: TResultRow[];
  width: number;
  height: number;
  isDataLoading: boolean;
  matrixData: { [dimName: string]: TDistribution };
  dimUserNames: { [key: string]: string };
  filters: TFilter;

  dimNames: string[];
  // baseSampleIndex?: number;
  // changeDimSamples: (dimName: string, sampleIndex: number) => void;
  setDimUserNames: (dictName: { [key: string]: string }) => void;
};

const ItemBrowser = (props: Props) => {
  const {
    matrixData,
    dimUserNames,
    setDimUserNames,
    dataset,
    width,
    samples,
    // changeDimSamples,
    // baseSampleIndex,
    dimNames,
    height,
    isDataLoading
  } = props;

  const padding = 24;
  const barHeight = 30; // height of bar chart
  const gap = 3; //horizontal gap between thumbnails
  const barLabelHeight = 14;
  const stepWidth = (width - 2 * padding) / STEP_NUM - gap;

  const rootStyle = getComputedStyle(document.documentElement),
    cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height')),
    cardBodyHeight = height - cardHeadHeight;

  if (isDataLoading)
    return (
      <Card
        title={'...'}
        size="small"
        bodyStyle={{ overflowY: 'scroll', height: cardBodyHeight }}
        loading={isDataLoading}
      >
        {' '}
      </Card>
    );

  const [dimX, changeDimX] = useState(`dim_0`);
  const [isShowConfig, toggleConfig] = useState(false);
  const [sampleIdx, changeSampleIdx] = useState(0);

  const baselineOptions = [...samples]
    // .sort((a, b) => +a['recons_loss'] - +b['recons_loss'])
    .slice(0, samples.length / 4);

  const iconWidth = 15;
  const imageSize = 64;

  const maxV = getMax(matrixData[dimX].histogram);
  const yScale = scaleLog().domain([0.1, maxV]).range([0, barHeight]);

  const dimUserName = dimUserNames[dimX] || dimX;

  // const inputName = (
  //   <>
  //     <input value={dimUserName} unselectable="on" onChange={e => setDimUserNames({ [dimX]: e.target.value })} />
  //   </>
  // );

  const dimXSelector = (
    <select
      id="xSelector"
      style={{ width: '100px' }}
      value={dimX}
      onChange={(e: any) => {
        changeDimX(e.target.value);
      }}
    >
      {dimNames
        .filter(d => d.includes('dim'))
        .map(dim => {
          return (
            <option key={dim} value={dim}>
              {dimUserNames[dim] || dim}
            </option>
          );
        })}
    </select>
  );

  const dimYSelector = (
    <select id="ySelector" style={{ width: '100px' }}>
      <option value="none">none</option>
      <option value="std">std</option>
      {dimNames.map(dimName => {
        return (
          <option key={dimName} value={dimName}>
            {dimUserNames[dimName] || dimName}
          </option>
        );
      })}
    </select>
  );

  const url = `${BASE_URL}/api/get_${dataset}_sample?id=${samples[sampleIdx].id}`;
  const image = (
    <img
      src={url}
      className="pixelated"
      alt={`sample_${samples[sampleIdx].id}`}
      height="64"
      width="64"
      style={{ border: 'solid 1px gray' }}
    />
  );

  const options = baselineOptions.map(sample => (
    <Option key={sample.id} value={sample.index} label={`${sample.id}.png`}>
      {dataset == 'IDC' ? sample.id.split('/')[2] : `${sample.id}.png`}
    </Option>
  ));

  const baselineSelector = (
    <>
      <label htmlFor="fname">Explore {dimUserNames[dimX] || dimX} based on image </label>
      <Select
        id="baseline"
        showSearch
        optionFilterProp="label"
        onChange={(idx: number) => changeSampleIdx(idx)}
        style={{ width: 'auto' }}
        placeholder="select an image"
        value={sampleIdx}
      >
        {options}
      </Select>
      {image}
    </>
  );

  const config = (
    <div className={styles.ConfigContainer}>
      <Button className={styles.ConfigBtn} type="default" size="small" onClick={() => toggleConfig(!isShowConfig)}>
        Config <LeftCircleOutlined />
      </Button>
      <div className={clsx(styles.ConfigPanel, isShowConfig ? 'show' : 'hide')}>
        <div className={styles.ConfigHideBtn} onClick={() => toggleConfig(false)}>
          <span>
            Config <RightCircleOutlined />
          </span>
        </div>
        {/* ----------Arrange----------- */}
        <hr className={styles.configHr} />
        <h5>Arrange</h5>
        <label>x</label>
        {dimXSelector}
        <br />
        <label>y</label>
        {dimYSelector}
        <br />
        <Button type="default" id="umapBtn" size="small">
          UMAP
        </Button>
        <br />
        <Button type="default" id="1dBtn" size="small">
          Grid
        </Button>
        {/* --------Group------------- */}
        <hr className={styles.configHr} />
        <h5>Group</h5>
        <Button type="default" id="stackXBtn" size="small">
          GroupX
        </Button>
        <Button type="default" id="groupBtn" size="small">
          AutoGroup
        </Button>
        <Button type="default" id="splitBtn" size="small">
          Split-All
        </Button>
        {/* --------Item------------- */}
        <hr className={styles.configHr} />
        <h5>Item</h5>
        <label> Size</label> <input id="itemSize" type="number" min="10" max="50" value="40" />
        {/* --------Summary------------- */}
        <hr className={styles.configHr} />
        <h5> Summary </h5>
        <select id="summarySelector" style={{ width: '100px' }}>
          <option value="foreshortened">Foreshortened</option>
          <option value="combining">Combining</option>
          <option value="combining2">Combining with offset</option>
          <option value="representative">Representative</option>
        </select>
        <Button type="primary" className={styles.updateBtn}>
          Update Concept
        </Button>
      </div>
    </div>
  );

  const z = samples[sampleIdx]['z'];
  const Row = (
    <DimRow
      row={matrixData[dimX]}
      dimName={dimX}
      stepWidth={stepWidth}
      yScale={yScale}
      barHeight={barHeight}
      barLabelHeight={barLabelHeight}
      gap={gap}
      imageSize={imageSize}
      dataset={props.dataset}
      latentZ={z}
    />
  );

  return (
    <Card
      title={`Items [${isDataLoading ? '...' : samples.length}]`}
      size="small"
      bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
      loading={isDataLoading}
    >
      {baselineSelector}
      <svg width={width - 2 * padding} height={barHeight + imageSize + barLabelHeight + gap * 2} id="ItemBrowser">
        <g>{Row}</g>
      </svg>
      <Piling dataset={dataset} samples={samples} dimNames={dimNames} dimUserNames={dimUserNames} />

      {config}
    </Card>
  );
};

const get_tool_icon = (width: number) => (
  <path
    transform={`scale(${(0.01 * width) / 8})`}
    d="M876.6 239.5c-.5-.9-1.2-1.8-2-2.5-5-5-13.1-5-18.1 0L684.2 409.3l-67.9-67.9L788.7 169c.8-.8 1.4-1.6 2-2.5 3.6-6.1 1.6-13.9-4.5-17.5-98.2-58-226.8-44.7-311.3 39.7-67 67-89.2 162-66.5 247.4l-293 293c-3 3-2.8 7.9.3 11l169.7 169.7c3.1 3.1 8.1 3.3 11 .3l292.9-292.9c85.5 22.8 180.5.7 247.6-66.4 84.4-84.5 97.7-213.1 39.7-311.3zM786 499.8c-58.1 58.1-145.3 69.3-214.6 33.6l-8.8 8.8-.1-.1-274 274.1-79.2-79.2 230.1-230.1s0 .1.1.1l52.8-52.8c-35.7-69.3-24.5-156.5 33.6-214.6a184.2 184.2 0 01144-53.5L537 318.9a32.05 32.05 0 000 45.3l124.5 124.5a32.05 32.05 0 0045.3 0l132.8-132.8c3.7 51.8-14.4 104.8-53.6 143.9z"
  ></path>
);

export default ItemBrowser;
