import { TDistribution, TResultRow, TFilter } from 'types';
import React, { useState } from 'react';
import { Card, Select, Button, message } from 'antd';
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
  const stepWidth = (width - 2 * padding - 200) / STEP_NUM - gap;

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
        {` `}
        <Button type="default" id="1dBtn" size="small">
          Grid
        </Button>
        {/* --------Group------------- */}
        <hr className={styles.configHr} />
        <h5>Group</h5>
        <Button type="default" id="XGroupBtn" size="small">
          Group by X
        </Button>
        <br />
        <Button type="default" id="groupBtn" size="small">
          Group by grid
        </Button>
        <br />
        <Button type="default" id="splitBtn" size="small">
          Split-All
        </Button>
        {/* --------Item------------- */}
        <hr className={styles.configHr} />
        <h5>Item</h5>
        <label> Size</label> <input id="itemSize" type="number" min="10" max="50" defaultValue={45} />
        {/* --------Summary------------- */}
        <hr className={styles.configHr} />
        <h5> Summary </h5>
        <select id="summarySelector" style={{ width: '100px' }}>
          <option value="foreshortened">Foreshortened</option>
          <option value="combining">Combining</option>
          <option value="combining2">Combining with offset</option>
          <option value="representative">Representative</option>
        </select>
        <Button
          type="primary"
          className={styles.updateBtn}
          onClick={() =>
            message.warning(
              'Update Concept is not supported in the online demo.\n Please download Drava and run it on your local computer.',
              5 //duration = 5s
            )
          }
        >
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
      imageSize={stepWidth}
      dataset={props.dataset}
      latentZ={z}
    />
  );

  const rowHeight = barHeight + imageSize + barLabelHeight + gap * 2;
  const pilingHeight = height - rowHeight - 150;

  return (
    <Card
      title={`Items [${isDataLoading ? '...' : samples.length}]`}
      size="small"
      bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
      loading={isDataLoading}
    >
      <Piling
        dataset={dataset}
        samples={samples}
        dimNames={dimNames}
        dimUserNames={dimUserNames}
        height={pilingHeight}
      />
      {baselineSelector}
      <svg width={width - 2 * padding} height={rowHeight} id="ItemBrowser">
        <g>{Row}</g>
      </svg>

      {config}
    </Card>
  );
};

export default ItemBrowser;
