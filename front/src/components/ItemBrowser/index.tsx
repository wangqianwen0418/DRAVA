import { TDistribution, TResultRow, TFilter } from 'types';
import React, { useState } from 'react';
import { Card, Select, Button } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import { datasetConfig } from 'config';

import styles from './index.module.css';
import clsx from 'clsx';

import { STEP_NUM } from 'Const';
import { scaleLinear, scaleLog, ScaleLogarithmic } from 'd3-scale';
import { getMax } from 'helpers';
import { DimRow } from 'components/LatentDim/DimRow';
import Piling from './Piling';
import { getItemURL } from 'dataService';

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
    isDataLoading,
    filters
  } = props;

  const padding = 24;
  const barHeight = 30; // height of bar chart
  const gap = 3; //horizontal gap between thumbnails
  const barLabelHeight = 14;
  const imageSize = 64;

  const rowHeight = barHeight + imageSize + barLabelHeight + gap * 2;

  const XStepWidth = (width - padding - rowHeight - 180) / STEP_NUM - gap; // 180 is the width of the configure panel

  const pilingHeight = height - rowHeight - 150;
  const YStepWidth = pilingHeight / STEP_NUM - gap;

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
  const [dimY, changeDimY] = useState(`none`);
  const [group, changeGroup] = useState('umap');
  const [sampleIdx, changeSampleIdx] = useState(0);
  const [isUpdating, changeUpadtingStatus] = useState(false);

  const baselineOptions = [...samples]
    // .sort((a, b) => +a['recons_loss'] - +b['recons_loss'])
    .slice(0, samples.length / 4);

  const maxV = getMax(matrixData[dimX].histogram);
  const yScaleLog = scaleLog()
    .domain([0.1, maxV])
    .range([barHeight / 10, barHeight]);
  const yScaleLinear = scaleLinear()
    .domain([0.1, maxV])
    .range([barHeight / 10, barHeight]);

  const dimXSelector = (
    <select
      id="xSelector"
      style={{ width: '100px' }}
      value={dimX}
      disabled={group == 'umap'}
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
    <select
      id="ySelector"
      style={{ width: '100px' }}
      value={dimY}
      onChange={(e: any) => {
        changeDimY(e.target.value);
      }}
      disabled={group != 'concept'}
    >
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

  const url = `${getItemURL(dataset, samples[sampleIdx].id)}&border=1`;
  const image = (
    <img
      src={url}
      className="pixelated"
      alt={`sample_${samples[sampleIdx].id}`}
      height="45"
      width="45"
      style={{ border: 'solid 1px gray' }}
    />
  );

  const options = baselineOptions.map(sample => (
    <Option key={sample.id} value={sample.index} label={`${sample.id}.png`}>
      {dataset == 'IDC' ? sample.id.split('/')[2] : `${sample.id}.png`}
    </Option>
  ));

  const baselineSelector = (
    <div style={{ transform: `translate(${rowHeight}px, 0px)` }}>
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
    </div>
  );

  const config = (
    <div className={styles.ConfigContainer}>
      <div className={clsx(styles.ConfigPanel, 'show')}>
        <div className={styles.ConfigHideBtn}>
          <span>
            <b>Config</b>
          </span>
        </div>
        {/* ----------Arrange----------- */}
        <hr className={styles.configHr} />
        <h5>Arrange</h5>
        <select
          id="groupSelector"
          style={{ width: '120px' }}
          value={group}
          onChange={(e: any) => {
            changeGroup(e.target.value);
          }}
        >
          <option value="umap">UMAP</option>
          <option value="grid">1D Grid</option>
          <option value="concept">Concept</option>
        </select>
        <br />
        <label>x</label>
        {dimXSelector}
        <br />
        <label>y</label>
        {dimYSelector}
        <Button type="default" id="gridBtn" size="small">
          Arrange in 2D Grid
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
        <select id="summarySelector" style={{ width: '120px' }} defaultValue="representative">
          <option value="foreshortened">Partial</option>
          <option value="foreshortened">Compression</option>
          <option value="see-through">See-Through</option>
          <option value="average">Average</option>
          <option value="representative">Representative</option>
        </select>
        <br />
        <label>Label</label>{' '}
        <select id="labelSelector" style={{ width: '100px' }} defaultValue="representative">
          <option value="none">none</option>
          {datasetConfig[dataset].labels.map((label, i) => {
            return (
              <option value={label} key={i}>
                {label}
              </option>
            );
          })}
        </select>
        <Button type="primary" id="updateConcept" className={styles.updateBtn} loading={isUpdating}>
          Update Concept
        </Button>
      </div>
    </div>
  );

  const z = samples[sampleIdx]['z'];
  const xRow = (
    <DimRow
      row={matrixData[dimX]}
      dimName={dimX}
      stepWidth={XStepWidth}
      yScale={dimX.includes('dim') ? yScaleLog : yScaleLinear}
      barHeight={barHeight}
      barLabelHeight={barLabelHeight}
      gap={gap}
      imageSize={XStepWidth}
      dataset={props.dataset}
      latentZ={z}
      rotate={false}
      hideHistogram={true}
    />
  );

  const yRow =
    dimY != 'none' && matrixData[dimY] && group === 'concept' ? (
      <DimRow
        row={matrixData[dimY]}
        dimName={dimY}
        stepWidth={YStepWidth}
        yScale={dimY.includes('dim') ? yScaleLog : yScaleLinear}
        barHeight={barHeight}
        barLabelHeight={barLabelHeight}
        gap={gap}
        imageSize={YStepWidth}
        dataset={props.dataset}
        // latentZ={z}
        rotate={true}
        hideHistogram={true}
      />
    ) : (
      <></>
    );

  return (
    <Card
      // title={`Items [${isDataLoading ? '...' : samples.length}]`}
      title="Item Browser"
      size="small"
      bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
      loading={isDataLoading}
    >
      <div>
        <svg width={120} height={pilingHeight} style={{ float: 'left' }} id="ItemBrowserY">
          <g transform={` translate(${rowHeight / 2} , ${pilingHeight - (rowHeight - YStepWidth)}) rotate(-90)`}>
            <g>{yRow}</g>
          </g>
        </svg>
        <Piling
          dataset={dataset}
          samples={samples}
          dimNames={dimNames}
          dimUserNames={dimUserNames}
          height={pilingHeight}
          changeUpadtingStatus={changeUpadtingStatus}
        />
      </div>
      {group != 'umap' ? (
        <>
          {baselineSelector}
          <svg width={width - 2 * padding} height={rowHeight} id="ItemBrowser">
            <g transform={`translate(${rowHeight}, 0)`}>
              <g>{xRow}</g>
            </g>
          </svg>
        </>
      ) : (
        <></>
      )}

      {config}

      {isUpdating && (
        <div
          style={{
            width,
            height: height - cardHeadHeight,
            backgroundColor: 'white',
            opacity: 0.5,
            position: 'absolute',
            top: cardHeadHeight,
            textAlign: 'center'
          }}
        >
          <LoadingOutlined style={{ fontSize: '50px', lineHeight: `${height - cardHeadHeight}px` }} />
        </div>
      )}
    </Card>
  );
};

export default ItemBrowser;
