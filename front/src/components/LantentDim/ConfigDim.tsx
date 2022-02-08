import { TDistribution, TResultRow } from 'types';
import React, { useState } from 'react';
import { Modal, Select, Button } from 'antd';
import { STEP_NUM } from 'Const';
import { scaleLinear, scaleLog, ScaleLogarithmic } from 'd3-scale';
import { getMax } from 'helpers';
import { DimRow } from './DimRow';

import { BASE_URL } from 'Const';

const { Option } = Select;

type Props = {
  row: TDistribution;
  dimName: string;
  dimUserNames: { [key: string]: string };
  dataset: string;
  samples: TResultRow[];
  setDimUserNames: (dictName: { [key: string]: string }) => void;
};

export const ConfigDim = (props: Props) => {
  const modalWidth = window.innerWidth * 0.8;
  const padding = 24;
  const barHeight = 30; // height of bar chart
  const gap = 3; //horizontal gap between thumbnails
  const barLabelHeight = 14;
  const stepWidth = (modalWidth - 2 * padding) / STEP_NUM - gap;

  const { row, dimName, dimUserNames, setDimUserNames, dataset, samples } = props;

  const [isModalVisible, setModalVisible] = useState(false);
  const [imageId, changeImageId] = useState(samples[0].id);

  const iconWidth = 15;
  const imageSize = 64;

  const maxV = getMax(row.histogram);
  const yScale = scaleLog().domain([0.1, maxV]).range([0, barHeight]);

  const inputName = (
    <>
      <label htmlFor="fname">Name: </label>
      <input
        value={dimUserNames[dimName] || dimName}
        unselectable="on"
        onChange={e => setDimUserNames({ [dimName]: e.target.value })}
      />
    </>
  );

  const url = `${BASE_URL}/api/get_${dataset}_sample?id=${imageId}`;
  const image = <img src={url} alt={`sample_${imageId}`} height="64" width="64" />;

  const options = samples.map(sample => (
    <Option key={sample.id} value={sample.id}>
      {`${sample.id}.png`}
    </Option>
  ));

  const baselineSelector = (
    <>
      <label htmlFor="fname">Explore this Dimension based on an image </label>
      <Select id="baseline" onChange={(id: string) => changeImageId(id)} style={{ width: '100px' }} value={imageId}>
        {options}
      </Select>
      <br />
      {image}
      <Button type="default">Update</Button>
    </>
  );

  const Row = (
    <DimRow
      row={row}
      dimName={dimName}
      stepWidth={stepWidth}
      yScale={yScale}
      barHeight={barHeight}
      barLabelHeight={barLabelHeight}
      gap={gap}
      imageSize={imageSize}
      dataset={props.dataset}
    />
  );
  return (
    <>
      <g className="configIcon pointer_cursor" fill="gray" onClick={() => setModalVisible(true)}>
        <rect width={iconWidth} height={iconWidth} fill="transparent" />
        {get_tool_icon(iconWidth)}
      </g>

      <Modal
        title={`Configure ${dimName}`}
        visible={isModalVisible}
        onCancel={() => setModalVisible(false)}
        onOk={() => setModalVisible(false)}
        okText="Save"
        width={modalWidth}
        destroyOnClose
      >
        {inputName}
        <br />
        {baselineSelector}
        <svg width={modalWidth - 2 * padding} height={barHeight + imageSize + barLabelHeight + gap * 2}>
          {Row}
        </svg>
      </Modal>
    </>
  );
};

const get_tool_icon = (width: number) => (
  <path
    transform={`scale(${(0.01 * width) / 8})`}
    d="M876.6 239.5c-.5-.9-1.2-1.8-2-2.5-5-5-13.1-5-18.1 0L684.2 409.3l-67.9-67.9L788.7 169c.8-.8 1.4-1.6 2-2.5 3.6-6.1 1.6-13.9-4.5-17.5-98.2-58-226.8-44.7-311.3 39.7-67 67-89.2 162-66.5 247.4l-293 293c-3 3-2.8 7.9.3 11l169.7 169.7c3.1 3.1 8.1 3.3 11 .3l292.9-292.9c85.5 22.8 180.5.7 247.6-66.4 84.4-84.5 97.7-213.1 39.7-311.3zM786 499.8c-58.1 58.1-145.3 69.3-214.6 33.6l-8.8 8.8-.1-.1-274 274.1-79.2-79.2 230.1-230.1s0 .1.1.1l52.8-52.8c-35.7-69.3-24.5-156.5 33.6-214.6a184.2 184.2 0 01144-53.5L537 318.9a32.05 32.05 0 000 45.3l124.5 124.5a32.05 32.05 0 0045.3 0l132.8-132.8c3.7 51.8-14.4 104.8-53.6 143.9z"
  ></path>
);
