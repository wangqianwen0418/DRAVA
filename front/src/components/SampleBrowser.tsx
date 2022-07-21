import React, { useState } from 'react';

import styles from './SampleBrowser.module.css';
import clsx from 'clsx';

import { Card, Select } from 'antd';

import { TFilter, TResultRow, TDistribution } from 'types';
import { BASE_URL } from 'Const';

const { Option } = Select;

interface Props {
  dataset: string;
  samples: TResultRow[];
  height: number;
  isDataLoading: boolean;
  matrixData: { [dimName: string]: TDistribution };
  dimUserNames: { [key: string]: string };
  filters: TFilter;
}

interface State {
  dimName: string;
}

const SampleBrowser = (props: Props) => {
  const [dimName, changeDimName] = useState<string>('none');
  const { samples, height, dataset, isDataLoading, filters, matrixData, dimUserNames } = props;

  const rootStyle = getComputedStyle(document.documentElement),
    cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

  const dimSelect = (
    <>
      grouped by{' '}
      <Select
        placeholder="Add a dimension"
        style={{ width: '150px', height: '30px' }}
        value={dimName}
        onChange={changeDimName}
      >
        <Option value="none"> none</Option>
        {Object.keys(filters).map(dimName => (
          <Option key={dimName} value={dimName}>
            {dimUserNames[dimName] || dimName}
          </Option>
        ))}
      </Select>
    </>
  );

  const filteredSampleIds = samples.map(d => d.id);
  const id2Image = (id: string) => {
    const url = `${BASE_URL}/api/get_item_sample?dataset=${dataset}&id=${id}`;
    return (
      <img
        loading="lazy"
        src={url}
        alt={`sample_${id}`}
        key={id}
        className={clsx(styles.sample, 'pixelated')}
        height="40"
        width="40"
      />
    );
  };

  const content =
    dimName == 'none'
      ? filteredSampleIds.map(id => id2Image(id))
      : matrixData[dimName]?.groupedSamples.map((sampleIds, idx) => (
          <div key={idx} className={styles.groupContainer}>
            {sampleIds.filter(id => filteredSampleIds.includes(id)).map(id => id2Image(id))}
          </div>
        ));

  return (
    <Card
      title={`Items [${isDataLoading ? '...' : samples.length}]`}
      size="small"
      bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
      loading={isDataLoading}
      extra={!isDataLoading && dimSelect}
    >
      {content}
    </Card>
  );
};

export default SampleBrowser;
