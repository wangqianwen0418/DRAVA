import React from 'react';

import styles from './SampleBrowser.module.css';

import { Card } from 'antd';

import { TResultRow } from 'types';
import { BASE_URL } from 'Const';

interface Props {
  dataset: string;
  samples: TResultRow[];
  height: number;
  isDataLoading: boolean;
}

export default class SampleBrowser extends React.Component<Props, {}> {
  render() {
    const { samples, height, dataset, isDataLoading } = this.props;

    const rootStyle = getComputedStyle(document.documentElement),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));
    return (
      <Card
        title={`Samples [${isDataLoading ? '...' : samples.length}]`}
        size="small"
        bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
        loading={isDataLoading}
      >
        {samples.map(sample => {
          const url = `${BASE_URL}/api/get_${dataset}_sample?id=${sample.id}`;
          return (
            <img
              loading="lazy"
              src={url}
              alt={`sample_${sample.id}`}
              key={sample.id}
              className={styles.sample}
              height="40"
              width="40"
            />
          );
        })}
      </Card>
    );
  }
}
