import React from 'react';

import styles from './SampleBrowser.module.css';

import { Card } from 'antd';

import { TResultRow } from 'types';

interface Props {
  dataset: string;
  samples: TResultRow[];
  height: number;
}

export default class SampleBrowser extends React.Component<Props, {}> {
  render() {
    const { samples, height, dataset } = this.props;

    const rootStyle = getComputedStyle(document.documentElement),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));
    return (
      <Card
        title={`Samples [${samples.length}]`}
        size="small"
        bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
      >
        {samples.map(sample => {
          const url = dataset == 'matrix' ? `assets/tad_imgs/chr5:${parseInt(sample.id) + 1}.jpg` : '';
          return (
            <img
              src={url}
              alt={`sample_${sample.id}`}
              key={sample.id}
              className={styles.sample}
              height={40}
              width={40}
            />
          );
        })}
      </Card>
    );
  }
}
