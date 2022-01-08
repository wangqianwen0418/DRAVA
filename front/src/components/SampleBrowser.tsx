import React from 'react';
import sampleVectors from 'assets/samples_vector.json';
import sampleLabels from 'assets/sample_labels.json';

import { Card } from 'antd';

import { TResultRow } from 'types';

interface Props {
  samples: TResultRow[];
  height: number;
}

export default class SampleBrowser extends React.Component<Props, {}> {
  render() {
    const { samples, height } = this.props;

    const rootStyle = getComputedStyle(document.documentElement),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));
    return (
      <Card
        title={`Samples [${samples.length}]`}
        size="small"
        bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
      >
        {samples.slice(0, 20).map(sample => {
          return (
            <img
              src={`assets/sample_imgs/${sample.id}.png`}
              alt={`sample_${sample.id}`}
              key={sample.id}
              style={{ border: 'solid black 1px' }}
            />
          );
        })}
      </Card>
    );
  }
}
