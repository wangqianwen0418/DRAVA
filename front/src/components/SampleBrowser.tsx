import React from 'react';

import styles from './SampleBrowser.module.css';

import { Card, Select } from 'antd';

import { TFilter, TResultRow } from 'types';
import { BASE_URL } from 'Const';

const { Option } = Select;

interface Props {
  dataset: string;
  samples: TResultRow[];
  height: number;
  isDataLoading: boolean;
  filters: TFilter;
}

interface State {
  dimName: string;
}

export default class SampleBrowser extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { dimName: 'dim_0' };
  }
  onChangeDim(dimName: string) {
    this.setState({ dimName });
  }
  render() {
    const { samples, height, dataset, isDataLoading, filters } = this.props;

    const rootStyle = getComputedStyle(document.documentElement),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const dimSelect = (
      <>
        grouped by{' '}
        <Select
          placeholder="Add a dimension"
          style={{ width: '150px', height: '30px' }}
          value={this.state.dimName}
          onChange={this.onChangeDim.bind(this)}
        >
          {Object.keys(filters).map(dimName => (
            <Option key={dimName} value={dimName}>
              {dimName}
            </Option>
          ))}
        </Select>
      </>
    );
    return (
      <Card
        title={`Samples [${isDataLoading ? '...' : samples.length}]`}
        size="small"
        bodyStyle={{ overflowY: 'scroll', height: height - cardHeadHeight }}
        loading={isDataLoading}
        extra={!isDataLoading && dimSelect}
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
