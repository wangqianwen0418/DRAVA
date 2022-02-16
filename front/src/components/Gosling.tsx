import { validateGoslingSpec, GoslingComponent, GoslingSpec } from 'gosling.js';
import * as React from 'react';

import { Card } from 'antd';
import { TResultRow } from 'types';
import { whatCHR } from 'dataService';
import { render } from '@testing-library/react';

interface Props {
  samples: TResultRow[];
  dataset: string;
  width: number;
  height: number;
  isDataLoading: boolean;
}

export default class GoslingVis extends React.Component<Props, {}> {
  shouldComponentUpdate(nextProps: Props) {
    if (nextProps.dataset != this.props.dataset || nextProps.samples.length !== this.props.samples.length) {
      return true;
    }
    return false;
  }
  render() {
    const { dataset, isDataLoading, width, height, samples } = this.props;
    const CHR = whatCHR(dataset);
    const labelJSON = samples
      .filter(sample => !sample.level || parseInt(sample.level) < 8) // only no level labels or labels whose level is less than 8
      .filter(sample => sample.chr == CHR)
      .map(sample => {
        return {
          chromosome: `chr${sample.chr}`,
          chromStart: sample.start,
          chromEnd: sample.end,

          ...sample
        };
      });

    const rootStyle = getComputedStyle(document.documentElement),
      cardPadding = parseInt(rootStyle.getPropertyValue('--card-body-padding')),
      cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const labelTrack: any = {
      title: 'samples',
      data: {
        values: labelJSON,
        type: 'json',
        chromosomeField: 'chromosome',
        genomicFields: ['start', 'end']
      },
      mark: 'rect',
      size: { value: 12 },
      height: 12,
      x: {
        field: 'start',
        type: 'genomic',
        axis: 'none',
        domain: { chromosome: `chr${CHR}` }
      },
      xe: { field: 'end', type: 'genomic' },
      stroke: { value: 'steelblue' },
      strokeWidth: { value: 1 }
    };

    if (dataset == 'matrix') {
      labelTrack['row'] = { field: 'level', type: 'nominal', legend: false };
      labelTrack['height'] = 12 * 8;
    }

    const MatrixTrack = {
      title: 'HFFc6_Hi-C',
      id: 'matrix-track',
      data: {
        url: 'https://server.gosling-lang.org/api/v1/tileset_info/?d=hffc6-hic-hg38',
        type: 'matrix'
      },
      mark: 'rect',
      x: {
        field: 'position1',
        type: 'genomic',
        axis: 'bottom'
      },
      y: {
        field: 'position2',
        type: 'genomic',
        axis: 'none'
      },
      color: {
        field: 'value',
        type: 'quantitative',
        range: 'grey'
      },
      width: 600,
      height: 600
    };

    const CTCFTrack = {
      title: 'CTCF',
      id: 'ctcf-track',
      layout: 'linear',
      data: {
        url: 'https://s3.amazonaws.com/gosling-lang.org/data/HFFC6_CTCF.mRp.clN.bigWig',
        type: 'bigwig',
        column: 'position',
        value: 'peak',
        binSize: 1
      },
      mark: 'area',
      x: {
        field: 'position',
        type: 'genomic',
        domain: { chromosome: CHR.toString() },
        axis: 'none'
      },
      y: { field: 'peak', type: 'quantitative' },
      color: { value: 'steelblue' },
      height: 20
    };

    const PeakTrack = {
      layout: 'linear',
      id: 'peak-track',
      data: {
        url: 'https://s3.amazonaws.com/gosling-lang.org/data/HFFc6_Atacseq.mRp.clN.bigWig',
        type: 'bigwig',
        column: 'position',
        value: 'peak',
        binSize: '2'
      },
      mark: 'area',
      x: {
        field: 'position',
        type: 'genomic',
        domain: { chromosome: CHR.toString() },
        axis: 'bottom'
      },
      y: { field: 'peak', type: 'quantitative' },
      color: { value: 'steelblue' },
      height: 40
    };

    const spec = {
      title: '',
      width: width - cardPadding * 2,
      height: height - cardPadding * 2 - cardHeadHeight - 24, // gosling vis axis: 24px
      tracks: [...(dataset == 'sequence' ? [PeakTrack] : [MatrixTrack, CTCFTrack]), labelTrack]
    };

    // validate the spec
    const validity = validateGoslingSpec(spec);

    if (validity.state === 'error') {
      console.warn('Gosling spec is invalid!', validity.message);
      return <></>;
    }

    return (
      <Card title="Genomic Browser" size="small" loading={isDataLoading}>
        <GoslingComponent spec={spec as GoslingSpec} experimental={{ reactive: true }} />
      </Card>
    );
  }
}
