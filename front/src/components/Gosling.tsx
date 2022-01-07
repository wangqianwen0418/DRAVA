import { validateGoslingSpec, GoslingComponent, GoslingSpec } from 'gosling.js';
import * as React from 'react';

import { Card } from 'antd';
import { TResultRow } from 'types';

interface Props {
  samples: TResultRow[];
  dataset: string;
  width: number;
  height: number;
}

export const GoslingVis = (props: Props) => {
  const CHR = props.dataset == 'sequence' ? 7 : 5;
  const labelJSON = props.samples
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

  if (props.dataset == 'matrix') {
    labelTrack['row'] = { field: 'level', type: 'nominal', legend: false };
    labelTrack['height'] = 12 * 8;
  }

  const MatrixTrack = {
    title: 'HFFc6_Hi-C',
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

  const PeakTrack = {
    layout: 'linear',
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
    width: props.width - cardPadding * 2,
    height: props.height - cardPadding * 2 - cardHeadHeight - 24, // gosling vis axis: 24px
    tracks: [props.dataset == 'sequence' ? PeakTrack : MatrixTrack, labelTrack]
  };

  // validate the spec
  const validity = validateGoslingSpec(spec);

  if (validity.state === 'error') {
    console.warn('Gosling spec is invalid!', validity.message);
    return <></>;
  }

  return (
    <Card title="Genomic Browser" size="small">
      <GoslingComponent spec={spec as GoslingSpec} />
    </Card>
  );
};
