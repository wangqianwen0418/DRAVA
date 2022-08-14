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

const ORANGE = '#E6A01B',
  LIGHT_ORANGE = '#fbd58f';

export default class GoslingVis extends React.Component<Props, {}> {
  shouldComponentUpdate(nextProps: Props) {
    if (
      nextProps.dataset != this.props.dataset ||
      nextProps.samples.length !== this.props.samples.length ||
      nextProps.width != this.props.width ||
      nextProps.height != this.props.height
    ) {
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

    const goslingComponentWidth = width - cardPadding * 2;
    const goslingComponentHeight = height - cardPadding * 2 - cardHeadHeight - 30; // gosling axis;
    const labelHeight = 18,
      peakHeight = 30,
      multiLabelHeight = 8 * 8;

    const labelTrack: any = {
      title: 'Samples',
      data: {
        values: labelJSON,
        type: 'json',
        chromosomeField: 'chromosome',
        genomicFields: ['start', 'end']
      },
      mark: 'rect',
      height: labelHeight,
      color: { value: LIGHT_ORANGE },
      x: { field: 'start', type: 'genomic' },
      xe: { field: 'end', type: 'genomic' },
      stroke: { value: ORANGE },
      strokeWidth: { value: 2 }
    };

    const MatrixTrack = {
      title: 'HFFc6_Hi-C',
      id: 'matrix-track',
      data: {
        url: 'https://server.gosling-lang.org/api/v1/tileset_info/?d=hffc6-hic-hg38',
        type: 'matrix'
      },
      mark: 'bar',
      x: {
        field: 'xs',
        type: 'genomic',
        domain: { chromosome: `chr${CHR}` }
      },
      xe: {
        field: 'xe',
        type: 'genomic'
      },
      y: {
        field: 'ys',
        type: 'genomic',
        domain: { chromosome: `chr${CHR}` },
        axis: 'none'
      },
      ye: {
        field: 'ye',
        type: 'genomic',
        axis: 'none'
      },
      color: {
        field: 'value',
        type: 'quantitative'
      },
      height: goslingComponentHeight
    };

    const annotationOnMatrix = {
      data: {
        values: labelJSON,
        type: 'json',
        chromosomeField: 'chromosome',
        genomicFields: ['start', 'end']
      },
      mark: 'bar',
      x: { field: 'start', type: 'genomic', axis: 'top' },
      xe: { field: 'end', type: 'genomic' },
      y: { field: 'start', type: 'genomic', axis: 'left' },
      ye: { field: 'end', type: 'genomic' },
      stroke: { value: ORANGE },
      strokeWidth: { value: 2 },
      color: { value: 'none' },
      opacity: { value: 1 },
      overlayOnPreviousTrack: true
    };

    const PeakTrack = {
      layout: 'linear',
      id: 'peak-track',
      data: {
        url: 'https://s3.amazonaws.com/gosling-lang.org/data/HFFc6_Atacseq.mRp.clN.bigWig',
        type: 'bigwig',
        column: 'position',
        value: 'peak',
        binSize: 2
      },
      mark: 'area',
      x: {
        field: 'position',
        type: 'genomic'
      },
      y: { field: 'peak', type: 'quantitative' },
      color: { value: 'gray' },
      height: peakHeight
    };

    const spec = {
      responsiveSize: { width: true },
      spacing: 0,
      xDomain: { chromosome: CHR.toString() },
      width: goslingComponentWidth,
      tracks: dataset == 'sequence' ? [labelTrack, PeakTrack] : [MatrixTrack, annotationOnMatrix]
    };

    // validate the spec
    const validity = validateGoslingSpec(spec);

    if (validity.state === 'error') {
      console.warn('Gosling spec is invalid!', validity.message);
      return <></>;
    }

    return (
      <Card title="Context View (Genomic)" size="small" loading={isDataLoading}>
        <GoslingComponent spec={spec as GoslingSpec} experimental={{ reactive: true }} />
      </Card>
    );
  }
}
