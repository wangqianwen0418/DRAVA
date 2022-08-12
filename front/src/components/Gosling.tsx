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
    
    const outerPoints: [number, number][] = [];
    const sorted = labelJSON.sort((a, b) => a.start - b.start);
    sorted.forEach(({ start, end }, i) => {
      const lb: [number, number] = [start, end]; // the left bottom point of a rectangle
      if(outerPoints.length === 0 || end > outerPoints[outerPoints.length - 1][1]) {
        // This means `lb` is an outer point
        outerPoints.push(lb);
      }
    });
    const overlayRects: Record<string, string | number>[] = [];
    const template = { chromosome: 'chr5' }; // Support other chromosomes as well?
    const CHR5_SIZE = 181538259;
    outerPoints.forEach(([start, end], i) => {
      const nextStart = i === outerPoints.length - 1 ? CHR5_SIZE : outerPoints[i + 1][0];
      /* left bottom side of the diagonal */
      overlayRects.push({ 
        ...template, 
        x: start, 
        xe: nextStart, 
        y: end, 
        ye: CHR5_SIZE
      });
      /* right top side of the diagonal */
      const nextNoOverlap = end < nextStart;
      if(nextNoOverlap) {
        overlayRects.push({ 
          ...template, 
          x: end, 
          xe: CHR5_SIZE, 
          y: start, 
          ye: end
        });
        overlayRects.push({ 
          ...template, 
          x: nextStart, 
          xe: CHR5_SIZE, 
          y: end, 
          ye: nextStart
        });
      } else {
        overlayRects.push({ 
          ...template, 
          x: end, 
          xe: CHR5_SIZE, 
          y: start, 
          ye: nextStart
        });
      }
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

    if (dataset == 'matrix') {
      labelTrack['title'] = 'Samples By Depth';
      labelTrack['row'] = {
        field: 'level',
        type: 'nominal',
        legend: false,
        domain: ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0'].reverse()
      };
      // labelTrack['stroke'] = { value: 'steelBlue' };
      // labelTrack['color'] = { value: ORANGE };
      labelTrack['height'] = multiLabelHeight;
    }

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
      height: goslingComponentHeight - peakHeight - multiLabelHeight
    };

    const annotationOnMatrix = {
      data: {
        values: labelJSON,
        type: 'json',
        chromosomeField: 'chromosome',
        genomicFields: ['start', 'end']
      },
      mark: 'bar',
      x: { field: 'start', type: 'genomic' },
      xe: { field: 'end', type: 'genomic' },
      y: { field: 'start', type: 'genomic' },
      ye: { field: 'end', type: 'genomic' },
      stroke: { value: ORANGE },
      strokeWidth: { value: 2 },
      color: { value: 'none' },
      opacity: { value: 1 },
      overlayOnPreviousTrack: true
    };

    const fadeOutOnMatrix = {
      data: {
        values: overlayRects,
        type: 'json',
        chromosomeField: 'chromosome',
        genomicFields: ['x', 'xe', 'y', 'ye']
      },
      mark: 'bar',
      x: { field: 'x', type: 'genomic' },
      xe: { field: 'xe', type: 'genomic' },
      y: { field: 'y', type: 'genomic' },
      ye: { field: 'ye', type: 'genomic' },
      // stroke: { value: ORANGE },
      strokeWidth: { value: 0 },
      color: { value: 'white' },
      opacity: { value: 0.3 },
      overlayOnPreviousTrack: true
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
        type: 'genomic'
      },
      y: { field: 'peak', type: 'quantitative', axis: 'none' },
      color: { value: 'gray' },
      height: peakHeight
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
      tracks: dataset == 'sequence' ? [labelTrack, PeakTrack] : [labelTrack, CTCFTrack, MatrixTrack, fadeOutOnMatrix, annotationOnMatrix]
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
