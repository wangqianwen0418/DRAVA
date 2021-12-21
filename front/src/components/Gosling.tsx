import { validateGoslingSpec, GoslingComponent, GoslingSpec } from 'gosling.js';
import * as React from 'react';

import { Card } from 'antd';
import { TResultRow } from 'types';

interface Props {
    samples: TResultRow[];
    width: number;
    height: number;
}

export const GoslingVis = (props: Props) => {
    const labelJSON = props.samples.map(sample => {
        return {
            Chromosome: `chr${sample.chr}`,
            chromStart: sample.start,
            chromEnd: sample.end
        };
    });
    const rootStyle = getComputedStyle(document.documentElement),
        cardPadding = parseInt(rootStyle.getPropertyValue('--card-body-padding')),
        cardHeadHeight = parseInt(rootStyle.getPropertyValue('--card-head-height'));

    const spec = {
        title: '',
        width: props.width - cardPadding * 2,
        height: props.height - cardPadding * 2 - cardHeadHeight - 24, // gosling vis axis: 24px
        tracks: [
            {
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
                    domain: { chromosome: '7' },
                    axis: 'bottom'
                },
                y: { field: 'peak', type: 'quantitative' },
                color: { value: 'black' },
                height: 40
            },
            {
                data: {
                    values: labelJSON,
                    type: 'json',
                    chromosomeField: 'Chromosome',
                    genomicFields: ['chromStart', 'chromEnd']
                },
                mark: 'rect',
                size: { value: 12 },
                height: 12,
                x: { field: 'chromStart', type: 'genomic', axis: 'none', domain: { chromosome: '7' } },
                xe: { field: 'chromEnd', type: 'genomic' },
                stroke: { value: 'orange' },
                strokeWidth: { value: 1 }
            }
        ]
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
