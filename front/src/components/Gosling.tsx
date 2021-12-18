import { validateGoslingSpec, GoslingComponent, GoslingSpec } from "gosling.js";
import * as React from "react";

import sampleLabels from 'assets/sample_labels.json'
import { Card } from "antd";


interface Props {
    sampleIdxs: number[],
    width: number,
    height: number
}

export const GoslingVis = (props: Props) => {
    const labelJSON = sampleLabels.filter((sample, idx) => props.sampleIdxs.includes(idx))
        .map(sample => {
            return {
                Chromosome: `chr${sample[0]}`,
                chromStart: sample[1],
                chromEnd: sample[2],
            }
        })

    const padding = 15

    const spec = {
        "title": "",
        "width": props.width - padding * 2,
        "height": props.height - padding * 2 - 40 - 24, // card header: 40px, gosling vis axis: 24px
        "tracks": [
          {
          "layout": "linear",
          "data": {
            "url": "https://s3.amazonaws.com/gosling-lang.org/data/HFFc6_Atacseq.mRp.clN.bigWig",
            "type": "bigwig",
            "column": "position",
            "value": "peak",
            "binSize": "2",
          },
            "mark": "area",
            "x": {
              "field": "position",
              "type": "genomic",
              "domain": {"chromosome": "7"},
              "axis": "bottom"
            },
            "y": {"field": "peak", "type": "quantitative"},
            "color": {"value": "black"},
            height: 40
          },
          {
              "data": {
                  "values": labelJSON,
                  "type":"json",
                    "chromosomeField":"Chromosome",
                    "genomicFields":[
                        "chromStart",
                        "chromEnd"
                    ]
              },
              "mark": "rect",
              "size": {"value": 12},
              height: 12,
              "x": {"field": "chromStart", "type": "genomic", "axis": "none", "domain": {"chromosome": "7"}},
              "xe": {"field": "chromEnd", "type": "genomic"},
              "stroke": {"value": "orange"},
              "strokeWidth": {"value": 1}
          }
        ]
      }

    // validate the spec
    const validity = validateGoslingSpec(spec);

    if(validity.state === 'error') {
        console.warn('Gosling spec is invalid!', validity.message);
        return <></>;
    }

    
    return <Card title='Genomic Browser' size="small">
      <GoslingComponent spec={spec as GoslingSpec} /> 
    </Card>
}