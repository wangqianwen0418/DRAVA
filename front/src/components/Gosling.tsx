import { validateGoslingSpec, GoslingComponent } from "gosling.js";
import * as React from "react";

import sampleLabels from 'assets/sample_labels.json'


interface Props {
    sampleIdxs: number[]
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

    const spec = `{
        "title": "",
        "alignment": "overlay",
        "width": ${window.innerWidth * 0.8},
        "height": 180,
        "tracks": [
          {
          "layout": "linear",
          "data": {
            "url": "https://s3.amazonaws.com/gosling-lang.org/data/HFFc6_Atacseq.mRp.clN.bigWig",
            "type": "bigwig",
            "column": "position",
            "value": "peak",
            "binSize": "2"
          },
            "mark": "area",
            "x": {
              "field": "position",
              "type": "genomic",
              "domain": {"chromosome": "7"},
              "axis": "bottom"
            },
            "y": {"field": "peak", "type": "quantitative"},
            "color": {"value": "black"}
          },
          {
              "data": {
                  "values": ${JSON.stringify(labelJSON)},
                  "type":"json",
                    "chromosomeField":"Chromosome",
                    "genomicFields":[
                        "chromStart",
                        "chromEnd"
                    ]
              },
              "mark": "rect",
              "size": {"value": 12},
              "x": {"field": "chromStart", "type": "genomic"},
              "xe": {"field": "chromEnd", "type": "genomic"},
              "stroke": {"value": "orange"},
              "strokeWidth": {"value": 1}
          }
        ]
      }`

    // validate the spec
    const validity = validateGoslingSpec(JSON.parse(spec));

    if(validity.state === 'error') {
        console.warn('Gosling spec is invalid!', validity.message);
        return <></>;
    }

    
    return <GoslingComponent spec={JSON.parse(spec)} />
}