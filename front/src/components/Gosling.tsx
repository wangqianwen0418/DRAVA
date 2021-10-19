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
        "width": ${window.innerWidth},
        "height": 180,
        "tracks": [
          {
          "layout": "linear", 
          "data": {
            "url": "https://s3.amazonaws.com/gosling-lang.org/data/HFFc6_Atacseq.mRp.clN.bigWig",
            "type": "bigwig",
            "column": "position",
            "value": "peak",
            "binSize": "8"
          },
            "mark": "area",
            "x": {
              "field": "position",
              "type": "genomic",
              "domain": {"chromosome": "7"},
              "axis": "bottom"
            },
            "y": {"field": "peak", "type": "quantitative"},
            "size": {"value": "2"},
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
              "size": {"value": 10},
              "x": {"field": "chromStart", "type": "genomic"},
              "stroke": {"value": "orange"},
              "strokeWidth": {"value": 1}
          }
        ]
      }`
    return <GoslingComponent spec={JSON.parse(spec)} />
}