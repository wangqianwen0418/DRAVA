import createPilingExample from './piling-interface';
import React, { useCallback, useEffect, useState } from 'react';
import { select as d3select } from 'd3-selection';
import styles from './Piling.module.css';
import { Button } from 'antd';
import { BASE_URL } from 'Const';

import { TResultRow } from 'types';

type Item = TResultRow & {
  src: string;
  [key: string]: any;
};
type Props = {
  samples: TResultRow[];
  dataset: string;
  dimX: string;
  dimUserNames: { [k: string]: string };
  dimNames: string[];
};
const Pilling = (props: Props) => {
  const { samples, dimNames, dimUserNames, dataset, dimX } = props;
  const items = samples.map(s => {
    const url = `${BASE_URL}/api/get_${dataset}_sample?id=${s.id}`;
    return { ...s, src: url }; // y = 0 in case dimYNum = null
  });
  const pileDragEnd = (e: any) => console.info('end of piling drag, ', e.target.items);

  const pilingOptions = {
    items,
    pileDragEnd,
    dims: [dimX, 'none'],
    getSvgGroup: () => d3select('svg#configDim').select(`g#${dimX}`), // pass a function rather than a selection in case the svg components have been rendered yet
    dataset
  };

  const pilingInitHandler = useCallback(async element => {
    if (element == null) return;

    const [piling, actions] = await createPilingExample(element, pilingOptions);
    // register action
    const reArrange = (event: Event) => {
      const dimY = (event.target as any).value;
      return actions.reArrange([dimX, dimY]);
    };
    const autoGroup = () => actions.group(dimX);
    document.querySelector('.ySelector')?.addEventListener('change', reArrange);
    document.getElementById('groupBtn')?.addEventListener('click', autoGroup);

    return () => {
      piling.destory();
      document.querySelector('.ySelector')?.removeEventListener('change', reArrange);
      document.getElementById('groupBtn')?.removeEventListener('click', autoGroup);
    };
  }, []);

  return (
    <div className={styles.piling_container}>
      <label>Oragnize samples vertically using</label>
      <select className="ySelector" style={{ width: '100px' }}>
        <option value="none">none</option>
        {dimNames.map(dimName => {
          return (
            <option key={dimName} value={dimName}>
              {dimUserNames[dimName] || dimName}
            </option>
          );
        })}
      </select>
      <Button type="default" id="groupBtn" size="small">
        Auto-Group
      </Button>
      <div className={styles.piling_wrapper} ref={pilingInitHandler} />
    </div>
  );
};

export default Pilling;
