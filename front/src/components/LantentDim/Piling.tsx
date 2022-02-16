import createPilingExample from './piling-interface';
import React, { useCallback, useEffect, useState } from 'react';
import styles from './Piling.module.css';
import { Button, Select } from 'antd';
import { BASE_URL } from 'Const';

import { TResultRow } from 'types';

const { Option } = Select;

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
    dims: [dimX, 'none']
  };

  const pilingInitHandler = useCallback(async element => {
    if (element !== null) {
      const [piling, actions] = await createPilingExample(element, pilingOptions);
      // register action
      document.querySelector('.ySelector')?.addEventListener('change', event => {
        const dimY = (event.target as any).value;
        return actions.reArrange([dimX, dimY]);
      });
      document.getElementById('groupBtn')?.addEventListener('click', () => actions.group(dimX));
    }
    return;
  }, []);

  return (
    <div className={styles.piling_container}>
      <label>Oragnize samples vertically using</label>
      <select className="ySelector" style={{ width: '100px' }}>
        {dimNames.map(dimName => {
          return (
            <option key={dimName} value={dimName}>
              {dimUserNames[dimName] || dimName}
            </option>
          );
        })}
      </select>
      <Button type="default" id="groupBtn">
        Auto Group
      </Button>
      <div className={styles.piling_wrapper} ref={pilingInitHandler} />
    </div>
  );
};

export default Pilling;
