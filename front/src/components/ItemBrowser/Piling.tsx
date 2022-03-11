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
  dimUserNames: { [k: string]: string };
  dimNames: string[];
};
const Pilling = (props: Props) => {
  const { samples, dimNames, dimUserNames, dataset } = props;
  const items = samples.map(s => {
    const url = `${BASE_URL}/api/get_${dataset}_sample?id=${s.id}`;
    return { ...s, src: url }; // y = 0 in case dimYNum = null
  });
  const pileDragEnd = (e: any) => console.info('end of piling drag, ', e.target.items);

  const pilingInitHandler = useCallback(async element => {
    if (element == null) return;

    const dimX = (document.getElementById('xSelector') as any).value;
    const dimY = (document.getElementById('ySelector') as any).value;

    const pilingOptions = {
      items,
      pileDragEnd,
      dims: [dimX, dimY],
      getSvgGroup: () => d3select('svg#ItemBrowser').select(`g`), // pass a function rather than a selection in case the svg components have been rendered yet
      dataset
    };

    const [piling, actions] = await createPilingExample(element, pilingOptions);

    // register action
    const reArrangeY = (event: any) => {
      const dimY = event.target.value;
      const dimX = (document.getElementById('xSelector') as any).value;
      return actions.reArrange([dimX, dimY]);
    };

    const reArrangeX = (event: any) => {
      const dimX = event.target.value;
      const dimY = (document.getElementById('ySelector') as any).value;
      return actions.reArrange([dimX, dimY]);
    };
    const stackX = () => {
      const dimX = (document.getElementById('xSelector') as any).value;
      actions.stackX(dimX);
    };

    const splitAll = () => {
      const dimX = (document.getElementById('xSelector') as any).value;
      const dimY = (document.getElementById('ySelector') as any).value;
      actions.splitAll([dimX, dimY]);
    };

    document.querySelector('#ySelector')?.addEventListener('change', reArrangeY);
    document.querySelector('#xSelector')?.addEventListener('change', reArrangeX);
    document.getElementById('stackXBtn')?.addEventListener('click', stackX);
    document.getElementById('splitBtn')?.addEventListener('click', splitAll);
    document.getElementById('umapBtn')?.addEventListener('click', actions.UMAP);
    document.getElementById('1dBtn')?.addEventListener('click', actions.grid);

    return () => {
      piling.destory();
      document.querySelector('#ySelector')?.removeEventListener('change', reArrangeY);
      document.querySelector('#xSelector')?.removeEventListener('change', reArrangeX);
      document.getElementById('stackXBtn')?.removeEventListener('click', stackX);
      document.getElementById('splitBtn')?.removeEventListener('click', splitAll);
      document.getElementById('umapBtn')?.removeEventListener('click', actions.UMAP);
      document.getElementById('1dBtn')?.addEventListener('click', actions.grid);
    };
  }, []);

  return (
    <div className={styles.piling_container}>
      <div className={styles.piling_wrapper} ref={pilingInitHandler} />
    </div>
  );
};

export default Pilling;