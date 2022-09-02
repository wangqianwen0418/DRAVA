import createPilingExample from './piling-interface';
import React, { useCallback, useEffect, useState } from 'react';
import { select as d3select } from 'd3-selection';
import styles from './Piling.module.css';

import { TFilter, TResultRow } from 'types';
import { getItemURL } from 'dataService';

type TItem = TResultRow & {
  src: string;
  [key: string]: any;
};
type Props = {
  samples: TResultRow[];
  dataset: string;
  dimUserNames: { [k: string]: string };
  dimNames: string[];
  height: number;
  changeUpadtingStatus: (f: boolean) => void;
};
const Pilling = (props: Props) => {
  const { samples, dimNames, dimUserNames, dataset, changeUpadtingStatus } = props;
  const items = samples.map(s => {
    const url = `${getItemURL(dataset, s.id)}&border=1`;
    // force item id to be undefined, so that piling will use item index as id
    return { ...s, src: url, id: undefined }; // y = 0 in case dimYNum = null
  });
  const [thisPiling, changePiling] = useState<any>('');

  const pilingInitHandler = useCallback(async element => {
    if (element == null) return;

    const dimX = (document.getElementById('xSelector') as any).value;
    const dimY = (document.getElementById('ySelector') as any).value;

    const pilingOptions = {
      items,
      dims: [dimX, dimY],
      getXSvgGroup: () => d3select('svg#ItemBrowser').select(`g`).select(`g`), // pass a function rather than a selection in case the svg components have been rendered yet
      getYSvgGroup: () => d3select('svg#ItemBrowserY').select(`g`).select(`g`),
      dataset
    };

    const { piling, actions } = await createPilingExample(element, pilingOptions);
    changePiling(piling);

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

    const gridGroup = () => {
      const dimX = (document.getElementById('xSelector') as any).value;
      const dimY = (document.getElementById('ySelector') as any).value;
      actions.gridGroup([dimX, dimY]);
    };

    const grid2D = () => {
      const dimX = (document.getElementById('xSelector') as any).value;
      const dimY = (document.getElementById('ySelector') as any).value;
      actions.grid2D([dimX, dimY]);
    };

    const changeSize = () => {
      const size = (document.getElementById('itemSize') as any).value;
      actions.changeSize(size);
    };

    const changeSummary = () => {
      const sType = (document.getElementById('summarySelector') as any).value;
      actions.changeSummary(sType);
    };

    const addLabel = () => {
      const label = (document.getElementById('labelSelector') as any).value;
      actions.addLabel(label);
    };

    const changeGroup = () => {
      const group = (document.getElementById('groupSelector') as any).value;
      const dimX = (document.getElementById('xSelector') as any).value;
      const dimY = (document.getElementById('ySelector') as any).value;
      if (group == 'umap') {
        actions.UMAP();
      } else if (group == 'grid') {
        actions.grid(dimX);
      } else {
        actions.reArrange([dimX, dimY]);
      }
    };

    const postNewGroups = () => {
      const dimX = (document.getElementById('xSelector') as any).value;
      const group = (document.getElementById('groupSelector') as any).value;
      changeUpadtingStatus(true);
      actions.postNewGroups(dataset, dimX, group).then(res => {
        if (res) {
          actions.updateGroups(res?.data);
        }
        changeUpadtingStatus(false);
      });
    };

    document.querySelector('#ySelector')?.addEventListener('change', reArrangeY);
    document.querySelector('#xSelector')?.addEventListener('change', reArrangeX);
    document.querySelector('#summarySelector')?.addEventListener('change', changeSummary);
    document.querySelector('#labelSelector')?.addEventListener('change', addLabel);
    document.querySelector('#groupSelector')?.addEventListener('change', changeGroup);

    document.getElementById('XGroupBtn')?.addEventListener('click', stackX);
    document.getElementById('groupBtn')?.addEventListener('click', gridGroup);
    document.getElementById('gridBtn')?.addEventListener('click', grid2D);
    document.getElementById('splitBtn')?.addEventListener('click', splitAll);
    document.getElementById('itemSize')?.addEventListener('change', changeSize);
    document.getElementById('updateConcept')?.addEventListener('click', postNewGroups);

    return () => {
      piling.destroy();
      document.querySelector('#ySelector')?.removeEventListener('change', reArrangeY);
      document.querySelector('#xSelector')?.removeEventListener('change', reArrangeX);
      document.querySelector('#summarySelector')?.removeEventListener('change', changeSummary);
      document.querySelector('#labelSelector')?.removeEventListener('change', addLabel);
      document.querySelector('#groupSelector')?.removeEventListener('change', changeGroup);

      document.getElementById('XGroupBtn')?.removeEventListener('click', stackX);
      document.getElementById('groupBtn')?.removeEventListener('click', gridGroup);
      document.getElementById('gridBtn')?.removeEventListener('click', grid2D);
      document.getElementById('splitBtn')?.removeEventListener('click', splitAll);
      document.getElementById('itemSize')?.removeEventListener('change', changeSize);
      document.getElementById('updateConcept')?.removeEventListener('click', postNewGroups);
    };
  }, []);

  useEffect(() => {
    if (thisPiling && props.samples.length > 0) {
      thisPiling.set({
        items: props.samples.map(s => {
          const url = `${getItemURL(dataset, s.id)}&border=1`;
          return { ...s, src: url, id: undefined }; // y = 0 in case dimYNum = null
        })
      });
    }
  }, [props.samples]);

  return (
    <div className={styles.piling_container}>
      <div className={styles.piling_wrapper} ref={pilingInitHandler} style={{ height: props.height }} />
    </div>
  );
};

export default Pilling;
