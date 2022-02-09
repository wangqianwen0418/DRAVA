import { TFilter, TResultRow } from 'types';
import { message, Modal, Select, Tooltip } from 'antd';
import React, { useState } from 'react';

import getScatter from './Scatter';
import { getCorrelation } from 'helpers/getCorrelation';
import { getDimValues } from 'helpers';

import { interpolateBlues } from 'd3-scale-chromatic';

import styles from './AddCorrelation.module.css';
const { Option } = Select;

type Props = {
  samples: TResultRow[];
  dimNames: string[];
  width: number;
  dimUserNames: { [key: string]: string };
};

export const Correlations = (props: Props) => {
  const [isModalVisible, setModalVisible] = useState(false);

  const { dimNames, dimUserNames, samples, width } = props;

  const scatterWidth = 200,
    scatterHeight = 200;
  const corrTable = (
    <table className={styles.corrTable}>
      {/* header */}
      <tr>
        <th> {` `} </th>
        {dimNames.map(d => (
          <th className={styles.corrCell} key={d}>
            {dimUserNames[d] || d}
          </th>
        ))}
      </tr>
      {/* rows */}
      {dimNames.map((dimNameY, i) => {
        return (
          <tr key={dimNameY}>
            <th className={styles.corrCell}> {dimUserNames[dimNameY] || dimNameY}</th>
            {dimNames.map((dimNameX, j) => {
              const corr = getCorrelation(getDimValues(samples, dimNameX), getDimValues(samples, dimNameY), 2);
              return (
                <Tooltip
                  mouseEnterDelay={0.5}
                  destroyTooltipOnHide
                  key={`${i}_${j}`}
                  title={
                    <svg height={scatterHeight} width={scatterWidth}>
                      {getScatter([dimNameX, dimNameY], scatterHeight, scatterWidth, samples, dimUserNames)}
                    </svg>
                  }
                >
                  <td
                    className={styles.corrCell}
                    key={dimNameY}
                    style={{
                      backgroundColor: interpolateBlues(Math.abs(corr)),
                      color: Math.abs(corr) > 0.5 ? 'white' : 'black'
                    }}
                  >
                    {corr}
                  </td>
                </Tooltip>
              );
            })}
          </tr>
        );
      })}
    </table>
  );

  return (
    <>
      <g className="addCorrelationButton" onClick={() => setModalVisible(!isModalVisible)}>
        <rect height={35} width={35} fill="white" stroke="gray" strokeWidth={2} strokeDasharray={'4 1'} rx={5} />
        <text fontSize={40} x={6} y={28}>
          +
        </text>
      </g>

      <Modal
        title="Add Dimension Correlations"
        visible={isModalVisible}
        onCancel={() => setModalVisible(false)}
        onOk={() => setModalVisible(false)}
        width={'80vw'}
        destroyOnClose
      >
        {corrTable}
        {/* {scatters.map(scatter => `x: ${scatter[0]}, y:${scatter[1]}; `)} */}
      </Modal>
    </>
  );
};
