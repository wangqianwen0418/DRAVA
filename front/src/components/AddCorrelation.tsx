import { TFilter, TResultRow } from 'types';
import { message, Modal, Select } from 'antd';
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
  let dimX = 'none',
    dimY = 'none';

  const [isModalVisible, setModalVisible] = useState(false);
  const [scatters, changeScatters] = useState<[nameOfDimX: string, nameOfDimY: string][]>([]);

  const { dimNames, dimUserNames, samples, width } = props;

  const selectX = (
    <Select
      placeholder="Add X dimension"
      style={{ width: '150px', height: '30px' }}
      onChange={(dimName: string) => {
        dimX = dimName;
      }}
    >
      {dimNames.map(dimName => (
        <Option key={dimName} value={dimName}>
          {dimUserNames[dimName] || dimName}
        </Option>
      ))}
    </Select>
  );

  const selectY = (
    <Select
      placeholder="Add X dimension"
      style={{ width: '150px', height: '30px' }}
      onChange={(dimName: string) => {
        dimY = dimName;
      }}
    >
      {dimNames.map(dimName => (
        <Option key={dimName} value={dimName}>
          {dimUserNames[dimName] || dimName}
        </Option>
      ))}
    </Select>
  );

  const addScatter = (dimX: string, dimY: string) => {
    scatters.push([dimX, dimY]);
    changeScatters(scatters);
  };

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
      {dimNames.map((dimNameX, i) => {
        return (
          <tr key={dimNameX}>
            <th className={styles.corrCell}> {dimUserNames[dimNameX] || dimNameX}</th>
            {dimNames.map((dimNameY, j) => {
              const corr = getCorrelation(getDimValues(samples, dimNameX), getDimValues(samples, dimNameY), 2);
              return (
                <td
                  className={styles.corrCell}
                  key={dimNameY}
                  style={{ backgroundColor: interpolateBlues(corr), color: corr > 0.5 ? 'white' : 'black' }}
                  onClick={() => addScatter(dimNameX, dimNameY)}
                >
                  {corr}
                </td>
              );
            })}
          </tr>
        );
      })}
    </table>
  );

  const scatterWidth = 200,
    scatterHeight = 200;
  return (
    <>
      {scatters.map((scatter, idx) => (
        <g key={idx} transform={`translate(${scatterWidth * idx}, ${Math.floor(width / (scatterWidth * idx))})`}>
          {getScatter(scatter, scatterHeight, scatterWidth, samples, dimUserNames)}
        </g>
      ))}

      <g
        className="addCorrelationButton"
        onClick={() => setModalVisible(!isModalVisible)}
        transform={`translate(${scatterWidth * scatters.length}, ${Math.floor(
          width / (scatterWidth * scatters.length)
        )})`}
      >
        <rect height={35} width={35} fill="white" stroke="gray" strokeWidth={2} strokeDasharray={'4 1'} rx={5} />
        <text fontSize={40} x={6} y={28}>
          +
        </text>
      </g>

      <Modal
        title="Add Dimension Correlations"
        visible={isModalVisible}
        okText="add"
        onCancel={() => setModalVisible(false)}
        width={'80vw'}
        onOk={() => {
          if (dimY != 'none' && dimX != 'none' && dimX != dimY) {
            scatters.push([dimX, dimY]);
            setModalVisible(false);
            changeScatters(scatters);
          } else {
            message.error(`unsuported input: dim x = ${dimX}, dim y = ${dimY}`);
          }
        }}
      >
        {corrTable}
        {scatters.map(scatter => `x: ${scatter[0]}, y:${scatter[1]}; `)}
      </Modal>
    </>
  );
};
