import React from 'react';
import './App.css';
import { Row, Col, Layout, Menu, Upload } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

import { RANGE_MAX, RANGE_MIN, STEP_NUM } from 'Const';
import { generateDistribution, getDimValues, range } from 'helpers';

import z_ranges_sequence from 'assets/z_range_sequence.json';
import z_ranges_matrix from 'assets/z_range_matrix.json';

import LatentDim from 'components/LantentDim/LatentDim';
import SampleBrowser from 'components/SampleBrowser';
import GoslingVis from 'components/Gosling';

import { queryResults } from 'dataService';
import { MenuInfo } from 'rc-menu/lib/interface';
import { TResultRow, TFilter, TDistribution, TMatrixData } from 'types';

const { Header, Sider, Content } = Layout;
const { SubMenu } = Menu;

/***
 * for learned latent dimensions, use the range used to generate simu images
 */
const Z_Ranges: { [k: string]: number[][] } = {
  sequence: z_ranges_sequence,
  matrix: z_ranges_matrix
};

const uploadProps = {
  name: 'file',
  action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
  headers: {
    authorization: 'authorization-text'
  },
  onChange(info: any): void {
    if (info.file.status !== 'uploading') {
      console.info(info.file, info.fileList);
    }
    if (info.file.status === 'done') {
      // message.success(`${info.file.name} file uploaded successfully`);
    } else if (info.file.status === 'error') {
      // message.error(`${info.file.name} file upload failed.`);
    }
  }
};

interface State {
  dataset: string;
  filters: TFilter;
  samples: TResultRow[];
  dimUserNames: { [key: string]: string }; // user can specify new names for latent dim
  isDataLoading: boolean;
  windowInnerSize?: { width: number, height: number };
}
export default class App extends React.Component<{}, State> {
  /****
   * the distribution of all samples on different dims
   * calculated based on samples
   ****/
  matrixData: TMatrixData = {};
  /***
   * whether a sample is shown based on each latent dim filter
   * if the flag from each dim is all true, show this sample
   * size: samples.length x Object.keys(matrixData).length
   * calculated based on samples, filters
   */
  filterMask: { [sampleId: string]: boolean[] } = {};
  constructor(prop: {}) {
    super(prop);
    this.state = {
      dataset: 'matrix',
      dimUserNames: {},
      filters: {},
      samples: [],
      isDataLoading: true,
      windowInnerSize: undefined
    };
    this.setFilters = this.setFilters.bind(this);
    this.updateDims = this.updateDims.bind(this);
    this.setDimUserNames = this.setDimUserNames.bind(this);
  }

  async onQueryResults(dataset: string) {
    const samples = await queryResults(dataset);
    const filters: TFilter = {};
    range(samples[0]['z'].length).forEach(dimNum => {
      filters[`dim_${dimNum}`] = range(STEP_NUM);
    });
    this.matrixData = this.calculateMatrixData(samples, dataset);

    // default show all samples
    samples.forEach(sample => {
      this.filterMask[sample.id] = Object.keys(this.matrixData).map(_ => true);
    });

    this.setState({ filters, samples, isDataLoading: false });
  }
  componentDidMount() {
    this.onQueryResults(this.state.dataset);

    window.addEventListener(
      'resize',
        () => {
            this.setState({ 
              ...this.state, 
              windowInnerSize: {
                width: window.innerWidth,
                height: window.innerHeight
              }
            });
        }
    );
  }
  // @update state
  onClickMenu(e: MenuInfo): void {
    const dataset = e.key;
    this.setState({
      dataset,
      samples: [],
      filters: {},
      dimUserNames: {},
      isDataLoading: true
    });
    this.onQueryResults(dataset);
  }
  /**
   * update the shown dims by changing the keys in state.filters
   * @state_update
   * */
  updateDims(dimNames: string[]): void {
    const { filters } = this.state;
    const currentDimNames = Object.keys(filters);
    const deleteDimNames = currentDimNames.filter(d => !dimNames.includes(d)),
      addDimNames = dimNames.filter(d => !currentDimNames.includes(d));

    deleteDimNames.forEach(n => {
      delete filters[n];
    });
    addDimNames.forEach(n => {
      filters[n] = range(STEP_NUM);
    });

    this.setState({ filters });
  }

  /**
   * update the sample filters by changing the object values in state.filters
   * @state_update
   * */
  setFilters(dimName: string, col: number): void {
    const { filters, samples } = this.state;
    const dimIndex = Object.keys(this.matrixData).indexOf(dimName);

    if (col === -1) {
      // set filters for the whole row
      if (filters[dimName].length > 0) {
        filters[dimName] = [];
        // update filter mask
        // this.filterMask = this.filterMask.map(d => {
        //   d[dimIndex] = false;
        //   return d;
        // });
        Object.values(this.filterMask).forEach(d => {
          d[dimIndex] = false;
        });
      } else {
        filters[dimName] = range(STEP_NUM);
        // update filter mask
        Object.values(this.filterMask).forEach(d => {
          d[dimIndex] = true;
        });
      }
    } else {
      // set filters for single grids
      const idx = filters[dimName].indexOf(col);
      if (idx === -1) {
        filters[dimName].push(col);
        // update filter mask
        this.matrixData[dimName].groupedSamples[col].forEach(sampleId => {
          this.filterMask[sampleId][dimIndex] = true;
        });
      } else {
        filters[dimName].splice(idx, 1);
        // update filter mask
        this.matrixData[dimName].groupedSamples[col].forEach(sampleId => {
          this.filterMask[sampleId][dimIndex] = false;
        });
      }
    }

    this.setState({ filters });
  }
  /**
   * user rename dimensions
   * @state_update
   * */
  setDimUserNames(nameDict: { [key: string]: string }) {
    this.setState({
      dimUserNames: { ...this.state.dimUserNames, ...nameDict }
    });
  }
  // @compute
  calculateMatrixData(samples: TResultRow[], dataset: string): { [dimName: string]: TDistribution } {
    console.info('call matrix calculation');
    var matrixData: { [k: string]: TDistribution } = {},
      row: TDistribution = { histogram: [], labels: [], groupedSamples: [] };
    const sampleIds = samples.map(d => d.id);

    let dimNames: string[] = [];
    if (samples.length > 0) {
      samples[0].z.forEach((_, idx) => {
        dimNames.push(`dim_${idx}`);
      });
      if (dataset == 'matrix') {
        dimNames = dimNames.concat([
          'size',
          'score',
          'ctcf_mean',
          'ctcf_left',
          'ctcf_right',
          'atac_mean',
          'atac_left',
          'atac_right'
        ]);
      }
    }
    dimNames.forEach((dimName, idx) => {
      const dimValues = getDimValues(samples, dimName);
      if (dimName.includes('dim')) {
        // use the same range we used to generate simu images
        const range = Z_Ranges[dataset] ? Z_Ranges[dataset][idx] : [RANGE_MIN, RANGE_MAX];
        row = generateDistribution(dimValues, false, STEP_NUM, sampleIds, 1, range);
      } else if (dimName == 'size') {
        row = generateDistribution(dimValues, false, STEP_NUM, sampleIds, 10);
      } else if (dimName == 'level') {
        row = generateDistribution(dimValues, true, STEP_NUM, sampleIds);
      } else {
        row = generateDistribution(dimValues, false, STEP_NUM, sampleIds);
      }
      matrixData[dimName] = row;
    });
    return matrixData;
  }

  render() {
    const { filters, dataset, isDataLoading, dimUserNames, samples, windowInnerSize } = this.state;

    const filteredSamples = samples.filter(sample => this.filterMask[sample.id].every(d => d));

    const siderWidth = 150,
      headerHeight = 0,
      contentPadding = 10,
      gutter = 16,
      appHeight = (windowInnerSize ? windowInnerSize.height : window.innerHeight) - headerHeight - 2 * contentPadding,
      colWidth = ((windowInnerSize ? windowInnerSize.width : window.innerWidth) - siderWidth - contentPadding * 2) * 0.5 - gutter;

    const sider = (
      <Sider width={siderWidth} collapsible>
        <div className="logo" style={{ height: 32, margin: 16, textAlign: 'center', color: 'white' }}>
          Drava
        </div>
        <Menu
          theme="dark"
          mode="inline"
          defaultOpenKeys={['dataset']}
          selectedKeys={[dataset]}
          onClick={this.onClickMenu.bind(this)}
        >
          <SubMenu key="dataset" title="Dataset">
            <Menu.Item key="sequence">Sequence</Menu.Item>
            <Menu.Item key="matrix">Matrix</Menu.Item>
            <Menu.Item key="celeb">Celeb</Menu.Item>
            <Menu.Item key="upload">
              <Upload {...uploadProps}>
                <UploadOutlined style={{ color: 'rgba(255, 255, 255, 0.65)' }} />
              </Upload>
            </Menu.Item>
          </SubMenu>
        </Menu>
      </Sider>
    );

    return (
      <Layout className="App">
        <Header style={{ height: headerHeight }}>Header</Header>
        <Layout>
          {sider}
          <Content style={{ padding: contentPadding, backgroundColor: 'white' }}>
            <Row gutter={gutter}>
              <Col span={12}>
                {dataset == 'celeb' ? (
                  <></>
                ) : (
                  <GoslingVis
                    dataset={dataset}
                    samples={filteredSamples}
                    width={colWidth}
                    height={appHeight * 0.5}
                    isDataLoading={isDataLoading}
                  />
                )}

                <SampleBrowser
                  dataset={dataset}
                  samples={filteredSamples}
                  height={appHeight * (dataset == 'celeb' ? 1 : 0.5)}
                  isDataLoading={isDataLoading}
                  matrixData={this.matrixData}
                  dimUserNames={dimUserNames}
                  filters={filters}
                />
              </Col>

              <Col span={12}>
                <LatentDim
                  dataset={dataset}
                  samples={samples}
                  filters={filters}
                  matrixData={this.matrixData}
                  height={appHeight}
                  width={colWidth}
                  isDataLoading={isDataLoading}
                  dimUserNames={dimUserNames}
                  setDimUserNames={this.setDimUserNames}
                  updateDims={this.updateDims}
                  setFilters={this.setFilters}
                />
              </Col>
            </Row>
          </Content>
        </Layout>
      </Layout>
    );
  }
}
