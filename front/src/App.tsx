import React from 'react';
import './App.css';
import { Row, Col, Layout, Menu, Upload } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

import { RANGE_MAX, RANGE_MIN, STEP_NUM } from 'Const';
import { generateDistribution, getDimValues, range } from 'helpers';

import z_ranges_sequence from 'assets/z_range_sequence.json';
import z_ranges_matrix from 'assets/z_range_matrix.json';

import LatentDim from 'components/LatentDim/LatentDim';
import ItemBrowser from 'components/ItemBrowser';
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

const non_genomic_dataset = ['celeb', 'IDC', 'dsprites'];

interface State {
  dataset: string;
  filters: TFilter;
  samples: TResultRow[];
  dimUserNames: { [key: string]: string }; // user can specify new names for latent dim
  isDataLoading: boolean;
  windowInnerSize?: { width: number; height: number };
}
export default class App extends React.Component<{}, State> {
  /****
   * the distribution of all samples on different dims
   * calculated based on samples
   * only update when query new dataset
   ****/
  matrixData: TMatrixData = {};
  filteredSamples: TResultRow[] = [];
  constructor(prop: {}) {
    super(prop);
    this.state = {
      dataset: 'IDC',
      dimUserNames: {},
      filters: {},
      samples: [],
      isDataLoading: true,
      windowInnerSize: undefined
    };
    this.setFilters = this.setFilters.bind(this);
    this.updateDims = this.updateDims.bind(this);
    this.setDimUserNames = this.setDimUserNames.bind(this);
    this.resize = this.resize.bind(this);
  }

  async onQueryResults(dataset: string) {
    const samples = await queryResults(dataset);
    const filters: TFilter = {};
    range(samples[0]['z'].length).forEach(dimNum => {
      filters[`dim_${dimNum}`] = range(STEP_NUM).map(_ => true);
    });
    const [matrixData, samplesWithAssign] = this.calculateMatrixData(samples, dataset);
    this.matrixData = matrixData;
    this.filteredSamples = samples;
    this.setState({ filters, samples: samplesWithAssign, isDataLoading: false });
  }
  resize() {
    this.setState({
      windowInnerSize: {
        width: window.innerWidth,
        height: window.innerHeight
      }
    });
  }
  componentDidMount() {
    this.onQueryResults(this.state.dataset);

    window.addEventListener('resize', this.resize);
  }
  componentWillUnmount() {
    window.removeEventListener('resize', this.resize);
  }
  // @update state
  onChangeDataset(e: MenuInfo): void {
    const dataset = e.key;
    if (dataset == 'upload') return;
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
    const { filters, samples } = this.state;
    const currentDimNames = Object.keys(filters);
    const deleteDimNames = currentDimNames.filter(d => !dimNames.includes(d)),
      addDimNames = dimNames.filter(d => !currentDimNames.includes(d));

    deleteDimNames.forEach(n => {
      delete filters[n];
    });
    addDimNames.forEach(n => {
      filters[n] = range(this.matrixData[n].histogram.length).map(d => true);
    });
    if (deleteDimNames.length > 0) {
      this.filteredSamples = this.getFilteredSamples(samples, filters);
    }

    this.setState({ filters });
  }

  /**
   * update the sample filters by changing the object values in state.filters
   * @state_update
   * */
  setFilters(dimName: string, col: number): void {
    const { filters, samples } = this.state;
    if (col === -1) {
      // set filters for the whole row
      if (filters[dimName].some(d => d)) {
        filters[dimName] = filters[dimName].map(d => false);
      } else {
        filters[dimName] = filters[dimName].map(d => true);
      }
    } else {
      filters[dimName][col] = !filters[dimName][col];
    }

    this.filteredSamples = this.getFilteredSamples(samples, filters);
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
  calculateMatrixData(samples: TResultRow[], dataset: string): [{ [dimName: string]: TDistribution }, TResultRow[]] {
    console.info('call matrix calculation');
    var matrixData: { [k: string]: TDistribution } = {},
      row: TDistribution = { histogram: [], labels: [], groupedSamples: [] },
      sampleAssignments: number[] = [],
      distributionResults = { row, sampleAssignments };
    const sampleIds = samples.map(d => d.id);

    let dimNames: string[] = [];
    if (samples.length > 0) {
      samples[0].z.forEach((_, idx) => {
        dimNames.push(`dim_${idx}`);
      });
      dimNames.push('recons_loss');
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
      } else if (dataset == 'IDC') {
        dimNames.push('label');
        dimNames.push('confidence');
        dimNames.push('prediction');
      } else if (dataset == 'sequence') {
        dimNames.push('peak_score');
      }
    }
    dimNames.forEach((dimName, idx) => {
      const dimValues = getDimValues(samples, dimName);
      if (dimName.includes('dim')) {
        // use the same range we used to generate simu images
        const range = Z_Ranges[dataset] ? Z_Ranges[dataset][idx] : [RANGE_MIN, RANGE_MAX];
        distributionResults = generateDistribution(dimValues, false, STEP_NUM, sampleIds, 1, range);
      } else if (dimName == 'size') {
        distributionResults = generateDistribution(dimValues, false, STEP_NUM, sampleIds, 10);
      } else if (dimName == 'level') {
        distributionResults = generateDistribution(dimValues, true, STEP_NUM, sampleIds);
      } else if (dimName == 'label' || dimName == 'prediction') {
        distributionResults = generateDistribution(dimValues, true, 2, sampleIds);
      } else {
        distributionResults = generateDistribution(dimValues, false, STEP_NUM, sampleIds);
      }

      matrixData[dimName] = distributionResults['row'];
      samples.forEach((sample, idx) => {
        sample.assignments[dimName] = distributionResults['sampleAssignments'][idx];
      });
    });
    return [matrixData, samples];
  }
  /**
   *
   * @compute
   */
  getFilteredSamples(samples: TResultRow[], filters: TFilter) {
    const filteredSamples = samples.filter(sample =>
      Object.keys(sample.assignments).every(dimName => {
        const col = sample.assignments[dimName];
        return filters[dimName] ? filters[dimName][col] : true;
      })
    );
    return filteredSamples;
  }

  render() {
    const { filters, dataset, isDataLoading, dimUserNames, samples, windowInnerSize } = this.state;

    const siderWidth = 150,
      headerHeight = 0,
      contentPadding = 10,
      gutter = 16,
      appHeight = (windowInnerSize ? windowInnerSize.height : window.innerHeight) - headerHeight - 2 * contentPadding,
      leftCol = 10,
      rightCol = 14,
      leftColWidth =
        ((windowInnerSize ? windowInnerSize.width : window.innerWidth) - siderWidth - contentPadding * 2) *
          (leftCol / 24) -
        gutter,
      rightColWidth =
        ((windowInnerSize ? windowInnerSize.width : window.innerWidth) - siderWidth - contentPadding * 2) *
          (rightCol / 24) -
        gutter;

    const sider = (
      <Sider width={siderWidth} collapsible>
        <div
          className="logo"
          style={{ height: 32, margin: 16, textAlign: 'left', color: 'white', fontSize: 30, fontWeight: 800 }}
        >
          DRAVA
        </div>
        <Menu
          theme="dark"
          mode="inline"
          defaultOpenKeys={['dataset']}
          selectedKeys={[dataset]}
          onClick={this.onChangeDataset.bind(this)}
        >
          <SubMenu key="dataset" title="Dataset">
            <Menu.Item key="sequence">Sequence</Menu.Item>
            <Menu.Item key="matrix">Matrix</Menu.Item>
            <Menu.Item key="celeb">Celeb</Menu.Item>
            <Menu.Item key="IDC">IDC</Menu.Item>
            <Menu.Item key="dsprites">shapes</Menu.Item>
            <Menu.Item key="upload">
              <Upload>
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
              <Col span={leftCol}>
                <LatentDim
                  dataset={dataset}
                  samples={samples}
                  filters={filters}
                  matrixData={this.matrixData}
                  height={appHeight * (non_genomic_dataset.includes(dataset) ? 1 : 0.6)}
                  width={leftColWidth}
                  isDataLoading={isDataLoading}
                  dimUserNames={dimUserNames}
                  setDimUserNames={this.setDimUserNames}
                  updateDims={this.updateDims}
                  setFilters={this.setFilters}
                />
                {non_genomic_dataset.includes(dataset) ? (
                  <></>
                ) : (
                  <GoslingVis
                    dataset={dataset}
                    samples={this.filteredSamples}
                    width={leftColWidth}
                    height={appHeight * 0.4}
                    isDataLoading={isDataLoading}
                  />
                )}
              </Col>
              <Col span={rightCol}>
                <ItemBrowser
                  dataset={dataset}
                  samples={this.filteredSamples}
                  height={appHeight}
                  width={rightColWidth}
                  isDataLoading={isDataLoading}
                  matrixData={this.matrixData}
                  dimUserNames={dimUserNames}
                  filters={filters}
                  dimNames={Object.keys(filters)}
                  setDimUserNames={this.setDimUserNames}
                />
              </Col>
            </Row>
          </Content>
        </Layout>
      </Layout>
    );
  }
}
