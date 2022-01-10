import React from 'react';
import './App.css';
import { Row, Col, Layout, Menu, Upload } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

import { STEP_NUM } from 'Const';
import { generateDistribution, range, withinRange, getRange } from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';
import { GoslingVis } from 'components/Gosling';

import { queryResults } from 'dataService';
import { MenuInfo } from 'rc-menu/lib/interface';
import { TResultRow, TFilter, TDistribution } from 'types';

const { Header, Sider, Content } = Layout;
const { SubMenu } = Menu;

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
}
export default class App extends React.Component<{}, State> {
  constructor(prop: {}) {
    super(prop);
    this.state = {
      dataset: 'matrix',
      filters: {},
      samples: []
    };
    this.setFilters = this.setFilters.bind(this);
    this.updateDims = this.updateDims.bind(this);
  }

  async onQueryResults(dataset: string) {
    const samples = await queryResults(dataset);
    const filters: TFilter = {};
    range(samples[0]['z'].length).forEach(dimNum => {
      filters[`dim_${dimNum}`] = range(STEP_NUM);
    });
    this.setState({ filters, samples });
  }
  componentDidMount() {
    this.onQueryResults(this.state.dataset);
  }
  // @update state
  onClickMenu(e: MenuInfo): void {
    const dataset = e.key;
    this.setState({
      dataset
    });
    this.onQueryResults(dataset);
  }
  // @state update
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

  // @state update
  setFilters(dimName: string, col: number): void {
    const { filters } = this.state;

    if (col === -1) {
      // set filters for the whole row
      if (filters[dimName].length > 0) {
        filters[dimName] = [];
      } else {
        filters[dimName] = range(STEP_NUM);
      }
    } else {
      // set filters for single grids
      const idx = filters[dimName].indexOf(col);
      if (idx === -1) {
        filters[dimName].push(col);
      } else {
        filters[dimName].splice(idx, 1);
      }
    }
    this.setState({ filters });
  }
  // @compute
  matrixData(): { [dimName: string]: TDistribution } {
    const { samples, filters } = this.state;

    var matrixData: { [k: string]: TDistribution } = {},
      row: TDistribution = { histogram: [], labels: [], groupedSamples: [] };
    Object.keys(filters).forEach((dimName, idx) => {
      if (dimName.includes('dim')) {
        const dimNum = parseInt(dimName.split('_')[1]);
        row = generateDistribution(
          samples.map(sample => sample['z'][dimNum]),
          false,
          STEP_NUM
        );
      } else if (dimName == 'size') {
        row = generateDistribution(
          samples.map(s => s.end - s.start),
          false,
          STEP_NUM
        );
      } else if (dimName == 'level') {
        row = generateDistribution(
          samples.map(s => s['level']),
          true
        );
      } else {
        row = generateDistribution(
          samples.map(s => s[dimName]),
          false,
          STEP_NUM
        );
      }
      matrixData[dimName] = row;
    });
    return matrixData;
  }
  // @compute
  filteredSamples(): TResultRow[] {
    const { samples, filters } = this.state;
    const userDims = Object.keys(filters).filter(d => !d.includes('dim_'));
    const matrixData = this.matrixData();

    const filteredSamples = samples.filter(sample => {
      // check latent dim z
      const inLatentSpace = sample.z.every((dimValue, dimIdx) => {
        const dimName = `dim_${dimIdx}`;
        if (!filters[dimName]) {
          return true;
        } else {
          return filters[dimName].some(groupIdx => matrixData[dimName]['groupedSamples'][groupIdx].includes(sample.id));
        }
      });
      //  check other user-defined dims
      const inUserDims = userDims.every(dimName => {
        return filters[dimName].some(groupIdx => matrixData[dimName]['groupedSamples'][groupIdx].includes(sample.id));
      });
      return inLatentSpace && inUserDims;
    });

    // const filteredSamples = samples.filter(sample => {
    //   const inRange = sample.z.every((dimensionValue, row_idx) => {
    //     const ranges = filters[`dim_${row_idx}`].map(i => getRange(i));
    //     return withinRange(dimensionValue, ranges);
    //   });
    //   return inRange;
    // });
    return filteredSamples;
  }

  render() {
    const { filters, samples, dataset } = this.state;
    if (samples.length == 0) return null;

    const filteredSamples = this.filteredSamples();

    const siderWidth = 150,
      headerHeight = 0,
      contentPadding = 10,
      gutter = 16,
      appHeight = window.innerHeight - headerHeight - 2 * contentPadding,
      colWidth = (window.innerWidth - siderWidth - contentPadding * 2) * 0.5 - gutter;

    const sider = (
      <Sider width={siderWidth} collapsible>
        <div className="logo" style={{ height: 32, margin: 16, textAlign: 'center', color: 'white' }}>
          LOGO
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
                <GoslingVis dataset={dataset} samples={filteredSamples} width={colWidth} height={appHeight * 0.5} />
                <SampleBrowser samples={filteredSamples} height={appHeight * 0.5} />
              </Col>

              <Col span={12}>
                <Grid
                  dataset={dataset}
                  samples={samples}
                  filters={filters}
                  matrixData={this.matrixData()}
                  height={appHeight}
                  width={colWidth}
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
