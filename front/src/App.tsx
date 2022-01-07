import React from 'react';
import './App.css';
import { Row, Col, Layout, Menu, Upload } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

import { STEP_NUM } from 'Const';
import { getSampleHist, range, withinRange, getRange } from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';
import { GoslingVis } from 'components/Gosling';

import { queryResults } from 'dataService';
import { MenuInfo } from 'rc-menu/lib/interface';
import { TResultRow } from 'types';

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
  filters: number[][];
  samples: TResultRow[];
}
export default class App extends React.Component<{}, State> {
  constructor(prop: {}) {
    super(prop);
    this.state = {
      dataset: 'matrix',
      filters: [],
      samples: []
    };
    this.setFilters = this.setFilters.bind(this);
  }

  async onQueryResults(dataset: string) {
    const samples = await queryResults(dataset);
    const filters = range(samples[0]['z'].length).map(_ => range(STEP_NUM));
    this.setState({ filters, samples });
  }
  componentDidMount() {
    this.onQueryResults(this.state.dataset);
  }

  setFilters(row: number, col: number) {
    const { filters } = this.state;

    if (col === -1) {
      // set filters for the whole row
      if (filters[row].length > 0) {
        filters[row] = [];
      } else {
        filters[row] = range(STEP_NUM);
      }
    } else {
      // set filters for single grids
      const idx = filters[row].indexOf(col);
      if (idx === -1) {
        filters[row].push(col);
      } else {
        filters[row].splice(idx, 1);
      }
    }
    this.setState({ filters });
  }
  // @compute
  filterSamples(samples: TResultRow[], filters: number[][]) {
    const filteredSamples = samples.filter(sample => {
      const inRange = sample.z.every((dimensionValue, row_idx) => {
        const ranges = filters[row_idx].map(i => getRange(i));
        return withinRange(dimensionValue, ranges);
      });
      return inRange;
    });
    return filteredSamples;
  }

  onClickMenu(e: MenuInfo) {
    const dataset = e.key;
    this.setState({
      dataset
    });
    this.onQueryResults(dataset);
  }

  render() {
    const { filters, samples, dataset } = this.state;
    if (filters.length === 0) return null;

    const filteredSamples = this.filterSamples(samples, filters);

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
            <Menu.Item key="sequence">ATAC</Menu.Item>
            <Menu.Item key="matrix">Hi-C</Menu.Item>
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
                  setFilters={this.setFilters}
                  filters={filters}
                  height={appHeight}
                  width={colWidth}
                  samples={samples}
                />
              </Col>
            </Row>
          </Content>
        </Layout>
      </Layout>
    );
  }
}
