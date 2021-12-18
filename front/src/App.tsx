import React from 'react';
import './App.css';
import { Row, Col, Layout, Menu } from 'antd';
import {UploadOutlined} from '@ant-design/icons';

import { stepNum } from 'Const';
import { range } from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';
import { GoslingVis } from 'components/Gosling';

import { requestHist } from 'dataService';

const { Header, Sider, Content } = Layout;
const { SubMenu } = Menu;

interface State {
  filters: number[][],
  hist: number[][]
}
export default class App extends React.Component<{}, State> {
  constructor(prop: {}) {
    super(prop)
    this.state = {
      filters: [],
      hist: []
    }
    this.setFilters = this.setFilters.bind(this)
  }

  async onRequestHist() {
    const hist = await requestHist()
    const filters = range(hist.length).map(_ => range(stepNum))
    this.setState({ filters, hist })
  }
  componentDidMount() {
    this.onRequestHist()
  }

  setFilters(row: number, col: number) {
    let { filters } = this.state

    if (col === -1) {
      // set filters for the whole row
      if (filters[row].length > 0) {
        filters[row] = []
      } else {
        filters[row] = range(stepNum)
      }
    } else {
      // set filters for single grids
      const idx = filters[row].indexOf(col)
      if (idx === -1) {
        filters[row].push(col)
      } else {
        filters[row].splice(idx, 1)
      }
    }
    this.setState({ filters })
  }

  render() {
    const { filters, hist } = this.state
    if (filters.length === 0) return null

    // const sampleIdxs = this.filterSamples()
    const sampleIdxs = range(400)
    const siderWidth = 150, headerHeight = 0, contentPadding = 10,
      gutter = 16, appHeight = window.innerHeight - headerHeight - 2 * contentPadding,
      colWidth = (window.innerWidth - siderWidth - contentPadding * 2) * 0.5 - gutter

    const sider = <Sider width={siderWidth} collapsible trigger={null} >
      <div className="logo" style={{height: 32, margin: 16, textAlign: 'center', color:'white'}}> 
        LOGO
      </div>
      <Menu theme="dark" mode="inline" defaultOpenKeys={['dataset']}>
        <SubMenu key="dataset" title="Dataset">
          <Menu.Item key="sequence">ATAC</Menu.Item>
          <Menu.Item key="matrix">Hi-C</Menu.Item>
          <Menu.Item key="upload" icon={<UploadOutlined />}> Upload </Menu.Item>
        </SubMenu>
      </Menu>
    </Sider>

    return (
      <Layout className="App" >
        <Header style={{ height: headerHeight }}>Header</Header>
        <Layout>
          {sider}
          <Content style={{ padding: contentPadding, backgroundColor: 'white' }}>
            <Row gutter={gutter}>

              <Col span={12}>
                <GoslingVis sampleIdxs={sampleIdxs} width={colWidth} height={appHeight * 0.5} />
                <SampleBrowser sampleIdxs={sampleIdxs} height={appHeight * 0.5} />
              </Col>

              <Col span={12}>
                <Grid setFilters={this.setFilters} filters={filters} height={appHeight} width={colWidth} hist={hist} />
              </Col>
            </Row>
          </Content>
        </Layout>

      </Layout>
    );
  }
}
