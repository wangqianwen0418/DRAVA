import React from 'react';
import './App.css';
import { Row, Col, Layout, Menu, Upload } from 'antd';
import {UploadOutlined} from '@ant-design/icons';

import { stepNum } from 'Const';
import { range } from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';
import { GoslingVis } from 'components/Gosling';

import { requestHist } from 'dataService';
import {MenuInfo} from 'rc-menu/lib/interface';

const { Header, Sider, Content } = Layout;
const { SubMenu } = Menu;

const uploadProps = {
  name: 'file',
  action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
  headers: {
    authorization: 'authorization-text',
  },
  onChange(info:any) {
    if (info.file.status !== 'uploading') {
      console.log(info.file, info.fileList);
    }
    if (info.file.status === 'done') {
      // message.success(`${info.file.name} file uploaded successfully`);
    } else if (info.file.status === 'error') {
      // message.error(`${info.file.name} file upload failed.`);
    }
  },
};

interface State {
  dataset: string,
  filters: number[][],
  hist: number[][]
}
export default class App extends React.Component<{}, State> {
  constructor(prop: {}) {
    super(prop)
    this.state = {
      dataset: 'sequence',
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

  onClickMenu(e: MenuInfo){
    this.setState({
      dataset: e.key
    })
  }

  render() {
    const { filters, hist } = this.state
    if (filters.length === 0) return null

    // const sampleIdxs = this.filterSamples()
    const sampleIdxs = range(400)
    const siderWidth = 150, headerHeight = 0, contentPadding = 10,
      gutter = 16, appHeight = window.innerHeight - headerHeight - 2 * contentPadding,
      colWidth = (window.innerWidth - siderWidth - contentPadding * 2) * 0.5 - gutter

    const sider = <Sider width={siderWidth} collapsible >
      <div className="logo" style={{height: 32, margin: 16, textAlign: 'center', color:'white'}}> 
        LOGO
      </div>
      <Menu theme="dark" mode="inline" defaultOpenKeys={['dataset']} onClick={this.onClickMenu.bind(this)}>
        <SubMenu key="dataset" title="Dataset">
          <Menu.Item key="sequence">ATAC</Menu.Item>
          <Menu.Item key="matrix">Hi-C</Menu.Item>
          <Menu.Item key="upload"> 
            <Upload {...uploadProps} >
              <UploadOutlined style={{color: 'rgba(255, 255, 255, 0.65)'}}/>
            </Upload>
          </Menu.Item>
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
