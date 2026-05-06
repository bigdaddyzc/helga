import { Layout, Typography } from 'antd'
import AgentChat from './pages/AgentChat'

const { Header, Content } = Layout
const { Title } = Typography

const App = () => {
  return (
    <Layout style={{ minHeight: '100vh', background: '#0a0e27' }}>
      <Header style={{
        background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        borderBottom: '1px solid rgba(0, 212, 255, 0.2)'
      }}>
        <div style={{
          width: 40,
          height: 40,
          borderRadius: '50%',
          background: 'radial-gradient(circle, #00d4ff 0%, #7c3aed 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginRight: 16,
          boxShadow: '0 0 20px rgba(0, 212, 255, 0.5)'
        }}>
          <span style={{ fontSize: 20 }}>⚡</span>
        </div>
        <Title level={3} style={{
          color: '#e2e8f0',
          margin: 0,
          fontFamily: 'Orbitron, sans-serif',
          letterSpacing: 2
        }}>
          HELGA
        </Title>
        <span style={{
          marginLeft: 12,
          color: '#64748b',
          fontSize: 12,
          fontFamily: 'Noto Sans SC, sans-serif'
        }}>
          认知智能体
        </span>
      </Header>
      <Content style={{ padding: '24px', background: '#0a0e27' }}>
        <AgentChat />
      </Content>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

        * {
          scrollbar-width: thin;
          scrollbar-color: #00d4ff #1a1f3a;
        }

        ::-webkit-scrollbar {
          width: 8px;
        }

        ::-webkit-scrollbar-track {
          background: #1a1f3a;
        }

        ::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, #00d4ff, #7c3aed);
          border-radius: 4px;
        }

        body {
          background: #0a0e27;
        }
      `}</style>
    </Layout>
  )
}

export default App