import { Card, Table, Tag, Progress, Collapse } from 'antd'
import { ValidationResults as ValidationResultsType } from '../api/helga'

interface Props {
  results: ValidationResultsType
}

export default function ValidationResults({ results }: Props) {
  const data = Object.entries(results.results).map(([key, value]) => ({
    name: key.replace('_', ' ').toUpperCase(),
    key,
    ...value
  }))

  const columns = [
    {
      title: '验证层',
      dataIndex: 'name',
      key: 'name',
      render: (text: string) => (
        <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{text}</span>
      )
    },
    {
      title: '状态',
      dataIndex: 'passed',
      key: 'passed',
      render: (passed: boolean) => (
        <Tag color={passed ? 'success' : 'error'} style={{
          background: passed ? 'rgba(82, 196, 26, 0.1)' : 'rgba(255, 78, 108, 0.1)',
          border: passed ? '1px solid rgba(82, 196, 26, 0.3)' : '1px solid rgba(255, 78, 108, 0.3)',
          color: passed ? '#52c41a' : '#ff4e6c'
        }}>
          {passed ? 'PASS' : 'FAIL'}
        </Tag>
      )
    },
    {
      title: '指标值',
      dataIndex: 'metric_value',
      key: 'metric_value',
      render: (val: number) => (
        <span style={{ color: '#00d4ff', fontFamily: 'Orbitron, monospace' }}>{val.toFixed(3)}</span>
      )
    },
    {
      title: '阈值',
      dataIndex: 'threshold',
      key: 'threshold',
      render: (val: number) => (
        <span style={{ color: '#64748b' }}>{val.toFixed(3)}</span>
      )
    },
    {
      title: '达成率',
      dataIndex: 'metric_value',
      key: 'progress',
      render: (val: number, record: any) => {
        const percent = Math.min(100, (val / record.threshold) * 100)
        return (
          <div style={{ width: 100 }}>
            <Progress
              percent={percent}
              size="small"
              showInfo={false}
              strokeColor={percent >= 100 ? '#52c41a' : percent >= 70 ? '#00d4ff' : '#ff4e6c'}
              trailColor="rgba(255,255,255,0.1)"
            />
          </div>
        )
      }
    }
  ]

  const collapseItems = data.map(item => ({
    key: item.key,
    label: (
      <span style={{ display: 'flex', alignItems: 'center' }}>
        <Tag
          color={item.passed ? 'success' : 'error'}
          style={{
            marginRight: 12,
            background: item.passed ? 'rgba(82, 196, 26, 0.1)' : 'rgba(255, 78, 108, 0.1)',
            border: item.passed ? '1px solid rgba(82, 196, 26, 0.3)' : '1px solid rgba(255, 78, 108, 0.3)',
            color: item.passed ? '#52c41a' : '#ff4e6c'
          }}
        >
          {item.passed ? 'PASS' : 'FAIL'}
        </Tag>
        <span style={{ color: '#e2e8f0' }}>{item.name}</span>
      </span>
    ),
    children: (
      <div style={{ padding: '8px 0' }}>
        <p style={{ color: '#94a3b8', marginBottom: 12 }}>
          <strong style={{ color: '#e2e8f0' }}>说明:</strong> {item.explanation}
        </p>
        {item.issues.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <strong style={{ color: '#e2e8f0' }}>问题:</strong>
            <ul style={{ color: '#94a3b8', marginTop: 6 }}>
              {item.issues.map((issue, i) => (
                <li key={i} style={{ marginBottom: 6 }}>
                  <Tag
                    color={
                      issue.severity === 'critical' ? 'error' :
                      issue.severity === 'warning' ? 'warning' : 'default'
                    }
                    style={{ marginRight: 8 }}
                  >
                    {issue.severity}
                  </Tag>
                  {issue.description}
                  {issue.fix_suggestion && (
                    <span style={{ color: '#64748b', fontStyle: 'italic' }}> → {issue.fix_suggestion}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
        {item.recommendations.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <strong style={{ color: '#e2e8f0' }}>建议:</strong>
            <ul style={{ color: '#94a3b8', marginTop: 6 }}>
              {item.recommendations.map((rec, i) => (
                <li key={i} style={{ marginBottom: 4 }}>{rec}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    )
  }))

  return (
    <Card style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 58, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%)',
      border: '1px solid rgba(0, 212, 255, 0.15)',
      borderRadius: 16,
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 24,
        paddingBottom: 16,
        borderBottom: '1px solid rgba(0, 212, 255, 0.1)'
      }}>
        <div style={{
          width: 40,
          height: 40,
          borderRadius: 10,
          background: 'linear-gradient(135deg, #00d4ff 0%, #0891b2 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ fontSize: 20 }}>✓</span>
        </div>
        <div>
          <div style={{ color: '#64748b', fontSize: 11, marginBottom: 2 }}>VALIDATION</div>
          <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>验证结果</div>
        </div>
      </div>

      <Table
        dataSource={data}
        columns={columns}
        rowKey="key"
        pagination={false}
        style={{ marginBottom: 16 }}
        rowClassName={() => 'transparent-row'}
      />

      <Collapse
        items={collapseItems}
        defaultActiveKey={data.map(d => d.key)}
        style={{
          background: 'rgba(0, 0, 0, 0.2)',
          border: '1px solid rgba(0, 212, 255, 0.1)',
          borderRadius: 12,
          overflow: 'hidden'
        }}
      />

      <style>{`
        .ant-table {
          background: transparent !important;
        }

        .ant-table-thead > tr > th {
          background: rgba(0, 0, 0, 0.3) !important;
          color: #64748b !important;
          border-bottom: 1px solid rgba(0, 212, 255, 0.1) !important;
        }

        .ant-table-tbody > tr > td {
          background: transparent !important;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        .ant-table-tbody > tr:hover > td {
          background: rgba(0, 212, 255, 0.05) !important;
        }

        .ant-collapse-header {
          color: #e2e8f0 !important;
        }

        .ant-collapse-content {
          background: transparent !important;
          border-top: 1px solid rgba(0, 212, 255, 0.1) !important;
        }
      `}</style>
    </Card>
  )
}