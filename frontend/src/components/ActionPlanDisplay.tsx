import { Card, Steps, Tag, List, Typography } from 'antd'

const { Title, Text } = Typography

interface ActionStep {
  step_number: number
  description: string
  details: Record<string, unknown>
  warning: string | null
}

interface ActionPlan {
  title: string
  action_name: string
  steps: ActionStep[]
  time_estimate: string
  resources: Record<string, unknown>
  alternatives: string[]
  reasoning: string
}

interface Props {
  actionPlan: ActionPlan
}

export default function ActionPlanDisplay({ actionPlan }: Props) {
  const items = actionPlan.steps.map((step, _index) => ({
    title: `步骤 ${step.step_number}`,
    description: (
      <div>
        <p style={{ margin: '4px 0', color: '#94a3b8' }}>{step.description}</p>
        {step.warning && (
          <Tag
            color="warning"
            style={{
              marginTop: 8,
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              color: '#f59e0b'
            }}
          >
            ⚠️ 警告: {step.warning}
          </Tag>
        )}
        {step.details && Object.keys(step.details).length > 0 && (
          <div style={{
            marginTop: 10,
            padding: 10,
            background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: 6,
            fontSize: 12
          }}>
            {Object.entries(step.details).map(([key, value]) => (
              <div key={key} style={{ display: 'flex', gap: 8 }}>
                <span style={{ color: '#64748b', minWidth: 80 }}>{key}:</span>
                <span style={{ color: '#94a3b8' }}>
                  {Array.isArray(value) ? value.join(', ') : String(value)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }))

  return (
    <Card style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 58, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%)',
      border: '1px solid rgba(245, 158, 11, 0.2)',
      borderRadius: 16,
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 24,
        paddingBottom: 16,
        borderBottom: '1px solid rgba(245, 158, 11, 0.1)'
      }}>
        <div style={{
          width: 40,
          height: 40,
          borderRadius: 10,
          background: 'linear-gradient(135deg, #f59e0b 0%, #f97316 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ fontSize: 20 }}>📋</span>
        </div>
        <div>
          <div style={{ color: '#64748b', fontSize: 11, marginBottom: 2 }}>ACTION PLAN</div>
          <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>
            {actionPlan.title || '执行方案'}
          </div>
        </div>
      </div>

      <Steps
        current={actionPlan.steps.length - 1}
        items={items}
        direction="vertical"
        style={{
          '--ant-primary-color': '#f59e0b'
        } as any}
      />

      <div style={{
        marginTop: 24,
        padding: 20,
        background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(249, 115, 22, 0.05) 100%)',
        borderRadius: 12,
        border: '1px solid rgba(245, 158, 11, 0.2)'
      }}>
        <Title level={5} style={{ marginBottom: 8, color: '#e2e8f0' }}>
          📖 执行说明
        </Title>
        <Text style={{ color: '#94a3b8' }}>{actionPlan.reasoning}</Text>
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 24,
        marginTop: 20
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: '#64748b', fontSize: 13 }}>预计时间:</span>
          <Tag
            color="warning"
            style={{
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              color: '#f59e0b'
            }}
          >
            ⏱️ {actionPlan.time_estimate}
          </Tag>
        </div>
      </div>

      {actionPlan.resources && Object.keys(actionPlan.resources).length > 0 && (
        <div style={{ marginTop: 20 }}>
          <Text strong style={{ color: '#e2e8f0' }}>📦 可用资源:</Text>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
            {Object.entries(actionPlan.resources).map(([key, value]) => (
              <Tag
                key={key}
                color="green"
                style={{
                  background: 'rgba(82, 196, 26, 0.1)',
                  border: '1px solid rgba(82, 196, 26, 0.3)',
                  color: '#52c41a'
                }}
              >
                {key}: {String(value)}
              </Tag>
            ))}
          </div>
        </div>
      )}

      {actionPlan.alternatives && actionPlan.alternatives.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <Text strong style={{ color: '#e2e8f0' }}>🔄 替代方案:</Text>
          <List
            size="small"
            bordered
            dataSource={actionPlan.alternatives}
            renderItem={(item) => (
              <List.Item style={{
                background: 'rgba(0, 0, 0, 0.2)',
                color: '#94a3b8'
              }}>
                {item}
              </List.Item>
            )}
            style={{
              marginTop: 8,
              background: 'rgba(0, 0, 0, 0.2)',
              border: '1px solid rgba(0, 212, 255, 0.1)',
              borderRadius: 8
            }}
          />
        </div>
      )}

      <style>{`
        .ant-steps-item-title {
          color: #e2e8f0 !important;
          font-weight: 600;
        }

        .ant-steps-item-description {
          color: #94a3b8;
        }

        .ant-steps-item-process .ant-steps-item-icon {
          background: linear-gradient(135deg, #f59e0b, #f97316) !important;
          border: none !important;
        }

        .ant-steps-item-finish .ant-steps-item-icon {
          background: rgba(82, 196, 26, 0.2) !important;
          border-color: #52c41a !important;
        }
      `}</style>
    </Card>
  )
}