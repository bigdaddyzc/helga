import { Card, Steps, Tag } from 'antd'
import type { ScenarioResult, DecisionIteration } from '../api/helga'

interface Props {
  reasoningChain: ScenarioResult['reasoning_chain']
  decisionIterations?: DecisionIteration[] | undefined
}

const stageLabels: Record<string, string> = {
  action_generation: '动作生成',
  value_evaluation: '价值评估',
  norm_check: '规范检查',
  complexity_check: '复杂度评估',
  decision: '最终决策'
}

const iterationStageLabels: Record<string, string> = {
  information_gathering: '信息收集',
  options_generation: '选项生成',
  evaluation: '方案评估',
  contingency_check: '突发检查',
  final_decision: '最终决策'
}

export default function ReasoningChain({ reasoningChain, decisionIterations }: Props) {
  const hasIterations = decisionIterations && decisionIterations.length > 0

  const items = hasIterations && decisionIterations
    ? decisionIterations.map((iter: DecisionIteration) => {
        const stageLabel = iterationStageLabels[iter.stage] || iter.stage
        return {
          title: stageLabel,
          description: (
            <div style={{
              fontSize: 13,
              background: 'rgba(0, 0, 0, 0.3)',
              padding: 12,
              borderRadius: 8,
              marginTop: 8
            }}>
              {iter.search_query && (
                <div style={{
                  marginBottom: 8,
                  padding: 10,
                  background: 'rgba(0, 212, 255, 0.1)',
                  borderRadius: 6,
                  border: '1px solid rgba(0, 212, 255, 0.2)'
                }}>
                  <strong style={{ color: '#00d4ff' }}>🔍 搜索查询:</strong>
                  <div style={{ color: '#e2e8f0', marginTop: 4 }}>{iter.search_query}</div>
                </div>
              )}
              {iter.search_result && (
                <div style={{
                  marginBottom: 8,
                  padding: 10,
                  background: 'rgba(168, 85, 247, 0.1)',
                  borderRadius: 6,
                  border: '1px solid rgba(168, 85, 247, 0.2)'
                }}>
                  <strong style={{ color: '#a855f7' }}>📋 搜索结果:</strong>
                  <pre style={{
                    margin: '4px 0 0 0',
                    whiteSpace: 'pre-wrap',
                    fontSize: 12,
                    color: '#94a3b8'
                  }}>{iter.search_result}</pre>
                </div>
              )}
              <div style={{ marginTop: 8 }}>
                <strong style={{ color: '#e2e8f0' }}>💭 推理:</strong>
                <div style={{ color: '#94a3b8', marginTop: 4 }}>{iter.reasoning_result}</div>
              </div>
              {iter.beliefs_updated && Object.keys(iter.beliefs_updated).length > 0 && (
                <div style={{
                  marginTop: 10,
                  padding: 10,
                  background: 'rgba(0, 0, 0, 0.3)',
                  borderRadius: 6
                }}>
                  <strong style={{ color: '#64748b' }}>📊 信念更新:</strong>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 6 }}>
                    {Object.entries(iter.beliefs_updated).map(([k, v]) => (
                      <span
                        key={k}
                        style={{
                          padding: '2px 8px',
                          background: 'rgba(0, 212, 255, 0.1)',
                          borderRadius: 4,
                          fontSize: 11,
                          color: '#00d4ff'
                        }}
                      >
                        {k}: {typeof v === 'number' ? v.toFixed(2) : v}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {iter.plans_considered.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <Tag color="purple" style={{
                    background: 'rgba(168, 85, 247, 0.1)',
                    border: '1px solid rgba(168, 85, 247, 0.3)',
                    color: '#a855f7'
                  }}>
                    考虑: {iter.plans_considered.join(', ')}
                  </Tag>
                </div>
              )}
            </div>
          )
        }
      })
    : reasoningChain.steps.map((step, _idx: number) => ({
        title: stageLabels[step.stage] || step.stage,
        description: (
          <div style={{
            background: 'rgba(0, 0, 0, 0.3)',
            padding: 12,
            borderRadius: 8,
            marginTop: 8
          }}>
            <p style={{ margin: '4px 0', color: '#94a3b8' }}>{step.rationale}</p>
            {_idx === reasoningChain.steps.length - 1 && (
              <Tag
                color="cyan"
                style={{
                  marginTop: 8,
                  background: 'rgba(0, 212, 255, 0.1)',
                  border: '1px solid rgba(0, 212, 255, 0.3)',
                  color: '#00d4ff'
                }}
              >
                备选: {reasoningChain.alternatives.join(', ')}
              </Tag>
            )}
          </div>
        )
      }))

  const currentStep = hasIterations && decisionIterations ? decisionIterations.length - 1 : reasoningChain.steps.length - 1

  return (
    <Card style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 58, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%)',
      border: '1px solid rgba(124, 58, 237, 0.2)',
      borderRadius: 16,
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 24,
        paddingBottom: 16,
        borderBottom: '1px solid rgba(124, 58, 237, 0.1)'
      }}>
        <div style={{
          width: 40,
          height: 40,
          borderRadius: 10,
          background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ fontSize: 20 }}>🔗</span>
        </div>
        <div>
          <div style={{ color: '#64748b', fontSize: 11, marginBottom: 2 }}>REASONING CHAIN</div>
          <div style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600 }}>推理链</div>
        </div>
      </div>

      <Steps
        current={currentStep}
        direction="vertical"
        items={items}
        style={{
          '--ant-primary-color': '#a855f7',
          '--ant-text-color': '#94a3b8',
          '--ant-text-color-secondary': '#475569'
        } as any}
      />

      <div style={{
        marginTop: 20,
        padding: 16,
        background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%)',
        borderRadius: 12,
        border: '1px solid rgba(0, 212, 255, 0.2)'
      }}>
        <div style={{
          color: '#64748b',
          fontSize: 11,
          marginBottom: 8,
          textTransform: 'uppercase',
          letterSpacing: 1
        }}>
          🎯 Final Decision
        </div>
        <div style={{
          color: '#e2e8f0',
          fontSize: 15,
          fontWeight: 500
        }}>
          {reasoningChain.final_decision}
        </div>
      </div>

      <style>{`
        .ant-steps-item-title {
          color: #e2e8f0 !important;
          font-weight: 600;
        }

        .ant-steps-item-description {
          color: #94a3b8;
        }

        .ant-steps-item-finish .ant-steps-item-icon {
          background: rgba(168, 85, 247, 0.2) !important;
          border-color: #a855f7 !important;
        }

        .ant-steps-item-process .ant-steps-item-icon {
          background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
          border: none !important;
        }

        .ant-steps-item-wait .ant-steps-item-icon {
          background: rgba(0, 0, 0, 0.3) !important;
          border-color: #475569 !important;
        }
      `}</style>
    </Card>
  )
}