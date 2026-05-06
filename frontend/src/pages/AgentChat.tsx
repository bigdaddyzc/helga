import { useState } from 'react'
import { Card, Input, Button, message } from 'antd'
import { helgaApi, ScenarioResult, ValidationResults } from '../api/helga'
import DecisionResult from '../components/DecisionResult'
import ReasoningChain from '../components/ReasoningChain'
import ValidationResultsDisplay from '../components/ValidationResults'
import ActionPlanDisplay from '../components/ActionPlanDisplay'

const { TextArea } = Input

export default function AgentChat() {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScenarioResult | null>(null)
  const [validationResults, setValidationResults] = useState<ValidationResults | null>(null)
  const [showValidation, setShowValidation] = useState(false)

  const handleSubmit = async (runValidation: boolean = false) => {
    if (!input.trim()) {
      message.warning('请输入场景描述')
      return
    }

    setLoading(true)
    setResult(null)
    setValidationResults(null)
    setShowValidation(false)
    try {
      const { data } = await helgaApi.runCustomScenario(input, runValidation)
      setResult(data.scenario_result)
      if (runValidation) {
        setValidationResults(data.validation_results)
        setShowValidation(true)
      }
      message.success('处理完成')
    } catch (err) {
      message.error('处理失败，请重试')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* 输入区域 */}
      <Card style={{
        background: 'linear-gradient(135deg, rgba(26, 31, 58, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)',
        border: '1px solid rgba(0, 212, 255, 0.2)',
        borderRadius: 16,
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
        marginBottom: 24
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: '#00d4ff',
              boxShadow: '0 0 10px #00d4ff'
            }} />
            <span style={{ color: '#94a3b8', fontSize: 14 }}>输入您的决策场景</span>
          </div>

          <TextArea
            rows={4}
            placeholder="例如：同事会议迟到 important meeting"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault()
                handleSubmit(false)
              }
            }}
            style={{
              background: 'rgba(0, 0, 0, 0.3)',
              border: '1px solid rgba(0, 212, 255, 0.3)',
              borderRadius: 12,
              color: '#e2e8f0',
              fontSize: 16,
              padding: '16px 20px',
              resize: 'none'
            }}
          />

          <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
            <Button
              onClick={() => handleSubmit(true)}
              loading={loading}
              style={{
                background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)',
                border: 'none',
                borderRadius: 8,
                color: 'white',
                fontWeight: 500,
                height: 40,
                paddingInline: 24
              }}
            >
              提交并运行验证
            </Button>
            <Button
              type="primary"
              onClick={() => handleSubmit(false)}
              loading={loading}
              style={{
                background: 'linear-gradient(135deg, #00d4ff 0%, #0891b2 100%)',
                border: 'none',
                borderRadius: 8,
                fontWeight: 500,
                height: 40,
                paddingInline: 24,
                boxShadow: '0 4px 15px rgba(0, 212, 255, 0.3)'
              }}
            >
              提交场景
            </Button>
          </div>
        </div>
      </Card>

      {/* 加载状态 */}
      {loading && (
        <Card style={{
          background: 'rgba(26, 31, 58, 0.8)',
          border: '1px solid rgba(0, 212, 255, 0.2)',
          borderRadius: 16,
          textAlign: 'center',
          padding: 48
        }}>
          <div style={{
            width: 80,
            height: 80,
            margin: '0 auto 24px',
            borderRadius: '50%',
            background: 'conic-gradient(from 0deg, #00d4ff, #7c3aed, #00d4ff)',
            animation: 'spin 1s linear infinite',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <div style={{
              width: 60,
              height: 60,
              borderRadius: '50%',
              background: '#0a0e27',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <span style={{ fontSize: 28 }}>🧠</span>
            </div>
          </div>
          <div style={{ color: '#00d4ff', fontSize: 18, fontWeight: 500, marginBottom: 8 }}>
            HELGA 思考中...
          </div>
          <div style={{ color: '#64748b', fontSize: 13 }}>
            正在分析场景、推理决策、执行验证
          </div>

          <style>{`
            @keyframes spin {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }
          `}</style>
        </Card>
      )}

      {/* 结果展示 */}
      {!loading && result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{
            animation: 'fadeInUp 0.5s ease-out'
          }}>
            <DecisionResult result={result} />
          </div>

          {result.action_plan && (
            <div style={{
              animation: 'fadeInUp 0.5s ease-out 0.1s both'
            }}>
              <ActionPlanDisplay actionPlan={result.action_plan} />
            </div>
          )}

          <div style={{
            animation: 'fadeInUp 0.5s ease-out 0.2s both'
          }}>
            <ReasoningChain
              reasoningChain={result.reasoning_chain}
              decisionIterations={result.plan_distribution?.decision_iterations}
            />
          </div>
        </div>
      )}

      {showValidation && validationResults && (
        <div style={{
          animation: 'fadeInUp 0.5s ease-out 0.3s both'
        }}>
          <ValidationResultsDisplay results={validationResults} />
        </div>
      )}

      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .ant-card {
          backdrop-filter: blur(10px);
        }

        .ant-input-textarea textarea {
          color: #e2e8f0 !important;
        }

        .ant-input-textarea textarea::placeholder {
          color: #475569 !important;
        }

        .ant-tabs-tab {
          color: #94a3b8 !important;
        }

        .ant-tabs-tab-active .ant-tabs-tab-btn {
          color: #00d4ff !important;
        }

        .ant-tabs-ink-bar {
          background: linear-gradient(90deg, #00d4ff, #7c3aed) !important;
        }
      `}</style>
    </div>
  )
}