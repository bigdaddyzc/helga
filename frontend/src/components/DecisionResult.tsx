import { Card, Tag, Row, Col, Progress } from 'antd'
import type { ScenarioResult } from '../api/helga'

interface Props {
  result: ScenarioResult
}

export default function DecisionResult({ result }: Props) {
  const { decision, emotion, plan_distribution } = result

  const utilityPercent = Math.min(100, Math.max(0, decision.utility * 10))
  const confidencePercent = Math.min(100, Math.max(0, (decision.confidence || 0) * 100))

  const hasPlanDistribution = plan_distribution && plan_distribution.plans && plan_distribution.plans.length > 0

  const planData = hasPlanDistribution
    ? plan_distribution!.plans.map((plan, idx) => ({
        name: plan.name,
        probability: plan_distribution!.probabilities[idx],
        duration: plan.estimated_duration,
        riskLevel: plan.overall_risk_level,
        routeDetails: plan.route_details,
        reasoning: plan.reasoning
      }))
    : []

  const probabilityData = (decision.top5_probabilities || []).map((item) => ({
    action: item.action,
    probability: item.probability
  }))

  return (
    <Card style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 58, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%)',
      border: '1px solid rgba(0, 212, 255, 0.15)',
      borderRadius: 16,
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 24,
        paddingBottom: 16,
        borderBottom: '1px solid rgba(0, 212, 255, 0.1)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 48,
            height: 48,
            borderRadius: 12,
            background: 'linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(0, 212, 255, 0.3)'
          }}>
            <span style={{ fontSize: 24 }}>🎯</span>
          </div>
          <div>
            <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>最终决策</div>
            <Tag color="cyan" style={{
              fontSize: 16,
              padding: '4px 20px',
              borderRadius: 8,
              background: 'rgba(0, 212, 255, 0.1)',
              border: '1px solid rgba(0, 212, 255, 0.3)',
              fontWeight: 600
            }}>
              {decision.action.toUpperCase()}
            </Tag>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ color: '#64748b', fontSize: 12, marginBottom: 4 }}>Utility Score</div>
          <div style={{
            fontSize: 28,
            fontWeight: 700,
            fontFamily: 'Orbitron, monospace',
            color: decision.utility > 0.6 ? '#00d4ff' : decision.utility > 0.3 ? '#a855f7' : '#f59e0b'
          }}>
            {decision.utility.toFixed(3)}
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <Row gutter={[24, 24]}>
        <Col span={8}>
          <div style={{
            background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: 12,
            padding: 16
          }}>
            <div style={{
              color: '#64748b',
              fontSize: 11,
              marginBottom: 8,
              textTransform: 'uppercase',
              letterSpacing: 1
            }}>
              🔢 Action ID
            </div>
            <div style={{ color: '#e2e8f0', fontSize: 14 }}>{decision.action_id}</div>
          </div>
        </Col>

        <Col span={8}>
          <div style={{
            background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: 12,
            padding: 16
          }}>
            <div style={{
              color: '#64748b',
              fontSize: 11,
              marginBottom: 8,
              textTransform: 'uppercase',
              letterSpacing: 1
            }}>
              💫 情感状态
            </div>
            <Tag color={emotion.valence > 0.5 ? 'green' : 'orange'} style={{ marginBottom: 8 }}>
              {emotion.dominant_emotion}
            </Tag>
            <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
              <div>
                <div style={{ color: '#475569', fontSize: 10 }}>VALENCE</div>
                <div style={{ color: '#e2e8f0', fontSize: 14 }}>{(emotion.valence * 100).toFixed(0)}%</div>
              </div>
              <div>
                <div style={{ color: '#475569', fontSize: 10 }}>AROUSAL</div>
                <div style={{ color: '#e2e8f0', fontSize: 14 }}>{(emotion.arousal * 100).toFixed(0)}%</div>
              </div>
            </div>
          </div>
        </Col>

        <Col span={8}>
          <div style={{
            background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: 12,
            padding: 16
          }}>
            <div style={{
              color: '#64748b',
              fontSize: 11,
              marginBottom: 8,
              textTransform: 'uppercase',
              letterSpacing: 1
            }}>
              ⚡ 效用分解
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: 12 }}>
                <span>价值</span>
                <span style={{ color: '#00d4ff' }}>{decision.utility_breakdown.value.toFixed(3)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: 12 }}>
                <span>规范</span>
                <span style={{ color: '#a855f7' }}>{decision.utility_breakdown.norm.toFixed(3)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: 12 }}>
                <span>复杂度</span>
                <span style={{ color: '#f59e0b' }}>-{decision.utility_breakdown.complexity.toFixed(3)}</span>
              </div>
              <Progress
                percent={utilityPercent}
                size="small"
                showInfo={false}
                strokeColor="linear-gradient(90deg, #00d4ff, #7c3aed)"
                style={{ marginTop: 8 }}
              />
            </div>
          </div>
        </Col>
      </Row>

      {/* Confidence Bar */}
      <div style={{
        marginTop: 24,
        padding: 16,
        background: 'rgba(0, 0, 0, 0.2)',
        borderRadius: 12
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12
        }}>
          <span style={{ color: '#94a3b8', fontSize: 13 }}>决策置信度</span>
          <span style={{
            color: confidencePercent > 50 ? '#52c41a' : '#faad14',
            fontWeight: 600
          }}>
            {(confidencePercent).toFixed(1)}%
          </span>
        </div>
        <Progress
          percent={confidencePercent}
          showInfo={false}
          strokeColor={confidencePercent > 50 ? '#52c41a' : '#faad14'}
          trailColor="rgba(255,255,255,0.1)"
          style={{ marginBottom: 8 }}
        />
        <div style={{ color: '#475569', fontSize: 11 }}>
          最优动作的概率
        </div>
      </div>

      {/* Top 5 Probability Distribution */}
      <div style={{ marginTop: 24 }}>
        <div style={{
          color: '#94a3b8',
          fontSize: 13,
          marginBottom: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 8
        }}>
          <span>📊</span> 决策方案概率分布
        </div>

        {hasPlanDistribution ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {planData.map((item, idx) => (
              <div
                key={item.name}
                style={{
                  padding: 16,
                  background: idx === 0 ? 'linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(124, 58, 237, 0.1) 100%)' : 'rgba(0, 0, 0, 0.2)',
                  borderRadius: 12,
                  border: idx === 0 ? '1px solid rgba(0, 212, 255, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                  {idx === 0 && (
                    <span style={{
                      marginRight: 8,
                      padding: '2px 8px',
                      background: '#00d4ff',
                      color: '#0a0e27',
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 700
                    }}>
                      BEST
                    </span>
                  )}
                  <span style={{
                    fontSize: 14,
                    fontWeight: 600,
                    color: idx === 0 ? '#00d4ff' : '#e2e8f0'
                  }}>
                    {item.name}
                  </span>
                  <div style={{ flex: 1, marginLeft: 16 }}>
                    <Progress
                      percent={item.probability * 100}
                      size="small"
                      showInfo={false}
                      strokeColor={idx === 0 ? '#00d4ff' : '#64748b'}
                      trailColor="rgba(255,255,255,0.1)"
                      style={{ marginBottom: 0 }}
                    />
                  </div>
                  <span style={{
                    width: 56,
                    textAlign: 'right',
                    fontSize: 14,
                    fontWeight: 700,
                    color: idx === 0 ? '#00d4ff' : '#94a3b8'
                  }}>
                    {(item.probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#64748b', marginBottom: 8 }}>
                  <span>⏱️ {item.duration}</span>
                  <span>⚠️ 风险:
                    <Tag
                      color={item.riskLevel === 'low' ? 'success' : item.riskLevel === 'high' ? 'error' : 'warning'}
                      style={{ margin: 0, marginLeft: 4 }}
                    >
                      {item.riskLevel}
                    </Tag>
                  </span>
                </div>

                {item.routeDetails && (
                  <div style={{
                    fontSize: 11,
                    color: '#475569',
                    marginBottom: 6,
                    padding: '6px 10px',
                    background: 'rgba(0,0,0,0.3)',
                    borderRadius: 6
                  }}>
                    🛣️ {item.routeDetails.route_name || '未知路线'}
                    {item.routeDetails.distance_km > 0 && <span> | {item.routeDetails.distance_km}km</span>}
                    {item.routeDetails.estimated_time_minutes > 0 && <span> | 约{item.routeDetails.estimated_time_minutes}分钟</span>}
                    {item.routeDetails.toll_cost > 0 && <span> | 通行费¥{item.routeDetails.toll_cost}</span>}
                    {item.routeDetails.has_congestion && <span> | 🔴拥堵</span>}
                  </div>
                )}

                {item.reasoning && (
                  <div style={{ fontSize: 11, color: '#64748b', fontStyle: 'italic' }}>
                    💭 {item.reasoning}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {probabilityData.map((item, idx) => (
              <div
                key={item.action}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '10px 14px',
                  background: idx === 0 ? 'rgba(0, 212, 255, 0.1)' : 'rgba(0, 0, 0, 0.2)',
                  borderRadius: 8,
                  border: idx === 0 ? '1px solid rgba(0, 212, 255, 0.2)' : '1px solid rgba(255, 255, 255, 0.05)'
                }}
              >
                <span style={{
                  width: 80,
                  fontSize: 12,
                  color: idx === 0 ? '#00d4ff' : '#94a3b8'
                }}>
                  {item.action}
                </span>
                <div style={{ flex: 1, marginRight: 12 }}>
                  <Progress
                    percent={item.probability * 100}
                    size="small"
                    showInfo={false}
                    strokeColor={idx === 0 ? '#00d4ff' : '#64748b'}
                    trailColor="rgba(255,255,255,0.1)"
                    style={{ marginBottom: 0 }}
                  />
                </div>
                <span style={{
                  width: 50,
                  textAlign: 'right',
                  fontSize: 12,
                  fontWeight: 600,
                  color: idx === 0 ? '#00d4ff' : '#94a3b8'
                }}>
                  {(item.probability * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
      `}</style>
    </Card>
  )
}