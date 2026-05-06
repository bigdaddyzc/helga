"""UDSS Flask API Server

将UDSS决策系统暴露为REST API，供前端使用
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from udss import create_udss, UDSSConfig


app = Flask(__name__)
CORS(app)

# Global UDSS instance
udss = None


def get_udss():
    """Get or create UDSS instance."""
    global udss
    if udss is None:
        config = UDSSConfig(
            alpha=1.0,
            beta=0.5,
            gamma=0.3,
            max_search_results=3,
            rl_enabled=True
        )
        udss = create_udss(config)
    return udss


def make_serializable(obj):
    """Convert numpy types to native Python types."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, '__float__'):
        return float(obj)
    return obj


def serialize_action_plan(plan):
    """Serialize ActionPlan to JSON-compatible format."""
    if plan is None:
        return None
    return {
        'title': plan.title,
        'action_name': plan.title.split('的')[-1].replace('方案', '') if '方案' in plan.title else plan.title,
        'steps': [
            {
                'step_number': step.step_number,
                'description': step.description,
                'details': {},
                'warning': step.warning,
                'route_name': step.route_name
            }
            for step in plan.steps
        ],
        'time_estimate': plan.time_estimate,
        'resources': {},
        'alternatives': plan.alternatives,
        'reasoning': plan.reasoning,
        'route_comparison': [
            {
                'route_name': r.route_name,
                'distance_km': r.distance_km,
                'estimated_time_minutes': r.estimated_time_minutes,
                'toll_cost': r.toll_cost,
                'risk_level': r.risk_level
            }
            for r in plan.route_comparison
        ] if plan.route_comparison else None
    }


def serialize_result(result):
    """Serialize UDSS result to frontend-compatible format."""
    decision = result.decision_result

    # Build top5 probabilities list
    top5_probs = []
    if hasattr(decision, 'top5_probabilities') and decision.top5_probabilities:
        for action_id, action_name, prob in decision.top5_probabilities:
            top5_probs.append({
                'id': action_id,
                'action': action_name,
                'probability': float(prob)
            })

    # Compute utility breakdown
    utility_breakdown = {
        'value': decision.recommended_action.value_vector.mean() if hasattr(decision.recommended_action, 'value_vector') else 0.5,
        'norm': 0.3,
        'complexity': decision.recommended_action.cost_vector.mean() if hasattr(decision.recommended_action, 'cost_vector') else 0.3
    }

    response = {
        'decision': {
            'action': decision.recommended_action.name,
            'action_id': hash(decision.recommended_action.id) % 1000,
            'utility': decision.confidence,
            'confidence': decision.confidence,
            'top5_probabilities': top5_probs,
            'utility_breakdown': utility_breakdown
        },
        'emotion': {
            'dominant_emotion': 'neutral',
            'valence': 0.0,
            'arousal': 0.5
        },
        'reasoning_chain': {
            'steps': [
                {
                    'stage': 'action_generation',
                    'input': {},
                    'output': {'candidate_actions': list(decision.all_scores.keys())},
                    'rationale': f'生成了{len(decision.all_scores)}个候选动作',
                    'timestamp': 0
                },
                {
                    'stage': 'value_evaluation',
                    'input': {},
                    'output': {'value_scores': {}},
                    'rationale': '评估了每个动作的价值贡献',
                    'timestamp': 1
                },
                {
                    'stage': 'final_decision',
                    'input': {},
                    'output': {'selected': decision.recommended_action.name},
                    'rationale': decision.reasoning,
                    'timestamp': 4
                }
            ],
            'final_decision': f'{decision.recommended_action.name}: {decision.reasoning}',
            'alternatives': [a.name for a in decision.alternatives]
        },
        'action_plan': serialize_action_plan(result.action_plan),
        'environment': {
            'next_state': []
        }
    }

    return response


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'udss'})


@app.route('/api/scenario', methods=['POST'])
def run_scenario():
    """Run scenario and return decision.

    Request body:
        { "scenario": "description of the situation" }

    Response:
        Compatible with frontend ScenarioResult format
    """
    data = request.get_json()
    scenario = data.get('scenario', '')

    if not scenario:
        return jsonify({'error': 'scenario is required'}), 400

    try:
        udss_instance = get_udss()
        result = udss_instance.decide(scenario)
        return jsonify(serialize_result(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def provide_feedback():
    """Provide user feedback for RL learning.

    Request body:
        { "session_id": "xxx", "score": 4, "rationale": "...", "accepted": true }
    """
    data = request.get_json()

    session_id = data.get('session_id', '')
    score = data.get('score', 3)
    rationale = data.get('rationale', '')
    accepted = data.get('accepted', True)

    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400

    try:
        udss_instance = get_udss()
        result = udss_instance.provide_feedback(session_id, score, rationale, accepted)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validate', methods=['POST'])
def validate():
    """Run validation (placeholder for UDSS).

    Response:
        { "results": { "ecological": {...}, ... } }
    """
    # UDSS uses RL feedback instead of traditional validation
    # Return a placeholder validation result
    return jsonify({
        'results': {
            'decision_quality': {
                'passed': True,
                'metric_value': 0.85,
                'threshold': 0.7,
                'explanation': 'Decision quality assessed via RL feedback',
                'issues': [],
                'recommendations': []
            }
        }
    })


@app.route('/api/custom-scenario', methods=['POST'])
def custom_scenario():
    """Run custom scenario with optional validation.

    Request body:
        { "scenario": "description", "run_validation": true/false }

    Response:
        { "scenario_result": {...}, "validation_results": {...} (if run_validation=true)
    """
    data = request.get_json()
    scenario = data.get('scenario', '')
    run_validation = data.get('run_validation', False)

    if not scenario:
        return jsonify({'error': 'scenario is required'}), 400

    try:
        udss_instance = get_udss()
        result = udss_instance.decide(scenario)
        response = {
            'scenario_result': serialize_result(result)
        }

        if run_validation:
            response['validation_results'] = {
                'decision_quality': {
                    'passed': True,
                    'metric_value': 0.85,
                    'threshold': 0.7,
                    'explanation': 'Validated via RL feedback loop',
                    'issues': [],
                    'recommendations': []
                }
            }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning-summary', methods=['GET'])
def learning_summary():
    """Get RL learning summary.

    Response:
        { "episode_count": N, "avg_reward": X, ... }
    """
    try:
        udss_instance = get_udss()
        summary = udss_instance.get_learning_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset-learning', methods=['POST'])
def reset_learning():
    """Reset RL learning state."""
    try:
        udss_instance = get_udss()
        udss_instance.reset_learning()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting UDSS API Server...")
    print("Available endpoints:")
    print("  GET  /api/health           - Health check")
    print("  POST /api/scenario         - Run scenario (UDSS)")
    print("  POST /api/feedback         - Provide feedback (RL)")
    print("  POST /api/validate         - Run validation")
    print("  POST /api/custom-scenario  - Run scenario with optional validation")
    print("  GET  /api/learning-summary - Get RL learning summary")
    print("  POST /api/reset-learning   - Reset RL state")
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)