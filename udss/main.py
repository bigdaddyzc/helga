"""UDSS 主入口

通用决策支持系统 - Universal Decision Support System

核心公式: Decision(E, Q, R) = argmax_a [α·V(a) + β·Sim(R,a) + δ·SemSim(Q,a) - γ·C(a)]

其中:
- V(a): 动作价值分数
- Sim(R,a): 搜索结果相似度
- SemSim(Q,a): 问题与动作的语义相似度
- C(a): 执行成本

使用示例:
    python main.py --question "我应该选择哪个offer"
    python main.py --interactive
"""

import argparse
import sys

from udss import UDSS, UDSSConfig, create_udss


def run_decision(udss: UDSS, question: str, context: dict = None):
    """执行单个决策"""
    print("\n" + "=" * 60)
    print("决策分析")
    print("=" * 60)
    print(f"\n问题: {question}")

    result = udss.decide(question, context)

    print(f"\n推荐动作: {result.decision_result.recommended_action.name}")
    print(f"置信度: {result.decision_result.confidence:.2%}")
    print(f"\n决策理由: {result.decision_result.reasoning}")
    print("\n" + "-" * 60)
    print("\n生成方案:")
    print(result.plan_text)

    return result


def run_interactive(udss: UDSS):
    """交互模式"""
    print("\n" + "=" * 60)
    print("UDSS 交互模式")
    print("=" * 60)
    print("输入您的问题，系统将生成决策方案")
    print("输入 'quit' 退出，输入 'feedback' 提供反馈")
    print()

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            if user_input.lower() == 'feedback':
                # 反馈模式
                session_id = input("请输入会话ID: ").strip()
                score = int(input("请输入评分(1-5): ").strip())
                rationale = input("请输入理由: ").strip()
                accepted = input("是否采纳(y/n): ").strip().lower() == 'y'

                feedback_result = udss.provide_feedback(session_id, score, rationale, accepted)
                print(f"\n反馈结果: {feedback_result}")
                continue

            if not user_input:
                continue

            result = run_decision(udss, user_input)
            print(f"\n会话ID: {result.session_id}")
            print(f"决策ID: {result.decision_id}")

        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


def run_feedback_demo(udss: UDSS):
    """反馈演示"""
    print("\n" + "=" * 60)
    print("反馈学习演示")
    print("=" * 60)

    # 执行初始决策
    question = "我应该接受哪个工作offer？"
    result = udss.decide(question)

    print(f"\n问题: {question}")
    print(f"推荐: {result.decision_result.recommended_action.name}")
    print(f"会话ID: {result.session_id}")

    # 模拟用户反馈
    print("\n--- 模拟用户反馈 ---")
    feedback_result = udss.provide_feedback(
        session_id=result.session_id,
        score=4,
        rationale="方案合理但缺少薪资对比信息",
        accepted=True
    )
    print(f"反馈结果: {feedback_result}")

    # 显示学习总结
    summary = udss.get_learning_summary()
    print(f"\n学习总结:")
    print(f"  训练轮次: {summary.get('episode_count', 0)}")
    print(f"  平均奖励: {summary.get('avg_reward', 0):.3f}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="UDSS - Universal Decision Support System"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="要决策的问题"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="额外上下文 (JSON格式)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互模式"
    )
    parser.add_argument(
        "--demo-feedback",
        action="store_true",
        help="反馈学习演示"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="价值权重 α"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="相似度权重 β"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.3,
        help="成本权重 γ"
    )

    args = parser.parse_args()

    # 创建配置
    config = UDSSConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )

    # 创建UDSS
    udss = create_udss(config)

    # 执行
    if args.demo_feedback:
        run_feedback_demo(udss)
    elif args.interactive:
        run_interactive(udss)
    elif args.question:
        context = None
        if args.context:
            import json
            try:
                context = json.loads(args.context)
            except:
                print("错误: context 必须是有效的JSON格式")
                return
        run_decision(udss, args.question, context)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python main.py -q \"我应该选择哪个offer\"")
        print("  python main.py --interactive")
        print("  python main.py --demo-feedback")


if __name__ == "__main__":
    main()