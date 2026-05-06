"""HELGA CLI interface."""

import sys
from typing import Optional
import numpy as np

from main import HELGAAgent, run_colleague_late_scenario, run_custom_scenario
from validation import run_all_validations, print_validation_summary


class HELGACLI:
    """Command-line interface for HELGA agent."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize CLI.

        Args:
            config_path: Path to config file
        """
        self.agent = HELGAAgent(config_path=config_path)

    def run_interactive(self) -> None:
        """Run interactive mode."""
        print("\n" + "=" * 60)
        print("HELGA Interactive Mode")
        print("=" * 60)
        print("Enter scenario descriptions or 'quit' to exit")
        print("Example: 'my colleague is late for an important meeting'")
        print()

        while True:
            try:
                user_input = input("> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                result = run_custom_scenario(self.agent, user_input)

                print(f"\n  Decision: {result['decision'].optimal_action.name}")
                print(f"  Emotion: {result['emotion'].dominant_emotion}")
                print(f"  Utility: {result['decision'].utility.total:.3f}")
                print(f"  Reasoning: {result['decision'].reasoning_chain.steps[-1].rationale}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def run_scenario(self, scenario: str) -> dict:
        """Run predefined scenario.

        Args:
            scenario: Scenario name

        Returns:
            Scenario results
        """
        if scenario == "colleague_late":
            return run_colleague_late_scenario(self.agent)
        else:
            print(f"Unknown scenario: {scenario}")
            return {}

    def run_validation(self) -> None:
        """Run all validation tests."""
        print("\nRunning all validations...")
        results = run_all_validations()
        print_validation_summary(results)


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HELGA CLI")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--scenario", help="Run scenario")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--validation", action="store_true", help="Run validation")

    args = parser.parse_args()

    cli = HELGACLI(config_path=args.config)

    if args.validation:
        cli.run_validation()
    elif args.scenario:
        cli.run_scenario(args.scenario)
    elif args.interactive:
        cli.run_interactive()
    else:
        # Default: run colleague_late scenario
        cli.run_scenario("colleague_late")


if __name__ == "__main__":
    main()