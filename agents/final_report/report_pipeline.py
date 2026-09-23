import json
from pathlib import Path
from typing import Dict, Any, Union

try:
    from agents.final_report.report_agent import ReportAgent
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agents.final_report.report_agent import ReportAgent


# Loads specialist evaluations from agents_analysis.json or input dictionary
def load_analysis_input(source: Union[str, Dict[str, Any]] = "agents_analysis.json") -> Dict[str, Any]:
    if isinstance(source, dict):
        return source

    with open(source, "r", encoding="utf-8") as file:
        return json.load(file)


# Saves the generated report to markdown file
def save_report(
    report_text: str,
    output_filepath: str = "final_report.md",
) -> None:
    with open(output_filepath, "w", encoding="utf-8") as file:
        file.write(report_text)


# Runs the final report pipeline to synthesize specialist evaluations into a profile
def run_report_pipeline(
    source: Union[str, Dict[str, Any]] = "agents_analysis.json",
    output_filepath: str = "final_report.md",
) -> str:
    analysis_data = load_analysis_input(source)
    report_agent = ReportAgent()
    report = report_agent.generate_report(analysis_data)
    save_report(report, output_filepath)
    return report


run = run_report_pipeline


if __name__ == "__main__":
    result = run()
    print(result)
