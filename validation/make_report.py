import argparse
from pathlib import Path
from cr.data.ingestion import from_excel, from_parquet, from_csv
from cr.automation import Runner
from cr.reporting import LatexWriter, Report
from cr.reporting.writers.latex.template import CR, Template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reports based on a previously recorded tape")
    parser.add_argument("tape", help="Path to a file containing the tape as yaml")
    parser.add_argument("report", help="Path to a file containing the report as yaml")
    parser.add_argument("-d", "--data", help="Path to a file that contains the root data", default=None)
    parser.add_argument("-o", "--output", help="Path to a directory to store the output", default="report")
    parser.add_argument("-t", "--template", help="Name of LatexTemplate to use", default="CR")
    args = parser.parse_args()

    datasets = {}
    if args.data:
        path = Path(args.data)
        # TODO: How to set dataset name?!?
        if path.suffix in [".xlsx", ".xlsm", ".xsl"]:
            datasets[path.stem] = from_excel(path)
        elif path.suffix in [".csv"]:
            datasets[path.stem] = from_csv(path)
        elif path.suffix in [".pq", ".parq", ".parquet"]:
            datasets[path.stem] = from_parquet(path)
        else:
            raise Exception(f"Unable to load data with extension {path.suffix}")

    runner = Runner(args.tape, datasets)
    report = Report.from_yaml(args.report)

    if args.template == "CR":
        template = CR()
    else:
        template = Template()

    writer = LatexWriter(args.output, template)
    writer.write(report, runner)

    print(f"""
######################
Created and exported report "{report.title}" to folder "{args.output}"
A pdf can be created by running the following commands:
> cd {Path(args.output).absolute()}
> lualatex report.tex
(the specific tex engine and options can differ)
######################
""")