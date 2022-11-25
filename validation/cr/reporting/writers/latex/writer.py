from pathlib import Path
import re
from datetime import date
from functools import partial
import importlib.resources as resources

from cr.plotting.plotly import export_figure
from .template import Template
from cr.testing.output import OutputType
from cr.automation.runner import Runner
from cr.reporting import Report
from cr.reporting.mapper import ContentMapper
from . import snippets as snip
from . import assets


def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def to_latex_path(path):
    if isinstance(path, Path):
        path = str(path)
    return path.replace("\\","/")

class LatexMapper(ContentMapper):

    def __init__(self, runner, context, assets, relative_assets_path, artifacts_path, relative_artifacts_path):
        super().__init__(runner, context)
        self.assets = assets
        self.relative_assets_path = relative_assets_path
        self.artifacts_path = artifacts_path
        self.relative_artifacts_path = relative_artifacts_path
        self.tag_maps.update({
            "ASSET": self.map_asset,
            "REF": self.map_cites
        })

    def map_asset(self, asset_tag:str):
        if asset_tag in self.assets:
            return to_latex_path(self.relative_assets_path.joinpath(self.assets[asset_tag]))
        else:
            # TODO: Raise Warning
            return asset_tag
        
    def map_cites(self, cite_tag:str):
        return f"\\ref{{res:{cite_tag}}}"

    def _map_output(self, output, test_uid, output_name, tag):
        if output.output_type == OutputType.FIGURE:
            fig = output.value
            path_figure = self.artifacts_path.joinpath(f"{test_uid}__{output_name}.png".replace(" ", "_"))
            export_figure(fig, path_figure, format="png")
            output_value = snip.figure(to_latex_path(self.relative_artifacts_path.joinpath(f"{test_uid}__{output_name}.png")), f"{test_uid}.{output_name}")
        else:
            output_value = str(output.value)       
        return f"""
%CR:RESULT:START:{tag}
{output_value}
%CR:RESULT:END:{tag}
"""

class LatexWriter():

    def __init__(self, output_directory:str, template:Template=None, overwrite="replace"):
        path = Path(output_directory)
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"LatexWriter needs an output directory, but got an existing path, which is not a directory: {path}")
            if path.iterdir():
                if overwrite == "replace":
                    rm_tree(path)
                    path.mkdir()
                else:
                    raise ValueError(f"LatexWriter found existing directory with items, and was set to neither 'replace' nor 'append' in arg overwrite")
        else:
            path.mkdir()

        self.path = path
        self.path_artifacts = path.joinpath("artifacts")
        self.path_artifacts.mkdir()
         # TODO: Move assets out of root current problem is in latex font not looking trough paths but only root
        self.path_assets = self.path.joinpath("")
        # self.path_assets.mkdir()

        if not template:
            template = Template()
        self.template = template
            
    def write(self, report:Report, runner:Runner, context_updates:dict=None):
        # Initialize the context
        if not context_updates:
            context_updates = {}
        context = {**report.context, **context_updates, **runner.get_run_context()}

        # Initialize a content mapper
        mapper = LatexMapper(runner, context, self.template.assets(), 
                            self.path_assets.relative_to(self.path), 
                            self.path_artifacts,
                            self.path_artifacts.relative_to(self.path)
                            )

        # Export sections and overall content
        chapters = {}
        for chapter in report.chapters:
            path_chapter = self.path.joinpath(chapter.title.replace(" ", "_"))
            path_chapter.mkdir()
            chapters[chapter.title] = {}
            for section in chapter.sections:
                path_section = path_chapter.joinpath(section.title.replace(" ","_") + ".tex") 
                path_section.write_text(mapper.map(section.content))
                chapters[chapter.title][section.title] = to_latex_path(path_section.relative_to(self.path))

        # Export asset files
        for key, path in self.template.assets().items():
            # Copy binary files otherwise parse content
            extension = path.split(".")[-1]
            if extension != "tex":
                with self.path_assets.joinpath(path).open("wb") as to_file:
                    with resources.open_binary(assets, path) as from_file:
                        to_file.write(from_file.read())
            else:
                with self.path_assets.joinpath(path).open("w", encoding="utf-8") as to_file:
                    with resources.open_text(assets, path) as from_file:
                        to_file.write(mapper.map(from_file.read()))


        # Generate the overall report structure 
        with self.path.joinpath("structure.tex").open('w') as file:
            for title, sections in chapters.items():
                file.write(f"\chapter{{{title}}}")
                for section_title, section_path in sections.items():
                    file.write(f"\section{{{section_title}}}")
                    file.write(f"\input{{{section_path}}}")

        # Generate main report file, preface and similar
        with self.path.joinpath("report.tex").open('w') as file:
            file.write(mapper.map(self.template.preface()))
            file.write(r"\begin{document}")
            file.write(mapper.map(self.template.document_start()))
            file.write(r"\input{structure.tex}")
            file.write(r"\end{document}")

