
from . import snippets as snip

class Template():

    def preface(self):
        return snip.BASE_PACKAGES

    def assets(self):
        return dict(
            CR_LOGO= "cr-logo.png",
        )

    def document_start(self):
        return ""

    def format_result(self, result):
        return result

    def format_content(self, content):
        return content

class CR(Template):

    def __init__(self, authors:dict=None) -> None:
        super().__init__()

    def assets(self):
        return dict(
            CR_LOGO= "cr-logo.png",
            BARLOW_BOLD = "Barlow-Bold.ttf",
            SECULARONE = "SecularOne-Regular.ttf",
            CRCOMMANDS = "CRCommands.tex",
            CRSTYLE = "CRSTYLE.tex",
            FRONTPAGE = "FRONTPAGE.tex"
        )

    def preface(self):
        return super().preface() + r"""
\usepackage{fontspec}
\usepackage{caption}
\usepackage{xcolor}
\usepackage[sc,compact,explicit]{titlesec}
\usepackage{enumitem}
\usepackage[
breaklinks=true,colorlinks=true,
linkcolor=black,urlcolor=black,citecolor=black,
bookmarks=true,bookmarksopenlevel=2
pdfauthor={CR Consulting},
pdftitle={!<CTX;TITLE>!}]{hyperref}

\geometry{total={210mm,297mm},
 left=1.25in,right=1.25in,
 bindingoffset=0mm, top=1in,bottom=1in}

\addtocontents{toc}{\protect\thispagestyle{empty}}
\OnehalfSpacing

\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.5}
\titlespacing*{\chapter}{0pt}{-15pt}{25pt}

\input{!<ASSET;CRSTYLE>!}    
\input{!<ASSET;CRCOMMANDS>!}    
        """   

    def document_start(self):
        return r"""
\input{!<ASSET;FRONTPAGE>!}    

\clearpage
\pagecolor{white}
\renewcommand*{\thefootnote}{[\arabic{footnote}]}
\setcounter{secnumdepth}{2}
\addtocontents{toc}{\protect\setcounter{tocdepth}{-1}}
\tableofcontents{}
\addtocontents{toc}{\vskip-40pt}
\addtocontents{toc}{~\hfill\textbf{Page}\par}
\addtocontents{toc}{\protect\setcounter{tocdepth}{2}}
 
\cleardoublepage%
\setcounter{page}{1}
\pagestyle{plain}
\input{!<ASSET;>!}  
"""

    @staticmethod
    def help_format_authors(names, roles, departments, mails, numbers):
        # TODO use a dict instead of list apocalypse
        authors = []
        for name, role, department, mail, number in zip(names, roles, departments, mails, numbers):
            authors.append(f"""
\\textbf{{{name}}} \\\\
\small{{{role}}} \\\\
\small{{{department}}} \\\\
\small{{{mail}}} \\\\
{{{number}}} \\\\          
            """)
        return r"\vspace{0.25em}\\".join(authors)