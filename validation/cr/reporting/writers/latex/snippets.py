
BASE_PACKAGES = r"""
\documentclass[11pt,a4paper, oneside]{memoir}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb}
\usepackage{tikz}
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}
"""

BASE_HDR = r"""
\pagestyle{plain}
\makepagestyle{plain}
\makeevenfoot{plain}{}{}{}
\makeoddfoot{plain}{}{Page \thepage \hspace{1pt} of \pageref{LastPage}}{}
\makeoddhead{plain}{}{\leftmark}{}
\makeevenhead{plain}{}{}{}
"""

def figure(path, id=None):
    if id:
        id = f"\label{{res:{id}}}"
    else:
        id = ""
    return r"""
\begin{figure}[htp]
\centering
\includegraphics[width=10cm,height=10cm,keepaspectratio]{""" + path.replace(" ","_") + r"""}
\caption{TEST}
""" + id + r"""
\end{figure}"""