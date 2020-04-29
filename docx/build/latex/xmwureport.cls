\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{xmwureport}[2014/09/15 v1.00 xmwu technology report document class]

%%载入基础类
\LoadClass[10.5pt,a4paper,openany]{book}


\newtoks\zhtitle
\newtoks\entitle
\newtoks\level
\newtoks\email
\newtoks\version
\newtoks\group
\newtoks\leader
\newtoks\completedata

%%数学字体宏包

%% 常用宏包
\RequirePackage{graphicx}
\graphicspath{{./figure/}{./figures/}{./image/}{./graphics/}{./graphic/}{./pictures/}{./picture/}}
\RequirePackage{enumerate}
\RequirePackage{amsmath,mathrsfs,amsfonts}
\RequirePackage{booktabs}
\RequirePackage{indentfirst}
\RequirePackage{lipsum}

%% 页面设置
\RequirePackage{hyperref}
\hypersetup{
 breaklinks,
 baseurl            =http://,
 unicode,
 bookmarksnumbered  =true,
 bookmarksopen      =true,
 pdfauthor          =\@author,
 pdftitle           =\@title note,
 pdfsubject         =\@author \@title note,
 pdfkeywords        ={notes},
 pdfcreator         ={XeLaTeX with elegantnote class},
 colorlinks,
 linkcolor          =black,
 plainpages         =false,
 pdfstartview       =FitH,
 pdfborder={0 0 0},
 linktocpage
 }

\RequirePackage{geometry}
\geometry{
  a4paper,
  left=31.7mm,  %% or inner=23mm
  right=31.7mm, %% or outer=18mm
  top=25.4mm,
  bottom=25.4mm,
  headheight=2.17cm,
  headsep=4mm,
  footskip=12mm
}

%% 字体设置
%\RequirePackage{lmodern,fix-cm,type1cm}
\RequirePackage[no-math]{fontspec}
%\defaultfontfeatures{Mapping=tex-text}

\RequirePackage{xunicode}
\RequirePackage{xltxtra}
\setmainfont[Ligatures=TeX]{Times New Roman}  %  (\textrm)
\setsansfont{Times New Roman}                 %  (\textsf)
\setmonofont{Times New Roman}              % Palatino Linotype
%-中文字体设置-%
\RequirePackage{xeCJK}
\setCJKmainfont[BoldFont={黑体},ItalicFont={楷体}]{宋体}
\setCJKsansfont[BoldFont={黑体}]{宋体}
\setCJKmonofont{宋体}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt

%%中文结构名字
\renewcommand{\contentsname}{\bfseries 目\hspace{2em}录}
\renewcommand{\figurename}{\textbf{图}}
\renewcommand{\tablename}{\text{表}}
\renewcommand{\partname}{部分}
\renewcommand{\listfigurename}{\bfseries 插图目录}
\renewcommand{\listtablename}{\bfseries 表格目录}
\renewcommand{\bibname}{\bfseries 参考文献}
\renewcommand{\appendixname}{\bfseries 附\hspace{2em}录}
\renewcommand{\indexname}{\bfseries 索\hspace{2em}引}


%% 章节设置
\RequirePackage[pagestyles]{titlesec}
\setcounter{secnumdepth}{3}
\linespread{1.5}
\titleformat{\chapter}[hang]{\fontsize{21pt}{\baselineskip}\bfseries}{\thechapter.}{0.5em}{}
\titlespacing*{\chapter}{0pt}{0pt}{16.5pt}
\titleformat{\section}[hang]{\fontsize{15.75pt}{\baselineskip} \bfseries}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{13pt}{13pt}
\titleformat{\subsection}[hang]{\fontsize{13.75pt}{\baselineskip} \bfseries}{\thesubsection.}{0.5em}{}
\titlespacing*{\subsection}{0pt}{13pt}{13pt}

%% 目录设置
\RequirePackage{titletoc}
\setcounter{tocdepth}{2}
\titlecontents{chapter}
              [0em] % left
              {\vskip 3pt \bfseries \fontsize{10.5pt}{10.5pt}} % above code, font and distance
              {\contentslabel[\thecontentslabel.]{1em}} % content label
              {}
              {\titlerule*[5pt]{.}\contentspage} % filler page format
\titlecontents{section}
              [2.5em]
              {\fontsize{10.5pt}{10.5pt}}
              {\contentslabel[\thecontentslabel.]{2em}}
              {}
              {\titlerule*[5pt]{.}\contentspage}
\titlecontents{subsection}
              [5em]
              {\fontsize{10.5pt}{10.5pt}}
              {\contentslabel[\thecontentslabel.]{3em}}
              {}
              {\titlerule*[5pt]{.}\contentspage}



%% 页眉页脚
\RequirePackage{lastpage}
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[CO]{\leftmark}
\fancyhead[CE]{\rightmark}
\fancyfoot[C]{--\thepage--}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand\chaptermark[1]{\markboth{\thechapter.\ #1}{}} % 页眉中的章标题
\renewcommand\sectionmark[1]{\markright{\thesection.\ #1}}  % 页眉中的节标题
%
\fancypagestyle{plain}{
  \fancyfoot[C]{--\thepage--}
}

%%封面
\RequirePackage{tabu}
\RequirePackage{multirow}
\def\maketitle{%
\thispagestyle{empty}
\begin{tabu} to \textwidth{|[1.2pt] X[c]|[0.8pt]X[c]|[1.2pt]}
  \tabucline[1.2pt]{-}
  \textbf{文档名称 Document Name} & \textbf{密级 Confidentiality Level}\\
  \tabucline[0.8pt]{-}
  \textbf{\the\zhtitle} & \textbf{\the\level}\\
  \tabucline[0.8pt]{-}
  \textbf{文档版本 Document Version} & \multirow{2}{*}{\textbf{Total Pages 共$\quad$ \pageref{LastPage} \quad 页}}\\
  \tabucline[0.8pt]{-1}
  \textbf{\the\version} & \\
  \tabucline[1.2pt]{-}
\end{tabu}
\vspace*{4cm}
\begin{center}
\fontsize{36pt}{36pt} {\textbf{\the\zhtitle} }
\end{center}
\vspace*{8cm}
\begin{center}
\tabulinesep = 3mm
\begin{tabu} to 0.6\textwidth{X[c] X[c] }
  \textbf{\Large 项目名称:} & \textbf{\Large \the\group}\\
  \textbf{\Large 项目组长:} & \textbf{\Large \the\leader}\\
  \textbf{\Large 完成日期:} & \textbf{\Large \the\completedata}
\end{tabu}
\end{center}


\vfill
\include{log}
}
