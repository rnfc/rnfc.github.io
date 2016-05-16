library(knitr)

#Set current directory
setwd(getwd())

# 1. Change knit function input directory to select the location of the RMarkdown file ("RMD/xxxxx/Name.Rmd")
# 2. Change knit function output directory of the Markdown file ("_posts/YYYY-MM-DD-Name.md")
# 3. Within the .Rmd file, change the fig.path to ('/public/images/YYYY-MM-DD-Name/')
# ***Note for 3. fig.path is an attribute within opts_chunk$set(fig.path=..., message=..., warning=..., echo=...)***
knit(input="RMD/FAA-Strategy/FAA_Explanation.Rmd", output="_posts/2016-05-16-FAA_Explanation.md")