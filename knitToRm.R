library(knitr)

# Set current directory
# Session -> Set Working Directory -> To Source File Location

knit.rmd.file <- function(input.path, output.path){
  VAR <- ""
  knitr::knit(input=input.path, output=output.path)
  # knitr::knit(input="RMD/FAA-Strategy/FAA_Explanation.Rmd", output="_posts/2016-05-16-FAA_Explanation.md")
  VAR <- "/"
  knitr::knit(input=input.path, output=output.path)
  #knitr::knit(input="RMD/FAA-Strategy/FAA_Explanation.Rmd", output="_posts/2016-05-16-FAA_Explanation.md")
}

# 1. Change knit function input directory to select the location of the RMarkdown file ("RMD/xxxxx/Name.Rmd")
# 2. Change knit function output directory of the Markdown file ("/_posts/YYYY-MM-DD-Name.md")
# 3. Within the .Rmd file, change the fig.path to paste0(VAR, 'public/images/YYYY-MM-DD-Name/')
# ***Note for 3. fig.path is an attribute within opts_chunk$set(fig.path=..., message=..., warning=..., echo=...)***
# 4. Add the following to the TOP of the .Rmd file: ***NEED THE LAYOUT: POST***
# ---
#  layout: post
#  title: "Name"
# ---

knit.rmd.file("RMD/FAA-Strategy/FAA_Explanation.Rmd", "_posts/2016-05-16-FAA_Explanation.md")
knit.rmd.file("RMD/Markowitz - Ilya/2015_06_05_Markowitz.Rmd", "_posts/2015-06-05-Markowitz.md")
knit.rmd.file("RMD/Rotation/Rotation_Strategy.Rmd", "_posts/2015-05-18-Rotation_Strategy.md")
knit.rmd.file("RMD/Momentum Markowitz - SIT/Momentum_Markowitz_Post.Rmd", "_posts/2015-06-04-Momentum_Markowitz.md")
knit.rmd.file("RMD/COGS_to_Revenue/COGS_to_Revenue.Rmd", "_posts/2016-05-16-COGS_to_Revenue.md")


