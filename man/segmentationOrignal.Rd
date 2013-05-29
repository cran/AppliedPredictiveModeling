\name{segmentationOriginal}
\docType{data}
\alias{segmentationOriginal}
 
\title{Cell Body Segmentation}
\description{
Hill, LaPan, Li and Haney (2007) develop models to predict which cells in a high content screen were well segmented. 
The data consists of 119 imaging measurements on 2019. The original analysis used 1009 for training and 1010 as a test set (see the column called \code{Case}). 

The outcome class is contained in a factor variable called \code{Class} with levels "PS" for poorly segmented and "WS" for well segmented.

A pre-processed version of these data can be found in the \pkg{caret} package.
}

\usage{data(segmentationOriginal)}
\value{
  \item{segmentationOriginal}{data frame of cells}
  }

\source{Hill, LaPan, Li and Haney (2007). Impact of image segmentation on high-content screening data quality for SK-BR-3 cells, 
\emph{BMC Bioinformatics}, Vol. 8, pg. 340, \url{http://www.biomedcentral.com/1471-2105/8/340}.
}

\keyword{datasets}  

