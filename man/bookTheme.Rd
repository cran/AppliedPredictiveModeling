\name{bookTheme}
\alias{bookTheme}
\alias{transparentTheme}
\title{
Lattice Themes
}
\description{
Two \pkg{lattice} themes used throughout the book.
}
\usage{
bookTheme(set = TRUE)

transparentTheme(set = TRUE, pchSize = 1, trans = 0.2)
}
\arguments{
  \item{set}{
a logical: should these settings be applied to the current device?
}
  \item{pchSize}{
the size of the plot symbols
}
  \item{trans}{
the amount of transparency (via the alpha channel). Note that transparency is not supported by all graphics devices.
}
}
\details{
When using these functions to save a plot, make sure to invoke them after the device has been opened (e.g. after calls such as \code{pdf()}.

}
\value{
Each function returns a list of theme parameters. See Sarkar (2008) or \code{\link[lattice]{trellis.par.get}} for specific details. 
}
\references{
Some of the colors are based on values from ColorBrewer \url{http://www.colorbrewer.org}.

Sarkar, D. (2008). Lattice: Multivariate Data Visualization with R. UseR! (1st ed. p. 286). Springer.
}
\author{
Max Kuhn
}

\examples{
library(lattice)

example <- quadBoundaryFunc(100)

bookTheme(set = TRUE)
xyplot(X2 ~ X1, data = example, groups = class, auto.key = TRUE)

transparentTheme(set = TRUE, trans = .6)
xyplot(X2 ~ X1, data = example, groups = class, auto.key = TRUE)
}

\keyword{hplot}

