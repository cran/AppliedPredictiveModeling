\name{getPackages}
\alias{getPackages}
\title{
Install Packages for Each Chapter
}
\description{
This function identifies the physical location on the user's computer where the chapter R scripts are located.
}
\usage{
getPackages(chapter, ...)
}

\arguments{
  \item{chapter}{an integer vector (or character versions of the integer) for the chapter number. See Details below:}
  \item{\ldots}{options to pass to \code{\link[utils]{install.packages}}}
}

\details{
Chapter names and packages. about dependencies. 
}

\author{
Max Kuhn
}

\examples{
\dontrun{
getPackages(2)
getPackages(2:3)
getPackages("4")
}
}

\keyword{utilities}
