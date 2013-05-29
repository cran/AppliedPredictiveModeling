\name{hepatic}
\docType{data}
\alias{bio}
\alias{chem}
\alias{injury}
\title{Hepatic Injury Data}
\description{
This  data set was used to develop a model for predicting compounds' probability of causing hepatic injury (i.e. liver damage). This data set consisted of 281 unique compounds; 376 predictors were measured or computed for each. The response was categorical (either "None", "Mild" or "Severe" ),and was highly unbalanced.

This kind of response often occurs in pharmaceutical data because companies steer away from creating molecules that have undesirable characteristics. Therefore, well-behaved molecules often greatly outnumber undesirable molecules. The predictors consisted of measurements from 184 biological screens and 192 chemical feature predictors. The biological predictors represent activity for each screen and take values between 0 and 10 with a mode of 4. The chemical feature predictors represent counts of important sub-structures as well as measures of physical properties that are thought to be associated with hepatic injury. 
  
}

\usage{data(hepatic)}

\value{
  \item{bio}{Biological screen results. }
  \item{chem}{Chemical fingerprints for sub-structures.}
  \item{injury}{A factor vector of outcomes.}
  }

\examples{
data(hepatic)
}

\keyword{datasets}  

