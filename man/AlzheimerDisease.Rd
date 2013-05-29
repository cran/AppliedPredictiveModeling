\name{AlzheimerDisease}
\docType{data}
\alias{diagnosis}
\alias{predictors}
\title{Alzheimer's Disease CSF Data}
\description{
Washington University conducted a clinical study to determine if biological measurements made from cerebrospinal fluid (CSF) can be used to diagnose or predict Alzheimer's disease (Craig-Schapiro et al. 2011). These data are a modified version of the values used for the publication.

The R factor vector \code{diagnosis} contains the outcome data for 333 of the subjects. The demographic and laboratory results are collected in the data frame \code{predictors}.

One important indicator of Alzheimer's disease is the genetic background of a subject. In particular, what versions of the Apolipoprotein E gene inherited from one's parents has an association with the disease. There are three variants of the gene: E2, E3 and E4. Since a child inherits a version of the gene from each parent, there are six possible combinations (e.g. E2/E2, E2/E3, and so on). This data is contained in the predictor column named \code{Genotype}.
  
}

\usage{data(AlzheimerDisease)}

\value{
  \item{diagnosis}{labels for the patients, either "Impaired" or "Control". }
  \item{predictors}{predictors for demographic data (eg. age, gender), genotype and assay results.}
  }


\source{Craig-Schapiro, R., Kuhn, M., Xiong, C., Pickering, E. H., Liu, J., Misko, T. P., Perrin, R. J., et al. (2011). Multiplexed Immunoassay Panel Identifies Novel CSF Biomarkers for Alzheimer's Disease Diagnosis and Prognosis. PLoS ONE, 6(4), e18850.

}

\examples{
data(AlzheimerDisease)
}

\keyword{datasets}  

