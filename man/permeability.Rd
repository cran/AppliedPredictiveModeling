\name{permeability}
\docType{data}
\alias{permeability}
\alias{fingerprints}
\title{Permeability Data}
\description{
This pharmaceutical data set was used to develop a model for predicting compounds' permeability.  In short, permeability is the measure of a molecule's ability to cross a membrane.  The body, for example, has notable membranes between the body and brain, known as the blood-brain barrier, and between the gut and body in the intestines.  These membranes help the body guard critical regions from receiving undesirable or detrimental substances.  For an orally taken drug to be effective in the brain, it first must pass through the intestinal wall and then must pass through the blood-brain barrier in order to be present for the desired neurological target.  Therefore, a compound's ability to permeate relevant biological membranes is critically important to understand early in the drug discovery process.  Compounds that appear to be effective for a particular disease in research screening experiments, but appear to be poorly permeable may need to be altered in order improve permeability, and thus the compound's ability to reach the desired target.  Identifying permeability problems can help guide chemists towards better molecules.

Permeability assays such as PAMPA and Caco-2 have been developed to help measure compounds' permeability (Kansy et al, 1998).  These screens are effective at quantifying a compound's permeability, but the assay is expensive labor intensive.  Given a sufficient number of compounds that have been screened, we could develop a predictive model for permeability in an attempt to potentially reduce the need for the assay.  In this project there were 165 unique compounds; 1107 molecular fingerprints were determined for each.  A molecular fingerprint is a binary sequence of numbers that represents the presence or absence of a specific molecular sub-structure.  The response is highly skewed, the predictors are sparse (15.5 percent are present), and many predictors are strongly associated.
}

\usage{data(permeability)}

\value{
  \item{permeability}{permeability values for each compound. }
  \item{fingerprints}{a matrix of binary fingerprint indicator variables.}
  }


\source{Kansy, M., Senner, F., and Gubernator, K. (1998). Physicochemical High Throughput Screening:  Parallel Artificial Membrane Permeation Assay in the Description of Passive Absorption Processes. J. Med. Chem, 41(7), 1007-1010.
}

\examples{
data(permeability)

hist(permeability)

summary(apply(fingerprints, 2, mean))
}

\keyword{datasets}  

