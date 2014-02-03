\name{NEWS}
\title{News for Package \pkg{AppliedPredictiveModeling}}
\newcommand{\cpkg}{\href{http://CRAN.R-project.org/package=#1}{\pkg{#1}}}


\section{Changes in version 1.1-5}{

The package dependencies were updated. Some were moved to 'Imports'

The chapter scripts were re-run with the latest versions of the \cpkg{AppliedPredictiveModeling} and\cpkg{caret} packages. For \cpkg{caret}, the names of the tuning parameter columns were changed to remove the dot. These were made to be consistent with the newer version of \cpkg{caret} and are not required. For example, \code{.sigma} was changed to \code{sigma} and so on. 

Additional changes are:

\itemize{
\item \code{03_Data_Pre_Processing.R} was changed to include code from the Computing section on creating dummy variables.

\item \code{04_Over_Fitting.R} was modified. Changes were made to:
   \itemize{
   \item avoid a warning message when the SVM grid was created
   \item \code{classProbs = TRUE} was added to the control function for the object \code{svmFit}
   \item some notes were made in the potential differences in SVM results between versions of \cpkg{caret}
   }
\item \code{06_Linear_Regression.R} was updated to use the newer "ridge" model in \cpkg{caret}.

\item In \code{07_Non-Linear_Reg.R}, some notes were made in the potential differences in SVM results between versions of \cpkg{caret}

\item In \code{16_Class_Imbalance.R}, verboseness (verbosity?) was turned off to make the results cleaner. Also, the code for \code{svmWtFit} was using the wrong tuning grid (\code{svmGrid2} instead of \code{svmGrid1}).

\item \code{19_Feature_Select.R} was changed so that the resampling values were only saved for the final model. 

}
}

\section{Changes in version 1.1-4}{
\itemize{
\item The data set \code{ChemicalManufacturingProcess} did not contain 
the rows with missing data. They were added back in.

\item Small changes to conform to R CMD check. 
}}

\section{Changes in version 1.1-2}{
\itemize{
\item Code to create the \code{carsSubset} object in Seciton 3.8 was added 
      to 03_Data_Pre_Processing.R
}}

\section{Changes in version 1.1-1}{
\itemize{
\item Initial Version
}}

