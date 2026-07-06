#' Abalone Data
#'
#' The Abalone data consist of data from 4177 abalones. The data consist of
#' measurements of the type (male, female and infant), the longest shell
#' measurement, the diameter, height and several weights (whole, shucked,
#' viscera and shell). The outcome is the number of rings. The age of the
#' abalone is the number of rings plus 1.5.
#'
#' The data are taken from the UCI database
#' (\url{https://archive.ics.uci.edu/ml/datasets/Abalone}).
#'
#'
#' @name abalone
#' @docType data
#' @return \item{abalone}{a data frame with 4177 rows and 9 columns}
#' @keywords datasets
#' @examples
#'
#' data(abalone)
#'
NULL

#' Alzheimer's Disease CSF Data
#'
#' Washington University conducted a clinical study to determine if biological
#' measurements made from cerebrospinal fluid (CSF) can be used to diagnose or
#' predict Alzheimer's disease (Craig-Schapiro et al. 2011). These data are a
#' modified version of the values used for the publication.
#'
#' The R factor vector \code{diagnosis} contains the outcome data for 333 of
#' the subjects. The demographic and laboratory results are collected in the
#' data frame \code{predictors}.
#'
#' One important indicator of Alzheimer's disease is the genetic background of
#' a subject. In particular, what versions of the Apolipoprotein E gene
#' inherited from one's parents has an association with the disease. There are
#' three variants of the gene: E2, E3 and E4. Since a child inherits a version
#' of the gene from each parent, there are six possible combinations (e.g.
#' E2/E2, E2/E3, and so on). This data is contained in the predictor column
#' named \code{Genotype}.
#'
#'
#' @name AlzheimerDisease
#' @aliases diagnosis predictors
#' @docType data
#' @return \item{diagnosis}{labels for the patients, either "Impaired" or
#' "Control". } \item{predictors}{predictors for demographic data (eg. age,
#' gender), genotype and assay results.}
#' @source Craig-Schapiro, R., Kuhn, M., Xiong, C., Pickering, E. H., Liu, J.,
#' Misko, T. P., Perrin, R. J., et al. (2011). Multiplexed Immunoassay Panel
#' Identifies Novel CSF Biomarkers for Alzheimer's Disease Diagnosis and
#' Prognosis. PLoS ONE, 6(4), e18850.
#' @keywords datasets
#' @examples
#'
#' data(AlzheimerDisease)
#'
NULL

#' Chemical Manufacturing Process Data
#'
#' This data set contains information about a chemical manufacturing process,
#' in which the goal is to understand the relationship between the process and
#' the resulting final product yield.  Raw material in this process is put
#' through a sequence of 27 steps to generate the final pharmaceutical product.
#' The starting material is generated from a biological unit and has a range of
#' quality and characteristics.  The objective in this project was to develop a
#' model to predict percent yield of the manufacturing process.  The data set
#' consisted of 177 samples of biological material for which 57 characteristics
#' were measured.  Of the 57 characteristics, there were 12 measurements of the
#' biological starting material, and 45 measurements of the manufacturing
#' process.  The process variables included measurements such as temperature,
#' drying time, washing time, and concentrations of by--products at various
#' steps.  Some of the process measurements can be controlled, while others are
#' observed.  Predictors are continuous, count, categorical; some are
#' correlated, and some contain missing values.  Samples are not independent
#' because sets of samples come from the same batch of biological starting
#' material.
#'
#'
#' @name ChemicalManufacturingProcess
#' @docType data
#' @return \code{ChemicalManufacturingProcess}: a data frame with columns for
#' the outcome (\code{Yield}) and the predictors (\code{BiologicalMaterial01}
#' though \code{BiologicalMaterial12} and \code{ManufacturingProcess01} though
#' \code{ManufacturingProcess45}
#' @keywords datasets
#' @examples
#'
#' data(ChemicalManufacturingProcess)
#'
NULL


#' Compressive Strength of Concrete from Yeh (1998)
#'
#' Yeh (1998) describes a collection of data sets from different sources that
#' can be used for modeling the compressive strength of concrete formulations
#' as a functions of their ingredients and age.
#'
#' The data are from Yeh (1998) and taken from the UCI ML website
#' \url{https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength}.
#'
#' There are 1030 data points from the UCI website, but the paper states that
#' approximately 1,000 samples were made, but only 727 were analyzed in the
#' source material. It is unclear which samples were excluded.
#'
#' @name concrete
#' @aliases concrete mixtures
#' @docType data
#' @return \item{concrete}{data frame of data with predictor columns
#' \code{Cement}, \code{BlastFurnaceSlag}, \code{FlyAsh}, \code{Water},
#' \code{Superplasticizer}, \code{CoarseAggregate}, \code{FineAggregate} and
#' \code{Age} with response column \code{CompressiveStrength}. These are the
#' amounts.} \item{mixtures}{The same data where all the ingredients have been
#' converted to proportions of the total amounts.}
#' @source Yeh, I. C. (1998). Modeling of strength of high-performance concrete
#' using artificial neural networks. \emph{Cement and Concrete Research},
#' 28(12), 1797-1808. Elsevier.
#' @keywords datasets
#' @examples
#'
#' if (require("caret")) {
#'   data(concrete)
#'
#'   library(caret)
#'
#'   ### Split used in the book:
#'   set.seed(975)
#'   inTrain <- createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
#'   training <- mixtures[ inTrain,]
#'   testing  <- mixtures[-inTrain,]
#' }
#'
#'
NULL


#' Fuel Economy Data
#'
#' The \url{https://fueleconomy.gov} website, run by the U.S. Department of
#' Energy's Office of Energy Efficiency and Renewable Energy and the U.S.
#' Environmental Protection Agency, lists different estimates of fuel economy
#' for passenger cars and trucks. For each vehicle, various characteristics are
#' recorded such as the engine displacement or number of cylinders. Along with
#' these values, laboratory measurements are made for the city and highway
#' miles per gallon (MPG) of the car.
#'
#' Predictors extracted from the website include: \code{EngDispl},
#' \code{NumCyl}, \code{Transmission}, \code{AirAspirationMethod},
#' \code{NumGears}, \code{TransLockup}, \code{TransCreeperGear},
#' \code{DriveDesc}, \code{IntakeValvePerCyl}, \code{ExhaustValvesPerCyl},
#' \code{CarlineClassDesc}, \code{VarValveTiming} and \code{VarValveLift}. The
#' outcome used in the book is in column \code{FE} and is the unadjusted
#' highway data.
#'
#'
#' @name FuelEconomy
#' @aliases cars2010 cars2011 cars2012
#' @docType data
#' @return \item{cars2010}{data in cars from model year 2010. }
#' \item{cars2011}{cars introduced in 2011 that were not in the model year 2010
#' data.} \item{cars2012}{cars introduced in 2012 that were not in the model
#' year 2010 or 2011 data }
#' @keywords datasets
#' @examples
#'
#' data(FuelEconomy)
#'
#' library(lattice)
#'
#' ### Plot shown in the text:
#'
#' cars2010 <- cars2010[order(cars2010$EngDispl),]
#' cars2011 <- cars2011[order(cars2011$EngDispl),]
#'
#' cars2010a <- cars2010
#' cars2010a$Year <- "2010 Model Year"
#' cars2011a <- cars2011
#' cars2011a$Year <- "2011 Model Year"
#'
#' plotData <- rbind(cars2010a, cars2011a)
#'
#' plotTheme <- bookTheme(FALSE)
#' plotTheme$plot.symbol$col <- rgb(.2, .2, .2, .5)
#' plotTheme$plot.symbol$cex <- 0.7
#' trellis.par.set(plotTheme)
#'
#' xyplot(FE ~ EngDispl|Year, plotData,
#'        xlab = "Engine Displacement",
#'        ylab = "Fuel Efficiency (MPG)",
#'        between = list(x = 1.2))
#'
#'
NULL


#' Hepatic Injury Data
#'
#' This data set was used to develop a model for predicting compounds'
#' probability of causing hepatic injury (i.e. liver damage). This data set
#' consisted of 281 unique compounds; 376 predictors were measured or computed
#' for each. The response was categorical (either "None", "Mild" or "Severe"
#' ),and was highly unbalanced.
#'
#' This kind of response often occurs in pharmaceutical data because companies
#' steer away from creating molecules that have undesirable characteristics.
#' Therefore, well-behaved molecules often greatly outnumber undesirable
#' molecules. The predictors consisted of measurements from 184 biological
#' screens and 192 chemical feature predictors. The biological predictors
#' represent activity for each screen and take values between 0 and 10 with a
#' mode of 4. The chemical feature predictors represent counts of important
#' sub-structures as well as measures of physical properties that are thought
#' to be associated with hepatic injury.
#'
#'
#' @name hepatic
#' @aliases bio chem injury
#' @docType data
#' @return \item{bio}{Biological screen results. } \item{chem}{Chemical
#' fingerprints for sub-structures.} \item{injury}{A factor vector of
#' outcomes.}
#' @keywords datasets
#' @examples
#'
#' data(hepatic)
#'
NULL


#' Internal Functions
#'
#' Internal functions
#'
#' @name apm-internal
#' @aliases lowerp upperp
#' @param \dots optional arguments to pass to internal functions
#' @author Max Kuhn
#' @keywords internal
NULL


#' HPC Job Scheduling Data
#'
#' These data consist of information on 4331 jobs in a high performance
#' computing environment. Seven attributes were recorded for each job along
#' with a discrete class describing the execution time.
#'
#' The predictors are: \code{Protocol} (the type of computation),
#' \code{Compounds} (the number of data points for each jobs),
#' \code{InputFields} (the number of characteristic being estimated),
#' \code{Iterations} (maximum number of iterations for the computations),
#' \code{NumPending} (the number of other jobs pending at the time of launch),
#' \code{Hour} (decimal hour of day for launch time) and \code{Day} (of launch
#' time).
#'
#' The classes are: \code{VF} (very fast), \code{F} (fast), \code{M} (moderate)
#' and \code{L} (long).
#'
#'
#' @name schedulingData
#' @docType data
#' @return \item{schedulingData}{a data frame with 4331 rows and 8 columns}
#' @keywords datasets
#' @examples
#'
#' data(schedulingData)
#'
#' library(caret)
#'
#' set.seed(1104)
#' inTrain <- createDataPartition(schedulingData$Class, p = .8, list = FALSE)
#'
#' schedulingData$NumPending <- schedulingData$NumPending + 1
#'
#' trainData <- schedulingData[ inTrain,]
#' testData  <- schedulingData[-inTrain,]
#'
#' modForm <- as.formula(Class ~ Protocol + log10(Compounds) +
#'                       log10(InputFields)+ log10(Iterations) +
#'                       log10(NumPending) + Hour + Day)
#'
#'
#'
NULL


#' Logistic Regression Predictions for the Credit Data
#'
#' add some notes
#'
#'
#' @name logisticCreditPredictions
#' @docType data
#' @return A data frame with columns \item{Bad}{The predicted class probability
#' for bad credit. } \item{Good}{The predicted class probability for good
#' credit.} \item{pred}{The predicted class. } \item{obs}{The observed class }
#' @keywords datasets
#' @examples
#'
#' ## show code to make the predictions
#'
NULL


#' Permeability Data
#'
#' This pharmaceutical data set was used to develop a model for predicting
#' compounds' permeability.  In short, permeability is the measure of a
#' molecule's ability to cross a membrane.  The body, for example, has notable
#' membranes between the body and brain, known as the blood-brain barrier, and
#' between the gut and body in the intestines.  These membranes help the body
#' guard critical regions from receiving undesirable or detrimental substances.
#' For an orally taken drug to be effective in the brain, it first must pass
#' through the intestinal wall and then must pass through the blood-brain
#' barrier in order to be present for the desired neurological target.
#' Therefore, a compound's ability to permeate relevant biological membranes is
#' critically important to understand early in the drug discovery process.
#' Compounds that appear to be effective for a particular disease in research
#' screening experiments, but appear to be poorly permeable may need to be
#' altered in order improve permeability, and thus the compound's ability to
#' reach the desired target.  Identifying permeability problems can help guide
#' chemists towards better molecules.
#'
#' Permeability assays such as PAMPA and Caco-2 have been developed to help
#' measure compounds' permeability (Kansy et al, 1998).  These screens are
#' effective at quantifying a compound's permeability, but the assay is
#' expensive labor intensive.  Given a sufficient number of compounds that have
#' been screened, we could develop a predictive model for permeability in an
#' attempt to potentially reduce the need for the assay.  In this project there
#' were 165 unique compounds; 1107 molecular fingerprints were determined for
#' each.  A molecular fingerprint is a binary sequence of numbers that
#' represents the presence or absence of a specific molecular sub-structure.
#' The response is highly skewed, the predictors are sparse (15.5 percent are
#' present), and many predictors are strongly associated.
#'
#'
#' @name permeability
#' @aliases permeability fingerprints
#' @docType data
#' @return \item{permeability}{permeability values for each compound. }
#' \item{fingerprints}{a matrix of binary fingerprint indicator variables.}
#' @source Kansy, M., Senner, F., and Gubernator, K. (1998). Physicochemical
#' High Throughput Screening: Parallel Artificial Membrane Permeation Assay in
#' the Description of Passive Absorption Processes. J. Med. Chem, 41(7),
#' 1007-1010.
#' @keywords datasets
#' @examples
#'
#' data(permeability)
#'
#' hist(permeability)
#'
#' summary(apply(fingerprints, 2, mean))
#'
NULL


#' Cell Body Segmentation
#'
#' Hill, LaPan, Li and Haney (2007) develop models to predict which cells in a
#' high content screen were well segmented.  The data consists of 119 imaging
#' measurements on 2019. The original analysis used 1009 for training and 1010
#' as a test set (see the column called \code{Case}).
#'
#' The outcome class is contained in a factor variable called \code{Class} with
#' levels "PS" for poorly segmented and "WS" for well segmented.
#'
#' A pre-processed version of these data can be found in the \pkg{caret}
#' package.
#'
#'
#' @name segmentationOriginal
#' @docType data
#' @return \item{segmentationOriginal}{data frame of cells}
#' @source Hill, LaPan, Li and Haney (2007). Impact of image segmentation on
#' high-content screening data quality for SK-BR-3 cells, \emph{BMC
#' Bioinformatics}, Vol. 8, pg. 340,
#' \url{https://link.springer.com/article/10.1186/1471-2105-8-340}.
#' @keywords datasets
NULL


#' Solubility Data
#'
#' Tetko et al. (2001) and Huuskonen (2000) investigated a set of compounds
#' with corresponding experimental solubility values using complex sets of
#' descriptors. They used linear regression and neural network models to
#' estimate the relationship between chemical structure and solubility. For our
#' analyses, we will use 1267 compounds and a set of more understandable
#' descriptors that fall into one of three groups: 208 binary "fingerprints"
#' that indicate the presence or absence of a particular chemical
#' sub-structure, 16 count descriptors (such as the number of bonds or the
#' number of Bromine atoms) and 4 continuous descriptors (such as molecular
#' weight or surface area).
#'
#'
#' @name solubility
#' @aliases trainX solTestXtrans solTrainY solTestX solTrainX solTrainXtrans solTestY
#' @docType data
#' @return \item{solTrainX}{training set predictors in their natural units. }
#' \item{solTrainXtrans}{training set predictors after transformations for
#' skewness and centering/scaling.} \item{solTrainY}{a vector of log10
#' solubility values for the training set. } \item{solTestX}{test set
#' predictors in their natural units. } \item{solTestXtrans}{test set
#' predictors after the same transformations used on the training set are
#' applied.} \item{solTestY}{a vector of log10 solubility values for the
#' training set. }
#' @source Tetko, I., Tanchuk, V., Kasheva, T., and Villa, A. (2001).
#' Estimation of aqueous solubility of chemical compounds using E-state
#' indices. \emph{Journal of Chemical Information and Computer Sciences},
#' 41(6), 1488-1493.
#'
#' Huuskonen, J. (2000). Estimation of aqueous solubility for a diverse set of
#' organic compounds based on molecular topology. \emph{Journal of Chemical
#' Information and Computer Sciences}, 40(3), 773-777.
#' @keywords datasets
#' @examples
#'
#' data(solubility)
#'
#' library(caret)
#'
#' ### Cross-validation splits used in the book:
#' set.seed(100)
#' indx <- createFolds(solTrainY, returnTrain = TRUE)
#'
#' ### To re-create the transformed version of the data:
#' \dontrun{
#' ## Find the predictors that are not fingerprints
#' contVars <- names(solTrainX)[!grepl("FP", names(solTrainX))]
#' ## Some have zero values, so we need to add one to them so that
#' ## we can use the Box-Cox transformation. Alternatively, we could
#' ## use the Yeo-Johnson transformation without altering the data.
#' contPredTrain <- solTrainX[,contVars] + 1
#' contPredTest  <-  solTestX[,contVars] + 1
#'
#' pp <- preProcess(contPredTrain, method = "BoxCox")
#' contPredTrain <- predict(pp, contPredTrain)
#' contPredTest  <- predict(pp, contPredTest)
#'
#' ## Reassemble the fingerprint data with the transformed values.
#' trainXtrans <- cbind(solTrainX[,grep("FP", names(solTrainX))], contPredTrain)
#' testXtrans  <- cbind( solTestX[,grep("FP", names(solTestX))],  contPredTest)
#'
#' all.equal(trainXtrans, solTrainXtrans)
#' all.equal(testXtrans, solTestXtrans)
#' 	}
#'
#'
NULL

#' Two Class Example Data
#'
#' These data contain two predictors measured for 208 samples. Of these, 111
#' samples are labeled as \code{Class1} and the remaining 97 are \code{Class2}.
#'
#'
#' @name twoClassData
#' @aliases twoClassData classes
#' @docType data
#' @return \item{predictors}{data frame of two predictors} \item{classes}{a
#' factor vector of class labeled}
#' @keywords datasets
#' @examples
#'
#' data(twoClassData)
#'
#' library(lattice)
#' xyplot(PredictorB ~ PredictorA,
#'        data = predictors,
#'        groups = classes,
#'        auto.key = TRUE)
#'
#'
NULL
