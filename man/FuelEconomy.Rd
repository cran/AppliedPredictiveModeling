\name{FuelEconomy}
\docType{data}
\alias{cars2010}
\alias{cars2011}
\alias{cars2012}
\title{Fuel Economy Data}
\description{
The \url{http://fueleconomy.gov} website, run by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy and the U.S. Environmental Protection Agency, lists different estimates of fuel economy for passenger cars and trucks. For each vehicle, various characteristics are recorded such as the engine displacement or number of cylinders. Along with these values, laboratory measurements are made for the city and highway miles per gallon (MPG) of the car.

Predictors extracted from the website include: \code{EngDispl}, \code{NumCyl}, \code{Transmission}, \code{AirAspirationMethod}, \code{NumGears}, \code{TransLockup}, \code{TransCreeperGear}, \code{DriveDesc}, \code{IntakeValvePerCyl}, \code{ExhaustValvesPerCyl}, \code{CarlineClassDesc}, \code{VarValveTiming} and \code{VarValveLift}. The outcome used in the book is in column  \code{FE} and is the unadjusted highway data.

}

\usage{data(FuelEconomy)}

\value{
  \item{cars2010}{data in cars from model year 2010. }
  \item{cars2011}{cars introduced in 2011 that were not in the model year 2010 data.}
  \item{cars2012}{cars introduced in 2012 that were not in the model year 2010 or 2011 data }
  }

\examples{
data(FuelEconomy)

library(lattice)

### Plot shown in the text:

cars2010 <- cars2010[order(cars2010$EngDispl),]
cars2011 <- cars2011[order(cars2011$EngDispl),]

cars2010a <- cars2010
cars2010a$Year <- "2010 Model Year"
cars2011a <- cars2011
cars2011a$Year <- "2011 Model Year"

plotData <- rbind(cars2010a, cars2011a)

plotTheme <- bookTheme(FALSE)
plotTheme$plot.symbol$col <- rgb(.2, .2, .2, .5)
plotTheme$plot.symbol$cex <- 0.7
trellis.par.set(plotTheme)

xyplot(FE ~ EngDispl|Year, plotData,
       xlab = "Engine Displacement",
       ylab = "Fuel Efficiency (MPG)",
       between = list(x = 1.2))

}

\keyword{datasets}  

