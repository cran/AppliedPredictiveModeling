#' Find Chapter Script Files
#' 
#' This function identifies the physical location on the user's computer where
#' the chapter R scripts are located.
#' 
#' 
#' @author Max Kuhn
#' @keywords utilities
#' @examples
#' 
#' scriptLocation()
#' 
#' @export scriptLocation
scriptLocation <-
  function()
    system.file("chapters", package = "AppliedPredictiveModeling")
