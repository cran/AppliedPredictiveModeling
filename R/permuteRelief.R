#' Permutation Statistics for the Relief Algorithm
#'
#' This function uses a permutation approach to determining the relative
#' magnitude of Relief scores (Kira and Rendell, 1992 and Kononenko, 1994).
#'
#' The scores for each predictor are computed using the original data and after
#' outcome data are randomly scrambled (\code{nprem} times). The mean and
#' standard deviation of the permuted values are determined and a standardized
#' version of the observed scores are determined by subtracting the permuted
#' means from the original values, then dividing each by the corresponding
#' standard deviation.
#'
#' @param x a data frame of predictor data
#' @param y a vector of outcomes
#' @param nperm the number of random permutations of the data
#' @param \dots options to pass to \code{\link[CORElearn]{attrEval}}, such as
#' the exact Relief algorithm, to use
#' @return a list with elements \item{standardized }{a vector of standardized
#' predictor scores} \item{permutations }{the values of the permuted scores,
#' for plotting to assess the permutation distribution} \item{observed}{the
#' observed scores} \item{options}{a list of options passed using \ldots{}}
#' @author Max Kuhn
#' @seealso \code{\link[CORElearn]{attrEval}}
#' @references Kira, K., & Rendell, L. (1992). The feature selection problem:
#' Traditional methods and a new algorithm. \emph{Proceedings of the Eleventh
#' International Conference on Machine Learning}, 129-129.
#'
#' Kononenko, I. (1994). Estimating attributes: analysis and extensions of
#' RELIEF. Machine Learning: ECML-94, 171-182.
#' @keywords htest
#' @examples
#'
#' # CORElearn was removed from CRAN in July 2026; please install from GitHub
#' # at https://github.com/cran/CORElearn
#' if (rlang::is_installed("CORElearn")) {
#'   set.seed(874)
#'   reliefEx3 <- easyBoundaryFunc(500)
#'   reliefEx3$X1 <- scale(reliefEx3$X1)
#'   reliefEx3$X2 <- scale(reliefEx3$X2)
#'   reliefEx3$prob <- NULL
#'
#'   standardized <- permuteRelief(
#'     reliefEx3[, 1:2],
#'     reliefEx3$class,
#'     ## For efficiency, a small number of
#'     ## permutations are used here.
#'     nperm = 50,
#'     estimator = "ReliefFequalK",
#'     ReliefIterations = 50
#'   )
#'   #'
#' }
#'
#' @export permuteRelief
permuteRelief <-
  function(x, y, nperm = 100, ...) {
    dat <- x
    dat$y <- y

    if (!rlang::is_installed("CORElearn")) {
      cli::cli_abort(
        "The {.pkg CORElearn} is require for this function. It is not on CRAN 
        but you can try to install it using the {.pkg pak} package via
        {.code pak::pak('cran/CORElearn')}."
      )
    }
    cl <- rlang::call2(
      "attrEval",
      .ns = "CORElearn",
      y ~ .,
      data = quote(dat),
      ...
    )
    obs <- rlang::eval_tidy(cl)
    permuted <- matrix(NA, ncol = length(obs), nrow = nperm)
    colnames(permuted) <- names(obs)
    for (i in 1:nperm) {
      dat$y <- sample(y)
      permuted[i, ] <- rlang::eval_tidy(cl)
    }
    means <- colMeans(permuted)
    sds <- apply(permuted, 2, stats::sd)
    permuted <- reshape2::melt(permuted)
    names(permuted)[2] <- "Predictor"
    permuted$X1 <- NULL
    list(
      standardized = (obs - means) / sds,
      permutations = permuted,
      observed = obs,
      options = list(...)
    )
  }
