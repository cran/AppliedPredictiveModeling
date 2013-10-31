permuteRelief <-
function(x, y, nperm = 100, ...)
  {
    dat <- x
    dat$y <- y
    
    obs <- attrEval(y ~ ., data = dat, ...)
    permuted <- matrix(NA, ncol = length(obs), nrow = nperm)
    colnames(permuted) <- names(obs)
    for(i in 1:nperm)
      {
        dat$y <- sample(y)
        permuted[i,] <- attrEval(y ~ ., data = dat, ...)
      }
    means <- colMeans(permuted)
    sds <- apply(permuted, 2, sd)
    permuted <- melt(permuted)
    names(permuted)[2] <- "Predictor"
    permuted$X1 <- NULL
    list(standardized = (obs - means)/sds,
         permutations = permuted,
         observed = obs,
         options = list(...))
  }

