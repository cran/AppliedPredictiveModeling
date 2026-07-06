upperp <- function(...) {
  args <- list(...)
  circ1 <- ellipse::ellipse(diag(rep(1, 2)), t = .1)
  lattice::panel.xyplot(
    circ1[, 1],
    circ1[, 2],
    type = "l",
    lty = lattice::trellis.par.get("reference.line")$lty,
    col = lattice::trellis.par.get("reference.line")$col,
    lwd = lattice::trellis.par.get("reference.line")$lwd
  )
  circ2 <- ellipse::ellipse(diag(rep(1, 2)), t = .2)
  lattice::panel.xyplot(
    circ2[, 1],
    circ2[, 2],
    type = "l",
    lty = lattice::trellis.par.get("reference.line")$lty,
    col = lattice::trellis.par.get("reference.line")$col,
    lwd = lattice::trellis.par.get("reference.line")$lwd
  )
  circ3 <- ellipse::ellipse(diag(rep(1, 2)), t = .3)
  lattice::panel.xyplot(
    circ3[, 1],
    circ3[, 2],
    type = "l",
    lty = lattice::trellis.par.get("reference.line")$lty,
    col = lattice::trellis.par.get("reference.line")$col,
    lwd = lattice::trellis.par.get("reference.line")$lwd
  )
  lattice::panel.xyplot(args$x,
                        args$y,
                        groups = args$groups,
                        subscripts = args$subscripts)
}

lowerp <- function(...) {
  
}
