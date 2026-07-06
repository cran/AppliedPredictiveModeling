#' Lattice Themes
#'
#' Two \pkg{lattice} themes used throughout the book.
#'
#' When using these functions to save a plot, make sure to invoke them after
#' the device has been opened (e.g. after calls such as \code{pdf()}.
#'
#' @aliases bookTheme transparentTheme
#' @param set a logical: should these settings be applied to the current
#' device?
#' @param pchSize the size of the plot symbols
#' @param trans the amount of transparency (via the alpha channel). Note that
#' transparency is not supported by all graphics devices.
#' @return Each function returns a list of theme parameters. See Sarkar (2008)
#' or \code{\link[lattice]{trellis.par.get}} for specific details.
#' @author Max Kuhn
#' @references Some of the colors are based on values from ColorBrewer
#' \url{https://colorbrewer2.org/}.
#'
#' Sarkar, D. (2008). Lattice: Multivariate Data Visualization with R. UseR!
#' (1st ed. p. 286). Springer.
#' @keywords hplot
#' @examples
#'
#' library(lattice)
#'
#' example <- quadBoundaryFunc(100)
#'
#' bookTheme(set = TRUE)
#' xyplot(X2 ~ X1, data = example, groups = class, auto.key = TRUE)
#'
#' transparentTheme(set = TRUE, trans = .6)
#' xyplot(X2 ~ X1, data = example, groups = class, auto.key = TRUE)
#'
#' @export bookTheme
bookTheme <- function(set = TRUE) {
  theme <- list(
    plot.polygon = list(
      alpha = 1,
      col = "aliceblue",
      border = "black",
      lty = 1,
      lwd = 1
    ),
    background = list(col = "transparent"),
    bar.fill = list(col = "#cce6ff"),
    box.rectangle = list(col = "black"),
    box.umbrella = list(col = "black"),
    dot.line = list(col = "#e8e8e8"),
    dot.symbol = list(col = "black"),
    plot.line = list(col = "black", lwd = 1, lty = 1),
    plot.symbol = list(col = "black", pch = 16),
    regions = list(
      col = c(
        "#FEF8FA",
        "#FDF6F9",
        "#FBF5F9",
        "#FAF3F8",
        "#F8F2F7",
        "#F7F0F7",
        "#F5EEF6",
        "#F4EDF5",
        "#F2EBF5",
        "#F1EAF4",
        "#EFE8F3",
        "#EDE7F2",
        "#ECE5F1",
        "#EAE4F1",
        "#E8E2F0",
        "#E6E1EF",
        "#E4DFEE",
        "#E2DEED",
        "#E0DCEC",
        "#DEDAEB",
        "#DCD9EA",
        "#D9D7E9",
        "#D7D6E8",
        "#D4D4E7",
        "#D1D2E6",
        "#CED1E5",
        "#CCCFE4",
        "#C8CEE3",
        "#C5CCE2",
        "#C2CAE1",
        "#BFC9E0",
        "#BBC7DF",
        "#B8C5DF",
        "#B4C4DE",
        "#B1C2DD",
        "#ADC0DC",
        "#A9BFDB",
        "#A6BDDA",
        "#A2BBD9",
        "#9EB9D9",
        "#9BB8D8",
        "#97B6D7",
        "#93B4D6",
        "#8FB2D5",
        "#8BB0D4",
        "#87AFD3",
        "#83ADD2",
        "#7FABD1",
        "#7AA9D0",
        "#76A7CF",
        "#71A5CE",
        "#6CA3CC",
        "#68A1CB",
        "#63A0CA",
        "#5D9EC9",
        "#589CC8",
        "#539AC6",
        "#4E98C5",
        "#4996C4",
        "#4493C3",
        "#3F91C1",
        "#3A8FC0",
        "#358DBF",
        "#308BBE",
        "#2C89BD",
        "#2887BC",
        "#2385BB",
        "#1F83BA",
        "#1C80B9",
        "#187EB7",
        "#157CB6",
        "#127AB5",
        "#0F78B3",
        "#0D76B2",
        "#0A73B0",
        "#0971AE",
        "#076FAC",
        "#066DAA",
        "#056AA7",
        "#0568A5"
      )
    ),
    strip.shingle = list(
      col = c(
        "#ff7f00",
        "#00ff00",
        "#00ffff",
        "#ff00ff",
        "#ff0000",
        "#ffff00",
        "#0080ff"
      )
    ),
    strip.background = list(
      col = c(
        "#ffe5cc",
        "#ccffcc",
        "#ccffff",
        "#ffccff",
        "#ffcccc",
        "#ffffcc",
        "#cce6ff"
      )
    ),
    reference.line = list(col = "#e8e8e8"),
    superpose.line = list(
      col = c(
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black"
      ),
      lty = rep(1:6, each = 6)
    ),
    superpose.symbol = list(
      pch = c(
        1,
        4,
        6,
        0,
        5,
        17,
        4,
        6,
        0,
        5,
        17,
        1,
        6,
        0,
        5,
        17,
        1,
        4,
        0,
        5,
        17,
        1,
        4,
        6,
        5,
        17,
        1,
        4,
        6,
        0,
        17,
        1,
        4,
        6,
        0,
        5
      ),
      cex = rep(0.7, 6 * 6),
      col = c(
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black",
        "#053061",
        "#B2182B",
        "#F46D43",
        "#5E4FA2",
        "#66C2A5",
        "black"
      )
    )
  )

  if (set) {
    lattice::trellis.par.set(theme)
  }
  invisible(theme)
}

#' @rdname bookTheme
#' @export
transparentTheme <-
  function(set = TRUE, pchSize = 1, trans = .2) {
    theme <- list(
      plot.polygon = list(
        alpha = 1,
        col = "aliceblue",
        border = "black",
        lty = 1,
        lwd = 1
      ),
      background = list(col = "transparent"),
      bar.fill = list(col = "#cce6ff"),
      box.rectangle = list(col = "black"),
      box.umbrella = list(col = "black"),
      dot.line = list(col = "#e8e8e8"),
      dot.symbol = list(col = "black"),
      plot.line = list(col = "black"),
      plot.symbol = list(col = "black"),
      regions = list(
        col = c(
          "#FEF8FA",
          "#FDF6F9",
          "#FBF5F9",
          "#FAF3F8",
          "#F8F2F7",
          "#F7F0F7",
          "#F5EEF6",
          "#F4EDF5",
          "#F2EBF5",
          "#F1EAF4",
          "#EFE8F3",
          "#EDE7F2",
          "#ECE5F1",
          "#EAE4F1",
          "#E8E2F0",
          "#E6E1EF",
          "#E4DFEE",
          "#E2DEED",
          "#E0DCEC",
          "#DEDAEB",
          "#DCD9EA",
          "#D9D7E9",
          "#D7D6E8",
          "#D4D4E7",
          "#D1D2E6",
          "#CED1E5",
          "#CCCFE4",
          "#C8CEE3",
          "#C5CCE2",
          "#C2CAE1",
          "#BFC9E0",
          "#BBC7DF",
          "#B8C5DF",
          "#B4C4DE",
          "#B1C2DD",
          "#ADC0DC",
          "#A9BFDB",
          "#A6BDDA",
          "#A2BBD9",
          "#9EB9D9",
          "#9BB8D8",
          "#97B6D7",
          "#93B4D6",
          "#8FB2D5",
          "#8BB0D4",
          "#87AFD3",
          "#83ADD2",
          "#7FABD1",
          "#7AA9D0",
          "#76A7CF",
          "#71A5CE",
          "#6CA3CC",
          "#68A1CB",
          "#63A0CA",
          "#5D9EC9",
          "#589CC8",
          "#539AC6",
          "#4E98C5",
          "#4996C4",
          "#4493C3",
          "#3F91C1",
          "#3A8FC0",
          "#358DBF",
          "#308BBE",
          "#2C89BD",
          "#2887BC",
          "#2385BB",
          "#1F83BA",
          "#1C80B9",
          "#187EB7",
          "#157CB6",
          "#127AB5",
          "#0F78B3",
          "#0D76B2",
          "#0A73B0",
          "#0971AE",
          "#076FAC",
          "#066DAA",
          "#056AA7",
          "#0568A5"
        )
      ),
      strip.shingle = list(
        col = c(
          "#ff7f00",
          "#00ff00",
          "#00ffff",
          "#ff00ff",
          "#ff0000",
          "#ffff00",
          "#0080ff"
        )
      ),
      strip.background = list(
        col = c(
          "#ffe5cc",
          "#ccffcc",
          "#ccffff",
          "#ffccff",
          "#ffcccc",
          "#ffffcc",
          "#cce6ff"
        )
      ),
      reference.line = list(col = "#e8e8e8"),
      superpose.line = list(
        col = c(
          grDevices::rgb(1, 0, 0, trans),
          grDevices::rgb(0, 0, 1, trans),
          grDevices::rgb(0.3984375, 0.7578125, 0.6445312, max(.6, trans)),
          grDevices::rgb(0, 0, 0, trans)
        ),
        lty = rep(1:2, 6)
      ),
      superpose.symbol = list(
        pch = c(16, 15, 17, 18, 16),
        cex = rep(pchSize, 5),
        col = c(
          grDevices::rgb(1, 0, 0, trans),
          grDevices::rgb(0, 0, 1, trans),
          grDevices::rgb(0.3984375, 0.7578125, 0.6445312, max(.6, trans)),
          grDevices::rgb(0, 0, 0, trans)
        )
      )
    )

    if (set) {
      lattice::trellis.par.set(theme, warn = FALSE)
    }
    invisible(theme)
  }
