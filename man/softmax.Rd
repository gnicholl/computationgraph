% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/operations.R
\name{softmax}
\alias{softmax}
\title{Row-wise softmax function}
\usage{
env_fwd$softmax(X)

env_bwd$softmax(X,prev)
}
\description{
\deqn{\text{softmax}(X) = \dfrac{\exp(X)}{\exp(X) \bigotimes 1_{J \times J}}}
where \eqn{\bigotimes} is standard matrix multiplication,
\eqn{J} is the number of columns in \eqn{X}, and
\eqn{1_{J \times J}} is a \eqn{J \times J} matrix of ones.
}
