
#' @name ForwardBackwardFunctions
#' @rdname ForwardBackwardFunctions
#' @title Forward and backward functions for operations
#'
#' @description Environments which hold forward and backward functions for each operation.
#' Operations should have the same name in both environments. The backward version
#' of the operation should have one additional argument `prev` which is the
#' partial derivative from the previous step in the backward propagation.
#' For binary operators, the backward function should output a list with
#' two entries representing the derivatives with respect to each argument.
#' See [computationgraph::Operators] for list of built-in operators.
#'
#' @usage names(env_fwd)
#' @usage env_fwd$operatorname
#'
#' @format `env_fwd` is an environment containing the forward functions of each operator.
#'
#' @examples
#' # Here's an example of how this package defines the
#' # forward and backward functions for matrix multiplication.
#' env_fwd$`%*%` = base::`%*%`
#' env_bwd$`%*%` = function(L,R,prev) {
#'   return(list(
#'     L=base::tcrossprod(prev,R),
#'     R=base::crossprod(L,prev)
#'   ))
#' }
#' env_fwd$crossprod = base::crossprod
#' env_bwd$crossprod = function(L,R,prev) {
#'   return(list(
#'     L=base::tcrossprod(R,prev),
#'     R=base::`%*%`(L,prev)
#'   ))
#' }
#' env_fwd$tcrossprod = base::tcrossprod
#' env_bwd$tcrossprod = function(L,R,prev) {
#'   return(list(
#'     L=base::`%*%`(prev,R),
#'     R=base::crossprod(prev,L)
#'   ))
#' }
#'
#' # And here's an example with a unary operator (softmax)
#' env_fwd$softmax = function(X) {
#'   ones = matrix(1,nrow=ncol(X),ncol=ncol(X))
#'   return(exp(X) / (exp(X) %*% ones))
#' }
#' env_bwd$softmax = function(X, prev) {
#'   ones = matrix(1,nrow=ncol(X),ncol=ncol(X))
#'   top = exp(X); bottom = exp(X) %*% ones
#'   tmp1 = prev*(top/bottom)
#'   tmp2 = prev*top*(-(1/bottom/bottom))
#'   tmp2 = tmp2 %*% ones
#'   tmp2 = tmp2*top
#'   return(tmp1 + tmp2)
#' }
#'
#' @export
env_fwd = new.env()

#' @rdname ForwardBackwardFunctions
#' @usage names(env_bwd)
#' @usage env_bwd$operatorname
#'
#' @format `env_bwd` is an environment containing the backward functions of each operator.
#'
#' @export
env_bwd = new.env()






#' @name Operators
#' @rdname Operators
#' @title Operators with pre-defined forward and backward functions.
#'
#' @description Below is listed the operators with pre-defined forward and backward
#' functions contained in `env_fwd` and `env_bwd`, respectively. See also \link[=elem_wise]{elem_wise}
#' for how to deal with operators (either unary or binary) which are applied element
#' by element to an input matrix.
#'
#' @details base R operators:
#' * `%*%`
#' * `crossprod`
#' * `tcrossprod`
#' * `colMeans`
#'
#' operators defined in this package:
#' * \link[=loss_L2]{L2 Loss}
#' * \link[=softmax]{softmax}
#' * \link[=add_colrep]{add matrix to column-repeated vector}
#' * \link[=add_rowrep]{add matrix to row-repeated vector}
#'
env_fwd$`%*%` = base::`%*%`
env_bwd$`%*%` = function(L,R,prev) {
  return(list(
    L=base::tcrossprod(prev,R),
    R=base::crossprod(L,prev)
  ))
}
env_fwd$crossprod = base::crossprod
env_bwd$crossprod = function(L,R,prev) {
  return(list(
    L=base::tcrossprod(R,prev),
    R=base::`%*%`(L,prev)
  ))
}
env_fwd$tcrossprod = base::tcrossprod
env_bwd$tcrossprod = function(L,R,prev) {
  return(list(
    L=base::`%*%`(prev,R),
    R=base::crossprod(prev,L)
  ))
}
env_fwd$colMeans = base::colMeans
env_bwd$colMeans = function(X, prev) {
  ones = matrix(1,nrow=nrow(X),ncol=1)
  (ones %*% prev)*X/nrow(X)
}





#' @name loss_L2
#' @title L2 Loss function
#' @description \deqn{f(L,R) = \frac{1}{2} \sum_i \sum_j (L_{ij} - R_{ij})^2}
#' @usage env_fwd$loss_L2(L,R)
#' @usage env_bwd$loss_L2(L,R,prev)
env_fwd$loss_L2 = function(L,R) return(0.5*sum((L-R)^2))
env_bwd$loss_L2 = function(L,R,prev) return(list(L=L-R,R=R-L))

#' @name softmax
#' @title Row-wise softmax function
#' @description \deqn{\text{softmax}(X) = \dfrac{\exp(X)}{\exp(X) \bigotimes 1_{J \times J}}}
#' where \eqn{\bigotimes} is standard matrix multiplication,
#' \eqn{J} is the number of columns in \eqn{X}, and
#' \eqn{1_{J \times J}} is a \eqn{J \times J} matrix of ones.
#' @usage env_fwd$softmax(X)
#' @usage env_bwd$softmax(X,prev)
env_fwd$softmax = function(X) {
  ones = matrix(1,nrow=ncol(X),ncol=ncol(X))
  return(exp(X) / (exp(X) %*% ones))
}
env_bwd$softmax = function(X, prev) {
  ones = matrix(1,nrow=ncol(X),ncol=ncol(X))
  top = exp(X); bottom = exp(X) %*% ones
  tmp1 = prev*(top/bottom)
  tmp2 = prev*top*(-(1/bottom/bottom))
  tmp2 = tmp2 %*% ones
  tmp2 = tmp2*top
  return(tmp1 + tmp2)
}

#' @name add_colrep
#' @title add matrix to a column-repeated vector
#' @usage env_fwd$add_colrep(L,R)
#' @usage env_bwd$add_colrep(L,R,prev)
env_fwd$add_colrep = function(L,R) {
  if(!(nrow(L)==nrow(R))) stop("matrices must have same number of rows")
  if(!(ncol(R)==1)) stop("R must be column vector")
  return(L + R %*% matrix(1,nrow=1,ncol=ncol(L)))
}
env_bwd$add_colrep = function(L,R,prev) {
  if(!(nrow(L)==nrow(R))) stop("matrices must have same number of rows")
  if(!(ncol(R)==1)) stop("R must be column vector")
  return(list(
    L=prev,
    R=prev %*% matrix(1,nrow=ncol(L),ncol=1)
  ))
}

#' @name add_rowrep
#' @title add matrix to a row-repeated vector
#' @usage env_fwd$add_rowrep(L,R)
#' @usage env_bwd$add_rowrep(L,R,prev)
env_fwd$add_rowrep = function(L,R) {
  if(!(ncol(L)==ncol(R))) stop("matrices must have same number of cols")
  if(!(nrow(R)==1)) stop("R must be row vector")
  return(L + matrix(1,nrow=nrow(L),ncol=1) %*% R)
}
env_bwd$add_rowrep = function(L,R,prev) {
  if(!(ncol(L)==ncol(R))) stop("matrices must have same number of cols")
  if(!(nrow(R)==1)) stop("R must be row vector")
  return(list(
    L=prev,
    R=matrix(1,nrow=1,ncol=nrow(L)) %*% prev
  ))
}





#' @name elem_wise
#' @title Element-by-element operations on matrices
#' @description Takes an operator which can be applied element-by-element
#' and converts it into forward and backward functions. The forward function
#' is trivial (it just returns the original operator). The backward function
#' uses [Deriv::Deriv] to automatically compute the derivative (for a unary operator)
#' or partial derivatives (for a binary operator).
#' @usage env_fwd$elem_wise(f)
#' @usage env_bwd$elem_wise(f)
#' @details Note that elem_wise counts the number of arguments in the supplied operator.
#' This can cause surprise errors. For example, `log` in base R actually has two arguments
#' (x and base) so `elem_wise` would treat it as a binary operator. To make it unary,
#' you'd want to define your own single-argument function `function (x) log(x)`.
#' @examples
#' X = matrix(runif(11*5),nrow=11,ncol=5)
#' Y = matrix(rnorm(11*5),nrow=11,ncol=5)
#' prev = matrix(rnorm(11*5),nrow=11,ncol=5)
#'
#' # unary operator example: exp
#' env_fwd$elem_wise(exp)(X)
#' env_bwd$elem_wise(exp)(X,prev)
#'
#' # unary operator example: log
#' f = function (x) log(x)
#' env_fwd$elem_wise(f)(X)
#' env_bwd$elem_wise(f)(X,prev)
#'
#' # binary operator example: multiplying matrices element by element
#' env_fwd$elem_wise(`*`)(X,Y)
#' env_bwd$elem_wise(`*`)(X,Y,prev)
#'
env_fwd$elem_wise = function(f) return(f)
env_bwd$elem_wise = function(f) {
  n_arg = length(formals(args(f)))
  if (n_arg==1) {
    return(
      function(X,prev) {
        return(prev*Deriv::Deriv(f)(X))
      }
    )
  } else if (n_arg==2) {
    return(
      function(L,R,prev) {
        res = Deriv::Deriv(f,combine="list")(L,R)
        res[[1]] = prev*res[[1]]
        res[[2]] = prev*res[[2]]
        names(res) = c("L","R")
        return(res)
      }
    )
  } else {
    stop("function supplied to elem_wise must have only 1 or 2 arguments")
  }
}

