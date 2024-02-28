
#' @import R6
#' @export
add_data_node = function(.cg,id,input) {
  new_node = DataNode$new(input)
  .cg$add(new_node,id)
  return(.cg)
}

#' @import R6
#' @export
add_embedding_node = function(.cg,id,input,dims,initfunc,...) {
  new_node = EmbeddingNode$new(input,dims,initfunc,...)
  .cg$add(new_node,id)
  return(.cg)
}

#' @import R6
#' @export
add_param_node = function(.cg,id,dims,initfunc,...) {
  new_node = ParameterNode$new(dims,initfunc,...)
  .cg$add(new_node,id)
  return(.cg)
}

#' @import R6
#' @export
add_binary_node = function(.cg,id,input,f) {
  finputs = as.character(substitute(f))
  if(length(finputs)==1) {
    new_node = OperatorNodeBinary$new(
      f  = env_fwd[[finputs]],
      df = env_bwd[[finputs]],
      lnode = .cg$get_node(input[1]),
      rnode = .cg$get_node(input[2])
    )
  } else {
    new_node = OperatorNodeBinary$new(
      f  = env_fwd[[finputs[1]]](get(finputs[2])),
      df = env_bwd[[finputs[1]]](get(finputs[2])),
      lnode = .cg$get_node(input[1]),
      rnode = .cg$get_node(input[2])
    )
  }
  .cg$add(new_node,id)
  return(.cg)
}

#' @import R6
#' @export
add_unary_node = function(.cg,id,input,f) {
  finputs = as.character(substitute(f))
  if(length(finputs)==1) {
    new_node = OperatorNodeUnary$new(
      f  = env_fwd[[finputs]],
      df = env_bwd[[finputs]],
      lnode = .cg$get_node(input)
    )
  } else {
    new_node = OperatorNodeUnary$new(
      f  = env_fwd[[finputs[1]]](get(finputs[2])),
      df = env_bwd[[finputs[1]]](get(finputs[2])),
      lnode = .cg$get_node(input)
    )
  }
  .cg$add(new_node,id)
  return(.cg)
}
