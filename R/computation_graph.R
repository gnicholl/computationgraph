
#' @import R6
#' @export
ComputationGraph = R6Class("ComputationGraph", list(
  AllNodes = NULL,
  DataNodes = NULL,
  ParameterNodes = NULL,
  OperatorNodes = NULL,
  learnrate = 0.00001,
  initialize=function() {
    self$AllNodes = new.env()
  },
  add=function(item,id) {
    if (class(item)[1]=="DataNode") {
      self$DataNodes = append(self$DataNodes,item)
    } else if (class(item)[1]=="ParameterNode") {
      self$ParameterNodes = append(self$ParameterNodes,item)
    } else if (class(item)[1]=="OperatorNodeBinary") {
      self$OperatorNodes = append(self$OperatorNodes,item)
    } else if (class(item)[1]=="OperatorNodeUnary") {
      self$OperatorNodes = append(self$OperatorNodes,item)
    } else if (class(item)[1]=="EmbeddingNode") {
      self$DataNodes = append(self$DataNodes,item)
      self$ParameterNodes = append(self$ParameterNodes,item)
    }
    self$AllNodes[[id]] = item
  },
  get_node = function(id) {
    return(self$AllNodes[[id]])
  },
  forward = function(i){
    for (item in self$DataNodes) {
      item$set_data(i)
    }
    for (item in self$OperatorNodes) {
      item$forward()
    }
  },
  backward = function() {
    # accumulate gradient backward through operator nodes
    L = length(self$OperatorNodes)
    for (l in L:1) {
      self$OperatorNodes[[l]]$compute_gradient()
    }

    # compute gradient for each weight
    for (item in self$ParameterNodes) {
      item$compute_gradient()
    }
  },
  update_weights = function() {
    for (item in self$ParameterNodes) {
      item$update(self$learnrate)
    }
  }
))

#' @import R6
#' @export
DataNode = R6Class("DataNode", list(
  data_matrix = NULL,
  output_value = NULL,
  output_nodes = NULL, # data, unlike operators, can be used in multiple places, and thus have multiple outputs
  initialize=function(input) {
    self$data_matrix = input
  },
  add_output = function(node) {
    self$output_nodes = append(self$output_nodes,node)
  },
  set_data = function(i) {
    self$output_value = self$data_matrix[[i]]
  }
))

#' @import R6
#' @export
EmbeddingNode = R6Class("EmbeddingNode",list(
  embedding_matrix = NULL,
  data_matrix = NULL,
  lookup_vals = NULL,
  output_value = NULL,
  output_nodes = NULL, # parameters, unlike operators, can be used in multiple places, and thus have multiple outputs
  gradient = NULL,
  initialize=function(input,dims,rfunc,...) {
    self$data_matrix = input
    self$embedding_matrix = matrix(rfunc(prod(dims),...),nrow=dims[1],ncol=dims[2])
    self$gradient = matrix(rep(0,prod(dims)),nrow=dims[1],ncol=dims[2])
  },
  set_data = function(i) {
    self$lookup_vals = self$data_matrix[[i]]
    self$output_value = self$embedding_matrix[self$lookup_vals,,drop=FALSE]
  },
  reset_gradient = function() {
    self$gradient[] = 0
  },
  compute_gradient = function() {
    # sum over gradients from all parent nodes
    if (!is.null(self$output_nodes)) {

      for (parent in self$output_nodes) {
        if (identical(self,parent$lnode)) {

          for (i in 1:length(self$lookup_vals)) {
            j = self$lookup_vals[i]
            self$gradient[j,] = self$gradient[j,] + parent$partial_derivs[[1]][i,]
          }

        } else if (identical(self,parent$rnode)) {

          for (i in 1:length(self$lookup_vals)) {
            j = self$lookup_vals[i]
            self$gradient[j,] = self$gradient[j,] + parent$partial_derivs[[2]][i,]
          }

        } else {
          stop("neither left nor right?")
        }

      } #endfor
    } #endif
  },
  update = function(learnrate) {
    self$embedding_matrix = self$embedding_matrix - learnrate*self$gradient
    self$reset_gradient()
  },
  add_output = function(node) {
    self$output_nodes = append(self$output_nodes,node)
  }
))

#' @import R6
#' @export
ParameterNode = R6Class("ParameterNode", list(
  output_value = NULL,
  output_nodes = NULL, # parameters, unlike operators, can be used in multiple places, and thus have multiple outputs
  gradient = NULL,
  initialize=function(dims,rfunc,...) {
    self$output_value = matrix(rfunc(prod(dims),...),nrow=dims[1],ncol=dims[2])
    self$gradient = matrix(rep(0,prod(dims)),nrow=dims[1],ncol=dims[2])
  },
  reset_gradient = function() {
    self$gradient[] = 0
  },
  compute_gradient = function() {
    # sum over gradients from all parent nodes
    if (!is.null(self$output_nodes)) {

      for (parent in self$output_nodes) {
        if (identical(self,parent$lnode)) {
          self$gradient = self$gradient + parent$partial_derivs[[1]]

        } else if (identical(self,parent$rnode)) {
          self$gradient = self$gradient + parent$partial_derivs[[2]]

        } else {
          stop("neither left nor right?")
        }

      } #endfor
    } #endif
  },
  update = function(learnrate) {
    self$output_value = self$output_value - learnrate*self$gradient
    self$reset_gradient()
  },
  add_output = function(node) {
    self$output_nodes = append(self$output_nodes,node)
  }
))

#' @import R6
#' @export
OperatorNodeBinary = R6Class("OperatorNodeBinary", list(
  f = NULL,
  df = NULL,
  output_value = NULL,
  partial_derivs = NULL,
  lnode = NULL,
  rnode = NULL,
  output_nodes = NULL, # operators can have at most one output node!
  initialize=function(f,df,lnode,rnode) {
    self$f = f
    self$df = df
    self$lnode = lnode
    self$rnode = rnode
    lnode$add_output(self)
    rnode$add_output(self)
  },
  forward=function() {
    self$output_value = self$f(
      self$lnode$output_value,
      self$rnode$output_value
    )
  },
  compute_gradient=function() {
    # get accumulated gradient from parent node
    if (!is.null(self$output_nodes)) {
      if (identical(self,self$output_nodes[[1]]$lnode)) {
        self$partial_derivs = self$df(
          self$lnode$output_value,
          self$rnode$output_value,
          self$output_nodes[[1]]$partial_derivs[[1]]
        )

      } else if (identical(self,self$output_nodes[[1]]$rnode)) {
        self$partial_derivs = self$df(
          self$lnode$output_value,
          self$rnode$output_value,
          self$output_nodes[[1]]$partial_derivs[[2]]
        )

      } else {
        stop("neither left nor right?")
      }
    } else {
      self$partial_derivs = self$df(
        self$lnode$output_value,
        self$rnode$output_value,
        1
      )
    }

  },
  add_output = function(node) {
    self$output_nodes = append(self$output_nodes,node)
  }
))

#' @import R6
#' @export
OperatorNodeUnary = R6Class("OperatorNodeUnary", list(
  f = NULL,
  df = NULL,
  output_value = NULL,
  partial_derivs = NULL,
  lnode = NULL,
  output_nodes = NULL, # operators can have at most one output node!
  initialize=function(f,df,lnode) {
    self$f = f
    self$df = df
    self$lnode = lnode
    lnode$add_output(self)
  },
  forward=function() {
    self$output_value = self$f(
      self$lnode$output_value
    )
  },
  compute_gradient=function() {

    # get accumulated gradient from parent node
    if (!is.null(self$output_nodes)) {
      if (identical(self,self$output_nodes[[1]]$lnode)) {
        self$partial_derivs[[1]] = self$df(
          self$lnode$output_value,
          self$output_nodes[[1]]$partial_derivs[[1]]
        )

      } else if (identical(self,self$output_nodes[[1]]$rnode)) {
        self$partial_derivs[[1]] = self$df(
          self$lnode$output_value,
          self$output_nodes[[1]]$partial_derivs[[2]]
        )

      } else {
        stop("neither left nor right?")
      }
    } else {
      self$partial_derivs[[1]] = self$df(
        self$lnode$output_value,
        1
      )
    }

  },
  add_output = function(node) {
    self$output_nodes = append(self$output_nodes,node)
  }
))
