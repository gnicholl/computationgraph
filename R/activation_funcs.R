

### ACTIVATION FUNCS

# sigmoid function
sigmoid = function(X) {
  1/(1+exp(-X))
}

# rectified linear units
ReLU = function(X) {
  ifelse(X>0,X,0)
}
