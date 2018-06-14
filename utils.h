#ifndef UTILS_H
#define UTILS_H

#include "armadillo"
#include "dynamic_het_types.h"
#include "dynamic_het.h"
#include "file_utils.h"

using namespace arma;

void fill_structure_info(network_structure &info){
  // old-fashioned way modifying stuff
  if(info.derived) return;

  info.time_length = (info.adj_matrix).n_slices;
  info.n_nodes = (info.adj_matrix).n_cols;

  Col<Q_Q> unique_types = unique(info.node_type_each);
  info.n_types = unique_types.n_elem;

  (info.n_nodes_each) = Col<n_n>(info.n_types);
  (info.each_type_start) = Col<n_n>(info.n_types + 1);
  (info.nodes_rearrange) = Col<n_n>(info.n_nodes);

  // this is not very efficient algorithm, but it will only run once..
  (info.n_nodes_each).zeros();
  (info.each_type_start).zeros();
  for(n_n node = 0; node < info.n_nodes; node++) ((info.n_nodes_each)((info.node_type_each)(node))) ++;
  for(Q_Q type = 1; type < info.n_types + 1; type++) (info.each_type_start)(type) = (info.each_type_start)(type - 1) + (info.n_nodes_each)(type - 1);

  for(n_n node = 0; node < info.n_nodes; node++){
    Q_Q &node_type = (info.node_type_each)(node);
    n_n &type_start = (info.each_type_start)(node_type);
    (info.nodes_rearrange)(type_start) = node;
    type_start++;
  }
  
  (info.each_type_start).zeros();
  for(Q_Q type = 1; type < info. n_types + 1; type++) (info.each_type_start)(type) = (info.each_type_start)(type - 1) + (info.n_nodes_each)(type - 1);
  info.derived = 0x01;
  return;
}

double r_hf_cauchy(double &scale){
  double ru = randu();
  return tan(ru * (datum::pi) / 2.)  * scale;
}

void r_dirichlet(mat &concentration){
  // FBI warning: changing input on spot
  concentration.for_each([](double &elem) {elem = randg(distr_param(elem,1.));});
  double c_sum = accu(concentration);
  concentration.for_each([&c_sum](double &elem) {elem /= c_sum;});
}

#endif
