#ifndef CONFIGURE_H
#define CONFIGURE_H

#include "armadillo"
#include "dynamic_het_types.h"
#include "utils.h"

network_info network_info_configure(Cube<Q_Q> &adj_mat, const Col<Q_Q> &n_type_each){
  // read files
  network_structure n_str;
  n_str.adj_matrix = adj_mat;
  n_str.node_type_each = n_type_each;
  n_str.space_dim = 2;
  n_str.link_type = LOGISTIC;
  n_str.derived = 0;
  fill_structure_info(n_str);

  network_update_pars n_up;
  n_up.total_steps = 4000;
  n_up.current_step = 0;
  n_up.burn_in_steps = 0;
  n_up.positions_step_size = vec(2);
  n_up.positions_step_size(0) = 0.03;
  n_up.positions_step_size(1) = 0.03;
  n_up.beta_step_size = 0.08;
  n_up.radius_kappa = 50000;

  network_prior_pars n_pp;
  n_pp.s2_scale = 0.5;
  n_pp.t2_scale = 0.02;
  n_pp.beta_scale = 1;

  network_info n_info = {n_str, n_pp, n_up};
  return n_info;
}

#endif