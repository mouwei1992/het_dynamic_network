#include<cmath>
#include<pthread.h>
#include <cstdlib>

#include "dynamic_het_types.h"
#include "dynamic_het.h"
#include "utils.h"
#include "logistic_approx.h"

using namespace std;
using namespace arma;

dy_het_network::dy_het_network(network_info &n_info){
  structure_pars = n_info.p_structure;
  update_pars = n_info.p_update;
  prior_pars = n_info.p_prior;
  init_draw_from_prior();
  init_buf();
  init_acceptance();
  update_current_dist();
  update_current_llh();
}

void dy_het_network::init_buf(void){
  const n_n &n_nodes = structure_pars.n_nodes;
  const n_n &t_length = structure_pars.time_length;

  current_buf.dist_matrix = cube(n_nodes,n_nodes,t_length);
  current_buf.log_likelihood_mat = cube(n_nodes,n_nodes,t_length);
  proposal_buf.dist_matrix = cube(n_nodes,n_nodes,t_length);
  proposal_buf.log_likelihood_mat = cube(n_nodes,n_nodes,t_length);

  current_buf.log_likelihood_mat.zeros();
  proposal_buf.log_likelihood_mat.zeros();
}

void dy_het_network::init_acceptance(void){
  const n_n &n_nodes = structure_pars.n_nodes;
  const n_n &t_length = structure_pars.time_length;
  const Q_Q &n_types = structure_pars.n_types;

  total_acceptance.positions_accepted = Mat<n_n>(n_nodes,t_length);
  total_acceptance.betas_accepted = Mat<n_n>(n_types,n_types);

  total_acceptance.positions_accepted.zeros();
  total_acceptance.betas_accepted.zeros();
  total_acceptance.radius_accepted = 0;
  total_acceptance.s2_accepted = 0;
  total_acceptance.t2_accepted = 0;
}

void dy_het_network::init_draw_from_prior(void){

  current_pars.s2 = r_hf_cauchy(prior_pars.s2_scale);
  current_pars.t2 = r_hf_cauchy(prior_pars.t2_scale);

  const Q_Q &n_types = structure_pars.n_types;
  current_pars.betas = mat(n_types,n_types);
  for(Q_Q type_1 = 0; type_1 < n_types; type_1++){
    for(Q_Q type_2 = 0; type_2 < n_types; type_2++)
      current_pars.betas(type_1,type_2) = r_hf_cauchy(prior_pars.beta_scale);
  }

  const n_n &t_length = structure_pars.time_length;
  const n_n &n_nodes = structure_pars.n_nodes;
  const Q_Q &universe_dimension = structure_pars.space_dim;
  current_pars.positions = cube(universe_dimension,n_nodes,t_length);
  current_pars.positions.slice(0).randn();
  current_pars.positions.slice(0).for_each([this](double &elem) {elem *= sqrt(current_pars.s2);});
  for(n_n t = 1; t < t_length; t++){
    current_pars.positions.slice(t).randn();
    current_pars.positions.slice(t).for_each([this](double &elem) {elem *= sqrt(current_pars.t2);});
    current_pars.positions.slice(t) += current_pars.positions.slice(t-1);
  }

  current_pars.radius = mat(n_types,n_types);
  current_pars.radius.ones();
  current_pars.radius.for_each([this](double &elem) {elem *= update_pars.radius_kappa;});
  r_dirichlet(current_pars.radius);

  proposal_pars = current_pars;
  return;
}

void dy_het_network::update_current_dist(void){
  cube &positions = current_pars.positions;
  cube &dist_mat = current_buf.dist_matrix;
  const n_n &n_nodes = structure_pars.n_nodes;

  for(Q_Q t = 0; t < structure_pars.time_length; t++){
    for(n_n node_1 = 0; node_1 < n_nodes; node_1++){
      for(n_n node_2 = 0; node_2 < n_nodes; node_2++){
        if(node_1 > node_2) continue;
        if(node_1 == node_2){
          if(t == 0) dist_mat(node_1,node_1,t) = norm(positions.slice(t).col(node_1));
          else dist_mat(node_1,node_1,t) = norm(positions.slice(t).col(node_1) - positions.slice(t-1).col(node_1));
          dist_mat(node_1,node_1,t) *= dist_mat(node_1,node_1,t);
        }else{
          // node_1 < node_2
          dist_mat(node_1,node_2,t) = norm(positions.slice(t).col(node_1) - positions.slice(t).col(node_2));
        }
      }
    }
  }
}

inline double dy_het_network::llh_cal(Q_Q &y, double &eta){
  double llh;
  switch(structure_pars.link_type){
    case LOGISTIC:
      if(eta > 25) llh = -eta;
      else if(eta < -25) llh = 0;
      else llh = -log(exp(float(eta)) + 1);
      if(y) llh += eta;
      break;
    case TEST:
      llh = eta;
      break;
    // a faster but less accurate version
    case LOGISTIC_F:
      llh = logistic_prox(y,eta);
      break;
  }
  return llh;
}

void *dy_het_network::update_current_llh_single_helper(void *context){
    thread_info *th_if = (thread_info *)context;
    ( ((dy_het_network *)(th_if->network_pt)) ->update_current_llh_single(th_if->thread_id));
    return 0;
}

void dy_het_network::update_current_llh(void){
  pthread_t tid1, tid2, tid3, tid4;
  thread_info t1_ = {this,0};
  thread_info t2_ = {this,1};
  thread_info t3_ = {this,2};
  thread_info t4_ = {this,3};
  pthread_create(&tid1, NULL, &dy_het_network::update_current_llh_single_helper, &t1_);
  pthread_create(&tid2, NULL, &dy_het_network::update_current_llh_single_helper, &t2_);
  pthread_create(&tid3, NULL, &dy_het_network::update_current_llh_single_helper, &t3_);
  pthread_create(&tid4, NULL, &dy_het_network::update_current_llh_single_helper, &t4_);
  pthread_join(tid1,NULL);
  pthread_join(tid2,NULL);
  pthread_join(tid3,NULL);
  pthread_join(tid4,NULL);
  return;
}

void dy_het_network::update_current_llh_single(Q_Q &n_th){
  cube &dist_mat = current_buf.dist_matrix;
  cube &llh_mat = current_buf.log_likelihood_mat;
  n_n &n_nodes = structure_pars.n_nodes;
  Q_Q node_1_type;
  Q_Q node_2_type;
  double dist;
  double eta;
  for(n_n node_1 = 0; node_1 < n_nodes; node_1++){
    if(node_1 % 4 != n_th) continue;
    node_1_type = structure_pars.node_type_each[node_1];
    for(n_n node_2 = 0; node_2 < n_nodes; node_2++){
      if(node_1 == node_2) continue;
      node_2_type = structure_pars.node_type_each[node_2];
      for(Q_Q t = 0; t < structure_pars.time_length; t++){
        const double &dist = node_1 < node_2 ? dist_mat.at(node_1,node_2,t):dist_mat.at(node_2,node_1,t);
        eta = eta_cal(dist, current_pars.betas.at(node_1_type,node_2_type), current_pars.radius.at(node_1_type,node_2_type));
        llh_mat.at(node_1,node_2,t) = llh_cal(structure_pars.adj_matrix.at(node_1,node_2,t), eta);
      }
    }

  }
}

int dy_het_network::update(n_n steps){
  // main MCMC update API
  for(int step = 0; step < steps; step++){
    update_pars.current_step++;

    if(update_pars.current_step == update_pars.burn_in_steps)
      cout << "Burn-in complets." << endl;

    if(update_pars.current_step == update_pars.total_steps){
      cout << "Sample complete." <<endl;
      return 0;
    }

    if(update_pars.current_step % 100 == 0)
      cout << "Current Step: " << update_pars.current_step << "/" << update_pars.total_steps << endl;

    update_positions();
    update_s2();
    update_t2();
    update_betas();
    update_radius();
  }
  return 1;
}

network_model_pars dy_het_network::duplicate_current_pars(void) const{
  return current_pars;
}
/*
void dy_het_network::update_positions(void){
  const n_n &n_nodes = structure_pars.n_nodes;
  const n_n &t_length = structure_pars.time_length;
  Col<n_n> update_seq = shuffle(linspace<Col<n_n>> (0,n_nodes - 1,n_nodes));

  cube &current_dist = current_buf.dist_matrix;
  cube &current_llh = current_buf.log_likelihood_mat;
  cube &proposal_dist = proposal_buf.dist_matrix;
  cube &proposal_llh = proposal_buf.log_likelihood_mat;

  double dist;
  double ru;
  double accept_prob;

  pthread_t tid1, tid2, tid3, tid4;
  pthread_mutex_t lk;
  pthread_cond_t position_proposed, llh_updated;

  pthread_mutex_init(&lk, NULL);
  pthread_cond_init(&position_proposed,NULL);
  pthread_cond_init(&llh_updated, NULL);

  thread_info_p t1_ = {this,0, &update_seq, &position_proposed, &llh_updated, &lk};
  thread_info_p t2_ = {this,1, &update_seq, &position_proposed, &llh_updated, &lk};
  thread_info_p t3_ = {this,2, &update_seq, &position_proposed, &llh_updated, &lk};
  thread_info_p t4_ = {this,3, &update_seq, &position_proposed, &llh_updated, &lk};

  pthread_create(&tid1, NULL, &dy_het_network::update_position_single_helper, &t1_);
  pthread_create(&tid2, NULL, &dy_het_network::update_position_single_helper, &t2_);
  pthread_create(&tid3, NULL, &dy_het_network::update_position_single_helper, &t3_);
  pthread_create(&tid4, NULL, &dy_het_network::update_position_single_helper, &t4_);

  for(n_n node_cnt = 0; node_cnt < n_nodes; node_cnt++){
    n_n &node = update_seq(node_cnt);
    for(n_n t = 0; t < t_length; t++){
      accept_prob = 0;
      ru = randu();
      Q_Q &node_type = structure_pars.node_type_each(node);
      mat &current_nodes_positions = current_pars.positions.slice(t);
      // actually i am not sure what i am doing..
      vec current_position = current_nodes_positions.unsafe_col(node);
      // same here
      vec proposal_position = proposal_pars.positions.slice(t).unsafe_col(node);

      proposal_position.randn();
      proposal_position.for_each([this, &node_type](double &elem) {elem *= (update_pars.positions_step_size)[node_type];});
      proposal_position += current_position;

      pthread_mutex_lock(&lk);
      pthread_mutex_lock(&lk);
      pthread_mutex_lock(&lk);
      pthread_mutex_lock(&lk);
      pthread_cond_broadcast(&position_proposed);

      if(t == 0){
        dist = dot(proposal_position,proposal_position);
        proposal_dist.at(node,node,0) = dist;
        accept_prob -= dist / (2 * current_pars.s2);
        double &old_dist = current_dist(node,node,0);
        accept_prob += old_dist / (2 * current_pars.s2);
      }else{
        // dangerous..
        vec last_position = current_pars.positions.slice(t - 1).unsafe_col(node);
        // it is redundant to do it here but it is not in the main loop..
        vec diff_position = proposal_position - last_position;
        dist = dot(diff_position, diff_position);
        proposal_dist.at(node,node,t) = dist;
        accept_prob -= dist * dist / (2 * current_pars.t2);
        double &old_dist = current_dist.at(node,node,t);
        accept_prob += old_dist * old_dist / (2 * current_pars.t2);
      }

      if(t < structure_pars.time_length - 1){
        // dangerous!
        vec next_position = current_pars.positions.slice(t + 1).unsafe_col(node);
        vec diff_position = proposal_position - next_position;
        dist = dot(diff_position, diff_position);
        proposal_dist.at(node,node,t + 1) = dist;
        accept_prob -= dist / (2 * current_pars.t2);
        double &old_dist = current_dist.at(node,node,t + 1);
        accept_prob += old_dist /(2 * current_pars.t2);
      }

      pthread_cond_wait(&llh_updated, &lk);
      pthread_cond_wait(&llh_updated, &lk);
      pthread_cond_wait(&llh_updated, &lk);
      pthread_cond_wait(&llh_updated, &lk);

      for(n_n node_2 = 0; node_2 < n_nodes; node_2++){
        if(node == node_2) continue;
        accept_prob -= (current_llh.at(node,node_2,t) + current_llh.at(node_2,node,t));
        accept_prob += (proposal_llh.at(node,node_2,t) + proposal_llh.at(node_2,node,t));
      }

      accept_prob = accept_prob > 0 ? 1:exp(accept_prob);

      if(ru < accept_prob){
        current_position = proposal_position;
        // update the buffers..
        current_dist.at(node,node,t) = proposal_dist.at(node,node,t);
        if(t < structure_pars.time_length - 1) current_dist(node,node,t + 1) = proposal_dist.at(node,node,t + 1);
        for(n_n node_2 = 0; node_2 < n_nodes; node_2++){
          current_llh.at(node,node_2,t) = proposal_llh.at(node,node_2,t);
          current_llh.at(node_2,node,t) = proposal_llh.at(node_2,node,t);
          if(node < node_2) current_dist.at(node,node_2,t) = proposal_dist.at(node,node_2,t);
          else current_dist.at(node_2,node,t) = proposal_dist.at(node_2,node,t);
        }
        if(update_pars.current_step >= update_pars.burn_in_steps) total_acceptance.positions_accepted.at(node,t) ++;
      }
    }
  }


  pthread_join(tid1,NULL);
  pthread_join(tid2,NULL);
  pthread_join(tid3,NULL);
  pthread_join(tid4,NULL);

  pthread_cond_destroy(&llh_updated);

  pthread_mutex_destroy(&lk);
  return;
}

void *dy_het_network::update_position_single_helper(void *context){
  thread_info_p *th_if_p = (thread_info_p *)context;
  ((dy_het_network *)(th_if_p -> network_pt)) -> update_position_single(th_if_p->update_seq , th_if_p -> thread_id, th_if_p -> position_proposed, th_if_p -> llh_updated, th_if_p -> lock);
  return 0;
}

void dy_het_network::update_position_single(Col<n_n> *update_seq, Q_Q &th_id, pthread_cond_t *position_proposed, pthread_cond_t *llh_updated, pthread_mutex_t *lock){
  const n_n &n_nodes = structure_pars.n_nodes;
  const n_n &t_length = structure_pars.time_length;

  cube &proposal_dist = proposal_buf.dist_matrix;
  cube &proposal_llh = proposal_buf.log_likelihood_mat;


  double dist;
  double eta;

  for(n_n node_cnt = 0; node_cnt < n_nodes; node_cnt++){
    n_n &node = (*update_seq)[node_cnt];
    Q_Q &node_type = structure_pars.node_type_each[node];

    for(n_n t = 0; t < t_length; t++){
      vec proposal_position = proposal_pars.positions.slice(t).unsafe_col(node);
      mat &current_nodes_positions = current_pars.positions.slice(t);

      pthread_mutex_lock(lock);
      pthread_cond_wait(position_proposed,lock);
      pthread_mutex_unlock(lock);

      for(n_n node_2 = 0; node_2 < n_nodes; node_2++){
        if(node == node_2) continue;
        if(node_2 % 4 != th_id) continue;
        Q_Q &node_2_type = structure_pars.node_type_each[node_2];

        dist = norm(proposal_position - current_nodes_positions.col(node_2));
        if(node < node_2) proposal_dist.at(node,node_2,t) = dist;
        else proposal_dist.at(node_2,node,t) = dist;

        // node -> node_2
        eta = eta_cal(dist,current_pars.betas.at(node_type,node_2_type),current_pars.radius.at(node_type,node_2_type));
        proposal_llh.at(node,node_2,t) = llh_cal(structure_pars.adj_matrix.at(node,node_2,t),eta);
        // the other way
        eta = eta_cal(dist,current_pars.betas.at(node_2_type,node_type),current_pars.radius.at(node_2_type,node_type));
        proposal_llh.at(node_2,node,t) = llh_cal(structure_pars.adj_matrix.at(node_2,node,t),eta);
      }

      pthread_cond_signal(llh_updated);
    }
  }
  return;
}
*/

void dy_het_network::update_positions(void){
  const n_n &n_nodes = structure_pars.n_nodes;
  const n_n &t_length = structure_pars.time_length;
  Col<n_n> update_seq = shuffle(linspace<Col<n_n>> (0,n_nodes - 1,n_nodes));
  for(n_n node_cnt = 0; node_cnt < n_nodes; node_cnt++){
    for(n_n t = 0; t < t_length; t++) update_position(update_seq(node_cnt),t);
  }
  return;
}

void dy_het_network::update_position(const n_n &node, const n_n &t){
  const n_n &n_nodes = structure_pars.n_nodes;
  const Q_Q &node_type = structure_pars.node_type_each(node);
  mat &current_nodes_positions = current_pars.positions.slice(t);
  // actually i am not sure what i am doing..
  vec current_position = current_nodes_positions.unsafe_col(node);
  // same here
  vec proposal_position = proposal_pars.positions.slice(t).unsafe_col(node);
  cube &current_dist = current_buf.dist_matrix;
  cube &current_llh = current_buf.log_likelihood_mat;
  cube &proposal_dist = proposal_buf.dist_matrix;
  cube &proposal_llh = proposal_buf.log_likelihood_mat;
  double dist;
  double eta;
  double accept_prob = 0;
  double ru = randu();

  proposal_position.randn();
  proposal_position.for_each([this, &node_type](double &elem) {elem *= (update_pars.positions_step_size)[node_type];});
  proposal_position += current_position;

  for(n_n node_2 = 0; node_2 < n_nodes; node_2++){
    if(node == node_2) continue;
    Q_Q &node_2_type = structure_pars.node_type_each[node_2];
    accept_prob -= (current_llh.at(node,node_2,t) + current_llh.at(node_2,node,t));
    dist = norm(proposal_position - current_nodes_positions.col(node_2));
    if(node < node_2) proposal_dist.at(node,node_2,t) = dist;
    else proposal_dist.at(node_2,node,t) = dist;

    // node -> node_2
    eta = eta_cal(dist,current_pars.betas.at(node_type,node_2_type),current_pars.radius.at(node_type,node_2_type));
    proposal_llh.at(node,node_2,t) = llh_cal(structure_pars.adj_matrix.at(node,node_2,t),eta);
    // the other way
    eta = eta_cal(dist,current_pars.betas.at(node_2_type,node_type),current_pars.radius.at(node_2_type,node_type));
    proposal_llh.at(node_2,node,t) = llh_cal(structure_pars.adj_matrix.at(node_2,node,t),eta);
    accept_prob += (proposal_llh.at(node,node_2,t) + proposal_llh.at(node_2,node,t));
  }

  if(t == 0){
    dist = dot(proposal_position,proposal_position);
    proposal_dist.at(node,node,0) = dist;
    accept_prob -= dist / (2 * current_pars.s2);
    double &old_dist = current_dist(node,node,0);
    accept_prob += old_dist / (2 * current_pars.s2);
  }else{
    // dangerous..
    vec last_position = current_pars.positions.slice(t - 1).unsafe_col(node);
    // it is redundant to do it here but it is not in the main loop..
    vec diff_position = proposal_position - last_position;
    dist = dot(diff_position, diff_position);
    proposal_dist.at(node,node,t) = dist;
    accept_prob -= dist * dist / (2 * current_pars.t2);
    double &old_dist = current_dist.at(node,node,t);
    accept_prob += old_dist * old_dist / (2 * current_pars.t2);
  }

  if(t < structure_pars.time_length - 1){
    // dangerous!
    vec next_position = current_pars.positions.slice(t + 1).unsafe_col(node);
    vec diff_position = proposal_position - next_position;
    dist = dot(diff_position, diff_position);
    proposal_dist.at(node,node,t + 1) = dist;
    accept_prob -= dist / (2 * current_pars.t2);
    double &old_dist = current_dist.at(node,node,t + 1);
    accept_prob += old_dist /(2 * current_pars.t2);
  }
  accept_prob = accept_prob > 0 ? 1:exp(accept_prob);

  if(ru < accept_prob){
    current_position = proposal_position;
    // update the buffers..
    current_dist.at(node,node,t) = proposal_dist.at(node,node,t);
    if(t < structure_pars.time_length - 1) current_dist(node,node,t + 1) = proposal_dist.at(node,node,t + 1);
    for(n_n node_2 = 0; node_2 < n_nodes; node_2++){
      current_llh.at(node,node_2,t) = proposal_llh.at(node,node_2,t);
      current_llh.at(node_2,node,t) = proposal_llh.at(node_2,node,t);
      if(node < node_2) current_dist.at(node,node_2,t) = proposal_dist.at(node,node_2,t);
      else current_dist.at(node_2,node,t) = proposal_dist.at(node_2,node,t);
    }
    if(update_pars.current_step >= update_pars.burn_in_steps) total_acceptance.positions_accepted.at(node,t) ++;
  }
  return;
}

void dy_het_network::update_t2(void){
  double &new_t2 = proposal_pars.t2;
  double &old_t2 = current_pars.t2;
  double ru = randu();
  double accept_prob = 0;
  const n_n &n_nodes = structure_pars.n_nodes;
  const n_n &t_length = structure_pars.time_length;
  new_t2 = r_hf_cauchy(prior_pars.t2_scale);
  double step_sum = 0;
  for(n_n t = 1; t < t_length; t++)
    step_sum += trace(current_buf.dist_matrix.slice(t));
  accept_prob = n_nodes * (t_length -1) * log(old_t2/new_t2) + step_sum/old_t2 - step_sum/new_t2;
  accept_prob /= 2.;
  accept_prob = accept_prob > 0 ? 1:exp(accept_prob);
  if(ru < accept_prob){
    old_t2 = new_t2;
    total_acceptance.t2_accepted++;
  }
  return;
}

void dy_het_network::update_s2(void){
  const n_n &n_nodes = structure_pars.n_nodes;
  double &new_s2 = proposal_pars.s2;
  double &old_s2 = current_pars.s2;
  double ru = randu();
  double accept_prob = 0;
  new_s2 = r_hf_cauchy(prior_pars.s2_scale);
  double step_sum = trace(current_buf.dist_matrix.slice(0));
  accept_prob = n_nodes * log(old_s2 / new_s2) + step_sum/old_s2 - step_sum/new_s2;
  accept_prob /= 2.;
  accept_prob = accept_prob > 0 ? 1:exp(accept_prob);
  if(ru < accept_prob){
    old_s2 = new_s2;
    total_acceptance.s2_accepted++;
  }
  return;
}

void dy_het_network::update_betas(void){
  const Q_Q &n_types = structure_pars.n_types;
  for(Q_Q type_1 = 0; type_1 < n_types; type_1++){
    for(Q_Q type_2 = 0; type_2 < n_types; type_2++)
      update_beta(type_1,type_2);
  }
  return;
}

void *dy_het_network::update_beta_single_helper(void *context){
  thread_info_b *th_if_b = (thread_info_b *) context;
  ((dy_het_network *)(th_if_b->network_pt))->update_beta_single(th_if_b->type_1, th_if_b -> type_2, th_if_b -> thread_id);
  return 0;
}

void dy_het_network::update_beta_single(const Q_Q &type_1, const Q_Q &type_2, const Q_Q &n_th){
  // only used to update the new llh_mat
  const cube &dist_mat = current_buf.dist_matrix;
  cube &new_llh_mat = proposal_buf.log_likelihood_mat;
  const n_n t_length = structure_pars.time_length;
  double &new_beta = proposal_pars.betas.at(type_1,type_2);
  const double &corr_radius = current_pars.radius.at(type_1,type_2);
  double eta;

  const n_n &type_1_begin = structure_pars.each_type_start[type_1];
  const n_n &type_1_end = structure_pars.each_type_start[type_1 + 1];
  const n_n &type_2_begin = structure_pars.each_type_start[type_2];
  const n_n &type_2_end = structure_pars.each_type_start[type_2 + 1];

  for(n_n type_1_node = type_1_begin; type_1_node < type_1_end; type_1_node++){
    n_n &node_1 = structure_pars.nodes_rearrange[type_1_node];
    if(node_1 % 4 != n_th) continue;
    for(n_n type_2_node = type_2_begin; type_2_node < type_2_end; type_2_node++){
      n_n &node_2 = structure_pars.nodes_rearrange[type_2_node];
      if(node_1 == node_2) continue;
      for(n_n t = 0; t < t_length; t++){
        const double &dist = node_1 < node_2 ? dist_mat.at(node_1,node_2,t):dist_mat.at(node_2,node_1,t);
        double &new_llh = new_llh_mat.at(node_1,node_2,t);
        eta = eta_cal(dist,new_beta,corr_radius);
        new_llh = llh_cal(structure_pars.adj_matrix.at(node_1,node_2,t), eta);
      }
    }
  }
  return;
}

void dy_het_network::update_beta(const Q_Q &type_1, const Q_Q &type_2){
  cube &old_llh_mat = current_buf.log_likelihood_mat;
  cube &new_llh_mat = proposal_buf.log_likelihood_mat;
  const n_n t_length = structure_pars.time_length;
  double &old_beta = current_pars.betas.at(type_1,type_2);
  double &new_beta = proposal_pars.betas.at(type_1,type_2);
  double ru = randu();
  double accept_prob = 0;

  new_beta = (2 * ru - 1) * update_pars.beta_step_size + 1;
  if( new_beta * (1 + update_pars.beta_step_size) < 1) return;
  // transition prob..
  accept_prob = - log(new_beta);
  new_beta *= old_beta;

  pthread_t tid1, tid2, tid3, tid4;
  thread_info_b t1_ = {this,0, type_1, type_2};
  thread_info_b t2_ = {this,1, type_1, type_2};
  thread_info_b t3_ = {this,2, type_1, type_2};
  thread_info_b t4_ = {this,3, type_1, type_2};

  pthread_create(&tid1, NULL, &dy_het_network::update_beta_single_helper, &t1_);
  pthread_create(&tid2, NULL, &dy_het_network::update_beta_single_helper, &t2_);
  pthread_create(&tid3, NULL, &dy_het_network::update_beta_single_helper, &t3_);
  pthread_create(&tid4, NULL, &dy_het_network::update_beta_single_helper, &t4_);

  pthread_join(tid1,NULL);
  pthread_join(tid2,NULL);
  pthread_join(tid3,NULL);
  pthread_join(tid4,NULL);

  const n_n &type_1_begin = structure_pars.each_type_start[type_1];
  const n_n &type_1_end = structure_pars.each_type_start[type_1 + 1];
  const n_n &type_2_begin = structure_pars.each_type_start[type_2];
  const n_n &type_2_end = structure_pars.each_type_start[type_2 + 1];

  for(n_n type_1_node = type_1_begin; type_1_node < type_1_end; type_1_node++){
    n_n &node_1 = structure_pars.nodes_rearrange[type_1_node];
    for(n_n type_2_node = type_2_begin; type_2_node < type_2_end; type_2_node++){
      n_n &node_2 = structure_pars.nodes_rearrange[type_2_node];
      if(node_1 == node_2) continue;
      for(n_n t = 0; t < t_length; t++){
        double &old_llh = old_llh_mat.at(node_1,node_2,t);
        double &new_llh = new_llh_mat.at(node_1,node_2,t);

        accept_prob += new_llh;
        accept_prob -= old_llh;
      }
    }
  }

  // times prior
  double &b_prior = prior_pars.beta_scale;
  accept_prob += log(1 + old_beta * old_beta / (b_prior * b_prior));
  accept_prob -= log(1 + new_beta * new_beta / (b_prior * b_prior));

  accept_prob = accept_prob < 0? exp(accept_prob):1;
  ru = randu();
  if(ru < accept_prob){
    old_beta = new_beta;
    for(n_n type_1_node = type_1_begin; type_1_node < type_1_end; type_1_node++){
      n_n &node_1 = structure_pars.nodes_rearrange[type_1_node];
      for(n_n type_2_node = type_2_begin; type_2_node < type_2_end; type_2_node++){
        if(type_1_node == type_2_node) continue;
        n_n &node_2 = structure_pars.nodes_rearrange[type_2_node];
        old_llh_mat.tube(node_1,node_2) = new_llh_mat.tube(node_1,node_2);
      }
    }
    total_acceptance.betas_accepted.at(type_1,type_2)++;
  }
  return;
}


void dy_het_network::update_radius(void){
  double &kappa = update_pars.radius_kappa;
  mat &radius = current_pars.radius;
  mat old_radius = current_pars.radius;
  mat &new_radius = proposal_pars.radius;
  n_n accept = 0;
  double accept_prob = 0;
  double rand_u = randu();
  cube old_llh = current_buf.log_likelihood_mat;

  new_radius = old_radius;
  new_radius.for_each([&kappa](double &elem) {elem *= kappa;});
  r_dirichlet(new_radius);

  accept_prob -= accu(old_llh);
  radius = new_radius;
  update_current_llh();
  accept_prob += accu(current_buf.log_likelihood_mat);

  accept_prob = accept_prob>0 ? 1:exp(accept_prob);
  if(rand_u < accept_prob){
    total_acceptance.radius_accepted += 1;
  }else{
    radius = old_radius;
  }
  return;
}

void dy_het_network::show_acceptance(void) const{
  cout << "positions accepted:" << endl << total_acceptance.positions_accepted;
  cout << "beta accepted:" << endl << total_acceptance.betas_accepted;
  cout << "radius accepted: " << total_acceptance.radius_accepted << endl;
  return;
}
