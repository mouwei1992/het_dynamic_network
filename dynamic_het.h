#ifndef DY_HET_NETWORK
#define DY_HET_NETWORK

#include "dynamic_het_types.h"
#include "armadillo"

using namespace std;
using namespace arma;

// n_n meaning non-negative
typedef unsigned int n_n;
// just to make it cute..
typedef unsigned char Q_Q;


class dy_het_network{
  private:
    // given network information..
    network_structure structure_pars;
    void init_buf(void);
    void init_acceptance(void);
    void init_draw_from_prior(void);
    // paramters
    network_update_pars update_pars;
    network_prior_pars prior_pars;
    network_model_pars current_pars;
    network_model_pars proposal_pars;
    acceptance_stat total_acceptance;

    // buff/cache for some calculated quantities
    network_cache current_buf;
    void update_current_llh(void);
    void update_current_dist(void);
    network_cache proposal_buf;

    // update parameters function
    inline double eta_cal(const double &dist,const double &beta,const double &radius) { return beta * (radius / dist - dist / radius); }
    inline double llh_cal(Q_Q &, double &);
    void update_positions(void);
    void update_position(const n_n &,const n_n &);
    void update_s2(void);
    void update_t2(void);
    void update_betas(void);
    void update_beta(const Q_Q &, const Q_Q &);
    void update_radius(void);
  public:
    // initialization
    dy_het_network(network_info &);
    // MCMC update functions pack
    int update(n_n);
    void show_acceptance(void) const;
    network_model_pars duplicate_current_pars(void) const;
    static void *update_beta_single_helper(void *);
    void update_beta_single(const Q_Q &, const Q_Q &, const Q_Q &);
    void update_current_llh_single(Q_Q &);
    static void *update_current_llh_single_helper(void *);
    // static void *update_position_single_helper(void *);
    // void update_position_single(Col<n_n> *, Q_Q &, pthread_cond_t *, pthread_cond_t *, pthread_mutex_t *);
};


#endif
