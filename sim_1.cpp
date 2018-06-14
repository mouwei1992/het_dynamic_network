#include<ctime>

#include "configure.h"
#include "file_utils.h"
#include "dynamic_het.h"
#include "dynamic_het.cpp"


using namespace std;
using namespace arma;

int main(){
    Cube<Q_Q> adj_mat(150,150,4);
    // small sub adjacency matrix for debuging..
    Cube<Q_Q> sub_adj_mat(5,5,4);
    Col<Q_Q> sub_types(5);
    sub_types = {0,0,0,1,1};
    Mat<Q_Q> adj_mat_1 = csv_to_mat<Q_Q>("sim_1_t1.csv");
    Mat<Q_Q> adj_mat_2 = csv_to_mat<Q_Q>("sim_1_t2.csv");
    Mat<Q_Q> adj_mat_3 = csv_to_mat<Q_Q>("sim_1_t3.csv");
    Mat<Q_Q> adj_mat_4 = csv_to_mat<Q_Q>("sim_1_t4.csv");

    adj_mat.slice(0) = adj_mat_1;
    adj_mat.slice(1) = adj_mat_2;
    adj_mat.slice(2) = adj_mat_3;
    adj_mat.slice(3) = adj_mat_4;

    sub_adj_mat.slice(0) = adj_mat_1.submat(0,0,4,4);
    sub_adj_mat.slice(1) = adj_mat_2.submat(0,0,4,4);
    sub_adj_mat.slice(2) = adj_mat_3.submat(0,0,4,4);
    sub_adj_mat.slice(3) = adj_mat_4.submat(0,0,4,4);

    Col<Q_Q> types = csv_to_mat<Q_Q>("sim_1_types.csv").t();
    // network_info n_i_1 = network_info_configure(adj_mat,types);
    network_info n_i_1 = network_info_configure(adj_mat, types);
    dy_het_network dy_het_sim_1(n_i_1);

    time_t t_begin, t_end;

    time(&t_begin);
    dy_het_sim_1.update(4000);
    time(&t_end);

    cout << "total time spent (in seconds): " << difftime(t_end, t_begin) << endl;

    cout << dy_het_sim_1.duplicate_current_pars().radius;
    /*
    cout << dy_het_sim_1.duplicate_current_pars().betas;
    cout << dy_het_sim_1.duplicate_current_pars().s2 << endl;
    cout << dy_het_sim_1.duplicate_current_pars().t2 << endl;
    mat pos_1 = dy_het_sim_1.duplicate_current_pars().positions.slice(0);
    dy_het_sim_1.show_acceptance();

    // writing to file
    // mat_to_csv("sim_1_res_pos_1.csv", pos_1);
    */

    return 0;
}
