#ifndef LOG_APP_H
#define LOG_APP_H

static const double log_2 = log(2);
static const double tylor8 = 17/16./56./720.;
static const double tylor10 = 7936/1024./720./720./7.;
static const double tylor12 = 86.375/132./7./720./720.;
static const double fac10 = 10 * 72 * 7 * 720;

inline double approx_exp_Tylor(const double x, const int &n){
  // this function approximate exp(x) for x small
  double result = 1;
  for(int i = n ; i > 0 ; i--){
    result = result * x/double(i) + 1;
  }
  return result;
}

inline double lookup_table1(const double &x){
  // this approxiamtion is for approximating exp(x) for -20 < x < -5
  // does not check the bound
  static int constructed = 0;
  // if not coded in rcpp, could use valarray<double> instead:
  // const int Length = (20 - 5)/0.01 + 1;
  // static valarray<double> table1(Length);
  static std::vector<double> table1;
  if(!constructed){
    cout<<"constructing look_up tabel1"<<std::endl;
    // table for exp(x) ranging -5 - -19
    for(double table_term = 20; table_term >= 4.999; table_term -= 0.01)  table1.insert(table1.begin(),exp(-table_term));
    constructed = 1;
  }

  int ind = (-x - 5)*100;
  double res = -x - 5-ind/100.;
  double result = table1[ind];
  if(ind <= 500) result /= (res * res/2. + res + 1);
  else result /= (res+1);
  return result;
}

// no problem now save time.. 
inline double lookup_table2(const double &x){
  // return log(exp(x) + 1) directly
  static int constructed = 0;
  // table for exp(x) ranging 0.9 - 5
  static std::vector<double> table2;
  static std::vector<double> tylor_coef1;
  static std::vector<double> tylor_coef2;
  if(!constructed){
    cout<<"constructing look_up tabel2"<<std::endl;
    for(double x = 5; x >= 0.8999; x -= 0.001){
      table2.insert(table2.begin(),log(exp(x)+1));
      tylor_coef1.insert(tylor_coef1.begin(),exp(x)/(1+exp(x)));
      tylor_coef2.insert(tylor_coef2.begin(),exp(x)/(1+exp(x))/(1+exp(x))/2.);
    }
    constructed = 1;
  }

  int ind = (x - 0.9)*1000;
  double res = x - 0.9 - ind/1000.;
  double result = table2[ind] + res * tylor_coef1[ind] + res * res * tylor_coef2[ind];
  return result;
}

inline double lookup_table3(const double &x){
  // return log(exp(x) + 1) directly
  static int constructed = 0;
  // table for exp(x) ranging 0.9 - 5
  static std::vector<double> table3;
  static std::vector<double> tylor_coef1;
  static std::vector<double> tylor_coef2;
  if(!constructed){
    cout<<"constructing look_up tabel3"<<std::endl;
    for(double x = 5; x >= 0.8999; x -= 0.001){
      table3.insert(table3.begin(),log(exp(-x)+1));
      tylor_coef1.insert(tylor_coef1.begin(),exp(-x)/(1+exp(-x)));
      tylor_coef2.insert(tylor_coef2.begin(),exp(-x)/(1+exp(-x))/(1+exp(-x))/2.);
    }
    constructed = 1;
  }

  int ind = (-x - 0.9)*1000;
  double res = x + 0.9 + ind/1000.;
  double result = table3[ind] + res * tylor_coef1[ind] + res * res * tylor_coef2[ind];
  return result;
}

inline double approx_llh(const double &eta){
  double llh = 0;
  if(eta > 5){
    if(eta > 20) return -eta;
    double exp_neta = lookup_table1(-eta);
    // double exp_neta = exp(-eta);
    if(eta > 10) llh = - (eta + exp_neta);
    else llh = - (eta + exp_neta * ( 1 - exp_neta/2. * ( 1 - 2 * exp_neta/3.)));
  }
    
  if(eta < -5){
    if(eta < -20) return 0;
    double exp_eta = lookup_table1(eta);
    if(eta < -10) llh = - exp_eta * ( 1 - exp_eta/2. * ( 1 - 2 * exp_eta /3.));
  }
  return llh;
}

// this should work well
inline void approx_llh2(const double &eta, double &llh){
  double eta2 = eta * eta;
  double eta4 = eta2 * eta2;
  double eta6 = eta4 * eta2;
  double eta8 = eta4 * eta4;
  double eta10 = eta8 *eta2;
  llh = -(log_2 + eta/2. + eta2/8. - eta4/192. + eta6/2880. - eta8*tylor8 + eta10 * tylor10);
  if(fabs(eta) > 0.7){
    double eta12 = eta10 * eta2;
    llh += eta12 * tylor12;
  } 
}

// black technology to calculate llh function..

inline double logistic_prox(const unsigned char &y, const double &eta){
  double llh = 0;
  if(fabs(eta) > 5) llh = approx_llh(eta);
  else if(fabs(eta) < 0.9) approx_llh2(eta,llh);
  else if(eta >0) llh = - lookup_table2(eta);
  else llh = - lookup_table3(eta);
  if(y) llh += eta;
  return llh;
}

#endif