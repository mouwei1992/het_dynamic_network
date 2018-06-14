#include<random>

class rand_u_generator{
  private:
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;
  public:
    rand_u_generator(void) {distribution = std::uniform_real_distribution<double>(0.0,1.0);}
    double randu(void){ return distribution(generator);}
};
