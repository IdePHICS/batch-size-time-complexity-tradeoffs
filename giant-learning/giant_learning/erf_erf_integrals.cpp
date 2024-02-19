#include <cmath>
#include <boost/math/constants/constants.hpp>

template<class T>
inline T square(T x){
  return x*x;
}

using namespace boost::math::constants;

namespace giant_learning{
namespace erf_erf {
  template<typename Real>
  inline Real I2(Real C11, Real C12, Real C22) {
    return std::asin(C12/std::sqrt((1.+C11)*(1.+C22)))/pi<Real>();
  }
  
  template<typename Real>
  inline Real I2_noise(Real C11, Real C12, Real C22) {
    return 2./(pi<Real>()*std::sqrt(1+C11+C22+(C11*C22)-C12*C12));
  }


  template<typename Real>
  inline Real I3(Real C11, Real C12, Real C13, Real C22, Real C23, Real C33) {
    Real L3 = (1+C11)*(1+C33)-C13*C13;
    return Real(2.)*(C23*(Real(1.)+C11)-C12*C13)/(pi<Real>()*(1+C11)*std::sqrt(L3));
  }

  template<typename Real>
  inline Real I4(Real C11, Real C12, Real C13, Real C14, Real C22, Real C23, Real C24, Real C33, Real C34, Real C44) {
    Real L4 = (1+C11)*(1+C22) - C12*C12;
    Real L0 = L4*C34 - C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23;
    Real L1 = L4*(1+C33) - C23*C23*(1+C11) - C13*C13*(1+C22) + 2*C12*C13*C23;
    Real L2 = L4*(1+C44) - C24*C24*(1+C11) - C14*C14*(1+C22) + 2*C12*C14*C24;
    return (4./(pi_sqr<Real>()*sqrt(L4)))*std::asin(L0/sqrt(L1*L2));
  }

}
}