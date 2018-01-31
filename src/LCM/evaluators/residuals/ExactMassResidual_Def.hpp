//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <MiniTensor_Mechanics.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>

#ifdef ALBANY_TIMER
#include <chrono>
#endif

//IKT: uncomment the following for debug output
//#define DEBUG_OUTPUT

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
ExactMassResidualBase<EvalT, Traits>::
ExactMassResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
 :
      w_bf_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      weights_("Weights", dl->qp_scalar),
      mass_(p.get<std::string>("Exact Mass Name"), dl->node_vector), 
      out_(Teuchos::VerboseObjectBase::getDefaultOStream())
{
#ifdef DEBUG_OUTPUT 
  *out_ << "IKT ExactMassResidualBase! \n"; 
#endif
  if (p.isParameter("Density"))  
    density_ = p.get<RealType>("Density"); 

  resid_using_cub_ = p.get<bool>("Residual Computed Using Cubature"); 
  use_composite_tet_ = p.get<bool>("Use Composite Tet 10"); 
  use_exact_mass_ = p.get<bool>("Use Exact Mass"); 

#ifdef DEBUG_OUTPUT 
  *out_ << "IKT resid_using_cub, use_composite_tet, use_exact_mass = " << resid_using_cub_ << ", " 
        << use_composite_tet_ << ", " << use_exact_mass_ << "\n"; 
#endif
  this->addDependentField(w_bf_);
  this->addDependentField(weights_);
  this->addEvaluatedField(mass_);

  if (p.isType<bool>("Disable Dynamics"))
    enable_dynamics_ = !p.get<bool>("Disable Dynamics");
  else
    enable_dynamics_ = true;

  if (enable_dynamics_) {
    accel_qps_ = decltype(accel_qps_)(
        p.get<std::string>("Acceleration Name"), dl->qp_vector);
    this->addDependentField(accel_qps_);
    accel_nodes_ = decltype(accel_nodes_)(
        p.get<std::string>("Acceleration Name"), dl->node_vector);
    this->addDependentField(accel_nodes_);
  }

  this->setName("ExactMassResidual" + PHX::typeAsString<EvalT>());


  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->node_qp_vector;
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
  num_cells_ = dims[0]; 
#ifdef DEBUG_OUTPUT
  *out_ << "IKT num_cells_, num_nodes, num_pts_, num_dims = " << num_cells_ << ", " << num_nodes_ << ", " << num_pts_ << ", " << num_dims_ << "\n"; 
#endif

  //Infer what is element type based on # of nodes 
  switch(num_nodes_) {
    case 4:
      elt_type = TET4; 
      break;
    case 8: 
      elt_type = HEX8; 
    case 10: 
      if (use_composite_tet_ == true) elt_type = CT10; 
      else elt_type = TET10; 
      break;
    default:
      elt_type = UNSUPPORTED;
  } 

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
 
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ExactMassResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(weights_, fm);
  this->utils.setFieldData(mass_, fm);
  if (enable_dynamics_) {
    this->utils.setFieldData(accel_qps_, fm);
    this->utils.setFieldData(accel_nodes_, fm);
  }
}

template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
tet4LocalMassRow(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(4);
  switch(row) {
    case 0: 
      mass_row[0] = 2.0; mass_row[1] = 1.0; 
      mass_row[2] = 1.0; mass_row[3] = 1.0; 
      break; 
    case 1: 
      mass_row[0] = 1.0; mass_row[1] = 2.0; 
      mass_row[2] = 1.0; mass_row[3] = 1.0; 
      break; 
    case 2: 
      mass_row[0] = 1.0; mass_row[1] = 1.0; 
      mass_row[2] = 2.0; mass_row[3] = 1.0; 
      break; 
    case 3: 
      mass_row[0] = 1.0; mass_row[1] = 1.0; 
      mass_row[2] = 1.0; mass_row[3] = 2.0; 
      break; 
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to tet4LocalMassRow! \n"
                                  << "Row must be between 0 and 3.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol * 6.0 * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] /= 120.0; 
    mass_row[i] *= scale;  
  }
  return mass_row; 
}


template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
tet4LocalMassRowLumped(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(4);
  switch(row) {
    case 0: 
      mass_row[0] = 1.0; 
      break; 
    case 1: 
      mass_row[1] = 1.0; 
      break; 
    case 2: 
      mass_row[2] = 1.0; 
      break; 
    case 3: 
      mass_row[3] = 1.0; 
      break; 
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to tet4LocalMassRowLumped! \n"
                                  << "Row must be between 0 and 3.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol * 6.0 * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] /= 24.0; 
    mass_row[i] *= scale;  
  }
  return mass_row; 
}

 
template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
tet10LocalMassRow(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(10);
  switch(row) {
    case 0:
      mass_row[0] = 6.0; mass_row[1] = 1.0; 
      mass_row[2] = 1.0; mass_row[3] = 1.0; 
      mass_row[4] = -4.0; mass_row[5] = -6.0; 
      mass_row[6] = -4.0; mass_row[7] = -4.0; 
      mass_row[8] = -6.0; mass_row[9] = -6.0;  
      break; 
    case 1:
      mass_row[0] = 1.0; mass_row[1] = 6.0; 
      mass_row[2] = 1.0; mass_row[3] = 1.0; 
      mass_row[4] = -4.0; mass_row[5] =-4.0; 
      mass_row[6] = -6.0; mass_row[7] = -6.0; 
      mass_row[8] = -4.0; mass_row[9] = -6.0; 
      break;
    case 2:
      mass_row[0] =1.0; mass_row[1] = 1.0; 
      mass_row[2] = 6.0; mass_row[3] = 1.0; 
      mass_row[4] = -6.0; mass_row[5] = -4.0; 
      mass_row[6] = -4.0; mass_row[7] = -6.0; 
      mass_row[8] = -6.0; mass_row[9] = -4.0; 
      break; 
    case 3: 
      mass_row[0] = 1.0; mass_row[1] = 1.0;
      mass_row[2] = 1.0; mass_row[3] = 6.0; 
      mass_row[4] = -6.0; mass_row[5] = -6.0; 
      mass_row[6] = -6.0; mass_row[7] = -4.0; 
      mass_row[8] = -4.0; mass_row[9] = -4.0; 
      break; 
    case 4: 
      mass_row[0] = -4.0; mass_row[1] = -4.0; 
      mass_row[2] = -6.0; mass_row[3] = -6.0; 
      mass_row[4] = 32.0; mass_row[5] = 16.0; 
      mass_row[6] = 16.0; mass_row[7] = 16.0; 
      mass_row[8] = 16.0; mass_row[9] = 8.0; 
      break; 
    case 5:
      mass_row[0] = -6.0; mass_row[1] = -4.0;
      mass_row[2] = -4.0; mass_row[3] = -6.0;
      mass_row[4] = 16.0; mass_row[5] = 32.0;
      mass_row[6] = 16.0; mass_row[7] = 8.0;
      mass_row[8] = 16.0; mass_row[9] = 16.0;
      break; 
    case 6: 
      mass_row[0] = -4.0; mass_row[1] = -6.0; 
      mass_row[2] = -4.0; mass_row[3] = -6.0; 
      mass_row[4] = 16.0; mass_row[5] = 16.0;  
      mass_row[6] = 32.0; mass_row[7] = 16.0; 
      mass_row[8] = 8.0; mass_row[9] = 16.0; 
      break; 
    case 7: 
      mass_row[0] = -4.0; mass_row[1] = -6.0; 
      mass_row[2] = -6.0; mass_row[3] = -4.0; 
      mass_row[4] = 16.0; mass_row[5] = 8.0; 
      mass_row[6] = 16.0; mass_row[7] = 32.0; 
      mass_row[8] = 16.0; mass_row[9] = 16.0; 
      break; 
    case 8: 
      mass_row[0] = -6.0; mass_row[1] = -4.0; 
      mass_row[2] = -6.0; mass_row[3] = -4.0; 
      mass_row[4] = 16.0; mass_row[5] = 16.0; 
      mass_row[6] = 8.0; mass_row[7] = 16.0; 
      mass_row[8] = 32.0; mass_row[9] = 16.0; 
      break; 
    case 9: 
      mass_row[0] = -6.0; mass_row[1] = -6.0; 
      mass_row[2] = -4.0; mass_row[3] = -4.0; 
      mass_row[4] = 8.0; mass_row[5] = 16.0; 
      mass_row[6] = 16.0; mass_row[7] = 16.0; 
      mass_row[8] = 16.0; mass_row[9] = 32.0; 
      break;  
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to tet10LocalMassRow! \n"
                                  << "Row must be between 0 and 9.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol * 6.0 * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] /= 2520.0 ; 
    mass_row[i] *= scale;  
  }
  return mass_row; 
}

template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
tet10LocalMassRowLumped(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(10);
  switch(row) {
    case 0:
      mass_row[0] = -1.0; 
      break; 
    case 1:
      mass_row[1] = -1.0; 
      break;
    case 2:
      mass_row[2] = -1.0; 
      break;
    case 3:
      mass_row[3] = -1.0; 
      break;
    case 4:
      mass_row[4] = 4.0; 
      break;
    case 5:
      mass_row[5] = 4.0; 
      break;
    case 6:
      mass_row[6] = 4.0; 
      break;
    case 7:
      mass_row[7] = 4.0; 
      break;
    case 8:
      mass_row[8] = 4.0; 
      break;
    case 9:
      mass_row[9] = 4.0; 
      break;
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to tet10LocalMassRowLumped! \n"
                                  << "Row must be between 0 and 9.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol * 6.0 * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] /= 120.0 ; 
    mass_row[i] *= scale;  
  }
  return mass_row; 
}

 
template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
hex8LocalMassRow(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(8);
  switch(row) {
    case 0: 
      mass_row[0] = 8.0; mass_row[1] = 4.0; mass_row[2] = 2.0; 
      mass_row[3] = 4.0; mass_row[4] = 4.0; mass_row[5] = 2.0; 
      mass_row[6] = 1.0; mass_row[7] = 2.0; 
      break; 
    case 1: 
      mass_row[0] = 4.0; mass_row[1] = 8.0; mass_row[2] = 4.0; 
      mass_row[3] = 2.0; mass_row[4] = 2.0; mass_row[5] = 4.0; 
      mass_row[6] = 2.0; mass_row[7] = 1.0; 
      break;
    case 2:
      mass_row[0] = 2.0; mass_row[1] = 4.0; mass_row[2] = 8.0; 
      mass_row[3] = 4.0; mass_row[4] = 1.0; mass_row[5] = 2.0; 
      mass_row[6] = 4.0; mass_row[7] = 2.0; 
      break; 
    case 3:
      mass_row[0] = 4.0; mass_row[1] = 2.0; mass_row[2] = 4.0; 
      mass_row[3] = 8.0; mass_row[4] = 2.0; mass_row[5] = 1.0; 
      mass_row[6] = 2.0; mass_row[7] = 4.0; 
      break; 
    case 4:
      mass_row[0] = 4.0; mass_row[1] = 2.0; mass_row[2] = 1.0; 
      mass_row[3] = 2.0; mass_row[4] = 8.0; mass_row[5] = 4.0; 
      mass_row[6] = 2.0; mass_row[7] = 4.0; 
      break; 
    case 5: 
      mass_row[0] = 2.0; mass_row[1] = 4.0; mass_row[2] = 2.0; 
      mass_row[3] = 1.0; mass_row[4] = 4.0; mass_row[5] = 8.0;
      mass_row[6] = 4.0; mass_row[7] = 2.0; 
      break; 
    case 6: 
      mass_row[0] = 1.0; mass_row[1] = 2.0; mass_row[2] = 4.0; 
      mass_row[3] = 2.0; mass_row[4] = 2.0; mass_row[5] = 4.0; 
      mass_row[6] = 8.0; mass_row[7] = 4.0; 
      break; 
    case 7: 
      mass_row[0] = 2.0; mass_row[1] = 1.0; mass_row[2] = 2.0; 
      mass_row[3] = 4.0; mass_row[4] = 4.0; mass_row[5] = 2.0; 
      mass_row[6] = 4.0; mass_row[7] = 8.0; 
      break; 
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to hex8LocalMassRow! \n"
                                  << "Row must be between 0 and 7.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol / 8.0 * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] /= 27.0 ;
    mass_row[i] *= scale;  
  }
  return mass_row; 
}
 
template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
hex8LocalMassRowLumped(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(8);
  switch(row) {
    case 0: 
      mass_row[0] = 1.0;  
      break; 
    case 1: 
      mass_row[1] = 1.0;  
      break; 
    case 2: 
      mass_row[2] = 1.0;  
      break; 
    case 3: 
      mass_row[3] = 1.0;  
      break; 
    case 4: 
      mass_row[4] = 1.0;  
      break; 
    case 5: 
      mass_row[5] = 1.0;  
      break; 
    case 6: 
      mass_row[6] = 1.0;  
      break; 
    case 7: 
      mass_row[7] = 1.0;  
      break; 
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to hex8LocalMassRowLumped! \n"
                                  << "Row must be between 0 and 7.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol / 8.0 * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] *= scale;  
  }
  return mass_row; 
}

 
template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
compositeTet10LocalMassRow(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(10); 
  switch(row) {
    case 0:
      mass_row[0] = 18.0; mass_row[1] = 0.0; 
      mass_row[2] = 0.0; mass_row[3] = 0.0; 
      mass_row[4] = 9.0; mass_row[5] = 0.0; 
      mass_row[6] = 9.0; mass_row[7] = 9.0; 
      mass_row[8] = 0.0; mass_row[9] = 0.0;  
      break; 
    case 1:
      mass_row[0] = 0.0; mass_row[1] = 18.0; 
      mass_row[2] = 0.0; mass_row[3] = 0.0; 
      mass_row[4] = 9.0; mass_row[5] = 9.0; 
      mass_row[6] = 0.0; mass_row[7] = 0.0; 
      mass_row[8] = 9.0; mass_row[9] = 0.0;  
      break;
    case 2: 
      mass_row[0] = 0.0; mass_row[1] = 0.0; 
      mass_row[2] = 18.0; mass_row[3] = 0.0; 
      mass_row[4] = 0.0; mass_row[5] = 9.0; 
      mass_row[6] = 9.0; mass_row[7] = 0.0;  
      mass_row[8] = 0.0; mass_row[9] = 9.0;  
      break; 
    case 3: 
      mass_row[0] = 0.0; mass_row[1] = 0.0; 
      mass_row[2] = 0.0; mass_row[3] = 18.0; 
      mass_row[4] = 0.0; mass_row[5] = 0.0; 
      mass_row[6] = 0.0; mass_row[7] = 9.0; 
      mass_row[8] = 9.0; mass_row[9] = 9.0; 
      break;
    case 4: 
      mass_row[0] = 9.0; mass_row[1] = 9.0; 
      mass_row[2] = 0.0; mass_row[3] = 0.0; 
      mass_row[4] = 80.0; mass_row[5] = 26.0; 
      mass_row[6] = 26.0; mass_row[7] = 26.0; 
      mass_row[8] = 26.0; mass_row[9] = 8.0;  
      break; 
    case 5:
      mass_row[0] = 0.0; mass_row[1] = 9.0; 
      mass_row[2] = 9.0; mass_row[3] = 0.0; 
      mass_row[4] = 26.0; mass_row[5] = 80.0; 
      mass_row[6] = 26.0; mass_row[7] = 8.0; 
      mass_row[8] = 26.0; mass_row[9] = 26.0;  
      break; 
    case 6:
      mass_row[0] = 9.0; mass_row[1] = 0.0; 
      mass_row[2] = 9.0; mass_row[3] = 0.0; 
      mass_row[4] = 26.0; mass_row[5] = 26.0; 
      mass_row[6] = 80.0; mass_row[7] = 26.0; 
      mass_row[8] = 8.0; mass_row[9] = 26.0; 
      break; 
    case 7:
      mass_row[0] = 9.0; mass_row[1] = 0.0; 
      mass_row[2] = 0.0; mass_row[3] = 9.0; 
      mass_row[4] = 26.0; mass_row[5] = 8.0; 
      mass_row[6] = 26.0; mass_row[7] = 80.0; 
      mass_row[8] = 26.0; mass_row[9] = 26.0;  
      break; 
    case 8:
      mass_row[0] = 0.0; mass_row[1] = 9.0; 
      mass_row[2] = 0.0; mass_row[3] = 9.0; 
      mass_row[4] = 26.0; mass_row[5] = 26.0; 
      mass_row[6] = 8.0; mass_row[7] = 26.0; 
      mass_row[8] = 80.0; mass_row[9] = 26.0;  
      break; 
    case 9:
      mass_row[0] = 0.0; mass_row[1] = 0.0; 
      mass_row[2] = 9.0; mass_row[3] = 9.0; 
      mass_row[4] = 8.0; mass_row[5] = 26.0; 
      mass_row[6] = 26.0; mass_row[7] = 26.0; 
      mass_row[8] = 26.0; mass_row[9] = 80.0; 
      break; 
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to compositeTet10LocalMassRow! \n"
                                  << "Row must be between 0 and 9.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] /= 1440.0; 
    mass_row[i] *= scale;  
  }
  return mass_row; 
}


template<typename EvalT, typename Traits>
std::vector<RealType> ExactMassResidualBase<EvalT, Traits>::
compositeTet10LocalMassRowLumped(const int cell, const int row) const 
{
  std::vector<RealType> mass_row(10); 
  switch(row) {
    case 0:
      mass_row[0] = 3.0; 
      break; 
    case 1:
      mass_row[1] = 3.0; 
      break;
    case 2:
      mass_row[2] = 3.0; 
      break;
    case 3:
      mass_row[3] = 3.0; 
      break;
    case 4:
      mass_row[4] = 14.0; 
      break;
    case 5:
      mass_row[5] = 14.0; 
      break;
    case 6:
      mass_row[6] = 14.0; 
      break;
    case 7:
      mass_row[7] = 14.0; 
      break;
    case 8:
      mass_row[8] = 14.0; 
      break;
    case 9:
      mass_row[9] = 14.0; 
      break;
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to compositeTet10LocalMassRowLumped! \n"
                                  << "Row must be between 0 and 9.\n"); 
  }
  const RealType elt_vol = computeElementVolume(cell); 
  const RealType scale = elt_vol * density_;
  for (int i=0; i<mass_row.size(); i++) {
    mass_row[i] /= 96.0; 
    mass_row[i] *= scale;  
  }
  return mass_row; 
}


template<typename EvalT, typename Traits>
RealType ExactMassResidualBase<EvalT, Traits>::
computeElementVolScaling(const int cell, const int node) const 
{
  RealType elt_vol_scale_at_node = 0.0; 
  for (int pt = 0; pt < num_pts_; ++pt) {
    elt_vol_scale_at_node += w_bf_(cell, node, pt);
  }
#ifdef DEBUG_OUTPUT
  if (cell == 0) 
    *out_ << "  IKT node, elt_vol_scale_at_node = " << node << ", " << elt_vol_scale_at_node << "\n"; 
#endif
  return elt_vol_scale_at_node; 
}

template<typename EvalT, typename Traits>
RealType ExactMassResidualBase<EvalT, Traits>::
computeElementVolume(const int cell) const 
{
  RealType elt_vol = 0.0; 
  for (int pt = 0; pt < num_pts_; ++pt) {
    elt_vol += weights_(cell, pt); 
  }
#ifdef DEBUG_OUTPUT
  if (cell == 0) 
    *out_ << "  IKT elt_vol = " << elt_vol << "\n"; 
#endif
  return elt_vol; 
}

template<typename EvalT, typename Traits>
void ExactMassResidualBase<EvalT, Traits>::
computeResidualValue(typename Traits::EvalData workset) const 
{
  //Zero out mass_ 
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < this->num_nodes_; ++node) {
      for (int dim = 0; dim < this->num_dims_; ++dim) {
        (this->mass_)(cell,node,dim) = ScalarT(0.0);
      }
    }
  }
  if (resid_using_cub_ == true) {
    //Approach 1: uses numerical cubature to compute residual contribution 
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < this->num_nodes_; ++node) {
        for (int pt = 0; pt < this->num_pts_; ++pt) {
          for (int dim = 0; dim < this->num_dims_; ++dim) {
            (this->mass_)(cell, node, dim) +=
              (this->density_) * (this->accel_qps_)(cell, pt, dim) * (this->w_bf_)(cell, node, pt);
          }
        }
      }
    }
  }
  else {
    //Approach 2: uses mass matrix to compute residual contribution (r = rho*M*a)
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < this->num_nodes_; ++node) { //loop over rows
        std::vector<RealType> mass_row;
        switch (this->elt_type) 
        {
          case this->TET4:
            mass_row = this->tet4LocalMassRow(cell, node);
            break;
          case this->HEX8: 
            mass_row = this->hex8LocalMassRow(cell, node);
            break; 
          case this->TET10: 
            mass_row = this->tet10LocalMassRow(cell, node);
            break;
          case this->CT10: 
            mass_row = this->compositeTet10LocalMassRow(cell, node);
            break; 
        } 
        for (int dim = 0; dim < this->num_dims_; ++dim) {
          ScalarT val = 0.0; 
          for (int i = 0; i < this->num_nodes_; ++i) { //loop over columns
            val += mass_row[i]*accel_nodes_(cell, i, dim); 
          }
          (this->mass_)(cell, node, dim) += val; 
        }
      }
    }
  }
#ifdef DEBUG_OUTPUT
  for (int cell = 0; cell < workset.numCells; ++cell) {
    if (cell == 0) {
      for (int node = 0; node < this->num_nodes_; ++node) {
        for (int dim = 0; dim < this->num_dims_; ++dim) {
          *(this->out_) << "IKT node, dim, ct_mass = " << node << ", " << dim << ", " << (this->mass_)(cell, node, dim) << "\n";
        }
      }
    }
  } 
#endif
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ExactMassResidual<PHAL::AlbanyTraits::Residual,Traits>::
ExactMassResidual(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : ExactMassResidualBase<PHAL::AlbanyTraits::Residual,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void ExactMassResidual<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef DEBUG_OUTPUT 
  *(this->out_) << "IKT ExactMassResidual Residual Specialization evaluateFields!\n";
#endif
  if (this->use_exact_mass_ == false) 
    return; 

  //Throw error is trying to call with unsupported element type
  if (this->elt_type == this->UNSUPPORTED) {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                               "Error! ExactMassResidual is being run with unsupported element having \n" 
                                << this->num_nodes_ << " nodes.  Please re-run with 'Use Exact Mass' = 'false'.\n"); 
  }

  this->computeResidualValue(workset);  
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ExactMassResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
ExactMassResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ExactMassResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void ExactMassResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef DEBUG_OUTPUT 
  *(this->out_) << "IKT ExactMassResidual Jacobian Specialization evaluateFields!\n";
#endif

  if (this->use_exact_mass_ == false) 
    return; 

  //Throw error is trying to call with unsupported element type
  if (this->elt_type == this->UNSUPPORTED) {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                               "Error! ExactMassResidual is being run with unsupported element having \n" 
                                << this->num_nodes_ << " nodes.  Please re-run with 'Use Exact Mass' = 'false'.\n"); 
  }

  //Compute residual value 
  this->computeResidualValue(workset);  
 
  //Set local Jacobian entries 
  double n_coeff = workset.n_coeff;
#ifdef DEBUG_OUTPUT
  *(this->out_) << "  IKT n_coeff = " << n_coeff << "\n"; 
#endif 
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < this->num_nodes_; ++node) { //loop over Jacobian rows
      std::vector<RealType> mass_row;
      switch (this->elt_type) 
      {
        case this->TET4:
          mass_row = this->tet4LocalMassRow(cell, node);
          break;
        case this->HEX8: 
          mass_row = this->hex8LocalMassRow(cell, node);
          break; 
        case this->TET10: 
          mass_row = this->tet10LocalMassRow(cell, node);
          break;
        case this->CT10: 
          mass_row = this->compositeTet10LocalMassRow(cell, node);
          break; 
      } 
      for (int dim = 0; dim < this->num_dims_; ++dim) {
        typename PHAL::Ref<ScalarT>::type valref = (this->mass_)(cell,node,dim); //get Jacobian row 
        int k;
        for (int i=0; i < this->num_nodes_; ++i) { //loop over Jacobian cols 
          k = i*this->num_dims_ + dim;
          valref.fastAccessDx(k) = n_coeff*mass_row[i];
        }
      }
    }
  }
  
#ifdef DEBUG_OUTPUT
  for (int cell = 0; cell < workset.numCells; ++cell) {
    if (cell == 0) {
      for (int node = 0; node < this->num_nodes_; ++node) {
        for (int dim = 0; dim < this->num_dims_; ++dim) {
          *(this->out_) << "IKT node, dim, ct_mass = " << node << ", " << dim << ", " << (this->mass_)(cell, node, dim) << "\n";
        }
      }
    }
  } 
#endif
//------------------------------------------------------------------------------
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ExactMassResidual<PHAL::AlbanyTraits::Tangent, Traits>::
ExactMassResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ExactMassResidualBase<PHAL::AlbanyTraits::Tangent,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void ExactMassResidual<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                             "Error! Tangent specialization of ExactMassResidual not implemented!\n"); 
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ExactMassResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
ExactMassResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : ExactMassResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void ExactMassResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                             "Error! DistParamDeriv specialization of ExactMassResidual not implemented!\n"); 
}

}

