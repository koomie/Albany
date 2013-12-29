//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"

template<class SizeField>
Teuchos::RCP<SizeField> AAdapt::MeshAdapt<SizeField>::szField = Teuchos::null;

template<class SizeField>
AAdapt::MeshAdapt<SizeField>::
MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
          const Teuchos::RCP<ParamLib>& paramLib_,
          Albany::StateManager& StateMgr_,
          const Teuchos::RCP<const Epetra_Comm>& comm_) :
  AAdapt::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
  remeshFileIndex(1) {

  disc = StateMgr_.getDiscretization();

  pumi_discretization = Teuchos::rcp_dynamic_cast<AlbPUMI::AbstractPUMIDiscretization>(disc);

  fmdbMeshStruct = pumi_discretization->getFMDBMeshStruct();

  mesh = fmdbMeshStruct->getMesh();

  szField = Teuchos::rcp(new SizeField(pumi_discretization));

  num_iterations = params_->get<int>("Max Number of Mesh Adapt Iterations", 1);

  // Save the initial output file name
  base_exo_filename = fmdbMeshStruct->outputFileName;

  adaptation_method = params_->get<std::string>("Method");

  if ( adaptation_method.compare(0,15,"RPI SPR Size") == 0 )
    checkValidStateVariable(params_->get<std::string>("State Variable",""));

  pPart part;
  PUMI_Mesh_GetPart(mesh,0,part);
  pGeomMdl model;
  PUMI_Mesh_GetGeomMdl(mesh,model);

  /** Type of the size field:
      - Application - the size field will be provided by the application (default).
      - TagDriven - tag driven size field.
      - Analytical - analytical size field.  */
  /** Type of model:
      - 0 - no model (not snap), 1 - mesh model (always snap), 2 - solid model (always snap)
  */

  //    rdr = Teuchos::rcp(new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 2 ));
  rdr = Teuchos::rcp(new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 0));

  // callback for solution transfer
  callback = Teuchos::rcp(new ma::FieldCallback((&(*rdr)),fmdbMeshStruct->apfMesh));

}

template<class SizeField>
AAdapt::MeshAdapt<SizeField>::
~MeshAdapt() {
}

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::queryAdaptationCriteria() {

  if(adapt_params_->get<std::string>("Remesh Strategy", "None").compare("Continuous") == 0){

    if(iter > 1)

      return true;

    else

      return false;

  }


  Teuchos::Array<int> remesh_iter = adapt_params_->get<Teuchos::Array<int> >("Remesh Step Number");

  for(int i = 0; i < remesh_iter.size(); i++)

    if(iter == remesh_iter[i])

      return true;

  return false;

}

template<class SizeField>
int
AAdapt::MeshAdapt<SizeField>::setSizeField(pPart part, pSField pSizeField, void* vp) {

  return szField->computeSizeField(part, pSizeField);

}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::printElementData() {

  Albany::StateArrays& sa = disc->getStateArrays();
  Albany::StateArrayVec& esa = sa.elemStateArrays;
  int numElemWorksets = esa.size();
  Teuchos::RCP<Albany::StateInfoStruct> stateInfo = state_mgr_.getStateInfoStruct();

  for(unsigned int i = 0; i < stateInfo->size(); i++) {

    const std::string stateName = (*stateInfo)[i]->name;
    const std::string init_type = (*stateInfo)[i]->initType;
    std::vector<int> dims;
    esa[0][stateName].dimensions(dims);
    int size = dims.size();

    std::cout << "Meshadapt: have element field \"" << stateName << "\" of type \"" << init_type << "\"" << std::endl;

    if(init_type == "scalar") {


      switch(size) {

        case 1:
          std::cout << "esa[ws][stateName](0)" << std::endl;
          break;

        case 2:
          std::cout << "esa[ws][stateName](cell, qp)" << std::endl;
          break;

        case 3:
          std::cout << "esa[ws][stateName](cell, qp, i)" << std::endl;
          break;

        case 4:
          std::cout << "esa[ws][stateName](cell, qp, i, j)" << std::endl;
          break;

      }
    }

    else if(init_type == "identity") {
      std::cout << "Have an identity matrix: " << "esa[ws][stateName](cell, qp, i, j)" << std::endl;
    }
  }
}

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol) {

  if(epetra_comm_->MyPID() == 0){
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "Adapting mesh using AAdapt::MeshAdapt method        " << std::endl;
    std::cout << "Iteration: " << iter                                  << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  }

 // Create a remeshed output file naming convention by adding the remesh_file_index_ ahead of the period
  std::size_t found = base_exo_filename.find("exo");
  if(found != std::string::npos){
    std::ostringstream ss;
    std::string str = base_exo_filename;
    ss << "_" << remeshFileIndex << ".";
    str.replace(str.find('.'), 1, ss.str());

    *output_stream_ << "Remeshing: renaming exodus output file to - " << str << std::endl;

    // Open the new exodus file for results
    pumi_discretization->reNameExodusOutput(str);

    remeshFileIndex++;

  }


  // display # entities before adaptation

  FMDB_Mesh_DspSize(mesh);

  apf::Mesh2* m = fmdbMeshStruct->apfMesh;
  apf::Field* solution = m->findField("solution");
  // replace nodes' coordinates with displaced coordinates
  if ( ! PCU_Comm_Self())
    fprintf(stderr,"assuming deformation problem: displacing coordinates\n");
  apf::displaceMesh(m,solution);

  szField->setParams(&sol, &ovlp_sol,
                     adapt_params_->get<double>("Target Element Size", 0.1),
		     adapt_params_->get<double>("Error Bound", 0.01),
		     adapt_params_->get<std::string>("State Variable", ""));

  szField->computeError();

  /** void meshAdapt::run(int niter,    // specify the maximum number of iterations
        int flag,           // indicate if a size field function call is available
        adaptSFunc sizefd)  // the size field function call  */

  rdr->run(num_iterations, 1, this->setSizeField);

  if ( adaptation_method.compare(0,15,"RPI SPR Size") == 0 ) {
    apf::destroyField(m->findField("size"));
  }
  
  // replace nodes' displaced coordinates with coordinates
  apf::displaceMesh(m,solution,-1.0);

  // display # entities after adaptation
  FMDB_Mesh_DspSize(mesh);

  // Reinitialize global and local ids in FMDB
  PUMI_Exodus_Init(mesh);  // generate global/local id

  // Throw away all the Albany data structures and re-build them from the mesh
  // Note that the solution transfer for the QP fields happens in this call
  pumi_discretization->updateMesh();

  return true;

}


//! Transfer solution between meshes.
template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution) {

// Lets check the output of the solution transfer, it needs to be complete here as once this function returns LOCA
// begins the equilibration step

  pumi_discretization->debugMeshWrite(newSolution, "debug_output.exo");

}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::checkValidStateVariable(const std::string name) {

  if (name.length() > 0) {

    // does state variable exist?
    std::string stateName;
    
    Albany::StateArrays& sa = disc->getStateArrays();
    Albany::StateArrayVec& esa = sa.elemStateArrays;
    Teuchos::RCP<Albany::StateInfoStruct> stateInfo = state_mgr_.getStateInfoStruct();
    bool exists = false;
    for(unsigned int i = 0; i < stateInfo->size(); i++) {
      stateName = (*stateInfo)[i]->name;
      if ( name.compare(0,100,stateName) == 0 ){
	exists = true; 
        break;
      }
    }
    if (!exists)
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				 "Error!    Invalid State Variable Parameter!");
    
    // is state variable a 3x3 tensor?
    
    std::vector<int> dims;
    esa[0][name].dimensions(dims);
    int size = dims.size();
    if (size != 4)
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      "Error! Invalid State Variable Parameter \"" << name << "\" looking for \"" << stateName << "\"" << std::endl);
  }
}

template<class SizeField>
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::MeshAdapt<SizeField>::getValidAdapterParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");

  Teuchos::Array<int> defaultArgs;

  validPL->set<Teuchos::Array<int> >("Remesh Step Number", defaultArgs, "Iteration step at which to remesh the problem");
  validPL->set<std::string>("Remesh Strategy", "", "Strategy to use when remeshing: Continuous - remesh every step.");
  validPL->set<int>("Max Number of Mesh Adapt Iterations", 1, "Number of iterations to limit meshadapt to");
  validPL->set<double>("Target Element Size", 0.1, "Seek this element size when isotropically adapting");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<std::string>("State Variable", "", "Error is estimated using this state variable at integration points. Must be a 3x3 tensor. If no state variable is specified during error-estimation based adaptivity, then the gradient of solution field will be recovered and used");
  
  return validPL;
}


