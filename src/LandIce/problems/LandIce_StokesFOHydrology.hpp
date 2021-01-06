//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_HYDROLOGY_PROBLEM_HPP
#define LANDICE_STOKES_FO_HYDROLOGY_PROBLEM_HPP 1

#include "LandIce_StokesFOBase.hpp"

#include "LandIce_HydrologyWaterDischarge.hpp"
#include "LandIce_HydrologyResidualMassEqn.hpp"
#include "LandIce_HydrologyResidualCavitiesEqn.hpp"
#include "LandIce_HydrologyMeltingRate.hpp"

namespace LandIce
{

/*
 * \brief The coupled problem StokesFO+Hydrology
 * 
 *   This problem is the union of StokesFO and Hydrology problems.
 *   You can look at StokesFO and StokesFOBase for more details on
 *   the ice stokes problem and Hydrology for details on the
 *   subglacial hydrology problem.
 *   This problem will solve the ice equation in 3d, while the
 *   hydrology problem is solved on the basal sideset of the mesh.
 */

class StokesFOHydrology : public StokesFOBase {
public:

  //! Default constructor
  StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<Teuchos::ParameterList>& discParams,
                     const Teuchos::RCP<ParamLib>& paramLib,
                     const int numDim_);

  //! Destructor
  ~StokesFOHydrology() = default;

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

private:

  //! Private to prohibit copying
  StokesFOHydrology(const StokesFOHydrology&) = delete;

  //! Private to prohibit copying
  StokesFOHydrology& operator=(const StokesFOHydrology&) = delete;

public:

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  template <typename EvalT>
  void constructHydrologyEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0); 

  // Flags to determine the kind of hydro problem
  bool eliminate_h;   // Solve for h in the cavity eqn, and eliminate the eqn (only for steady)
  bool unsteady;      // Whether the hydrology is steady or unsteady.
  bool has_h_till;    // Adds a third eqn for the water thickness in the till storage
  bool has_p_dot;     // Include C*d(p_w)/dt in the cavity equation (englacial porosity term)

  int hydro_neq;      // Number of hydrology equations
  int offset_hydro;   // Offset for hydrology eqn (equal to number of ice equations)

  // Diagnostic dofs do not appear under time derivative. The most common hydro case
  // only has h (water thickness) under time derivative, and not p (the water pressure).
  // Some pb config, however, may have p_dot.
  Teuchos::ArrayRCP<std::string> diagnostic_hydro_dofs_names;
  Teuchos::ArrayRCP<std::string> prognostic_hydro_dofs_names;
  Teuchos::ArrayRCP<std::string> prognostic_hydro_dofs_names_dot;

  static constexpr char hydraulic_potential_name[] = "hydraulic_potential";
  static constexpr char water_thickness_name[]     = "water_thickness";
  static constexpr char water_thickness_dot_name[] = "water_thickness_dot";
};

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOHydrology::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                               const Albany::MeshSpecsStruct& meshSpecs,
                                               Albany::StateManager& stateMgr,
                                               Albany::FieldManagerChoice fieldManagerChoice,
                                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // --- StokesFOBase evaluators --- //
  constructStokesFOBaseEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // --- Hydrology evaluators --- //
  constructHydrologyEvaluators<EvalT> (fm0);

  // --- Gather dofs --- //

  // Gather ice velocity field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names[0], dof_offsets[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather hydro dofs (possibly with time derivatives)
  if (unsteady) {
    if (diagnostic_hydro_dofs_names.size()>0) {
      ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, diagnostic_hydro_dofs_names);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    int offset = diagnostic_hydro_dofs_names.size();
    ev = evalUtils.constructGatherSolutionEvaluator (false, prognostic_hydro_dofs_names, prognostic_hydro_dofs_names_dot, offset);
    fm0.template registerEvaluator<EvalT> (ev);
  } else {
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, dof_names);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // --- Scatter residuals --- //

  // Scatter stokes residual
  // ev = evalUtils.constructScatterResidualEvaluator(true, stokes_resid_names, offsetStokes, "Scatter Stokes");
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter hydrology residual
       // const Teuchos::RCP<shards::CellTopology>& cellType,
       // const std::string& sideSetName,
       // bool residualsAreVolumeFields,
       // bool isVectorField,
       // const std::string& resid_name,
       // int offsetToFirstDOF=0, std::string scatterName="Scatter") const {
  ev = evalUtils.constructScatterSideEqnResidualEvaluator(cellType);
  fm0.template registerEvaluator<EvalT> (ev);
  // // ---------------------------- Registering state variables ------------------------- //

  // std::string stateName, fieldName, param_name;

  // // Getting the names of the distributed parameters (they won't have to be loaded as states)
  // std::map<std::string,bool> is_dist_param;
  // std::map<std::string,std::string> dist_params_name_to_mesh_part;
  // std::map<std::string,bool> is_extruded_param;
  // // The following are used later to check that the needed fields (depending on the simulation options) are successfully loaded/gathered
  // std::set<std::string> inputs_found;
  // std::map<std::string,std::set<std::string>> ss_inputs_found;

  // if (this->params->isSublist("Parameters")) {
  //   const Teuchos::ParameterList& parameterParams = this->params->sublist("Parameters");
  //   int total_num_param_vecs, num_param_vecs, num_dist_param_vecs;
  //   Albany::getParameterSizes(parameterParams, total_num_param_vecs, num_param_vecs, num_dist_param_vecs);

  //   for (unsigned int p_index=0; p_index< num_dist_param_vecs; ++p_index)
  //   {
  //     std::string parameter_sublist_name = Albany::strint("Parameter", p_index+num_param_vecs);
  //     param_list = parameterParams.sublist(parameter_sublist_name);
  //     param_name = param_list->get<std::string>("Name");
  //     dist_params_name_to_mesh_part[param_name] = param_list->get<std::string>("Mesh Part","");
  //     is_extruded_param[param_name] = param_list->get<bool>("Extruded",false);
  //     int extruded_param_level = 0;
  //     extruded_params_levels->insert(std::make_pair(param_name, extruded_param_level));
  //     is_dist_param[param_name] = true;
  //   }
  // }

  // // Registering 3D states and building their load/save/gather evaluators
  // if (discParams->isSublist("Required Fields Info"))
  // {
  //   Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  //   int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  //   std::string fieldType, fieldUsage, meshPart;
  //   bool nodal_state, outputToExodus;
  //   for (unsigned int ifield=0; ifield<num_fields; ++ifield)
  //   {
  //     Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

  //     // Get current state specs
  //     stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
  //     fieldType  = thisFieldList.get<std::string>("Field Type");
  //     fieldUsage = thisFieldList.get<std::string>("Field Usage", "Input");

  //     if (fieldUsage == "Unused")
  //       continue;

  //     inputs_found.insert(stateName);
  //     outputToExodus = (fieldUsage == "Output" || fieldUsage == "Input-Output");
  //     meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

  //     if(fieldType == "Elem Scalar") {
  //       entity = Albany::StateStruct::ElemData;
  //       p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, outputToExodus, &entity, meshPart);
  //       nodal_state = false;
  //     }
  //     else if(fieldType == "Node Scalar") {
  //       entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
  //       p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, outputToExodus, &entity, meshPart);
  //       nodal_state = true;
  //     }
  //     else if(fieldType == "Elem Vector") {
  //       entity = Albany::StateStruct::ElemData;
  //       p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, outputToExodus, &entity, meshPart);
  //       nodal_state = false;
  //     }
  //     else if(fieldType == "Node Vector") {
  //       entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
  //       p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, outputToExodus, &entity, meshPart);
  //       nodal_state = true;
  //     }

  //     // Do we need to save the state?
  //     if (outputToExodus)
  //     {
  //       // An output: save it.
  //       p->set<bool>("Nodal State", nodal_state);
  //       ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
  //       fm0.template registerEvaluator<EvalT>(ev);

  //       // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
  //       if (ev->evaluatedFields().size()>0)
  //         fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
  //     }

  //     // Do we need to load/gather the state/parameter?
  //     if (is_dist_param[stateName])
  //     {
  //       // A parameter: gather it
  //       ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
  //       fm0.template registerEvaluator<EvalT>(ev);
  //     }
  //     else if (fieldUsage == "Input" || fieldUsage == "Input-Output")
  //     {
  //       // Not a parameter but still required as input: load it.
  //       p->set<std::string>("Field Name", fieldName);
  //       ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  //       fm0.template registerEvaluator<EvalT>(ev);
  //     }
  //   }
  // }

  // // Registering 2D states and building their load/save/gather evaluators
  // // Note: we MUST have 'Side Set Discretizations', since this is the StokesFOHydrology coupling and we MUST have a basal mesh...
  // Teuchos::Array<std::string> ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
  // for (unsigned int i=0; i<ss_names.size(); ++i)
  // {
  //   const std::string& ss_name = ss_names[i];
  //   Teuchos::ParameterList& req_fields_info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
  //   int num_fields = req_fields_info.get<int>("Number Of Fields",0);
  //   Teuchos::RCP<PHX::DataLayout> dl_temp;
  //   Teuchos::RCP<PHX::DataLayout> sns;
  //   std::string fieldType, fieldUsage, meshPart;
  //   bool nodal_state, outputToExodus;
  //   int numLayers;

  //   const std::string& sideEBName = meshSpecs.sideSetMeshSpecs.at(ss_name)[0]->ebName;
  //   Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
  //   for (unsigned int ifield=0; ifield<num_fields; ++ifield)
  //   {
  //     Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

  //     // Get current state specs
  //     stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
  //     fieldType  = thisFieldList.get<std::string>("Field Type");
  //     fieldUsage = thisFieldList.get<std::string>("Field Usage", "Input");

  //     if (fieldUsage == "Unused")
  //       continue;

  //     ss_inputs_found[ss_name].insert(stateName);
  //     outputToExodus = (fieldUsage == "Output" || fieldUsage == "Input-Output");
  //     meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

  //     numLayers = thisFieldList.isParameter("Number Of Layers") ? thisFieldList.get<int>("Number Of Layers") : -1;
  //     fieldType  = thisFieldList.get<std::string>("Field Type");

  //     if(fieldType == "Elem Scalar") {
  //       entity = Albany::StateStruct::ElemData;
  //       p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_scalar2, sideEBName, outputToExodus, &entity, meshPart);
  //       nodal_state = false;
  //     }
  //     else if(fieldType == "Node Scalar") {
  //       entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
  //       p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_scalar, sideEBName, outputToExodus, &entity, meshPart);
  //       nodal_state = true;
  //     }
  //     else if(fieldType == "Elem Vector") {
  //       entity = Albany::StateStruct::ElemData;
  //       p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_vector, sideEBName, outputToExodus, &entity, meshPart);
  //       nodal_state = false;
  //     }
  //     else if(fieldType == "Node Vector") {
  //       entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
  //       p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_vector, sideEBName, outputToExodus, &entity, meshPart);
  //       nodal_state = true;
  //     }
  //     else if(fieldType == "Elem Layered Scalar") {
  //       entity = Albany::StateStruct::ElemData;
  //       sns = ss_dl->cell_scalar2;
  //       dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,LayerDim>(sns->extent(0),sns->extent(1),numLayers));
  //       stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, outputToExodus, &entity, meshPart);
  //     }
  //     else if(fieldType == "Node Layered Scalar") {
  //       entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
  //       sns = ss_dl->node_scalar;
  //       dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2),numLayers));
  //       stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, outputToExodus, &entity, meshPart);
  //     }
  //     else if(fieldType == "Elem Layered Vector") {
  //       entity = Albany::StateStruct::ElemData;
  //       sns = ss_dl->cell_vector;
  //       dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Dim,LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2),numLayers));
  //       stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, outputToExodus, &entity, meshPart);
  //     }
  //     else if(fieldType == "Node Layered Vector") {
  //       entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
  //       sns = ss_dl->node_vector;
  //       dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,Dim,LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2),
  //                                                                              sns->extent(3),numLayers));
  //       stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, outputToExodus, &entity, meshPart);
  //     }

  //     if (fieldUsage == "Unused")
  //       continue;

  //     if (outputToExodus)
  //     {
  //       // An output: save it.
  //       p->set<bool>("Nodal State", nodal_state);
  //       p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
  //       ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,ss_dl));
  //       fm0.template registerEvaluator<EvalT>(ev);

  //       // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
  //       if (ev->evaluatedFields().size()>0)
  //         fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
  //     }

  //     if (is_dist_param[stateName])
  //     {
  //       // A parameter: gather it
  //       if (is_extruded_param[stateName])
  //       {
  //         ev = evalUtils.constructGatherScalarExtruded2DNodalParameter(stateName,fieldName);
  //         fm0.template registerEvaluator<EvalT>(ev);
  //       }
  //       else
  //       {
  //         ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
  //         fm0.template registerEvaluator<EvalT>(ev);
  //       }
  //     }
  //     else if (fieldUsage == "Input" || fieldUsage == "Input-Output")
  //     {
  //       // Not a parameter but required as input: load it.
  //       p->set<std::string>("Field Name", fieldName);
  //       ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
  //       fm0.template registerEvaluator<EvalT>(ev);
  //     }
  //   }
  // }

  // // ------------------- Interpolations and utilities ------------------ //

  // int offsetStokes = 0;
  // int offsetHydro  = stokes_neq;

  // // Gather stokes solution field
  // ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, stokes_dof_names, offsetStokes);
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Gather hydrology solution field
  // if (has_h_equation && unsteady)
  // {
  //   ev = evalUtils.constructGatherSolutionEvaluator (false, hydro_dof_names, hydro_dof_names_dot, offsetHydro);
  //   fm0.template registerEvaluator<EvalT> (ev);
  // }
  // else
  // {
  //   ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, hydro_dof_names, offsetHydro);
  //   fm0.template registerEvaluator<EvalT> (ev);
  // }

  // // Scatter stokes residual
  // ev = evalUtils.constructScatterResidualEvaluator(true, stokes_resid_names, offsetStokes, "Scatter Stokes");
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Scatter hydrology residual
  // ev = evalUtils.constructScatterResidualEvaluator(false, hydro_resid_names, offsetHydro, "Scatter Hydrology");
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Interpolate stokes solution field
  // ev = evalUtils.constructDOFVecInterpolationEvaluator(stokes_dof_names[0]);
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Interpolate stokes solution gradient
  // ev = evalUtils.constructDOFVecGradInterpolationEvaluator(stokes_dof_names[0]);
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Interpolate effective pressure
  // ev = evalUtils.constructDOFInterpolationSideEvaluator("effective_pressure", basalSideName);
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Compute basis funcitons
  // ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Gather coordinates
  // ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Map to physical frame
  // ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Intepolate surface height
  // ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("surface_height");
  // fm0.template registerEvaluator<EvalT> (ev);

  // // Intepolate surface height gradient
  // ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("surface_height");
  // fm0.template registerEvaluator<EvalT> (ev);

  // // If temperature is loaded as node-based field, then interpolate it as a cell-based field
  // std::string interpolationType = "Value At Cell Barycenter";
  // ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator("temperature", interpolationType, false, cellBasis);
  // fm0.template registerEvaluator<EvalT> (ev);

  // // -------------------- Special evaluators for basal side handling ----------------- //

  // //---- Restrict vertex coordinates from cell-based to cell-side-based on basalside
  // ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector",cellType,"Coord Vec " + basalSideName);
  // fm0.template registerEvaluator<EvalT> (ev);

  // //---- Restrict ice velocity from cell-based to cell-side-based on basal side
  // ev = evalUtils.constructDOFCellToSideEvaluator(stokes_dof_names[0],basalSideName,"Node Vector",cellType, "basal_ice_velocity");
  // fm0.template registerEvaluator<EvalT> (ev);

  // //---- Restrict hydraulic potential from cell-based to cell-side-based on basal side
  // ev = evalUtils.constructDOFCellToSideEvaluator(hydro_dof_names[0],basalSideName,"Node Scalar", cellType);
  // fm0.template registerEvaluator<EvalT> (ev);

  // if (has_h_equation)
  // {
  //   //---- Restrict water thickness from cell-based to cell-side-based on basal side
  //   //TODO: need to write GatherSolutionSide evaluator
  //   ev = evalUtils.constructDOFCellToSideEvaluator(hydro_dof_names[1],basalSideName,"Node Scalar", cellType);
  //   fm0.template registerEvaluator<EvalT> (ev);

  //   // Interpolate drainage sheet depth
  //   ev = evalUtils.constructDOFInterpolationSideEvaluator(hydro_dof_names[1], basalSideName);
  //   fm0.template registerEvaluator<EvalT> (ev);

  //   if (unsteady)
  //   {
  //     // Interpolate drainage sheet depth time derivative
  //     ev = evalUtils.constructDOFInterpolationSideEvaluator(hydro_dof_names_dot[0], basalSideName);
  //     fm0.template registerEvaluator<EvalT> (ev);
  //   }
  // }
  // else
  // {
  //   // If only potential equation is solved, the water_thickness MUST be loaded/gathered
  //   TEUCHOS_TEST_FOR_EXCEPTION (ss_inputs_found[basalSideName].find(water_thickness_name)==ss_inputs_found[basalSideName].end(),
  //                               std::logic_error, "Error! You did not specify the '" << water_thickness_name << "' requirement in the basal mesh.\n");

  //   // Interpolate water thickness (no need to restrict it to the side, cause we loaded it as a side state already)
  //   ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(water_thickness_name, basalSideName);
  //   fm0.template registerEvaluator<EvalT> (ev);
  // }

  // //---- Compute side basis functions
  // ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName);
  // fm0.template registerEvaluator<EvalT> (ev);

  // //---- Restrict ice velocity from cell-based to cell-side-based and interpolate on quad points
  // ev = evalUtils.constructDOFVecInterpolationSideEvaluator("basal_ice_velocity", basalSideName);
  // fm0.template registerEvaluator<EvalT> (ev);

  // //---- Interpolate ice thickness on QP on side
  // ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("ice_thickness", basalSideName);
  // fm0.template registerEvaluator<EvalT>(ev);

  // //---- Interpolate surface height on QP on side
  // ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("surface_height", basalSideName);
  // fm0.template registerEvaluator<EvalT>(ev);

  // //---- Interpolate hydraulic potential gradient
  // ev = evalUtils.constructDOFGradInterpolationSideEvaluator(hydraulic_potential_name, basalSideName);
  // fm0.template registerEvaluator<EvalT> (ev);

  // //---- Interpolate surface water input
  // ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("surface_water_input", basalSideName);
  // fm0.template registerEvaluator<EvalT> (ev);

  // //---- Interpolate geothermal flux
  // ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("geothermal_flux", basalSideName);
  // fm0.template registerEvaluator<EvalT> (ev);

  // //---- Extend hydrology potential equation residual from cell-side-based to cell-based
  // ev = evalUtils.constructDOFSideToCellEvaluator(hydro_resid_names[0],basalSideName,"Node Scalar", cellType);
  // fm0.template registerEvaluator<EvalT> (ev);

  // if (has_h_equation)
  // {
  //   //---- Extend hydrology thickness equation residual from cell-side-based to cell-based
  //   ev = evalUtils.constructDOFSideToCellEvaluator(hydro_resid_names[1],basalSideName,"Node Scalar", cellType);
  //   fm0.template registerEvaluator<EvalT> (ev);
  // }

  // // -------------------- Special evaluators for surface side handling ----------------- //

  // if (surfaceSideName!="INVALID")
  // {
  //   //---- Compute side basis functions
  //   ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, surfaceSideBasis, surfaceCubature, surfaceSideName);
  //   fm0.template registerEvaluator<EvalT> (ev);

  //   //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side and interpolate on quad points
  //   ev = evalUtils.constructDOFCellToSideEvaluator(ice_velocity_name,surfaceSideName,"Node Vector", cellType,"surface_ice_velocity");
  //   fm0.template registerEvaluator<EvalT> (ev);

  //   //---- Interpolate surface velocity on QP on side
  //   ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity", surfaceSideName);
  //   fm0.template registerEvaluator<EvalT>(ev);

  //   //---- Interpolate surface velocity rms on QP on side
  //   ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity_RMS", surfaceSideName);
  //   fm0.template registerEvaluator<EvalT>(ev);
  // }

  // // -------------------------------- LandIce evaluators ------------------------- //

  // // --- FO Stokes Resid --- //
  // p = rcp(new ParameterList("Stokes Resid"));

  // //Input
  // p->set<std::string>("Weighted BF Variable Name", "wBF");
  // p->set<std::string>("Weighted Gradient BF Variable Name", "wGrad BF");
  // p->set<std::string>("Velocity QP Variable Name", stokes_dof_names[0]);
  // p->set<std::string>("Velocity Gradient QP Variable Name", stokes_dof_names[0] + " Gradient");
  // p->set<std::string>("Body Force Variable Name", "Body Force");
  // p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  // p->set<std::string>("Coordinate Vector Name", "Coord Vec");
  // p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));
  // p->set<std::string>("Basal Residual Variable Name", "Basal Residual");
  // p->set<bool>("Needs Basal Residual", true);

  // //Output
  // p->set<std::string>("Residual Variable Name", stokes_resid_names[0]);

  // ev = rcp(new StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  // fm0.template registerEvaluator<EvalT>(ev);

  // // --- Basal Stokes Residual --- //
  // p = rcp(new ParameterList("Stokes Basal Resid"));

  // //Input
  // p->set<std::string>("BF Side Name", "BF "+basalSideName);
  // p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
  // p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "beta");
  // p->set<std::string>("Velocity Side QP Variable Name", "basal_ice_velocity");
  // p->set<std::string>("Side Set Name", basalSideName);
  // p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Basal Friction Coefficient"));

  // //Output
  // p->set<std::string>("Basal Residual Variable Name", "Basal Residual");

  // ev = rcp(new StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  // fm0.template registerEvaluator<EvalT>(ev);

  // //--- Sliding velocity calculation ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

  // // Input
  // p->set<std::string>("Field Name","basal_ice_velocity");
  // p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
  // p->set<std::string>("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

  // // Output
  // p->set<std::string>("Field Norm Name","sliding_velocity");

  // ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  // fm0.template registerEvaluator<EvalT>(ev);

  // //--- Effective pressure calculation ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure"));

  // // Input
  // p->set<std::string>("Ice Thickness Variable Name","ice_thickness");
  // p->set<std::string>("Surface Height Variable Name","surface_height");
  // p->set<std::string>("Hydraulic Potential Variable Name",hydro_dof_names[0]);
  // p->set<std::string>("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

  // // Output
  // p->set<std::string>("Effective Pressure Variable Name","effective_pressure");

  // ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl_basal));
  // fm0.template registerEvaluator<EvalT>(ev);

  // //--- LandIce basal friction coefficient ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient"));

  // //Input
  // p->set<std::string>("Sliding Velocity QP Variable Name", "sliding_velocity");
  // p->set<std::string>("BF Variable Name", "BF "+basalSideName);
  // p->set<std::string>("Effective Pressure QP Variable Name", "effective_pressure");
  // p->set<std::string>("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Basal Friction Coefficient"));
  // p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  // p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

  // //Output
  // p->set<std::string>("Basal Friction Coefficient Variable Name", "beta");

  // p->set<bool>("Nodal",false);
  // ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_basal, FieldScalarType::Scalar, FieldScalarType::Scalar,FieldScalarType::ParamScalar);
  // fm0.template registerEvaluator<EvalT>(ev);

  // //--- Shared Parameter for basal friction coefficient: lambda ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

  // param_name = "Bed Roughness";
  // p->set<std::string>("Parameter Name", param_name);
  // p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  // Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>> ptr_lambda;
  // ptr_lambda = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>(*p,dl));
  // ptr_lambda->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
  // fm0.template registerEvaluator<EvalT>(ptr_lambda);

  // //--- Shared Parameter for basal friction coefficient: mu ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

  // param_name = "Coulomb Friction Coefficient";
  // p->set<std::string>("Parameter Name", param_name);
  // p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  // Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>> ptr_mu;
  // ptr_mu = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>(*p,dl));
  // ptr_mu->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
  // fm0.template registerEvaluator<EvalT>(ptr_mu);

  // //--- Shared Parameter for basal friction coefficient: power ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

  // param_name = "Power Exponent";
  // p->set<std::string>("Parameter Name", param_name);
  // p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  // Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>> ptr_power;
  // ptr_power = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>(*p,dl));
  // ptr_power->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
  // fm0.template registerEvaluator<EvalT>(ptr_power);

  // //--- LandIce basal friction coefficient at nodes ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient Node"));

  // //Input
  // p->set<std::string>("Sliding Velocity Variable Name", "Sliding Velocity");
  // p->set<std::string>("Effective Pressure Variable Name", "effective_pressure");
  // p->set<std::string>("Bed Roughness Variable Name", "Bed Roughness");
  // p->set<std::string>("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Basal Friction Coefficient"));

  // //Output
  // p->set<std::string>("Basal Friction Coefficient Variable Name", "Beta");
  // p->set<bool>("Nodal",true);
  // ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_basal, FieldScalarType::Scalar, FieldScalarType::Scalar,FieldScalarType::ParamScalar);
  // fm0.template registerEvaluator<EvalT>(ev);

  // //--- Sliding velocity at nodes calculation ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

  // // Input
  // p->set<std::string>("Field Name","Basal Velocity");
  // p->set<std::string>("Field Layout","Cell Side Node Vector");
  // p->set<std::string>("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

  // // Output
  // p->set<std::string>("Field Norm Name","Sliding Velocity");

  // ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  // fm0.template registerEvaluator<EvalT>(ev);

  // //--- LandIce viscosity ---//
  // p = rcp(new ParameterList("LandIce Viscosity"));

  // //Input
  // p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  // p->set<std::string>("Velocity QP Variable Name", stokes_dof_names[0]);
  // p->set<std::string>("Velocity Gradient QP Variable Name", stokes_dof_names[0] + " Gradient");
  // p->set<std::string>("Temperature Variable Name", "temperature");
  // p->set<std::string>("Flow Factor Variable Name", "flow_factor");
  // p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  // p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));

  // //Output
  // p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  // p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

  // ev = rcp(new LandIce::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT, typename EvalT::ParamScalarT>(*p,dl));
  // fm0.template registerEvaluator<EvalT>(ev);

  // //--- Shared Parameter for Continuation:  ---//
  // p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  // param_name = "Glen's Law Homotopy Parameter";
  // p->set<std::string>("Parameter Name", param_name);
  // p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  // Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_homotopy;
  // ptr_homotopy = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
  // ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));
  // fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  // //--- Body Force ---//
  // p = rcp(new ParameterList("Body Force"));

  // //Input
  // p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");
  // p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  // p->set<std::string>("Surface Height Gradient Name", "surface_height Gradient");
  // p->set<std::string>("Surface Height Name", "surface_height");
  // p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  // p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  // p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

  // //Output
  // p->set<std::string>("Body Force Variable Name", "Body Force");

  // ev = rcp(new LandIce::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl));
  // fm0.template registerEvaluator<EvalT>(ev);

  // // Checking whether the body force calculation requires 'surface_height Gradient'. If so, make sure it is loaded/gathered
  // for (auto tag_ptr : ev->dependentFields())
  // {
  //   if (tag_ptr->name()=="surface_height Gradient")
  //   {
  //     TEUCHOS_TEST_FOR_EXCEPTION (inputs_found.find("surface_height")==inputs_found.end(),std::logic_error,
  //                                 "Error! StokesFOBodyForce requires 'surface_height', but you did not specify it in the discretization requirements.\n");
  //   }
  // }

  // // ------- Hydrology Water Discharge -------- //
  // p = rcp(new Teuchos::ParameterList("Hydrology Water Discharge"));

  // //Input
  // p->set<std::string> ("Water Thickness QP Variable Name","water_thickness");
  // p->set<std::string> ("Hydraulic Potential Gradient QP Variable Name", hydro_dof_names[0] + " Gradient");
  // p->set<std::string> ("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*> ("LandIce Hydrology",&params->sublist("LandIce Hydrology"));
  // p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

  // //Output
  // p->set<std::string> ("Water Discharge QP Variable Name","Water Discharge");

  // if (has_h_equation)
  //   ev = rcp(new LandIce::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
  // else
  //   ev = rcp(new LandIce::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));
  // fm0.template registerEvaluator<EvalT>(ev);

  // // ------- Hydrology Melting Rate -------- //
  // p = rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  // //Input
  // p->set<std::string> ("Geothermal Heat Source QP Variable Name","geothermal_flux");
  // p->set<std::string> ("Sliding Velocity QP Variable Name","sliding_velocity");
  // p->set<std::string> ("Basal Friction Coefficient QP Variable Name","beta");
  // p->set<std::string> ("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

  // //Output
  // p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");

  // ev = rcp(new LandIce::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
  // fm0.template registerEvaluator<EvalT>(ev);


  // // ------- Hydrology Residual Potential Eqn-------- //
  // p = rcp(new Teuchos::ParameterList("Hydrology Residual Potential Eqn"));

  // //Input
  // p->set<std::string> ("BF Name", "BF " + basalSideName);
  // p->set<std::string> ("Gradient BF Name", "Grad BF " + basalSideName);
  // p->set<std::string> ("Weighted Measure Name", "Weighted Measure " + basalSideName);
  // p->set<std::string> ("Water Discharge QP Variable Name", "Water Discharge");
  // p->set<std::string> ("Effective Pressure QP Variable Name", "effective_pressure");
  // p->set<std::string> ("Water Thickness QP Variable Name", water_thickness_name);
  // p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
  // p->set<std::string> ("Surface Water Input QP Variable Name","surface_water_input");
  // p->set<std::string> ("Sliding Velocity QP Variable Name","sliding_velocity");
  // p->set<std::string> ("Side Set Name", basalSideName);
  // p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));
  // p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&params->sublist("LandIce Hydrology"));

  // //Output
  // p->set<std::string> ("Potential Eqn Residual Name",hydro_resid_names[0]);

  // if (has_h_equation)
  //   ev = rcp(new LandIce::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
  // else
  //   ev = rcp(new LandIce::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));

  // fm0.template registerEvaluator<EvalT>(ev);

  // if (has_h_equation)
  // {
  //   // ------- Hydrology Evolution Residual -------- //
  //   p = rcp(new Teuchos::ParameterList("Hydrology Residual Evolution"));

  //   //Input
  //   p->set<std::string> ("BF Name", "BF " + basalSideName);
  //   p->set<std::string> ("Weighted Measure Name", "Weighted Measure " + basalSideName);
  //   p->set<std::string> ("Water Thickness QP Variable Name",water_thickness_name);
  //   p->set<std::string> ("Water Thickness Dot QP Variable Name",water_thickness_dot_name);
  //   p->set<std::string> ("Effective Pressure QP Variable Name","effective_pressure");
  //   p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
  //   p->set<std::string> ("Sliding Velocity QP Variable Name","sliding_velocity");
  //   p->set<std::string> ("Side Set Name", basalSideName);
  //   p->set<Teuchos::ParameterList*> ("LandIce Hydrology",&params->sublist("LandIce Hydrology"));
  //   p->set<Teuchos::ParameterList*> ("LandIce Physics",&params->sublist("LandIce Physics"));

  //   //Output
  //   p->set<std::string> ("Thickness Eqn Residual Name", hydro_resid_names[1]);

  //   ev = rcp(new HydrologyResidualThicknessEqn<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
  //   fm0.template registerEvaluator<EvalT>(ev);
  // }

  // if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  // {
  //   // Require scattering of stokes residual
  //   PHX::Tag<typename EvalT::ScalarT> stokes_res_tag("Scatter Stokes", dl->dummy);
  //   fm0.requireField<EvalT>(stokes_res_tag);

  //   // Require scattering of hydrology residual
  //   PHX::Tag<typename EvalT::ScalarT> hydro_res_tag("Scatter Hydrology", dl->dummy);
  //   fm0.requireField<EvalT>(hydro_res_tag);
  // }
  // else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  // {
  //   // ----------------------- Responses --------------------- //
  //   RCP<ParameterList> paramList = rcp(new ParameterList("Param List"));
  //   RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
  //   paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
  //   paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);
  //   paramList->set<std::string>("Surface Velocity Side QP Variable Name","surface_icei_velocity");
  //   paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","observed_surface_velocity");
  //   paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","observed_surface_velocity_RMS");
  //   paramList->set<std::string>("BF Surface Name","BF " + surfaceSideName);
  //   paramList->set<std::string>("Weighted Measure Surface Name","Weighted Measure " + surfaceSideName);
  //   paramList->set<std::string>("Surface Side Name", surfaceSideName);
  //   // The regularization with the gradient of beta is available only for GIVEN_FIELD or GIVEN_CONSTANT, which do not apply here
  //   paramList->set<std::string>("Basal Friction Coefficient Gradient Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
  //   paramList->set<std::string>("Basal Side Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
  //   paramList->set<std::string>("Inverse Metric Basal Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
  //   paramList->set<std::string>("Weighted Measure Basal Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");

  //   ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
  //   return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  // }


  // Finally, construct responses, and return the tags
  return constructStokesFOBaseResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, responseList);
}

} // Namespace LandIce

#endif // LANDICE_STOKES_FO_HYDROLOGY_PROBLEM_HPP
