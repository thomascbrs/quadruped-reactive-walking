///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for FootstepPlannerQP class
///
/// \details FootstepPlannerQP data structure 
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef FOOTSTEPPLANNERQP_H_INCLUDED
#define FOOTSTEPPLANNERQP_H_INCLUDED


#include "qrw/Surface.hpp"
#include "qrw/Types.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

#include "pinocchio/math/rpy.hpp"
#include "qrw/Gait.hpp"

#include "eiquadprog/eiquadprog-fast.hpp"
using namespace eiquadprog::solvers;


// typedef std::vector<std::vector> MyList;
typedef std::vector<Surface> SurfaceDataList;
typedef std::vector<SurfaceDataList> SurfaceDataList2;
typedef std::vector<SurfaceDataList2> SurfaceDataList3;


struct optimData{
  int i;
  int j;
  Surface surface;
  Vector3 next_pos;
};

class FootstepPlannerQP
{
private :
  Gait* gait_;  // Gait object to hold the gait informations

  double dt;      // Time step of the contact sequence (time step of the MPC)
  double T_gait;  // Gait period
  double T_mpc;   // MPC period (prediction horizon)
  double h_ref;   // Reference height for the trunk

  // Predefined quantities
  double k_feedback;  // Feedback gain for the feedback term of the planner
  double g;           // Value of the gravity acceleartion
  double L;           // Value of the maximum allowed deviation due to leg length

  // Number of time steps in the prediction horizon
  int n_steps;  // T_mpc / time step of the MPC

  // Constant sized matrices
  Matrix34 shoulders_;        // Position of shoulders in local frame
  Matrix34 currentFootstep_;  // Feet matrix in world frame
  Vector3 nextFootstep_;     // Feet matrix in world frame
  Matrix34 targetFootstep_;
  std::vector<Matrix34> footsteps_;

  Matrix3 Rz;  // Predefined matrices for compute_footstep function
  Matrix3 Rz_tmp;  // Predefined matrices for compute_footstep function
  VectorN dt_cum;
  VectorN yaws;
  VectorN dx;
  VectorN dy;
  int k;
  int k_mpc;   
  double dt_tsid; 

  Vector3 q_tmp;
  Vector3 q_dxdy;
  Vector3 RPY_;
  Eigen::Quaterniond quat_;
  Vector3 b_v;
  Vector3 b_voptim;
  Vector6 b_vref;

  // QP problem : 
  const int n = 14; // Number of variables
  const int m = 6*4; // Number of constraints inequalities

  // The problem is in the form:
  // min (1/2)x' P x + q0' x
  // subject to  G x <= h

  // Weight Matrix
 
  Eigen::MatrixXd P;
  Eigen::VectorXd q0;

  Eigen::MatrixXd G;
  Eigen::VectorXd h;

  Eigen::MatrixXd Aeq;
  Eigen::VectorXd Beq;

  Eigen::VectorXd weights;
  Vector3 V_optim;

  // qp solver
  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;
  EiquadprogFast_status status;
  Eigen::VectorXd x;
  EiquadprogFast qp;

  std::vector<int> feet;
  Vector4 t0s;
  Vector4 t_swing;
  bool sl1m_status ;
  int sl1m_iteration;
  Surface floor_surface;
  Surface surface_tmp;

  std::vector<optimData> List_optim;
  std::vector<Surface> surfaces_selected;
  void computeFootsteps(VectorN const& q, Vector6 const& v, Vector6 const& vref, const SurfaceDataList2 surface_sl1m);
  void update_remaining_time();
  Vector3 computeNextFootstep(int i, int j,Vector3 b_vref_in, bool feedback_ref);
  void updateTargetFootsteps();
  MatrixN vectorToMatrix(std::vector<Matrix34> const& array);
  Surface select_surface_fromXY(Vector3 next_ft, int phase,int moving_foot_index);

public:
    // Constructor
    FootstepPlannerQP() ; 
    // ~FootstepPlannerQP() {}

    int surface_inequalities(int i_start , const Surface surface, const Eigen::Matrix<double,3,1> next_ft, int id_foot);
    int solve_qp();


    void initialize(double dt_in,
                    double T_mpc_in,
                    double h_ref_in,
                    int const& k_mpc_in,
                    double const& dt_tsid_in,
                    MatrixN const& shouldersIn,
                    Gait& gaitIn,
                    int N_gait,
                    Surface floor_surface_in);

    MatrixN computeTargetFootstep(int k_in, VectorN const& q, Vector6 const& v, Vector6 const& b_vref , const SurfaceDataList2 surface_sl1m , bool sl1m_status_in, int sl1m_iteration_in);
    void updateNewContact();

    MatrixN getFootsteps();
    MatrixN getTargetFootsteps();
    Vector3 getVrefResults();
    std::vector<Surface> getSurfacesSelected();
   
};



#endif  // FOOTSTEPPLANNERQP_H_INCLUDED
