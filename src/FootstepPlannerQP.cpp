#include "qrw/FootstepPlannerQP.hpp"
#include <chrono>
using namespace std::chrono;

FootstepPlannerQP::FootstepPlannerQP()
    : gait_(NULL)
    , k_feedback(0.03)
    , g(9.81)
    , L(0.155)
    , nextFootstep_(Vector3::Zero())
    , footsteps_()
    , Rz(Matrix3::Zero())
    , Rz_tmp(Matrix3::Zero())
    , dt_cum()
    , yaws()
    , dx()
    , dy()
    , q_tmp(Vector3::Zero())
    , q_dxdy(Vector3::Zero())
    , RPY_(Vector3::Zero())
    , b_v(Vector3::Zero())
    , b_vref(Vector6::Zero())
    , feet()
    , t0s(Vector4::Zero())
    , t_swing(Vector4::Zero())
{
  //empty
}

void FootstepPlannerQP::initialize(double dt_in,
                    double T_mpc_in,
                    double h_ref_in,
                    int const& k_mpc_in,
                    double const& dt_tsid_in,
                    MatrixN const& shouldersIn,
                    Gait& gaitIn,
                    int N_gait,
                    Surface floor_surface_in)
{  
  k = 0;
  dt_tsid = dt_tsid_in;
  sl1m_status = false;
  sl1m_iteration =0 ;
  floor_surface = floor_surface_in;
  surface_tmp = Surface();
  k_mpc = k_mpc_in;
  dt = dt_in;
  T_mpc = T_mpc_in;
  h_ref = h_ref_in;
  n_steps = (int)std::lround(T_mpc_in / dt_in);
  shoulders_ = shouldersIn;
  currentFootstep_ = shouldersIn.block(0, 0, 3, 4);
  gait_ = &gaitIn;
  targetFootstep_ = shouldersIn;
  dt_cum = VectorN::Zero(N_gait);
  yaws = VectorN::Zero(N_gait);
  dx = VectorN::Zero(N_gait);
  dy = VectorN::Zero(N_gait);
  V_optim.setZero();
  for (int i = 0; i < N_gait; i++)
  {
    footsteps_.push_back(Matrix34::Zero());
  }
  Rz(2, 2) = 1.0;

  for (int j = 0; j<4;j++){
    surfaces_selected.push_back(floor_surface_in);
  }
  // QP initialization
  // weight
  q0.setZero(n);
  G.setZero(m,n);
  h.setZero(m);
  P.setZero(n,n);
  x.setZero(n);
  qp.reset(n,0,m);
  Aeq.setZero(n,0);
  Beq.setZero(0);
  weights.setZero(14); 
  weights << 0.05,0.05, 1.0,1.0,1.0, 1.0,1.0,1.0 ,1.0,1.0,1.0 , 1.0,1.0,1.0;
  P.diagonal() << weights;
  b_voptim.setZero();
}

int FootstepPlannerQP::surface_inequalities(int i_start , const Surface surface, const Eigen::Matrix<double,3,1> next_ft, int id_foot)
{ 
  int i_end = i_start + 6;
  
  // Update G in GX <= h :
  G.block(i_start,0,6,2) = -k_feedback*surface.A.block(0,0,6,2);
  G.block(i_start,2+3*id_foot,6,3) = surface.A;
  // Update h vector
  h.segment(i_start,6) = surface.b - surface.A * next_ft;
  // double epsilon = 0.0001;
  // h(i_start+5) += epsilon;
  
  return i_end;
}

int FootstepPlannerQP::solve_qp(){
  /**
  * Eiquadprog -Fast solves the problem :
  * min. 0.5 * x' Hess x + g0' x
  * s.t. CE x + ce0 = 0
  *      CI x + ci0 >= 0
  */
  status = qp.solve_quadprog(P, q0.transpose(), Aeq, Beq, -G, h, x);
  return 0;
}

Surface FootstepPlannerQP::select_surface_fromXY(Vector3 next_ft, int phase,int moving_foot_index){
  double h_selected = 0.;
  bool reduced_surface_found = false;
  Surface sf = Surface(); 
  if (sl1m_iteration != 0){
    // TODO
  }
  else{
    sf = floor_surface ;
  }
  return sf;
}

void FootstepPlannerQP::computeFootsteps(VectorN const& q, Vector6 const& v, Vector6 const& vref,const SurfaceDataList2 surface_sl1m)
{
  for (uint i = 0; i < footsteps_.size(); i++){
        footsteps_[i] = Matrix34::Zero();
  }
  MatrixN gait = gait_->getCurrentGait();

  // Set current position of feet for feet in stance phase
  for (int j = 0; j < 4; j++){
    if (gait(0, j) == 1.0){
          footsteps_[0].col(j) = currentFootstep_.col(j);
    }
  }

  // Cumulative time by adding the terms in the first column (remaining number of timesteps)
  // Get future yaw yaws compared to current position
  dt_cum(0) = dt;
  yaws(0) = vref(5) * dt_cum(0) + RPY_(2);
  for (uint j = 1; j < footsteps_.size(); j++)
  {
      dt_cum(j) = gait.row(j).isZero() ? dt_cum(j - 1) : dt_cum(j - 1) + dt;
      yaws(j) = vref(5) * dt_cum(j) + RPY_(2);
  }

  // Displacement following the reference velocity compared to current position
  if (vref(5, 0) != 0){
    for (uint j = 0; j < footsteps_.size(); j++){
      dx(j) = (v(0) * std::sin(vref(5) * dt_cum(j)) + v(1) * (std::cos(vref(5) * dt_cum(j)) - 1.0)) / vref(5);
      dy(j) = (v(1) * std::sin(vref(5) * dt_cum(j)) - v(0) * (std::cos(vref(5) * dt_cum(j)) - 1.0)) / vref(5);
    }
  }
  else    {
    for (uint j = 0; j < footsteps_.size(); j++){
      dx(j) = v(0) * dt_cum(j);
      dy(j) = v(1) * dt_cum(j);
    }
  }

  // Get current and reference velocities in base frame (rotated yaw)
  b_v = Rz.transpose() * v.head(3);  //linear velocity in base frame
  b_vref.head(3) = Rz.transpose() * vref.head(3);
  b_vref.tail(3) = Rz.transpose() * vref.tail(3);

  // Current position without height
  Vector3 q_tmp = q.head(3);
  q_tmp(2) = 0.0;

  update_remaining_time();

  // Update the footstep matrix depending on the different phases of the gait (swing & stance)
  int i = 1;
  int phase = 0;
  List_optim.clear();
  while (!gait.row(i).isZero()){
    // Feet that were in stance phase and are still in stance phase do not move
    for (int j = 0; j < 4; j++){
      if (gait(i - 1, j) * gait(i, j) > 0){
        footsteps_[i].col(j) = footsteps_[i - 1].col(j);
      }
    }
    
    int moving_foot_index = 0;
    // Feet that were in swing phase and are now in stance phase need to be updated
    for (int j = 0; j < 4; j++){
      if ((1 - gait(i - 1, j)) * gait(i, j) > 0){
        // Offset to the future position
        q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;        

        // Get future desired position of footsteps
        Vector3 nextFootstep_tmp = computeNextFootstep(i, j, b_v, true);

        Vector3 nextFootstep_qp = computeNextFootstep(i, j, b_v , false);        

        // Get desired position of footstep compared to current position
        Rz_tmp.setZero();
        double c = std::cos(yaws(i - 1));
        double s = std::sin(yaws(i - 1));
        Rz_tmp.topLeftCorner<2, 2>() << c, -s, s, c;

        nextFootstep_qp = (Rz_tmp * nextFootstep_qp + q_tmp + q_dxdy).transpose();
        // footsteps_[i].col(j) = (Rz_tmp * nextFootstep_tmp + q_tmp + q_dxdy).transpose();

        bool j_is_flying = false;
        for (int i_feet = 0; i_feet < (int)feet.size(); i_feet++){
          if (feet[i_feet] == j ){j_is_flying = true;}
        }

        if (j_is_flying && phase == 0){
          // feet currently in flying phase
          if (t0s[j] < 10e-4 and k % k_mpc == 0){
            // Beginning of flying phase, selection of surface
            if (sl1m_status){
              surfaces_selected[j] = surface_sl1m[phase][moving_foot_index];
            }
            else{
              Surface sf_ = Surface();
              // Select surface with heuristic
              surfaces_selected[j] = select_surface_fromXY(nextFootstep_tmp, phase,moving_foot_index);
              
            }               
          }
          optimData optim_data = {i,j,surfaces_selected[j],nextFootstep_qp};
          List_optim.push_back(optim_data);
        }          
        else{
          Surface sf_ = Surface();
          if (sl1m_status){
            sf_ = surface_sl1m[phase][moving_foot_index];
          }
          else{
            // Select surface with heuristic
            sf_ = select_surface_fromXY(nextFootstep_tmp, phase,moving_foot_index);            
          }
          optimData optim_data = {i,j,sf_,nextFootstep_qp};
          List_optim.push_back(optim_data);
        }
        moving_foot_index += 1;
      }
    }
    if (!( gait.row(i-1) - gait.row(i) ).isZero()){
      phase += 1;
    }                
    i++;
  }  
  
  // Reset matrix Inequalities
  G.setZero();
  h.setZero();
  x.setZero();

  // Adapting q0 vector with reference velocity
  q0(0) = -weights(0)*vref(0);
  q0(1) = -weights(1)*vref(1);

  // Convert problem to inequalities
  int i_start = 0;
  int id_foot = 0;
  for (uint id_l =0; id_l < List_optim.size() ;id_l++){
    i_start = surface_inequalities( i_start , List_optim[id_l].surface , List_optim[id_l].next_pos, id_foot);
    id_foot++;
  }

   // Solve QP
  solve_qp();

  // Retrieve results
  V_optim(0) = x(0);
  V_optim(1) = x(1);

  // Get new reference velocity in base frame to recompute the new footsteps
  b_voptim = Rz.transpose() * V_optim ; // lin velocity in base frame (rotated yaw)

  // Update the foostep matrix with the position optimised, for changing phase index 
  for (uint id_l =0; id_l < List_optim.size() ;id_l++){
    int i = List_optim[id_l].i;
    int j = List_optim[id_l].j;

    // Offset to the future position
    q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;

    // Get future desired position of footsteps with k_feedback
    Vector3 nextFootstep_qp = computeNextFootstep(i, j, b_voptim , true);

    // Get desired position of footstep compared to current position
    Rz_tmp.setZero();
    double c = std::cos(yaws(i - 1));
    double s = std::sin(yaws(i - 1));
    Rz_tmp.topLeftCorner<2, 2>() << c, -s, s, c;

    nextFootstep_qp = ( Rz_tmp * nextFootstep_qp + q_tmp + q_dxdy).transpose();
    nextFootstep_qp(0) += x(2 + 3*id_l);
    nextFootstep_qp(1) += x(2 + 3*id_l + 1);
    nextFootstep_qp(2) += x(2 + 3*id_l + 2);
    footsteps_[i].col(j) = nextFootstep_qp ;
  }

  // Update the next stance phase after the changing phase
  i = 1;
  while (!gait.row(i).isZero()){
    // Feet that were in stance phase and are still in stance phase do not move
    for (int j = 0; j < 4; j++){
      if (gait(i - 1, j) * gait(i, j) > 0){
        footsteps_[i].col(j) = footsteps_[i - 1].col(j);
      }
    }
    i++;
  }  
}

MatrixN FootstepPlannerQP::computeTargetFootstep(int k_in, VectorN const& q, Vector6 const& v, Vector6 const& b_vref, const SurfaceDataList2 surface_sl1m, bool sl1m_status_in, int sl1m_iteration_in)
{   
    k = k_in;
    sl1m_status = sl1m_status_in;
    sl1m_iteration = sl1m_iteration_in;
    // Get the reference velocity in world frame (given in base frame)
    quat_ = { q(6) , q(3), q(4), q(5) };  // w, x, y, z
    RPY_ << pinocchio::rpy::matrixToRpy(quat_.toRotationMatrix());

    double c = std::cos(RPY_(2));
    double s = std::sin(RPY_(2));
    Rz.topLeftCorner<2, 2>() << c, -s, s, c;

    Vector6 vref = b_vref;
    // o_vref is given 
    // vref.head(3) = Rz * b_vref.head(3);
    // vref.tail(3) = Rz * b_vref.tail(3);

    // Compute the desired location of footsteps over the prediction horizon
    computeFootsteps(q, v, vref, surface_sl1m);

    // Update desired location of footsteps on the ground
    updateTargetFootsteps();

    return targetFootstep_;
}

void FootstepPlannerQP::update_remaining_time(){
  if ((k % k_mpc) == 0){
    feet.clear();
    t0s.setZero();

    // Indexes of feet in swing phase
    for (int i = 0; i < 4; i++){
      if (gait_->getCurrentGait()(0, i) == 0)
        feet.push_back(i);
    }
    // If no foot in swing phase
    if (feet.size() == 0)
      return;

    // For each foot in swing phase get remaining duration of the swing phase
    for (int j = 0; j < (int)feet.size(); j++){
      int i = feet[j];
      t_swing[i] = gait_->getPhaseDuration(0, feet[j], 0.0);  // 0.0 for swing phase
      double value = t_swing[i] - (gait_->getRemainingTime() * k_mpc - ((k + 1) % k_mpc)) * dt_tsid - dt_tsid;
      t0s[i] = std::max(0.0, value);
    }
  }
  else{
    // If no foot in swing phase
    if (feet.size() == 0)
        return;

    // Increment of one time step for feet in swing phase
    for (int i = 0; i < (int)feet.size(); i++){
      double value = t0s[feet[i]] + dt_tsid;
      t0s[feet[i]] = std::max(0.0, value);
    }
  }
}

Vector3 FootstepPlannerQP::computeNextFootstep(int i, int j, Vector3 b_vref_in, bool feedback_ref){
  nextFootstep_.setZero();

  double t_stance = gait_->getPhaseDuration(i, j, 1.0);  // 1.0 for stance phase

  // Add symmetry term
  nextFootstep_ = t_stance * 0.5 * b_v;
  // Add feedback term
  nextFootstep_ += k_feedback * b_v ;
  if (feedback_ref){
    nextFootstep_ += - k_feedback*b_vref_in.head(3);
  }

  // Add centrifugal term
  Vector3 cross;
  cross << b_v(1) * b_vref(5) - b_v(2) * b_vref(4), b_v(2) * b_vref(3) - b_v(0) * b_vref(5), 0.0;
  nextFootstep_ += 0.5 * std::sqrt(h_ref / g) * cross;

  // Legs have a limited length so the deviation has to be limited
  nextFootstep_(0) = std::min(nextFootstep_(0), L);
  nextFootstep_(0) = std::max(nextFootstep_(0), -L);
  nextFootstep_(1) = std::min(nextFootstep_(1), L);
  nextFootstep_(1) = std::max(nextFootstep_(1), -L);

  // Fix problem :
  nextFootstep_[1] = 0.;

  // Add shoulders
  Vector3 RP_ = RPY_;
  RP_[2] = 0; // Yaw taken into account later
  nextFootstep_ += pinocchio::rpy::rpyToMatrix(RP_) * shoulders_.col(j);

  // Remove Z component (working on flat ground)
  nextFootstep_(2) = 0.;

  return nextFootstep_;
}


void FootstepPlannerQP::updateTargetFootsteps(){
  for (int i = 0; i < 4; i++){
      int index = 0;
      while (footsteps_[index](0, i) == 0.0){
        index++;
      }
      targetFootstep_.col(i) << footsteps_[index](0, i), footsteps_[index](1, i), 0.0;
  }
}


void FootstepPlannerQP::updateNewContact(){
  for (int i = 0; i < 4; i++){
    if (gait_->getCurrentGaitCoeff(0, i) == 1.0){
      currentFootstep_.col(i) = (footsteps_[1]).col(i);
    }
  }
}

MatrixN FootstepPlannerQP::getFootsteps() { return vectorToMatrix(footsteps_); }
MatrixN FootstepPlannerQP::getTargetFootsteps() { return targetFootstep_; }
Vector3  FootstepPlannerQP::getVrefResults(){return V_optim;}

MatrixN FootstepPlannerQP::vectorToMatrix(std::vector<Matrix34> const& array){
  MatrixN M = MatrixN::Zero(array.size(), 12);
  for (uint i = 0; i < array.size(); i++){
    for (int j = 0; j < 4; j++){
      M.row(i).segment<3>(3 * j) = array[i].col(j);
    }
  }
  return M;
}

std::vector<Surface> FootstepPlannerQP::getSurfacesSelected(){return surfaces_selected;};




