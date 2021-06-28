///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for FootstepPlannerQP class
///
/// \details Planner that outputs current and future locations of footsteps
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef FOOTSTEPPLANNERQP_H_INCLUDED
#define FOOTSTEPPLANNERQP_H_INCLUDED


#include "qrw/Gait.hpp"
#include "qrw/Surface.hpp"
#include "qrw/Types.h"

#include "pinocchio/math/rpy.hpp"

#include "eiquadprog/eiquadprog-fast.hpp"
#include <vector>

using namespace eiquadprog::solvers;

typedef std::vector<Surface> SurfaceVector;
typedef std::vector<std::vector<Surface>> SurfaceVectorVector;

struct optimData
{
    int phase;
    int foot;
    Surface surface;
    Vector3 next_pos;
};

class FootstepPlannerQP
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Empty constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    FootstepPlannerQP();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Initializer
    ///
    /// \param[in] dt_in Time step of the contact sequence (time step of the MPC)
    /// \param[in] T_mpc_in MPC period (prediction horizon)
    /// \param[in] h_ref_in Reference height for the trunk
    /// \param[in] k_mpc_in
    /// \param[in] dt_tsid_in
    /// \param[in] shoulderIn Position of shoulders in local frame
    /// \param[in] gaitIn Gait object to hold the gait informations
    /// \param[in] N_gait
    /// \param[in] floor_surface_in
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(double dt_in,
                    double T_mpc_in,
                    double h_ref_in,
                    int const& k_mpc_in,
                    double const& dt_tsid_in,
                    MatrixN const& shouldersIn,
                    Gait& gaitIn,
                    int N_gait,
                    Surface initialSurface_in);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~FootstepPlannerQP() {}

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the desired location of footsteps and update relevant matrices
    ///
    ///  \param[in] q  current position vector of the flying base in world frame (linear and angular stacked)
    ///  \param[in] v  current velocity vector of the flying base in world frame (linear and angular stacked)
    ///  \param[in] b_vref  desired velocity vector of the flying base in base frame (linear and angular stacked)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    MatrixN computeTargetFootstep(int const k_in,
                                  VectorN const& q,
                                  Vector6 const& v,
                                  Vector6 const& b_vref,
                                  SurfaceVectorVector const& potentialSurfaces,
                                  SurfaceVector const& surfaces,
                                  bool const surfaceStatus,
                                  int const surfaceIteration);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Refresh feet position when entering a new contact phase
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void updateNewContact();

    MatrixN getFootsteps();
    MatrixN getTargetFootsteps();
    Vector3 getVrefResults();
    SurfaceVector getSelectedSurfaces() const;
    Surface getSelectedSurface(int const foot) const;

    int surfaceInequalities(int i_start, Surface const& surface, Vector3 const& next_ft, int id_foot);

private:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
    ///        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
    ///        For feet currently touching the ground the desired position is where they currently are.
    ///
    /// \param[in] q current position vector of the flying base in world frame(linear and angular stacked)
    /// \param[in] v current velocity vector of sthe flying base in world frame(linear and angular stacked)
    /// \param[in] vref desired velocity vector of the flying base in world frame(linear and angular stacked)
    /// \param[in] surface_sl1m list of surfaces
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void computeFootsteps(VectorN const& q, Vector6 const& v, Vector6 const& vref);

    void update_remaining_time();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the target location on the ground of a given foot for an upcoming stance phase
    ///
    /// \param[in] i considered phase (row of the gait matrix)
    /// \param[in] j considered foot (col of the gait matrix)
    ///
    /// \retval Matrix with the next footstep positions
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void computeNextFootstep(int i, int j, Vector3 b_vref_in, Vector3& nextFootstep, bool const feedback_ref);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Update desired location of footsteps using information coming from the footsteps planner
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void updateTargetFootsteps();

    MatrixN vectorToMatrix(std::vector<Matrix34> const& array);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Surface selectSurfaceFromPoint(Vector3 const& point, int phase, int moving_foot_index);

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
    Vector3 nextFootstep_;      // Feet matrix in world frame
    Vector3 nextFootstepQP_;    // Feet matrix in world frame
    Matrix34 targetFootstep_;
    std::vector<Matrix34> footsteps_;

    Matrix3 Rz;      // Predefined matrices for compute_footstep function
    Matrix3 Rz_tmp;  // Predefined matrices for compute_footstep function
    VectorN dt_cum;
    VectorN yaws;
    VectorN dx;
    VectorN dy;
    int k_;
    int k_mpc;
    double dt_tsid;

    Vector3 q_tmp;
    Vector3 q_dxdy;
    Vector3 RPY_;
    Eigen::Quaterniond quat_;

    Vector3 b_v_;
    Vector6 b_vRef_;

    // QP problem :
    const int N = 14;     // Number of variables vx, vy, p1, p2, p3
    const int M = 6 * 4;  // Number of constraints inequalities

    // min. 1/2 * x' C_ x + q_' x
    // s.t. C_ x + d_ = 0
    //      G_ x + h_ >= 0

    // Weight Matrix
    MatrixN P_;
    VectorN q_;

    MatrixN G_;
    VectorN h_;

    MatrixN C_;
    VectorN d_;

    VectorN weights_;
    Vector3 vOptim_;
    Vector3 b_vOptim_;

    // qp solver
    EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;
    EiquadprogFast_status status;
    VectorN x;
    EiquadprogFast qp;

    std::vector<int> feet_;
    Vector4 t0s;
    Vector4 t_swing;

    bool surfaceStatus_;
    int surfaceIteration_;
    SurfaceVector surfaces_;
    SurfaceVectorVector potentialSurfaces_;
    Surface initialSurface_;

    std::vector<optimData> optimVector_;
    SurfaceVector selectedSurfaces_;
};


#endif  // FOOTSTEPPLANNERQP_H_INCLUDED
