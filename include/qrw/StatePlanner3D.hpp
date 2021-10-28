///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for StatePlanner class
///
/// \details Planner that outputs the reference trajectory of the base based on the reference
///          velocity given by the user and the current position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STATEPLANNER3D_H_INCLUDED
#define STATEPLANNER3D_H_INCLUDED

#include "pinocchio/math/rpy.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "qrw/Heightmap.hpp"
#include "qrw/Params.hpp"
#include "qrw/Types.h"

class StatePlanner3D
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Empty constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    StatePlanner3D();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Initializer
    ///
    /// \param[in] params Object that stores parameters
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(Params& params);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~StatePlanner3D() {}

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the reference trajectory of the CoM for each time step of the
    ///        predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
    ///        of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
    ///        linear velocity and angular velocity vertically stacked. The first column contains
    ///        the current state while the remaining N columns contains the desired future states.
    ///
    /// \param[in] q current position vector of the flying base in horizontal frame (linear and angular stacked)
    /// \param[in] v current velocity vector of the flying base in horizontal frame (linear and angular stacked)
    /// \param[in] vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
    /// \param[in] z_average average height of feet currently in stance phase
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void computeReferenceStates(VectorN const& q, Vector6 const& v, Vector6 const& vref, int is_new_step);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute references configuration for MIP planner
    ///
    /// \param[in] q current position vector of the flying base in horizontal frame (linear and angular stacked)
    /// \param[in] vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void computeConfigurations(VectorN const& q, Vector6 const& vref);

    MatrixN getReferenceStates() { return referenceStates_; }
    int getNSteps() { return nSteps_; }
    MatrixN get_configurations() { return configs_; }

private:
    double dt_;               // Time step of the contact sequence (time step of the MPC)
    double referenceHeight_;  // Reference height for the trunk
    int nSteps_;              // Number of timesteps in the prediction horizon
    double stepDuration_;     // Duration of a timestep

    Vector3 rpy_;   // Current roll, pitch and yaw angles
    Matrix3 Rz_;    // Rotation matrix z-axis, yaw angle
    Vector3 DxDy_;  // Temporary vector for displacement on x and y axis

    // Reference trajectory matrix of size 12 by (1 + N)  with the current state of
    // the robot in column 0 and the N steps of the prediction horizon in the others
    MatrixN referenceStates_;

    VectorN dtVector_;  // Vector containing all time steps in the prediction horizon
    Heightmap heightmap_;
    Vector3 rpyMap_;
    Vector3 meanSurface_;  // Vector3 such as [a,b,c], such as ax + by -z + c = 0

    double velocityMax_;   // rad.s-1
    double velocityMaxZ_;  // rad.s-1
    int nSurfaceConfigs_;
    MatrixN configs_;
    Vector7 config_;
    Vector3 rpyConfig_;  // Current roll, pitch and yaw angles
};

#endif  // STATEPLANNER3D_H_INCLUDED
