#include "qrw/StatePlanner3D.hpp"

StatePlanner3D::StatePlanner3D()
    : dt_(0.)
    , referenceHeight_(0.)
    , nSteps_(0)
    , stepDuration_(0.)
    , rpy_(Vector3::Zero())
    , Rz_(Matrix3::Zero())
    , DxDy_(Vector3::Zero())
    , referenceStates_()
    , dtVector_()
    , heightmap_()
    , rpyMap_(Vector3::Zero())
    , meanSurface_(Vector3::Zero())
    , velocityMax_(0.4)
    , velocityMaxZ_(0.1)
    , nSurfaceConfigs_(3)
    , configs_()
    , config_(Vector7::Zero())
    , rpyConfig_(Vector3::Zero())
{
    // Empty
}

void StatePlanner3D::initialize(Params& params)
{
    dt_ = params.dt_mpc;
    referenceHeight_ = params.h_ref;
    nSteps_ = static_cast<int>(params.gait.rows());
    stepDuration_ = params.T_gait / 2;
    referenceStates_ = MatrixN::Zero(12, 1 + nSteps_);
    dtVector_ = VectorN::LinSpaced(nSteps_, dt_, static_cast<double>(nSteps_) * dt_);
    heightmap_.initialize(std::getenv("SOLO3D_ENV_DIR") + params.environment_heightmap);
    configs_ = MatrixN::Zero(7, nSurfaceConfigs_);
}

void StatePlanner3D::computeReferenceStates(VectorN const& q, Vector6 const& v, Vector6 const& vRef, int is_new_step)
{
    if (q.rows() != 6)
    {
        throw std::runtime_error("q should be a vector of size 6");
    }

    if (is_new_step)
    {
        meanSurface_ = heightmap_.computeMeanSurface(q(0), q(1));  // Update surface equality before new step
        rpyMap_(0) = -std::atan2(meanSurface_(1), 1.);
        rpyMap_(1) = -std::atan2(meanSurface_(0), 1.);
        computeConfigurations(q, vRef);
    }

    rpy_ = q.tail(3);
    double c = std::cos(rpy_(2));
    double s = std::sin(rpy_(2));
    Rz_.topLeftCorner<2, 2>() << c, -s, s, c;

    // Update the current state
    referenceStates_(0, 0) = 0.0;                       // In horizontal frame x = 0.0
    referenceStates_(1, 0) = 0.0;                       // In horizontal frame y = 0.0
    referenceStates_(2, 0) = q(2, 0);                   // We keep height
    referenceStates_.block(3, 0, 2, 1) = rpy_.head(2);  // We keep roll and pitch
    referenceStates_(5, 0) = 0.0;                       // In horizontal frame yaw = 0.0
    referenceStates_.block(6, 0, 3, 1) = v.head(3);
    referenceStates_.block(9, 0, 3, 1) = v.tail(3);

    for (int i = 0; i < nSteps_; i++)
    {
        if (std::abs(vRef(5)) >= 0.001)
        {
            referenceStates_(0, 1 + i) = (vRef(0) * std::sin(vRef(5) * dtVector_(i)) + vRef(1) * (std::cos(vRef(5) * dtVector_(i)) - 1.0)) / vRef(5);
            referenceStates_(1, 1 + i) = (vRef(1) * std::sin(vRef(5) * dtVector_(i)) - vRef(0) * (std::cos(vRef(5) * dtVector_(i)) - 1.0)) / vRef(5);
        }
        else
        {
            referenceStates_(0, 1 + i) = vRef(0) * dtVector_(i);
            referenceStates_(1, 1 + i) = vRef(1) * dtVector_(i);
        }
        referenceStates_(0, 1 + i) += referenceStates_(0, 0);
        referenceStates_(1, 1 + i) += referenceStates_(1, 0);

        referenceStates_(5, 1 + i) = vRef(5) * dtVector_(i);

        referenceStates_(6, 1 + i) = vRef(0) * std::cos(referenceStates_(5, 1 + i)) - vRef(1) * std::sin(referenceStates_(5, 1 + i));
        referenceStates_(7, 1 + i) = vRef(0) * std::sin(referenceStates_(5, 1 + i)) + vRef(1) * std::cos(referenceStates_(5, 1 + i));

        referenceStates_(11, 1 + i) = vRef(5);

        // Update according to heightmap
        DxDy_(0) = referenceStates_(0, i + 1);
        DxDy_(1) = referenceStates_(1, i + 1);
        DxDy_ = Rz_ * DxDy_ + q.head(3);  // world frame

        referenceStates_(2, 1 + i) = meanSurface_(0) * DxDy_(0) + meanSurface_(1) * DxDy_(1) + meanSurface_(2) + referenceHeight_;

        referenceStates_(3, 1 + i) = rpyMap_[0] * std::cos(-rpy_[2]) - rpyMap_[1] * std::sin(-rpy_[2]);
        referenceStates_(4, 1 + i) = rpyMap_[0] * std::sin(-rpy_[2]) + rpyMap_[1] * std::cos(-rpy_[2]);
    }

    // Update velocities according to heightmap
    for (int i = 0; i < nSteps_; i++)
    {
        if (i == 0)
        {
            referenceStates_(8, 1 + i) = std::max(std::min((referenceStates_(2, 1) - q[2]) / dt_, velocityMaxZ_), -velocityMaxZ_);
            referenceStates_(9, 1 + i) = std::max(std::min((referenceStates_(3, 1) - rpy_[0]) / dt_, velocityMax_), -velocityMax_);
            referenceStates_(10, 1 + i) = std::max(std::min((referenceStates_(4, 1) - rpy_[1]) / dt_, velocityMax_), -velocityMax_);
        }
        else
        {
            referenceStates_(9, 1 + i) = 0.;
            referenceStates_(10, 1 + i) = 0.;
            referenceStates_(8, 1 + i) = (referenceStates_(2, 2) - referenceStates_(2, 1)) / dt_;
        }
    }
}

void StatePlanner3D::computeConfigurations(VectorN const& q, Vector6 const& vRef)
{
    Vector3 meanSurfaceTmp = Vector3::Zero();  // Temporary mean surface vector
    Vector3 rpyMapTmp = Vector3::Zero();       // Temporary rpy configuration from surface

    for (int i = 0; i < nSurfaceConfigs_; i++)
    {
        double dt_config = stepDuration_ * (i + 2);  // Delay of 2 phase of contact for MIP

        if (std::abs(vRef(5)) >= 0.001)
        {
            configs_(0, i) = (vRef(0) * std::sin(vRef(5) * dt_config) + vRef(1) * (std::cos(vRef(5) * dt_config) - 1.0)) / vRef(5);
            configs_(1, i) = (vRef(1) * std::sin(vRef(5) * dt_config) - vRef(0) * (std::cos(vRef(5) * dt_config) - 1.0)) / vRef(5);
        }
        else
        {
            configs_(0, i) = vRef(0) * dt_config;
            configs_(1, i) = vRef(1) * dt_config;
        }
        configs_(0, i) = std::cos(q(5)) * configs_(0, i) - std::sin(q(5)) * configs_(1, i);  // Yaw rotation for dx
        configs_(1, i) = std::sin(q(5)) * configs_(0, i) + std::cos(q(5)) * configs_(1, i);  // Yaw rotation for dy
        configs_.block(0, i, 2, 1) += q.head(2);                                             // Add initial position

        // Compute the mean surface according to the prediction
        // meanSurfaceTmp = heightmap_.computeMeanSurface(configs_(0, i), configs_(1, i));
        // rpyMapTmp(0) = -std::atan2(meanSurfaceTmp(1), 1.);
        // rpyMapTmp(1) = -std::atan2(meanSurfaceTmp(0), 1.);
        // Update according to heightmap
        // configs_(2, i) = meanSurfaceTmp(0) * configs_(0, i) + meanSurfaceTmp(1) * configs_(1, i) + meanSurfaceTmp(2) + referenceHeight_;
        // rpyConfig_(2) = q(5) + vRef(5) * dt_config;
        // rpyConfig_(0) = rpyMapTmp(0) * std::cos(rpyConfig_(2)) - rpyMapTmp(1) * std::sin(rpyConfig_(2));
        // rpyConfig_(1) = rpyMapTmp(0) * std::sin(rpyConfig_(2)) + rpyMapTmp(1) * std::cos(rpyConfig_(2));

        configs_(2, i) = meanSurface_(0) * configs_(0, i) + meanSurface_(1) * configs_(1, i) + meanSurface_(2) + referenceHeight_;
        rpyConfig_(2) = q(5) + vRef(5) * dt_config;
        rpyConfig_(0) = rpyMap_(0) * std::cos(rpyConfig_(2)) - rpyMap_(1) * std::sin(rpyConfig_(2));
        rpyConfig_(1) = rpyMap_(0) * std::sin(rpyConfig_(2)) + rpyMap_(1) * std::cos(rpyConfig_(2));

        configs_.block(3, i, 4, 1) = pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(rpyConfig_)).coeffs();
    }
}
