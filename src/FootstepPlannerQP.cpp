#include "qrw/FootstepPlannerQP.hpp"
#include <chrono>
using namespace std::chrono;

FootstepPlannerQP::FootstepPlannerQP()
    : gait_(NULL)
    , k_(003)
    , k_feedback(0.03)
    , g(9.81)
    , L(0.155)
    , nextFootstep_(Vector3::Zero())
    , nextFootstepQP_(Vector3::Zero())
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
    , b_v_(Vector3::Zero())
    , b_vRef_(Vector6::Zero())
    , feet_()
    , t0s(Vector4::Zero())
    , t_swing(Vector4::Zero())
    , P_ {MatrixN::Zero(N, N)}
    , q_ {VectorN::Zero(N)}
    , G_ {MatrixN::Zero(M, N)}
    , h_ {VectorN::Zero(M)}
    , C_ {MatrixN::Zero(N, 0)}
    , d_ {VectorN::Zero(0)}
    , x {VectorN::Zero(N)}
    , vOptim_ {Vector3::Zero(N)}
    , b_vOptim_ {Vector3::Zero(N)}
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
                                   Surface initialSurface_in)
{
    dt_tsid = dt_tsid_in;
    surfaceStatus_ = false;
    surfaceIteration_ = 0;
    initialSurface_ = initialSurface_in;
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
    for (int i = 0; i < N_gait; i++)
    {
        footsteps_.push_back(Matrix34::Zero());
    }
    Rz(2, 2) = 1.0;

    for (int foot = 0; foot < 4; foot++)
    {
        selectedSurfaces_.push_back(initialSurface_in);
    }

    // QP initialization
    qp.reset(N, 0, M);
    weights_.setZero(N);
    weights_ << 0.05, 0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    P_.diagonal() << weights_;
}


MatrixN FootstepPlannerQP::computeTargetFootstep(int const k_in,
                                                 VectorN const& q,
                                                 Vector6 const& v,
                                                 Vector6 const& b_vref,
                                                 SurfaceVectorVector const& potentialSurfaces,
                                                 SurfaceVector const& surfaces,
                                                 bool const surfaceStatus,
                                                 int const surfaceIteration)
{
    k_ = k_in;
    surfaceStatus_ = surfaceStatus;
    surfaceIteration_ = surfaceIteration;
    surfaces_ = surfaces;
    potentialSurfaces_ = potentialSurfaces;

    // Get the reference velocity in world frame (given in base frame)
    quat_ = {q(6), q(3), q(4), q(5)};  // w, x, y, z
    RPY_ << pinocchio::rpy::matrixToRpy(quat_.toRotationMatrix());

    double c = std::cos(RPY_(2));
    double s = std::sin(RPY_(2));
    Rz.topLeftCorner<2, 2>() << c, -s, s, c;

    // Compute the desired location of footsteps over the prediction horizon
    computeFootsteps(q, v, b_vref);

    // Update desired location of footsteps on the ground
    updateTargetFootsteps();

    return targetFootstep_;
}


void FootstepPlannerQP::computeFootsteps(VectorN const& q,
                                         Vector6 const& v,
                                         Vector6 const& vref)
{
    MatrixN gait = gait_->getCurrentGait();

    // Set current position of feet for feet in stance phase
    std::fill(footsteps_.begin(), footsteps_.end(), Matrix34::Zero());
    for (int foot = 0; foot < 4; foot++)
    {
        if (gait(0, foot) == 1.0)
            footsteps_[0].col(foot) = currentFootstep_.col(foot);
    }

    // Cumulative time by adding the terms in the first column (remaining number of timesteps)
    // Get future yaw yaws compared to current position
    dt_cum(0) = dt;
    yaws(0) = vref(5) * dt_cum(0) + RPY_(2);
    for (int foot = 1; foot < footsteps_.size(); foot++)
    {
        dt_cum(foot) = gait.row(foot).isZero() ? dt_cum(foot - 1) : dt_cum(foot - 1) + dt;
        yaws(foot) = vref(5) * dt_cum(foot) + RPY_(2);

        if (vref(5) != 0)
        {
            dx(foot) = (v(0) * std::sin(vref(5) * dt_cum(foot)) + v(1) * (std::cos(vref(5) * dt_cum(foot)) - 1.0)) / vref(5);
            dy(foot) = (v(1) * std::sin(vref(5) * dt_cum(foot)) - v(0) * (std::cos(vref(5) * dt_cum(foot)) - 1.0)) / vref(5);
        }
        else
        {
            dx(foot) = v(0) * dt_cum(foot);
            dy(foot) = v(1) * dt_cum(foot);
        }
    }


    // Get current and reference velocities in base frame (rotated yaw)
    b_v_ = Rz.transpose() * v.head(3);  //linear velocity in base frame
    b_vRef_.head(3) = Rz.transpose() * vref.head(3);
    b_vRef_.tail(3) = Rz.transpose() * vref.tail(3);

    // Current position without height
    q_tmp = q.head(3);
    q_tmp(2) = 0.0;

    update_remaining_time();

    // Update the footstep matrix depending on the different phases of the gait (swing & stance)
    int i = 1;
    int phase = 0;
    optimVector_.clear();
    while (!gait.row(i).isZero())
    {
        // Feet that were in stance phase and are still in stance phase do not move
        for (int foot = 0; foot < 4; foot++)
        {
            if (gait(i - 1, foot) * gait(i, foot) > 0)
            {
                footsteps_[i].col(foot) = footsteps_[i - 1].col(foot);
            }
        }

        int moving_foot_index = 0;
        // Feet that were in swing phase and are now in stance phase need to be updated
        for (int foot = 0; foot < 4; foot++)
        {
            if ((1 - gait(i - 1, foot)) * gait(i, foot) > 0)
            {
                // Offset to the future position
                q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;

                // Get future desired position of footsteps
                computeNextFootstep(i, foot, b_vRef_.head(3), nextFootstep_, true);
                computeNextFootstep(i, foot, b_vRef_.head(3), nextFootstepQP_, false);

                // Get desired position of footstep compared to current position
                Rz_tmp.setZero();
                double c = std::cos(yaws(i - 1));
                double s = std::sin(yaws(i - 1));
                Rz_tmp.topLeftCorner<2, 2>() << c, -s, s, c;

                nextFootstepQP_ = (Rz_tmp * nextFootstepQP_ + q_tmp + q_dxdy).transpose();

                bool flyingFoot = false;
                for (int other_foot = 0; other_foot < (int)feet_.size(); other_foot++)
                {
                    if (feet_[other_foot] == foot)
                    {
                        flyingFoot = true;
                    }
                }

                if (flyingFoot && phase == 0)
                {
                    // feet currently in flying phase
                    if (t0s[foot] < 10e-4 and k_ % k_mpc == 0)
                    {
                        // Beginning of flying phase, selection of surface
                        if (surfaceStatus_)
                        {
                            selectedSurfaces_[foot] = surfaces_[moving_foot_index];
                        }
                        else
                        {
                            // Select surface with heuristic
                            selectedSurfaces_[foot] = selectSurfaceFromPoint(nextFootstep_, phase, moving_foot_index);
                        }
                    }
                    optimData optim_data = {i, foot, selectedSurfaces_[foot], nextFootstepQP_};
                    optimVector_.push_back(optim_data);
                }
                else
                {
                    Surface sf_ = Surface();
                    if (surfaceStatus_)
                    {
                        sf_ = surfaces_[moving_foot_index];
                    }
                    else
                    {
                        // Select surface with heuristic
                        sf_ = selectSurfaceFromPoint(nextFootstep_, phase, moving_foot_index);
                    }
                    optimData optim_data = {i, foot, sf_, nextFootstepQP_};
                    optimVector_.push_back(optim_data);
                }
                moving_foot_index += 1;
            }
        }
        if (!(gait.row(i - 1) - gait.row(i)).isZero())
        {
            phase += 1;
        }
        i++;
    }

    // Reset matrix Inequalities
    G_.setZero();
    h_.setZero();
    x.setZero();

    // Adapting q_ vector with reference velocity
    q_(0) = -weights_(0) * vref(0);
    q_(1) = -weights_(1) * vref(1);

    // Convert problem to inequalities
    int iStart = 0;
    int foot = 0;
    for (uint id_l = 0; id_l < optimVector_.size(); id_l++)
    {
        iStart = surfaceInequalities(iStart, optimVector_[id_l].surface, optimVector_[id_l].next_pos, foot);
        foot++;
    }

    // Eiquadprog-Fast solves the problem :
    // min. 1/2 * x' C_ x + q_' x
    // s.t. C_ x + d_ = 0
    //      G_ x + h_ >= 0
    status = qp.solve_quadprog(P_, q_, C_, d_, G_, h_, x);

    // Retrieve results
    vOptim_.head(2) = x.head(2);

    // Get new reference velocity in base frame to recompute the new footsteps
    b_vOptim_ = Rz.transpose() * vOptim_;  // lin velocity in base frame (rotated yaw)

    // Update the foostep matrix with the position optimised, for changing phase index
    for (uint id_l = 0; id_l < optimVector_.size(); id_l++)
    {
        int i = optimVector_[id_l].phase;
        int foot = optimVector_[id_l].foot;

        // Offset to the future position
        q_dxdy << dx(i - 1, 0), dy(i - 1, 0), 0.0;

        // Get future desired position of footsteps with k_feedback
        computeNextFootstep(i, foot, b_vOptim_, nextFootstepQP_, true);

        // Get desired position of footstep compared to current position
        Rz_tmp.setZero();
        double c = std::cos(yaws(i - 1));
        double s = std::sin(yaws(i - 1));
        Rz_tmp.topLeftCorner<2, 2>() << c, -s, s, c;

        nextFootstepQP_ = (Rz_tmp * nextFootstepQP_ + q_tmp + q_dxdy).transpose();
        nextFootstepQP_(0) += x(2 + 3 * id_l);
        nextFootstepQP_(1) += x(2 + 3 * id_l + 1);
        nextFootstepQP_(2) += x(2 + 3 * id_l + 2);
        footsteps_[i].col(foot) = nextFootstepQP_;
    }

    // Update the next stance phase after the changing phase
    i = 1;
    while (!gait.row(i).isZero())
    {
        // Feet that were in stance phase and are still in stance phase do not move
        for (int foot = 0; foot < 4; foot++)
        {
            if (gait(i - 1, foot) * gait(i, foot) > 0)
            {
                footsteps_[i].col(foot) = footsteps_[i - 1].col(foot);
            }
        }
        i++;
    }
}

void FootstepPlannerQP::update_remaining_time()
{
    if ((k_ % k_mpc) == 0)
    {
        feet_.clear();
        t0s.setZero();

        // Indexes of feet in swing phase
        for (int i = 0; i < 4; i++)
        {
            if (gait_->getCurrentGait()(0, i) == 0)
                feet_.push_back(i);
        }
        // If no foot in swing phase
        if (feet_.size() == 0)
            return;

        // For each foot in swing phase get remaining duration of the swing phase
        for (int foot = 0; foot < (int)feet_.size(); foot++)
        {
            int i = feet_[foot];
            t_swing[i] = gait_->getPhaseDuration(0, feet_[foot], 0.0);  // 0.0 for swing phase
            double value = t_swing[i] - (gait_->getRemainingTime() * k_mpc - ((k_ + 1) % k_mpc)) * dt_tsid - dt_tsid;
            t0s[i] = std::max(0.0, value);
        }
    }
    else
    {
        // If no foot in swing phase
        if (feet_.size() == 0)
            return;

        // Increment of one time step for feet_ in swing phase
        for (int i = 0; i < (int)feet_.size(); i++)
        {
            double value = t0s[feet_[i]] + dt_tsid;
            t0s[feet_[i]] = std::max(0.0, value);
        }
    }
}

void FootstepPlannerQP::computeNextFootstep(int i, int foot, Vector3 b_vRef_in, Vector3& nextFootstep, bool feedback_ref)
{
    nextFootstep.setZero();

    double t_stance = gait_->getPhaseDuration(i, foot, 1.0);  // 1.0 for stance phase

    // Add symmetry term
    nextFootstep = t_stance * 0.5 * b_v_;
    // Add feedback term
    nextFootstep += k_feedback * b_v_;
    if (feedback_ref)
    {
        nextFootstep += -k_feedback * b_vRef_in.head(3);
    }

    // Add centrifugal term
    Vector3 cross;
    cross << b_v_(1) * b_vRef_in(5) - b_v_(2) * b_vRef_in(4), b_v_(2) * b_vRef_in(3) - b_v_(0) * b_vRef_in(5), 0.0;
    nextFootstep += 0.5 * std::sqrt(h_ref / g) * cross;

    // Legs have a limited length so the deviation has to be limited
    nextFootstep(0) = std::min(nextFootstep(0), L);
    nextFootstep(0) = std::max(nextFootstep(0), -L);
    nextFootstep(1) = std::min(nextFootstep(1), L);
    nextFootstep(1) = std::max(nextFootstep(1), -L);

    // Fix problem :
    nextFootstep[1] = 0.;

    // Add shoulders
    Vector3 RP_ = RPY_;
    RP_[2] = 0;  // Yaw taken into account later
    nextFootstep += pinocchio::rpy::rpyToMatrix(RP_) * shoulders_.col(foot);

    // Remove Z component (working on flat ground)
    nextFootstep(2) = 0.;
}


void FootstepPlannerQP::updateTargetFootsteps()
{
    for (int i = 0; i < 4; i++)
    {
        int index = 0;
        while (footsteps_[index](0, i) == 0.0)
        {
            index++;
        }
        targetFootstep_.col(i) << footsteps_[index](0, i), footsteps_[index](1, i), 0.0;
    }
}


void FootstepPlannerQP::updateNewContact()
{
    for (int i = 0; i < 4; i++)
    {
        if (gait_->getCurrentGaitCoeff(0, i) == 1.0)
        {
            currentFootstep_.col(i) = (footsteps_[1]).col(i);
        }
    }
}

MatrixN FootstepPlannerQP::getFootsteps()
{
    return vectorToMatrix(footsteps_);
}
MatrixN FootstepPlannerQP::getTargetFootsteps()
{
    return targetFootstep_;
}
Vector3 FootstepPlannerQP::getVrefResults()
{
    return vOptim_;
}

Surface FootstepPlannerQP::selectSurfaceFromPoint(Vector3 const& point, int phase, int moving_foot_index)
{
    double sfHeight = 0.;
    bool surfaceFound = false;

    Surface sf;

    if (surfaceIteration_)
    {
        SurfaceVector potentialSurfaces = potentialSurfaces_[moving_foot_index];
        sf = potentialSurfaces[0];
        for (int i = 0; i < potentialSurfaces.size(); i++)
        {
            if (potentialSurfaces[i].hasPoint(point.head(2)))
            {
                double height = sf.getHeight(point.head(2));
                if (height > sfHeight)
                {
                    sfHeight = height;
                    sf = potentialSurfaces[i];
                    surfaceFound = true;
                }
            }
        }

        if (not surfaceFound)
        {
            // TODO

            // obj1 = simple_object(np.array([[point[0], point[1], 0.]]),  [[0, 0, 0]])
            // o1 = fclObj_trimesh(obj1)
            // t1 = hppfcl.Transform3f()
            // t2 = hppfcl.Transform3f()
            // distance = 100

            // for sf in potential_surfaces:
            //     vert = np.zeros(sf.vertices.shape)
            //     vert[:, :2] = sf.vertices[:, :2]
            //     tri = Delaunay(vert[:, :2])
            //     obj2 = simple_object(vert,  tri.simplices.tolist())
            //     o2 = fclObj_trimesh(obj2)
            //     gg = gjk(o1, o2, t1, t2)

            //     if gg.distance <= distance:
            //         surface_selected = sf
            //         distance = gg.distance
        }
    }
    else
    {
        sf = initialSurface_;
    }
    return sf;
}


MatrixN FootstepPlannerQP::vectorToMatrix(std::vector<Matrix34> const& array)
{
    MatrixN M = MatrixN::Zero(array.size(), 12);
    for (uint i = 0; i < array.size(); i++)
    {
        for (int foot = 0; foot < 4; foot++)
        {
            M.row(i).segment<3>(3 * foot) = array[i].col(foot);
        }
    }
    return M;
}

int FootstepPlannerQP::surfaceInequalities(int i_start, Surface const& surface, Vector3 const& next_ft, int foot)
{
    G_.block(i_start, 0, 6, 2) = k_feedback * surface.A_.block(0, 0, 6, 2);
    G_.block(i_start, 2 + 3 * foot, 6, 3) = -surface.A_;
    h_.segment(i_start, 6) = surface.b_ - surface.A_ * next_ft;

    return i_start + 6;
}

SurfaceVector FootstepPlannerQP::getSelectedSurfaces() const
{
    return selectedSurfaces_;
};

Surface FootstepPlannerQP::getSelectedSurface(int const foot) const
{
    return selectedSurfaces_[foot];
};
