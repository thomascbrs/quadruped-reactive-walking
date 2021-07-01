#include "qrw/gepadd.hpp"
#include "qrw/FootTrajectoryGenerator.hpp"
#include "qrw/FootTrajectoryGeneratorBezier.hpp"
#include "qrw/FootstepPlanner.hpp"
#include "qrw/FootstepPlannerQP.hpp"
#include "qrw/Gait.hpp"
#include "qrw/InvKin.hpp"
#include "qrw/MPC.hpp"
#include "qrw/Params.hpp"
#include "qrw/QPWBC.hpp"
#include "qrw/StatePlanner.hpp"
#include "qrw/Surface.hpp"

#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace bp = boost::python;

template <typename MPC>
struct MPCPythonVisitor : public bp::def_visitor<MPCPythonVisitor<MPC>>
{
    template <class PyClassMPC>
    void visit(PyClassMPC& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))
            .def(bp::init<double, int, double, int>(bp::args("dt_in", "n_steps_in", "T_gait_in", "N_gait"),
                                                    "Constructor with parameters."))

            // Run MPC from Python
            .def("run", &MPC::run, bp::args("num_iter", "xref_in", "fsteps_in"), "Run MPC from Python.\n")
            .def("get_latest_result", &MPC::get_latest_result,
                 "Get latest result (predicted trajectory  forces to apply).\n")
            .def("get_gait", &MPC::get_gait, "Get gait matrix.\n")
            .def("get_Sgait", &MPC::get_Sgait, "Get S_gait matrix.\n")
            .add_property("I",
                          bp::make_function(&MPC::get_I, bp::return_value_policy<bp::return_by_value>()),
                          bp::make_function(&MPC::set_I), "Inertia matrix \n");
    }

    static void expose()
    {
        bp::class_<MPC>("MPC", bp::no_init).def(MPCPythonVisitor<MPC>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};

void exposeMPC() { MPCPythonVisitor<MPC>::expose(); }

/////////////////////////////////
/// Binding StatePlanner class
/////////////////////////////////
template <typename StatePlanner>
struct StatePlannerPythonVisitor : public bp::def_visitor<StatePlannerPythonVisitor<StatePlanner>>
{
    template <class PyClassStatePlanner>
    void visit(PyClassStatePlanner& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getReferenceStates", &StatePlanner::getReferenceStates, "Get xref matrix.\n")
            .def("getNSteps", &StatePlanner::getNSteps, "Get number of steps in prediction horizon.\n")

            .def("initialize", &StatePlanner::initialize, bp::args("dt_in", "T_mpc_in", "h_ref_in"),
                 "Initialize StatePlanner from Python.\n")

            // Run StatePlanner from Python
            .def("computeReferenceStates", &StatePlanner::computeReferenceStates, bp::args("q", "v", "b_vref", "z_average"),
                 "Run StatePlanner from Python.\n");
    }

    static void expose()
    {
        bp::class_<StatePlanner>("StatePlanner", bp::no_init).def(StatePlannerPythonVisitor<StatePlanner>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeStatePlanner() { StatePlannerPythonVisitor<StatePlanner>::expose(); }

/////////////////////////////////
/// Binding Surface class
/////////////////////////////////
template <typename Surface>
struct SurfacePythonVisitor : public bp::def_visitor<SurfacePythonVisitor<Surface>>
{
    template <class PyClassSurface>
    void visit(PyClassSurface& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))
            .def(bp::init<MatrixN, VectorN, MatrixN>(bp::args("A", "b", "vertices"), "Constructor with parameters."))

            .def("get_height", &Surface::getHeight, bp::args("point"),
                 "get the height of a point of the surface.\n")

            .add_property("A",
                          bp::make_function(&Surface::getA, bp::return_value_policy<bp::return_by_value>()))
            .add_property("b",
                          bp::make_function(&Surface::getb, bp::return_value_policy<bp::return_by_value>()))
            .add_property("vertices",
                          bp::make_function(&Surface::getVertices, bp::return_value_policy<bp::return_by_value>()))

            .def("has_point", &Surface::hasPoint, bp::args("point"),
                 "return true if the point is in the surface.\n");
    }

    static void expose()
    {
        bp::class_<Surface>("Surface", bp::no_init).def(SurfacePythonVisitor<Surface>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeSurface() { SurfacePythonVisitor<Surface>::expose(); }

/////////////////////////////////
/// Binding Gait class
/////////////////////////////////
template <typename Gait>
struct GaitPythonVisitor : public bp::def_visitor<GaitPythonVisitor<Gait>>
{
    template <class PyClassGait>
    void visit(PyClassGait& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getCurrentGait", &Gait::getCurrentGait, "Get currentGait_ matrix.\n")
            .def("isNewPhase", &Gait::isNewPhase, "Get newPhase_ boolean.\n")
            .def("getIsStatic", &Gait::getIsStatic, "Get is_static_ boolean.\n")

            .def("initialize", &Gait::initialize, bp::args("dt_in", "T_gait_in", "T_mpc_in", "N_gait"),
                 "Initialize Gait from Python.\n")

            .def("updateGait", &Gait::updateGait, bp::args("k", "k_mpc", "q", "joystickCode"),
                 "Update current gait matrix from Python.\n")

            .def("setGait", &Gait::setGait, bp::args("gaitMatrix"),
                 "Set current gait matrix from Python.\n")

            .def("getPhaseDuration", &Gait::getPhaseDuration, bp::args("i", "j", "value"),
                 "Compute the remaining and total duration of a phase.\n")

            .def("getRemainingTime", &Gait::getRemainingTime, "get remaining time of the current phase\n");
    }

    static void expose()
    {
        bp::class_<Gait>("Gait", bp::no_init).def(GaitPythonVisitor<Gait>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeGait() { GaitPythonVisitor<Gait>::expose(); }

/////////////////////////////////
/// Binding FootstepPlanner class
/////////////////////////////////
template <typename FootstepPlanner>
struct FootstepPlannerPythonVisitor : public bp::def_visitor<FootstepPlannerPythonVisitor<FootstepPlanner>>
{
    template <class PyClassFootstepPlanner>
    void visit(PyClassFootstepPlanner& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getFootsteps", &FootstepPlanner::getFootsteps, "Get footsteps_ matrix.\n")

            .def("initialize", &FootstepPlanner::initialize, bp::args("dt_in", "T_mpc_in", "h_ref_in", "shouldersIn", "gaitIn", "N_gait"),
                 "Initialize FootstepPlanner from Python.\n")

            // Compute target location of footsteps from Python
            .def("computeTargetFootstep", &FootstepPlanner::computeTargetFootstep, bp::args("q", "v", "b_vref"),
                 "Compute target location of footsteps from Python.\n")

            .def("updateNewContact", &FootstepPlanner::updateNewContact, "Refresh feet position when entering a new contact phase.\n");
    }

    static void expose()
    {
        bp::class_<FootstepPlanner>("FootstepPlanner", bp::no_init).def(FootstepPlannerPythonVisitor<FootstepPlanner>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeFootstepPlanner() { FootstepPlannerPythonVisitor<FootstepPlanner>::expose(); }


/////////////////////////////////
/// Binding FootstepPlannerQP class
/////////////////////////////////
template <typename FootstepPlannerQP>
struct FootstepPlannerQPPythonVisitor : public bp::def_visitor<FootstepPlannerQPPythonVisitor<FootstepPlannerQP>>
{
    template <class PyClassFootstepPlannerQP>
    void visit(PyClassFootstepPlannerQP& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))


            .def("initialize", &FootstepPlannerQP::initialize, bp::args("dt_in", "T_mpc_in", "h_ref_in", "k_mpc", "dt_tsid", "shouldersIn", "gaitIn", "N_gait", "floor_surface_in"),
                 "Initialize FootstepPlannerQP from Python.\n")

            // Compute target location of footsteps from Python
            .def("computeTargetFootstep", &FootstepPlannerQP::computeTargetFootstep, bp::args("k_in", "q", "v", "b_vref", "potentialSurfaces", "surfaces", "status", "iteration"),
                 "Compute target location of footsteps from Python.\n")

            .def("updateNewContact", &FootstepPlannerQP::updateNewContact, "Refresh feet position when entering a new contact phase.\n")
            .def("get_selected_surfaces", &FootstepPlannerQP::getSelectedSurfaces, "get the selected surfaces \n")
            .def("get_selected_surface", &FootstepPlannerQP::getSelectedSurface, bp::args("foot"), "get the selected surfaces \n")
            .def("getFootsteps", &FootstepPlannerQP::getFootsteps, "Get footsteps_ matrix.\n");
    }

    static void expose()
    {
        bp::class_<FootstepPlannerQP>("FootstepPlannerQP", bp::no_init).def(FootstepPlannerQPPythonVisitor<FootstepPlannerQP>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeFootstepPlannerQP()
{
    bp::class_<SurfaceVector>("SurfaceVector")
        .def(bp::vector_indexing_suite<SurfaceVector>());
    bp::class_<SurfaceVectorVector>("SurfaceVectorVector")
        .def(bp::vector_indexing_suite<SurfaceVectorVector>());
    FootstepPlannerQPPythonVisitor<FootstepPlannerQP>::expose();
}

/////////////////////////////////
/// Binding FootTrajectoryGenerator class
/////////////////////////////////
template <typename FootTrajectoryGenerator>
struct FootTrajectoryGeneratorPythonVisitor : public bp::def_visitor<FootTrajectoryGeneratorPythonVisitor<FootTrajectoryGenerator>>
{
    template <class PyClassFootTrajectoryGenerator>
    void visit(PyClassFootTrajectoryGenerator& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getFootPosition", &FootTrajectoryGenerator::getFootPosition, "Get position_ matrix.\n")
            .def("getFootVelocity", &FootTrajectoryGenerator::getFootVelocity, "Get velocity_ matrix.\n")
            .def("getFootAcceleration", &FootTrajectoryGenerator::getFootAcceleration, "Get acceleration_ matrix.\n")

            .def("initialize", &FootTrajectoryGenerator::initialize, bp::args("maxHeightIn", "lockTimeIn", "targetFootstepIn", "initialFootPosition", "dt_tsid_in", "k_mpc_in", "gaitIn"),
                 "Initialize FootTrajectoryGenerator from Python.\n")

            // Compute target location of footsteps from Python
            .def("update", &FootTrajectoryGenerator::update, bp::args("k", "targetFootstep"),
                 "Compute target location of footsteps from Python.\n");
    }

    static void expose()
    {
        bp::class_<FootTrajectoryGenerator>("FootTrajectoryGenerator", bp::no_init).def(FootTrajectoryGeneratorPythonVisitor<FootTrajectoryGenerator>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeFootTrajectoryGenerator() { FootTrajectoryGeneratorPythonVisitor<FootTrajectoryGenerator>::expose(); }

/////////////////////////////////
/// Binding FootTrajectoryGeneratorBezier class
/////////////////////////////////
template <typename FootTrajectoryGeneratorBezier>
struct FootTrajectoryGeneratorBezierPythonVisitor : public bp::def_visitor<FootTrajectoryGeneratorBezierPythonVisitor<FootTrajectoryGeneratorBezier>>
{
    template <class PyClassFootTrajectoryGeneratorBezier>
    void visit(PyClassFootTrajectoryGeneratorBezier& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getFootPosition", &FootTrajectoryGeneratorBezier::getFootPosition, "Get position_ matrix.\n")
            .def("getFootVelocity", &FootTrajectoryGeneratorBezier::getFootVelocity, "Get velocity_ matrix.\n")
            .def("getFootAcceleration", &FootTrajectoryGeneratorBezier::getFootAcceleration, "Get acceleration_ matrix.\n")
            .def("evaluateBezier", &FootTrajectoryGeneratorBezier::evaluateBezier, "Evaluate Bezier curve by foot.\n")

            .def("initialize", &FootTrajectoryGeneratorBezier::initialize, bp::args("maxHeightIn", "lockTimeIn", "targetFootstepIn", "initialFootPosition", "dt_tsid_in", "k_mpc_in", "gaitIn"),
                 "Initialize FootTrajectoryGeneratorBezier from Python.\n")
            
            .add_property("t0s",
                          bp::make_function(&FootTrajectoryGeneratorBezier::get_t0s, bp::return_value_policy<bp::return_by_value>()))
            .add_property("t_swing",
                          bp::make_function(&FootTrajectoryGeneratorBezier::get_t_swing, bp::return_value_policy<bp::return_by_value>()))

            // Compute target location of footsteps from Python
            .def("update", &FootTrajectoryGeneratorBezier::update, bp::args("k", "targetFootstep"),
                 "Compute target location of footsteps from Python.\n");
    }

    static void expose()
    {
        bp::class_<FootTrajectoryGeneratorBezier>("FootTrajectoryGeneratorBezier", bp::no_init).def(FootTrajectoryGeneratorBezierPythonVisitor<FootTrajectoryGeneratorBezier>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeFootTrajectoryGeneratorBezier() { FootTrajectoryGeneratorBezierPythonVisitor<FootTrajectoryGeneratorBezier>::expose(); }

/////////////////////////////////
/// Binding InvKin class
/////////////////////////////////
template <typename InvKin>
struct InvKinPythonVisitor : public bp::def_visitor<InvKinPythonVisitor<InvKin>>
{
    template <class PyClassInvKin>
    void visit(PyClassInvKin& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))
            .def(bp::init<double>(bp::args("dt_in"), "Constructor with parameters."))

            .def("get_q_step", &InvKin::get_q_step, "Get velocity goals matrix.\n")
            .def("get_dq_cmd", &InvKin::get_dq_cmd, "Get acceleration goals matrix.\n")

            // Run InvKin from Python
            .def("refreshAndCompute", &InvKin::refreshAndCompute,
                 bp::args("x_cmd", "contacts", "goals", "vgoals", "agoals", "posf", "vf", "wf", "af", "Jf",
                          "posb", "rotb", "vb", "ab", "Jb"),
                 "Run InvKin from Python.\n");
    }

    static void expose()
    {
        bp::class_<InvKin>("InvKin", bp::no_init).def(InvKinPythonVisitor<InvKin>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};

void exposeInvKin() { InvKinPythonVisitor<InvKin>::expose(); }

/////////////////////////////////
/// Binding QPWBC class
/////////////////////////////////
template <typename QPWBC>
struct QPWBCPythonVisitor : public bp::def_visitor<QPWBCPythonVisitor<QPWBC>>
{
    template <class PyClassQPWBC>
    void visit(PyClassQPWBC& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("get_f_res", &QPWBC::get_f_res, "Get velocity goals matrix.\n")
            .def("get_ddq_res", &QPWBC::get_ddq_res, "Get acceleration goals matrix.\n")
            .def("get_H", &QPWBC::get_H, "Get H weight matrix.\n")

            // Run QPWBC from Python
            .def("run", &QPWBC::run, bp::args("M", "Jc", "f_cmd", "RNEA", "k_contacts"), "Run QPWBC from Python.\n");
    }

    static void expose()
    {
        bp::class_<QPWBC>("QPWBC", bp::no_init).def(QPWBCPythonVisitor<QPWBC>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};
void exposeQPWBC() { QPWBCPythonVisitor<QPWBC>::expose(); }

/////////////////////////////////
/// Binding Params class
/////////////////////////////////
template <typename Params>
struct ParamsPythonVisitor : public bp::def_visitor<ParamsPythonVisitor<Params>>
{
    template <class PyClassParams>
    void visit(PyClassParams& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("initialize", &Params::initialize, bp::args("file_path"),
                 "Initialize Params from Python.\n")

            // Read Params from Python
            .def_readwrite("interface", &Params::interface)
            .def_readwrite("SIMULATION", &Params::SIMULATION)
            .def_readwrite("LOGGING", &Params::LOGGING)
            .def_readwrite("PLOTTING", &Params::PLOTTING)
            .def_readwrite("dt_wbc", &Params::dt_wbc)
            .def_readwrite("N_gait", &Params::N_gait)
            .def_readwrite("envID", &Params::envID)
            .def_readwrite("velID", &Params::velID)
            .def_readwrite("dt_mpc", &Params::dt_mpc)
            .def_readwrite("T_gait", &Params::T_gait)
            .def_readwrite("T_mpc", &Params::T_mpc)
            .def_readwrite("N_SIMULATION", &Params::N_SIMULATION)
            .def_readwrite("type_MPC", &Params::type_MPC)
            .def_readwrite("use_flat_plane", &Params::use_flat_plane)
            .def_readwrite("predefined_vel", &Params::predefined_vel)
            .def_readwrite("kf_enabled", &Params::kf_enabled)
            .def_readwrite("enable_pyb_GUI", &Params::enable_pyb_GUI);
    }

    static void expose()
    {
        bp::class_<Params>("Params", bp::no_init).def(ParamsPythonVisitor<Params>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeParams() { ParamsPythonVisitor<Params>::expose(); }

/////////////////////////////////
/// Exposing classes
/////////////////////////////////
BOOST_PYTHON_MODULE(libquadruped_reactive_walking)
{
    boost::python::def("add", gepetto::example::add);
    boost::python::def("sub", gepetto::example::sub);

    eigenpy::enableEigenPy();

    exposeMPC();
    exposeStatePlanner();
    exposeSurface();
    exposeGait();
    exposeFootstepPlanner();
    exposeFootstepPlannerQP();
    exposeFootTrajectoryGenerator();
    exposeFootTrajectoryGeneratorBezier();
    exposeInvKin();
    exposeQPWBC();
    exposeParams();
}