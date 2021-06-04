import pinocchio as pin
import numpy as np
import os
from time import perf_counter as clock

from sl1m.problem_definition import Problem
from sl1m.generic_solver import solve_MIP

from solo_rbprm.solo_abstract import Robot as SoloAbstract

from hpp.corbaserver.affordance.affordance import AffordanceTool
from hpp.corbaserver.rbprm.tools.surfaces_from_path import getAllSurfacesDict
from hpp.corbaserver.problem_solver import ProblemSolver
from hpp.gepetto import ViewerFactory

# --------------------------------- PROBLEM DEFINITION ---------------------------------------------------------------

paths = [os.environ["INSTALL_HPP_DIR"] + "/share/solo-rbprm/com_inequalities/feet_quasi_flat/",
         os.environ["INSTALL_HPP_DIR"] + "/share/solo-rbprm/relative_effector_positions/"]
limbs = ['FLleg', 'FRleg', 'HLleg', 'HRleg']
others = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
rom_names = ['solo_LFleg_rom', 'solo_RFleg_rom', 'solo_LHleg_rom', 'solo_RHleg_rom']

# --------------------------------- METHODS ---------------------------------------------------------------


class SurfacePlanner:
    """
    Choose the next surface to use by solving a MIP problem
    """

    def __init__(self, environment_URDF, T_gait):
        """
        Initialize the affordance tool and save the solo abstract rbprm builder, and surface dictionary
        """
        self.T_gait = T_gait
        self.solo_abstract = SoloAbstract()
        self.solo_abstract.setJointBounds("root_joint", [-5., 5., -5., 5., 0.241, 1.5])
        self.solo_abstract.boundSO3([-3.14, 3.14, -0.01, 0.01, -0.01, 0.01])
        self.solo_abstract.setFilter(rom_names)
        for limb in rom_names:
            self.solo_abstract.setAffordanceFilter(limb, ['Support'])
        ps = ProblemSolver(self.solo_abstract)
        vf = ViewerFactory(ps)
        afftool = AffordanceTool()
        afftool.setAffordanceConfig('Support', [0.5, 0.03, 0.00005])
        afftool.loadObstacleModel(environment_URDF, "environment", vf, reduceSizes=[0.05, 0., 0.])
        ps.selectPathValidation("RbprmPathValidation", 0.05)

        self.all_surfaces = getAllSurfacesDict(afftool)

    def compute_gait(self, gait_in):
        """
        Get a gait matrix with only one line per phase
        :param gait_in: gait matrix with several line per phase
        :return: gait matrix
        """
        gait = [gait_in[0, :]]
        for i in range(1, gait_in.shape[0] - 1):
            if (gait_in[i, :] != gait[-1]).any():
                gait.append(gait_in[i, :])

        # TODO only works if we the last phase is not a flying phase
        gait.pop(-1)
        gait = np.roll(gait, -1, axis=0)

        return gait

    def compute_step_length(self, o_v_ref):
        """
        Get a gait matrix with only one line per phase
        :param gait_in: gait matrix with several line per phase
        :return: gait matrix
        """
        # TODO: Divide by number of phases in gait
        step_length = o_v_ref * self.T_gait/4

        return np.array([step_length[i][0] for i in range(2)])

    def get_potential_surfaces(self, configs, gait):
        """
        Get the rotation matrix and surface condidates for each configuration in configs
        :param configs: a list of successive configurations of the robot
        :param gait: a gait matrix
        :return: a list of surface candidates
        """
        surfaces_list = []
        for id, config in enumerate(configs):
            stance_feet = np.nonzero(gait[id % len(gait)] == 1)[0]
            previous_swing_feet = np.nonzero(gait[(id-1) % len(gait)] == 0)[0]
            moving_feet = stance_feet[np.in1d(stance_feet, previous_swing_feet, assume_unique=True)]
            roms = np.array(rom_names)[moving_feet]

            foot_surfaces = []
            for rom in roms:
                surfaces = []
                surfaces_names = self.solo_abstract.clientRbprm.rbprm.getCollidingObstacleAtConfig(config.tolist(), rom)
                for name in surfaces_names:
                    surfaces.append(self.all_surfaces[name][0])

                # Sort and then convert to array
                surfaces = sorted(surfaces)
                surfaces_array = []
                for surface in surfaces:
                    surfaces_array.append(np.array(surface).T)

                # Add to surfaces list
                foot_surfaces.append(surfaces_array)
            surfaces_list.append(foot_surfaces)

        return surfaces_list

    def run(self, configs, gait_in, current_contacts, o_v_ref):
        """
        Select the nex surfaces to use
        :param xref: successive states
        :param gait: a gait matrix
        :param current_contacts: the initial_contacts to use in the computation
        :param o_v_ref: the desired velocity for the cost
        :return: the selected surfaces for the first phase
        """
        print("------------------------------------------------------------------------------------")
        t0 = clock()

        R = [pin.XYZQUATToSE3(np.array(config)).rotation for config in configs]

        gait = self.compute_gait(gait_in)

        step_length = self.compute_step_length(o_v_ref)

        surfaces = self.get_potential_surfaces(configs, gait)

        initial_contacts = [current_contacts[:, i].tolist() for i in range(4)]

        pb = Problem(limb_names=limbs, other_names=others, constraint_paths=paths)
        pb.generate_problem(R, surfaces, gait, initial_contacts, com=False)

        costs = {"step_size": [1.0, step_length]}
        pb_data = solve_MIP(pb, costs=costs, com=False)

        if pb_data.success:
            surface_indices = pb_data.surface_indices

            selected_surfaces = []
            for foot, index in enumerate(surface_indices[0]):
                selected_surfaces.append(surfaces[0][foot][index])
            
            t1 = clock()
            print("Run took ", 1000. * (t1-t0))

            return selected_surfaces

        else:
            # TODO what if the problem did not converge ???
            return None
