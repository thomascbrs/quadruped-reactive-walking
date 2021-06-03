import numpy as np
import os

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
limbs = ['HRleg', 'HLleg', 'FLleg', 'FRleg']
others = ['HR_FOOT', 'HL_FOOT', 'FL_FOOT', 'FR_FOOT']
rom_names = ['solo_RHleg_rom', 'solo_LHleg_rom', 'solo_LFleg_rom', 'solo_RFleg_rom']
offsets = {'FRleg':  [0.1946, -0.0875, -0.241], 'FLleg': [0.1946, 0.0875, -0.241],
           'HRleg': [-0.1946, -0.0875, -0.241], 'HLleg': [-0.1946, 0.0875, -0.241]}

# --------------------------------- METHODS ---------------------------------------------------------------


class SurfacePlanner:
    """
    Choose the next surface to use by solving a MIP problem
    """

    def __init__(self, environment_URDF):
        """
        Initialize the affordance tool and save the solo abstract rbprm builder, and surface dictionary
        """
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

    def run(self, states, R, gait, current_contacts, step_length):
        """
        Select the nex surfaces to use
        :param states: a list of successive base positions
        :param R: a list of successive orientations of the robot
        :param gait: a gait matrix
        :param current_contacts: the initial_contacts to use in the computation
        :param step_length: the desired step_length for the cost
        :return: the selected surfaces for the first phase
        """
        surfaces = self.get_potential_surfaces(configs, gait)

        pb = Problem(limb_names=limbs, other_names=others, constraint_paths=paths)
        pb.generate_problem(R, surfaces, gait, current_contacts, com=False)

        costs = {"step_size": [1.0, step_length]}
        pb_data = solve_MIP(pb, costs=costs, com=False)

        if pb_data.success:
            surface_indices = pb_data.surface_indices

            selected_surfaces = []
            for foot, index in enumerate(surface_indices[0]):
                selected_surfaces.append[surfaces[0][foot][index]]

            return selected_surfaces
