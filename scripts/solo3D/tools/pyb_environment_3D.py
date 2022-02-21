import numpy as np
from IPython.terminal.embed import embed
import pybullet as pyb
import pybullet_data
import os
import pinocchio as pin
from scipy.linalg.matfuncs import fractional_matrix_power
from sl1m.constants_and_tools import default_transform_from_pos_normal
from sl1m.tools.obj_to_constraints import load_obj, rotate_inequalities, as_inequalities
from example_robot_data.robots_loader import Solo12Loader


class PybEnvironment3D():
    ''' Class to vizualise the 3D environment and foot trajectory and in PyBullet simulation.
    '''

    def __init__(self, params, gait, statePlanner, footStepPlanner, footTrajectoryGenerator, model):
        """
        Store the solo3D class, used for trajectory vizualisation.
        Args:
        - params: parameters of the simulation.
        - gait: Gait class.
        - statePlanner: State planner class.
        - footStepPlannerQP: Footstep planner class (QP version).
        - footTrajectoryGenerator: Foot trajectory class (Bezier version).
        """
        self.enable_pyb_GUI = params.enable_pyb_GUI
        self.URDF = os.environ["SOLO3D_ENV_DIR"] + params.environment_URDF
        self.params = params

        # Parameters for display
        self._SL1M = True  # Display SL1M position.
        self._TRAJ = True  # Display 3D foot trajectory.
        self._CAMERA = False  # Update Pyb camera.
        self._COM = True  # Display CoM constraints.

        # Parameters for Errors printing
        self._COM_CHECK = True  # Print if error between current CoM position and CoM constraints in Sl1M.

        # Check errors
        if (params.enable_multiprocessing_mip and self._SL1M):
            raise AttributeError("MIP Multiprocessing activated, not possible to display SL1M targets.")

        # Solo3D python class
        self.footStepPlanner = footStepPlanner
        self.statePlanner = statePlanner
        self.gait = gait
        self.footTrajectoryGenerator = footTrajectoryGenerator
        self.refresh = 1  # Int to determine when refresh object position (k % refresh == 0)

        self.n_points = 15
        self.trajectory_Ids = np.zeros((3, 4, self.n_points))
        self.sl1m_target_Ids = np.zeros((7, 4))
        self.com_Ids = np.zeros(4)

        self.com_objects = []  # CoM inequalities
        self.model = model
        self.data = model.createData()
        self.max_distance = 0.
        self._feet_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]

    def update(self, k, all_feet_pos, is_new_step, o_feet, gait, q):
        ''' Update position of the objects in pybullet environment.
        Args :
            - k (int) : step of the simulation.
            - all_feet_pos (list): List of optimized position such as : [[Foot 1 next_pos, None , Foot1 next_pos] ,
                                                                    [Foot 2 next_pos, None , Foot2 next_pos] ...]
            - is_new_step (int): Boolean if a new flying phase has just started.
            - o_feet : (Array 3x4), Position of the feet in world frame
            - gait : (Array x4), Current gait matrix.
            - q : (Array 18x1), State of the robot in world frame.
        '''
        # On iteration 0, PyBullet env has not been started
        if k == 1:
            self.initializeEnv()

        if self.enable_pyb_GUI and k > 1 and not self.params.enable_multiprocessing_mip:
            if k % self.refresh == 0:
                # Update target trajectory, current and next phase
                if self._TRAJ:
                    self.updateTargetTrajectory()

                if self._COM:
                    self.updateCoMConstraints(o_feet, gait, q)

                if self._COM_CHECK:
                    self.check_CoM_inequalities(o_feet, gait, q)

                if self._CAMERA:
                    self.updateCamera(k)

            if self._SL1M and is_new_step:
                self.update_target_SL1M(all_feet_pos)

        return 0

    def updateCoMConstraints(self, o_feet, gait, q):
        ''' Update the pybullet simulation with the CoM constraints.
        Args :
            - o_feet : (Array 3x4), Position of the feet in world frame
            - gait : (Array x4), Current gait matrix.
            - q : (Array 18x1), State of the robot in world frame.
        '''
        quat = pin.Quaternion(pin.rpy.rpyToMatrix(q[3:6, 0]))

        for foot in range(4):
            if gait[foot] == 1:  # Foot in contact with the ground
                pyb.resetBasePositionAndOrientation(int(self.com_Ids[foot]),
                                                    posObj=o_feet[:, foot],
                                                    ornObj=quat.coeffs())
            else:
                pyb.resetBasePositionAndOrientation(int(self.com_Ids[foot]),
                                                    posObj=np.array([0., 0., -1.]),
                                                    ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

        return 0

    def check_CoM_inequalities(self, o_feet, gait, q):
        ''' Print if the CoM inequalities are not respected.

        Args:
            - o_feet : (Array 3x4), Position of the feet in horizontal frame
            - gait : (Array x4), Current gait matrix.
            - q : (Array x18) : state of the robot
        '''
        R = pin.rpy.rpyToMatrix(q[3:6, 0])
        o_com = q[:3, 0]
        normal = np.array([0., 0., 1.])
        transform = default_transform_from_pos_normal(np.zeros(3), normal, R)
        K = []
        foot_contact = []
        for foot in range(4):
            if gait[foot] == 1.:
                foot_contact.append(foot)
                ine = rotate_inequalities(self.com_objects[foot], transform.copy())
                K.append((ine.A, ine.b))
        flag = 0
        for i in range(len(K)):
            A = K[i][0]
            b = K[i][1]
            ERR = A @ (o_com - o_feet[:, foot_contact[i]]) > b
            if np.any(ERR):
                idx = np.where(ERR)[0][0]
                c = A @ (o_com - o_feet[:, foot_contact[i]])
                print("--------------------------------------------------")
                print("Exceeding CoM kinematic constraints, FOOT : ", self._feet_names[foot_contact[i]])
                flag += 1

        # TOOL to compute the distance contact point --> shoulder (usefull to reshape the inequalities)
        #  Compute the distance of the leg FL
        # quat = pin.Quaternion(pin.rpy.rpyToMatrix(q[3:6, 0]))
        # q_19 = np.zeros((19, 1))
        # q_19[:3, 0] = q[:3, 0]
        # q_19[3:7, 0] = quat.coeffs()
        # q_19[7:, 0] = q[6:, 0]
        # pin.forwardKinematics(self.model, self.data, q_19)
        # pin.updateFramePlacements(self.model, self.data)
        # distance = np.linalg.norm(self.data.oMf[10].translation - self.data.oMf[6].translation)
        # if distance > self.max_distance:
        #     self.max_distance = distance
        #     print("MAX DISTANCE : ", self.max_distance)

        return 0

    def updateCamera(self, k):
        """ Update pybullet camera.
        """
        # Update position of PyBullet camera on the robot position to do as if it was attached to the robot
        if k > 10 and self.enable_pyb_GUI:
            pyb.resetDebugVisualizerCamera(cameraDistance=0.95,
                                           cameraYaw=357,
                                           cameraPitch=-29,
                                           cameraTargetPosition=[0.6, 0.14, -0.22])
        return 0

    def update_target_SL1M(self, all_feet_pos):
        ''' Update position of the SL1M target
        Args :
        -  all_feet_pos : list of optimized position such as : [[Foot 1 next_pos, None , Foot1 next_pos] , [Foot 2 next_pos, None , Foot2 next_pos] ]
        '''
        if len(all_feet_pos) > 0:
            for i in range(len(all_feet_pos[0])):
                for j in range(len(all_feet_pos)):
                    if all_feet_pos[j][i] is None:
                        pyb.resetBasePositionAndOrientation(int(self.sl1m_target_Ids[i, j]),
                                                            posObj=np.array([0., 0., -0.5]),
                                                            ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

                    else:
                        pyb.resetBasePositionAndOrientation(int(self.sl1m_target_Ids[i, j]),
                                                            posObj=all_feet_pos[j][i],
                                                            ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

        return 0

    def updateTargetTrajectory(self):
        ''' Update the target trajectory for current and next phases. Hide the unnecessary spheres.
        '''

        gait = self.gait.getCurrentGait()

        for j in range(4):
            # Count the position of the plotted trajectory in the temporal horizon
            # c = 0 --> Current trajectory/foot pos
            # c = 1 --> Next trajectory/foot pos
            c = 0
            i = 0

            for i in range(gait.shape[0]):
                # footsteps = fsteps[i].reshape((3, 4), order="F")
                if i > 0:
                    if (1 - gait[i - 1, j]) * gait[i, j] > 0:  # from flying phase to stance
                        if c == 0:
                            # Current flying phase, using coeff store in Bezier curve class
                            t0 = self.footTrajectoryGenerator.t0s[j]
                            t1 = self.footTrajectoryGenerator.t_swing[j]
                            t_vector = np.linspace(t0, t1, self.n_points)

                            for id_t, t in enumerate(t_vector):
                                # Bezier trajectory
                                pos = self.footTrajectoryGenerator.evaluateBezier(j, 0, t)
                                # Polynomial Curve 5th order
                                # pos = self.footTrajectoryGenerator.evaluatePoly(j, 0, t)
                                pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[c, j, id_t]),
                                                                    posObj=pos,
                                                                    ornObj=np.array([0.0, 0.0, 0.0, 1.0]))
                            c += 1

                else:
                    if gait[i, j] == 1:
                        # not hidden in the floor, traj
                        if not pyb.getBasePositionAndOrientation(int(self.trajectory_Ids[0, j, 0]))[0][2] == -0.1:
                            for t in range(self.n_points):
                                pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[0, j, t]),
                                                                    posObj=np.array([0., 0., -0.1]),
                                                                    ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

                        c += 1

                i += 1

            # Hide the sphere objects not used
            while c < self.trajectory_Ids.shape[0]:

                # not hidden in the floor, traj
                if not pyb.getBasePositionAndOrientation(int(self.trajectory_Ids[c, j, 0]))[0][2] == -0.1:
                    for t in range(self.n_points):
                        pyb.resetBasePositionAndOrientation(int(self.trajectory_Ids[c, j, t]),
                                                            posObj=np.array([0., 0., -0.1]),
                                                            ornObj=np.array([0.0, 0.0, 0.0, 1.0]))

                c += 1

        return 0

    def initializeEnv(self):
        '''
        Load objects in pybullet simulation.
        '''
        print("Loading PyBullet object ...")
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load 3D environment.
        tmpId = pyb.loadURDF(self.URDF)
        pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

        # Load sphere objects for trajectories.
        if self._TRAJ:
            for i in range(self.trajectory_Ids.shape[0]):

                # rgbaColor : [R , B , G , alpha opacity]
                if i == 0:
                    rgba = [0.41, 1., 0., 1.]
                else:
                    rgba = [0.41, 1., 0., 0.5]

                mesh_scale = [0.0035, 0.0035, 0.0035]
                visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                      fileName="sphere_smooth.obj",
                                                      halfExtents=[0.5, 0.5, 0.1],
                                                      rgbaColor=rgba,
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0.0, 0.0, 0.0],
                                                      meshScale=mesh_scale)
                for j in range(4):
                    for id_t in range(self.n_points):
                        self.trajectory_Ids[i, j, id_t] = pyb.createMultiBody(baseMass=0.0,
                                                                              baseInertialFramePosition=[0, 0, 0],
                                                                              baseVisualShapeIndex=visualShapeId,
                                                                              basePosition=[0.0, 0.0, -0.1],
                                                                              useMaximalCoordinates=True)
        # Load sphere objects for SL1M (5 phases + init pos).
        if self._SL1M:
            for i in range(6):
                rgba_list = [[1., 0., 0., 1.], [1., 0., 1., 1.], [1., 1., 0., 1.], [0., 0., 1., 1.]]
                mesh_scale = [0.01, 0.01, 0.01]

                for j in range(4):
                    rgba = rgba_list[j]
                    rgba[-1] = 1 - (1 / 9) * i
                    visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                          fileName="sphere_smooth.obj",
                                                          halfExtents=[0.5, 0.5, 0.1],
                                                          rgbaColor=rgba,
                                                          specularColor=[0.4, .4, 0],
                                                          visualFramePosition=[0.0, 0.0, 0.0],
                                                          meshScale=mesh_scale)

                    self.sl1m_target_Ids[i, j] = pyb.createMultiBody(baseMass=0.0,
                                                                     baseInertialFramePosition=[0, 0, 0],
                                                                     baseVisualShapeIndex=visualShapeId,
                                                                     basePosition=[0.0, 0.0, -0.1],
                                                                     useMaximalCoordinates=True)

        # Load CoM constraint objects.
        if self._COM:
            kinematic_constraints_path = os.environ["SOLO3D_ENV_DIR"] + "/com_inequalities/"
            suffix_com = "_effector_frame.obj"
            limbs_names = ["FLleg", "FRleg", "HLleg", "HRleg"]
            rgba_front = [0.2, 1., 0., 0.3]
            rgba_back = [1., 0.2, 0., 0.3]
            mesh_scale = [1., 1., 1.]

            for foot in range(4):
                foot_name = limbs_names[foot]
                filekin = kinematic_constraints_path + "COM_constraints_in_" + \
                    foot_name + suffix_com

                self.com_objects.append(as_inequalities(load_obj(filekin)))

                if foot < 2:
                    rgba = rgba_front
                else:
                    rgba = rgba_back
                visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                      fileName=filekin,
                                                      halfExtents=[0.5, 0.5, 0.1],
                                                      rgbaColor=rgba,
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0.0, 0.0, 0.0],
                                                      meshScale=mesh_scale)

                self.com_Ids[foot] = pyb.createMultiBody(baseMass=0.0,
                                                         baseInertialFramePosition=[0, 0, 0],
                                                         baseVisualShapeIndex=visualShapeId,
                                                         basePosition=[0.0, 0.0, -0.],
                                                         useMaximalCoordinates=True)

        return 0
