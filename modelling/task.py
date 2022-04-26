from dm_control import composer
from dm_control.composer import variation
from dm_control.utils import rewards
import numpy as np


class RunThroughCorridor(composer.Task):
    """A task that requires a walker to run to the end of a corridor.
    Based on: https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/tasks/corridors.py
    """
    
    def __init__(self,
                walker,
                arena,
                walker_spawn_position=(.5, 0, 0),
                walker_spawn_rotation=None,
                contact_termination=True,
                terminate_at_height=-0.5,
                physics_timestep=0.005,
                control_timestep=0.025):
        """Initializes this task.
        Args:
        walker: an instance of `locomotion.walkers.base.Walker`.
        arena: an instance of `locomotion.arenas.corridors.Corridor`.
        walker_spawn_position: a sequence of 3 numbers, or a `composer.Variation`
            instance that generates such sequences, specifying the position at
            which the walker is spawned at the beginning of an episode.
        walker_spawn_rotation: a number, or a `composer.Variation` instance that
            generates a number, specifying the yaw angle offset (in radians) that is
            applied to the walker at the beginning of an episode.
        target_velocity: a number specifying the target velocity (in meters per
            second) for the walker.
        contact_termination: whether to terminate if a non-foot geom touches the
            ground.
        terminate_at_height: a number specifying the height of end effectors below
            which the episode terminates.
        physics_timestep: a number specifying the timestep (in seconds) of the
            physics simulation.
        control_timestep: a number specifying the timestep (in seconds) at which
            the agent applies its control inputs (in seconds).
        """

        self._arena = arena
        self._walker = walker
        self._walker.create_root_joints(self._arena.attach(self._walker))
        self._walker_spawn_position = walker_spawn_position
        self._walker_spawn_rotation = walker_spawn_rotation

        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        enabled_observables.append(self._walker.observables.sensors_touch)
        # enabled_observables.append(self._walker.observables.egocentric_camera)
        for observable in enabled_observables:
            observable.enabled = True

        self._contact_termination = contact_termination
        self._terminate_at_height = terminate_at_height

        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep)

        self._lhand_body = walker.mjcf_model.find('body', 'hand_L')
        self._rhand_body = walker.mjcf_model.find('body', 'hand_R')
        self._head_body = walker.head
        self._prev_action = None

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        self._arena.regenerate(random_state)
        self._arena.mjcf_model.visual.map.znear = 0.00025
        self._arena.mjcf_model.visual.map.zfar = 4.

    def initialize_episode(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

        self._failure_termination = False
        walker_foot_geoms = set(self._walker.ground_contact_geoms)

        walker_nonfoot_geoms = [
            geom for geom in self._walker.mjcf_model.find_all('geom')
            if geom not in walker_foot_geoms]
        self._walker_nonfoot_geomids = set(
            physics.bind(walker_nonfoot_geoms).element_id)
        self._ground_geomids = set(
            physics.bind(self._arena.ground_geoms).element_id)

    def _is_disallowed_contact(self, contact):
        set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def before_step(self, physics, action, random_state):
        if isinstance(action, tuple):
            action = action[0]
        self._walker.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        self._failure_termination = False
        # if self._contact_termination:
        #     for c in physics.data.contact:
        #         if self._is_disallowed_contact(c):
        #             self._failure_termination = True
        #             break

        if self._terminate_at_height is not None:
            # if any(physics.bind(self._walker.end_effectors).xpos[:, -1] <
            #         self._terminate_at_height):
            if self._walker.observables.body_height(physics) < self._terminate_at_height:
                self._failure_termination = True
        
    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.
        else:
            return 1.

    def get_reward(self, physics):
        
        act = physics.data.act
        if self._prev_action is None:
            act_rew = np.linalg.norm(act)
        else:
            act_rew = np.linalg.norm(act - self._prev_action)
            self._prev_action = act

        up =  _upright_reward(physics, self._walker, deviation_angle=5)
        time = physics.data.time
        speed = _speed_reward(physics, self._walker)
        # return .1 * act_rew
        speed = speed if speed > .5 else (0 if speed > 0 else -1)
        return 5 * speed + up + .01 * time + 0.1 * act_rew
        # return speed
        


def _body_height_reward(physics, walker):
    return walker.observables.body_height(physics)
    
def _speed_reward(physics, walker):
    walker_xvel = physics.bind(walker.root_body).subtree_linvel[0]
    xvel_term = rewards.tolerance(
        walker_xvel, (1, 1),
        margin=1,
        sigmoid='linear',
        value_at_margin=0.0)
    return xvel_term  

def _actuators_activation(physics, walker):
    activations = walker.observables.actuator_activation(physics)
    return np.linalg.norm(np.ones_like(activations)) - np.linalg.norm(activations**2)

def _upright_reward(physics, walker, deviation_angle=0):
    """Returns a reward proportional to how upright the torso is.
    Args:
    physics: an instance of `Physics`.
    walker: the focal walker.
    deviation_angle: A float, in degrees. The reward is 0 when the torso is
        exactly upside-down and 1 when the torso's z-axis is less than
        `deviation_angle` away from the global z-axis.
    """
    deviation = np.cos(np.deg2rad(deviation_angle))
    upright_torso = physics.bind(walker.root_body).xmat[-1]
    # if hasattr(walker, 'pelvis_body'):
    upright_pelvis = physics.bind(walker.pelvis_body).xmat[-1]
    upright_zz = np.stack([upright_torso, upright_pelvis])
    # else:
    #     upright_zz = upright_torso
    upright = rewards.tolerance(upright_zz,
                                bounds=(deviation, float('inf')),
                                sigmoid='linear',
                                margin=1 + deviation,
                                value_at_margin=0)
    return np.min(upright)