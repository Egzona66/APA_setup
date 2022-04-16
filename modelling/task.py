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
                walker_spawn_position=(0, 0, 0),
                walker_spawn_rotation=None,
                target_velocity=3.0,
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
        enabled_observables.append(self._walker.observables.egocentric_camera)
        for observable in enabled_observables:
            observable.enabled = True

        self._vel = target_velocity
        self._contact_termination = contact_termination
        self._terminate_at_height = terminate_at_height

        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep)

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
        self._walker.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        self._failure_termination = False
        if self._contact_termination:
            for c in physics.data.contact:
                if self._is_disallowed_contact(c):
                    self._failure_termination = True
                    break
        if self._terminate_at_height is not None:
            if any(physics.bind(self._walker.end_effectors).xpos[:, -1] <
                    self._terminate_at_height):
                self._failure_termination = True
        
    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.
        else:
            return 1.

    def get_reward(self, physics):
        # walker_xvel = physics.bind(self._walker.root_body).subtree_linvel[0]
        # xvel_term = rewards.tolerance(
        #     walker_xvel, (self._vel, self._vel),
        #     margin=self._vel,
        #     sigmoid='linear',
        #     value_at_margin=0.0)
        # return xvel_term

        xpos = physics.bind(self._walker.root_body).xpos[0]
        L = self._arena._corridor_length
        return 1.0 - (L-xpos)/L