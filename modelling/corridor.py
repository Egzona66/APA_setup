from dm_control.locomotion.arenas.corridors import Corridor


_SIDE_WALLS_GEOM_GROUP = 3
_CORRIDOR_X_PADDING = 0.0
_WALL_THICKNESS = 0.16
_SIDE_WALL_HEIGHT = 4.0
_DEFAULT_ALPHA = 0.5


class Forceplate(Corridor):
    """
    A small empty corridor, based on dm_control EmptyCorridor environment
    https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/arenas/corridors.py
    """

    def _build(self,
                corridor_width=.4,
                corridor_length=10,
                visible_side_planes=True,
                name='empty_corridor'):
        """Builds the corridor.
        Args:
        corridor_width: A number or a `composer.variation.Variation` object that
            specifies the width of the corridor.
        corridor_length: A number or a `composer.variation.Variation` object that
            specifies the length of the corridor.
        visible_side_planes: Whether to the side planes that bound the corridor's
            perimeter should be rendered.
        name: The name of this arena.
        """
        super()._build(name=name)

        self._corridor_width = corridor_width
        self._corridor_length = corridor_length

        self._walls_body = self._mjcf_root.worldbody.add('body', name='walls')

        self._mjcf_root.visual.map.znear = 0.0005
        self._mjcf_root.asset.add(
            'texture', type='skybox', builtin='gradient',
            rgb1=[1.0, 0.85, 0.85], rgb2=[.5, .5, .5], width=100, height=600)
        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1])

        alpha = _DEFAULT_ALPHA if visible_side_planes else 0.0
        self._ground_plane = self._mjcf_root.worldbody.add(
            'geom', type='plane', rgba=[0.1, 0.1, 0.1, 1], size=[1, 1, 1])
        self._left_plane = self._mjcf_root.worldbody.add(
            'geom', type='plane', xyaxes=[1, 0, 0, 0, 0, 1], size=[1, 1, 1],
            rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
        self._right_plane = self._mjcf_root.worldbody.add(
            'geom', type='plane', xyaxes=[-1, 0, 0, 0, 0, 1], size=[1, 1, 1],
            rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
        self._near_plane = self._mjcf_root.worldbody.add(
            'geom', type='plane', xyaxes=[0, 1, 0, 0, 0, 1], size=[1, 1, 1],
            rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
        self._far_plane = self._mjcf_root.worldbody.add(
            'geom', type='plane', xyaxes=[0, -1, 0, 0, 0, 1], size=[1, 1, 1],
            rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)

        self._current_corridor_length = None
        self._current_corridor_width = None

    def regenerate(self, random_state):
        """Regenerates this corridor.
        New values are drawn from the `corridor_width` and `corridor_height`
        distributions specified in `_build`. The corridor is resized accordingly.
        Args:
        random_state: A `numpy.random.RandomState` object that is passed to the
            `Variation` objects.
        """
        self._walls_body.geom.clear()
        corridor_width = self._corridor_width
        corridor_length = self._corridor_length
        self._current_corridor_length = corridor_length
        self._current_corridor_width = corridor_width

        self._ground_plane.pos = [corridor_length / 2, 0, 0]
        self._ground_plane.size = [
            corridor_length / 2 + _CORRIDOR_X_PADDING, corridor_width / 2, 1]

        self._left_plane.pos = [
            corridor_length / 2, corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
        self._left_plane.size = [
            corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

        self._right_plane.pos = [
            corridor_length / 2, -corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
        self._right_plane.size = [
            corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

        self._near_plane.pos = [
            -_CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
        self._near_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

        self._far_plane.pos = [
            corridor_length + _CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
        self._far_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

    @property
    def corridor_length(self):
        return self._current_corridor_length

    @property
    def corridor_width(self):
        return self._current_corridor_width

    @property
    def ground_geoms(self):
        return (self._ground_plane,)