""" Base class supplying basic serialization methods and
used to mark things that are intended to be serialzable.
By serializable, we mean 'able to be rendered to dict of base types'
such that this state can be easily pickled. """
from __future__ import absolute_import

import attr


class Serializable(object):
    """ Base class supplying basic serialization methods and
    used to mark things that are intended to be serialzable.
    By serializable, we mean 'able to be rendered to dict of base types'
    such that this state can be easily pickled. """
    VERSION = 1

    @classmethod
    def deserialize(cls, state):
        """ Deserialize the object from the dict of basic types.
        :param state dict: dict (serialized) representation of the object
        :return Serializable: the deserialized object
        """
        ver = state.pop('version')
        assert ver == cls.VERSION
        return cls(**state)

    def serialize(self):
        """ Serialize object to dict of basic types: int, np.ndarray, etc.
        :return dict: dict (serialized) representation of the object
        """
        state = attr.asdict(self)
        state['version'] = self.VERSION
        return state
