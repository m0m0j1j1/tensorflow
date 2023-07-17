import os
from tensorflow.python.eager import context
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework.ops import get_default_graph

# For simplicity consideration, user should always turn on/off multi_stream through env var?
# @tf_export("cuda.enable_multi_stream")
# def enable_multi_stream():
#     multi_stream_is_enabled = True
# @tf_export("cuda.disable_multi_stream")
# def disable_multi_stream():
#     multi_stream_is_enabled = False

multi_stream_is_enabled = bool(os.environ.get("TF_NODE_LEVEL_MULTISTREAM", False))

@tf_export("cuda.multi_stream_is_enabled")
def multi_stream_is_enabled():
    return multi_stream_is_enabled


@tf_export("cuda.stream_scope")
def stream_scope(stream=None, include_grad=True):
    _check_if_in_eager()
    g = get_default_graph()
    return g.stream_manager.stream_scope(stream=stream, include_grad=include_grad)


@tf_export("cuda.get_stream")
def get_stream(stream_id=None, name=None):
    if all([stream_id, name]):
        raise RuntimeError("Please choose only one arg to specify")
    g = get_default_graph()
    stream = None
    if stream_id:
        stream = g.stream_manager.id_to_stream.get(stream_id, None)
    if name:
        stream = g.stream_manager.name_to_stream.get(name, None)
    if stream is None:
        # create a new one
        stream = Stream(stream_id=stream_id, name=name)
    return stream


def _check_if_in_eager():
    if context.executing_eagerly():
        raise RuntimeError(
            "multi-stream function does not support eager execution yet")


@tf_export("cuda.Stream")
class Stream(object):
    def __init__(self, stream_id=None, name=None):
        self.stream_id = stream_id
        self.name = name
        if stream_id is not None:
            if type(stream_id) is not int:
                raise ValueError(
                    "stream_id {} invalid. Must be int".format(stream_id))
            if stream_id < 0:
                raise ValueError(
                    "stream_id {} invalid. Must >= 0".format(stream_id))
            self.explicit_assigned_stream_id = True
        else:
            self.explicit_assigned_stream_id = False

        if name is not None:
            if type(name) is not str:
                raise ValueError("name {} invalid. Must be str".format(name))
            if name.startswith(GraphStreamManager.auto_name_pattern):
                raise ValueError(
                    "name {} invalid. {}* is reserved for auto generated stream names".format(
                        name, GraphStreamManager.auto_name_pattern))

        _check_if_in_eager()
        g = get_default_graph()
        g.stream_manager.register_stream(self)


class GraphStreamManager(object):
    auto_name_pattern = "_auto_assigned_name_"

    def __init__(self):
        self.id_to_stream = {}
        self.name_to_stream = {}
        self._stream_stack = [None, None]

    def _get_a_valid_id(self):
        occupied_ids = sorted(list(self.id_to_stream.keys()))
        valid_id = 1  # according to cpp design, auto assigned stream id should > 0
        while valid_id in occupied_ids:
            valid_id += 1
        return valid_id

    def _auto_assigned_name(self, stream_id):
        return self.__class__.auto_name_pattern + str(stream_id)

    def _move_registeration(self, stream):
        if stream.explicit_assigned_stream_id or \
                not stream.name.startswith(self.__class__.auto_name_pattern):
            raise RuntimeError(
                "should never be called on a explicit assigned stream")
        old_id = stream.stream_id
        old_name = stream.name
        new_id = self._get_a_valid_id()
        new_name = self._auto_assigned_name(new_id)
        del self.id_to_stream[old_id]
        del self.name_to_stream[old_name]
        stream.stream_id = new_id
        stream.name = new_name
        self.id_to_stream[new_id] = stream
        self.name_to_stream[new_name] = stream

    def register_stream(self, stream):
        new_id = None
        if stream.explicit_assigned_stream_id:
            occupied = self.id_to_stream.get(stream.stream_id, None)
            if occupied is None:
                new_id = stream.stream_id
            else:
                if occupied.explicit_assigned_stream_id:
                    raise RuntimeError(
                        "There is an existing stream whose id is {}. Conflict.".format(stream.stream_id))
                self._move_registeration(occupied)
                new_id = stream.stream_id
        else:
            new_id = self._get_a_valid_id()

        new_name = stream.name
        if new_name is None:
            new_name = self._auto_assigned_name(new_id)

        if new_name in self.name_to_stream:
            raise RuntimeError(
                "There is an existing stream whose name is {}. Conflict".format(new_name))

        stream.stream_id = new_id
        stream.name = new_name
        self.id_to_stream[new_id] = stream
        self.name_to_stream[new_name] = stream

    @tf_contextlib.contextmanager
    def stream_scope(self, stream=None, include_grad=True):
        if stream is None:
            stream = Stream()

        old_stack = self._stream_stack
        self._stream_stack = [stream, include_grad]
        try:
            yield stream
        finally:
            self._stream_stack = old_stack

    def get_cur_stream_id(self):
        if self._stream_stack[0]:
            return self._stream_stack[0].stream_id
        return None

    def get_cur_include_grad(self):
        return self._stream_stack[1]
