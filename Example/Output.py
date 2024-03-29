# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Example

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Output(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Output()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsOutput(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Output
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Output
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Output
    def Result(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def OutputStart(builder):
    builder.StartObject(2)

def Start(builder):
    OutputStart(builder)

def OutputAddType(builder, type):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(type), 0)

def AddType(builder, type):
    OutputAddType(builder, type)

def OutputAddResult(builder, result):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(result), 0)

def AddResult(builder, result):
    OutputAddResult(builder, result)

def OutputEnd(builder):
    return builder.EndObject()

def End(builder):
    return OutputEnd(builder)
