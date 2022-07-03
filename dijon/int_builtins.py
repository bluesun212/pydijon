from interpreter import *
import sys


class DenominatorVariable(Variable):
    @Variable.value.setter
    def value(self, value: StackValue):
        if isinstance(value, NumericalValue):
            self.val = NumericalValue(value.den, 1)
        else:
            self.val = DEFAULT_VALUE


class FracVariable(Variable):
    @Variable.value.setter
    def value(self, value: StackValue):
        if isinstance(value, NumericalValue):
            self.val = NumericalValue(value.num % value.den, value.den)
        else:
            self.val = DEFAULT_VALUE


class TypeVariable(Variable):
    @Variable.value.setter
    def value(self, value: StackValue):
        the_type = 0

        if isinstance(value, DefaultNumericalValue):
            the_type = 1
        elif isinstance(value, NumericalValue):
            the_type = 2
        elif isinstance(value, DanglingRef):
            the_type = 3
        elif isinstance(value, LocatedRef):
            the_type = 4

        self.val = NumericalValue(the_type, 1)


class ErrorVariable(Variable):
    @property
    def value(self):
        raise DijonException("Error triggered by code")


def generate_builtins(interp: Interpreter):
    builtin_frame = interp.run_code("% builtins")  # Create an empty frame

    # Add the builtins to the frame
    builtin_frame.scope.indices['den'] = DenominatorVariable()
    builtin_frame.scope.indices['frac'] = FracVariable()

    builtin_frame.scope.indices['type'] = TypeVariable()
    builtin_frame.scope.indices['error'] = ErrorVariable()
    builtin_frame.scope.indices['stderr'] = OutVariable(sys.stderr)

    version = Variable()
    version.indices['major'] = Variable(NumericalValue(interp.version[0], 1))
    version.indices['minor'] = Variable(NumericalValue(interp.version[1], 1))
    version.indices['build'] = Variable(NumericalValue(interp.version[2], 1))
    builtin_frame.scope.indices['version'] = version

    # Add the above variables to the export list and return the frame
    for (name, var) in builtin_frame.scope.indices.items():
        builtin_frame.exports.append(LocatedRef(var, builtin_frame, name))
    return builtin_frame
