from __future__ import annotations
import sys

from typing import List, Dict, Tuple, Optional
from os import getcwd
from enum import Enum
from pathlib import Path


# These are the valid identifier characters
def is_var_char(ch: str) -> bool:
    return ('a' <= ch <= 'z') or ('0' <= ch <= '9') or ch == '.' or ch == '_'


# Find the GCD of a and b using Euler's algorithm
def _gcd(a, b):
    while b != 0:
        t = b
        b = a % b
        a = t
    return a


class DijonException(Exception):
    """Base exception for all errors that could occur in Dijon."""
    pass


class DijonImportException(DijonException):
    """Class for exceptions that occur during an import."""

    def __init__(self, message, location, line_no, char_no):
        new_message = f"{message} [at {location}, line {line_no+1}, chr {char_no+1}]"
        super().__init__(new_message)


# Source code symbols
class Symbol:
    """A representation of one code symbol, i.e. a name or operation."""

    def __init__(self, line: int, pos: int):
        self.line = line
        self.pos = pos


class Name(Symbol):
    """A symbol that represents a variable name."""

    def __init__(self, line: int, pos: int, name: str):
        """
        Creates a new Name object
        :param line: the line number
        :param pos: the character index in the line
        :param name: the string representation of the name symbol
        """

        super().__init__(line, pos)
        self.name = name

    def __str__(self):
        return self.name


class Operation(Symbol):
    """A symbol that represents one operation."""

    def __init__(self, line: int, pos: int, text: str):
        """
        Creates a new Operation object
        :param line: the line number
        :param pos: the character index in the line
        :param text: A textual representation of this symbol
        """

        super().__init__(line, pos)
        self.text = text

    def run_initial(self, frame: Frame):
        """Executes the operation before any branches occur."""
        pass

    def run_final(self, frame: Frame):
        """Executes the operation after any branches occur."""
        pass

    def __str__(self):
        return self.text


class Subtract(Operation):
    """Represents subtraction between two numbers."""

    def __init__(self, line: int, pos: int):
        super().__init__(line, pos, '-')

    def run_initial(self, frame: Frame):
        # Dereference the top two stack values if applicable
        if len(frame.stack) >= 2:
            frame.queue_deref(1)
            frame.queue_deref(2)

    def run_final(self, frame: Frame):
        # If len(stack) < 2, push 0.  Otherwise, pop two values, ensure they are numbers, and push their difference

        if len(frame.stack) < 2:
            frame.stack.append(ZERO)
        else:
            b = frame.stack.pop()
            a = frame.stack.pop()
            if isinstance(a, NumericalValue) and isinstance(b, NumericalValue):
                frame.stack.append(a.subtract(b))
            else:
                raise DijonException(f"Invalid arguments for - operator: {a}, {b}")


class Divide(Operation):
    """Represents division between two numbers"""

    def __init__(self, line: int, pos: int):
        super().__init__(line, pos, '/')

    def run_initial(self, frame: Frame):
        # Dereference the top two stack values if applicable
        if len(frame.stack) >= 2:
            frame.queue_deref(1)
            frame.queue_deref(2)

    def run_final(self, frame: Frame):
        # If len(stack) < 2, push 1.  Otherwise, pop two values, ensure they are numbers, and push their quotient

        if len(frame.stack) < 2:
            frame.stack.append(ONE)
        else:
            b = frame.stack.pop()
            a = frame.stack.pop()
            if isinstance(a, NumericalValue) and isinstance(b, NumericalValue):
                frame.stack.append(a.divide(b))
            else:
                raise DijonException("Invalid arguments for / operator")


class Dereference(Operation):
    """Represents a dereference operation, which resolves the value of the reference on the top of the stack."""

    def __init__(self, line: int, pos: int):
        super().__init__(line, pos, '$')

    def run_initial(self, frame: Frame):
        # Dereference the top value on the stack
        if len(frame.stack) < 1:
            raise DijonException("Not enough arguments for $")
        else:
            frame.queue_deref(1)


class Reference(Operation):
    """Represents a reference operation, which pushes a reference to the value on the top of the stack."""

    def __init__(self, line: int, pos: int):
        super().__init__(line, pos, '&')

    def run_initial(self, frame: Frame):
        if len(frame.stack) < 1:
            raise DijonException("Not enough arguments for &")
        else:
            a = frame.stack.pop()

            # If the value is an unresolved reference, then resolve it and push the resulting resolved reference
            # The variable must actually exist in this instance, otherwise nothing happens
            if isinstance(a, DanglingRef):
                var = frame.get_ref_var(a)
                if var:
                    frame.stack.append(LocatedRef(var, frame, a.name))
                    return

            # Otherwise, create a reference by wrapping the value in an anonymous variable and pushing the result
            var = Variable()
            var.value = a
            frame.stack.append(LocatedRef(var))


class Concat(Operation):
    """Represents a concatenation operation.  References can be concatenated to refer to a child variable."""

    def __init__(self, line: int, pos: int):
        super().__init__(line, pos, '#')

    def run_initial(self, frame: Frame):
        if len(frame.stack) < 2:
            raise DijonException("Not enough arguments for #")
        else:
            b = frame.stack.pop()
            a = frame.stack.pop()
            b_val = ""

            # Ensure 'a' is a reference already
            if not isinstance(a, ReferenceValue):
                raise DijonException("Argument 1 must be a reference")

            # Find the suffix that will be appended depending on the type of b
            if isinstance(b, LocatedRef):
                raise DijonException("Argument 2 must not be a resolved reference")
            elif isinstance(b, DanglingRef):
                b_val = b.name
            elif isinstance(b, NumericalValue):
                if b.is_integer() and b.num >= 0:
                    b_val = str(b.num)
                else:
                    return  # Silently ignore negative numbers and fractions

            # Add a new reference to the stack representing the concatenation
            if isinstance(a, DanglingRef):
                frame.stack.append(DanglingRef(a.name + '.' + b_val))
            elif isinstance(a, LocatedRef):
                frame.stack.append(a.extend(b_val))


class Shortcut(Operation):
    """Represents the shortcut operation, which occurs at the end of every line."""

    def __init__(self, line: int, pos: int):
        super().__init__(line, pos, '~')

    def run_initial(self, frame: Frame):
        # Dereference the value at the top of the stack if applicable
        if len(frame.stack) >= 1:
            frame.queue_deref(1)

    def run_final(self, frame: Frame):
        # Set the variable
        if len(frame.stack) >= 2:
            # Resolve the reference of the set variable
            a = frame.stack[-2]
            if isinstance(a, ReferenceValue):
                var = frame.get_ref_var(a, True)
                if var:
                    var.value = frame.stack[-1]

        frame.stack.clear()


# A list of above operations
OPERATIONS = {'-': Subtract, '/': Divide, '#': Concat, '~': Shortcut, '$': Dereference, '&': Reference}


# Shorthand to invoke the constructor of the operation
def get_op(line: int, pos: int, ch: str) -> Operation:
    return OPERATIONS[ch](line, pos)


# Source code reading and conversion
class Trigger:
    """Represents a trigger in the source code, which holds the code when a certain variable is changed."""

    def __init__(self, name: str, source: SourceFile):
        """
        Creates a new Trigger object
        :param name: the identifier of this trigger
        :param source: the SourceFile this trigger is associated with
        """

        self.name = name
        self.source = source

        self.children: List[Trigger] = []
        self.code: List[Symbol] = []
        self.parent: Optional[Trigger] = None

    def add_child(self, child: Trigger):
        """Adds a nested Trigger."""

        self.children.append(child)
        child.parent = self

    def add_symbol(self, sym: Symbol):
        """Appends a code symbol to the list of code symbols associated with this Trigger."""

        # Don't append multiple shortcut symbols
        if not isinstance(sym, Shortcut) or (self.code and not isinstance(self.code[-1], Shortcut)):
            self.code.append(sym)


class SourceFile:
    """A container class that contains information about a given Dijon file."""

    def __init__(self, name: str, code: str, path: Optional[Path] = None):
        """
        Creates a new SourceFile object
        :param name: The file name
        :param path: a Path object representing this file if applicable
        :param code: a string containing the source code
        """

        # Initialize fields
        self.name = name
        self.path = path
        self.root_trigger = Trigger("<main>", self)
        self.imports = []
        self.global_imports = []
        path_name = self.name if not self.path else str(self.path)

        # State variables for parsing
        code = code + '\n'
        curr_trigger = self.root_trigger
        line_no = 0
        char_no = -1

        buffer = ""
        global_flag = False
        state = 0  # 0 -> not in name, 1 -> in name, 2 -> in trigger, 3 -> in import, 4 -> in comment

        # Read through each character
        for c in code:
            char_no += 1

            # Handle names - ignore if in a comment
            if state != 4:
                if is_var_char(c):  # Is part of a name
                    if not state:  # A variable/name is being read
                        state = 1
                    buffer += c
                elif state:  # The end of the name has been reached
                    # Global import has 2 '@' signs, so handle the 2nd if present
                    if state == 3 and c == '@':
                        global_flag = True
                        continue

                    if not buffer:
                        raise DijonImportException("Empty name for trigger/import", path_name, line_no, char_no)

                    if state == 3:  # Add an import to the list
                        self.imports.append(buffer)
                        if global_flag:
                            self.global_imports.append(buffer)
                    elif state == 2:  # Create new trigger
                        tr = Trigger(buffer, self)
                        curr_trigger.add_child(tr)
                        curr_trigger = tr
                    elif state == 1:  # Add name to symbol list
                        curr_trigger.add_symbol(Name(line_no, char_no, buffer))

                    # Clean up
                    state = 0
                    buffer = ""
                    global_flag = False

            # Handle end of line
            if c == '\n':
                curr_trigger.add_symbol(get_op(line_no, char_no, '~'))
                line_no += 1
                char_no = -1
                state = 0
            elif state == 4:  # If in a comment, ignore any of the following logic
                continue

            # Handle operations, comments, imports, and triggers, adding shortcut characters when necessary
            if c in OPERATIONS:  # Read in operation
                curr_trigger.add_symbol(get_op(line_no, char_no, c))
            elif c == '%':  # Comment
                state = 4
            elif c == '@':  # Import, but only at beginning of file before any code
                if curr_trigger == self.root_trigger and not curr_trigger.code:
                    state = 3
                else:
                    raise DijonImportException("Import not at beginning of file", path_name, line_no, char_no)
            elif c == ':':  # A trigger started
                curr_trigger.add_symbol(get_op(line_no, char_no, '~'))
                state = 2
            elif c == ';':  # A trigger ended
                curr_trigger.add_symbol(get_op(line_no, char_no, '~'))
                if curr_trigger.parent is None:
                    raise DijonImportException("Non-matching number of :s and ;s", path_name, line_no, char_no)
                curr_trigger = curr_trigger.parent

        if curr_trigger != self.root_trigger:
            raise DijonImportException("Non-matching number of :s and ;s", path_name, line_no, char_no)


# Stack values
class StackValue:
    """A container representing data on the stack."""
    pass


class ReferenceValue(StackValue):
    """Reference to a variable by name."""

    def __init__(self, name: Optional[str]):
        """
        Creates a new ReferenceValue object
        :param name: the string path to the object if any
        """

        self.name = name

    def get_name(self) -> str:
        return '' if not self.name else self.name


class DanglingRef(ReferenceValue):
    """Represents an unresolved reference."""

    pass


class LocatedRef(ReferenceValue):
    """Represents a reference that has been resolved."""

    def __init__(self, ref: Variable, callee_frame: Optional[Frame] = None, name: Optional[str] = None):
        """
        Creates a new LocatedRef object
        :param ref: the located variable this reference points to
        :param callee_frame: the frame that located this reference
        :param name: the name of the object, if any
        """

        super().__init__(name)
        if bool(callee_frame) != bool(name):
            raise DijonException("callee_frame and name must both be either set or unset")

        self.ref = ref
        self.callee_frame = callee_frame
        self.extension = None

    def extend(self, suffix):
        """
        Gets a reference starting at this variable and pointing to a child variable with string suffix
        :param suffix: the variable string path representing the relative path to the child object
        :return: the new reference pointing to the child variable
        """

        new_ref = LocatedRef(self.ref, self.callee_frame, self.name)
        new_ref.extension = suffix
        return new_ref

    def get_name(self):
        return '.'.join(filter(None, (self.name, self.extension)))


class NumericalValue(StackValue):
    """Represents a fraction value on the stack.  Note that this class is immutable."""

    def __init__(self, num: int, den: int):
        """
        Creates a new NumericalValue object
        :param num: the numerator
        :param den: the denominator
        """

        self.num = num
        self.den = den
        if self.den > 1:
            self._simplify()

    def _simplify(self):
        mult = -1 if self.num < 0 else 1  # _gcd can't handle negative numbers
        gcd = _gcd(mult*self.num, self.den)
        self.num //= mult*gcd
        self.den //= gcd

    def is_integer(self) -> bool:
        """
        :return: whether the fraction is an integer
        """

        return self.den == 1

    def negate(self) -> NumericalValue:
        """
        :return: a new NumericalValue representing the additive inverse of this object.
        """

        return NumericalValue(-1*self.num, self.den)

    def invert(self) -> NumericalValue:
        """
        :return: a new NumericalValue representing the multiplicative inverse of this object.
        """

        mult = -1 if self.num < 0 else 1  # Ensure the denominator of the result is positive
        if self.num == 0:
            raise DijonException("Division by zero")

        return NumericalValue(mult*self.den, mult*self.num)

    def add(self, other: NumericalValue) -> NumericalValue:
        """
        Adds
        :param other: the object to be added with
        :return: a new NumericalValue representing the sum of this object and the provided object.
        """

        # a/b + c/d = (ad+bc)/(bd)
        return NumericalValue(self.num*other.den+other.num*self.den, self.den*other.den)

    def subtract(self, other: NumericalValue) -> NumericalValue:
        """
        Subtracts
        :param other: the object to be subtracted
        :return: a new NumericalValue representing the difference of this object and the provided object.
        """

        return self.add(other.negate())

    def multiply(self, other: NumericalValue) -> NumericalValue:
        """
        Multiplies
        :param other: the object to be multiplied with
        :return: a new NumericalValue representing the product of this object and the provided object.
        """

        return NumericalValue(self.num*other.num, self.den*other.den)

    def divide(self, other: NumericalValue) -> NumericalValue:
        """
        Divides
        :param other: the object to be divided
        :return: a new NumericalValue representing the quotient of this object and the provided object.
        """

        return self.multiply(other.invert())

    def to_float(self):
        """
        :return: the float representation of this number
        """
        return self.num/self.den


# Numerical constants
ZERO = NumericalValue(0, 1)
ONE = NumericalValue(1, 1)
DEFAULT_VALUE = ZERO


# Variable structure
class Variable:
    """Representation of a variable in memory.  Contains a value and a list of children variables."""

    def __init__(self):
        self._value: StackValue = DEFAULT_VALUE
        self.indices: Dict[str, Variable] = dict()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: StackValue):
        self._value = value


class Scope(Variable):
    """
    A subclass of Variable that also stores the triggers and its parent scope.
    """

    def __init__(self, parent, trigger):
        """
        Creates a new Scope object
        :param parent: the parent scope that contains this scope
        :param trigger: the trigger that this scope is associated with
        """

        super().__init__()
        self.parent: Optional[Scope] = parent
        self.trigger: Trigger = trigger


# Special variables
class OutVariable(Variable):
    @Variable.value.setter
    def value(self, value: StackValue):
        # Find a stack representation of the value based on its type
        output = None
        if isinstance(value, ReferenceValue):
            output = '?' if not value.name else value.name
        elif isinstance(value, NumericalValue):
            # Convert whole numbers to ASCII, otherwise print their fractional form
            if value.is_integer() and 0 <= value.num <= 127:
                output = chr(value.num)
            else:
                output = f"{value.num}/{value.den}"

        if output:
            print(output, end='', flush=True)


class NumberOutVariable(Variable):
    @Variable.value.setter
    def value(self, value: StackValue):
        if isinstance(value, NumericalValue):
            if value.is_integer():
                print(value.num, end='', flush=True)
            else:
                print(value.to_float())


class ExportVariable(Variable):
    def __init__(self, frame: Frame):
        super().__init__()
        self.frame = frame

    @Variable.value.setter
    def value(self, value: StackValue):
        if not isinstance(value, LocatedRef):
            raise DijonException("export must be passed a resolved reference")
        if value not in self.frame.exports:
            self.frame.exports.append(value)


# Execution
class FrameState(Enum):
    """An enumeration that represents the states of execution a Frame can be in."""

    INITIAL = 1     # The frame hasn't been run yet.
    RUNNING = 2     # The frame is currently running code.
    CONTINUE = 3    # The frame needs to check for dereferences.
    BREAK = 4       # The frame needs to break to run a trigger.
    FINISHED = 5    # Execution has completed on the frame.


# Represents the state of reference objects
STATE_TRIGGER = 0
STATE_DEREF = 1


class Frame:
    """An execution frame on the stack which runs all code in a trigger."""

    def __init__(self, parent: Scope or None, trigger: Trigger):
        """
        Creates a new Frame object
        :param parent: the parent Scope, if any
        :param trigger: the trigger this frame is associated with
        """

        self.scope = Scope(parent, trigger)
        self.stack: List[StackValue] = []
        self.pos = 0

        # Branching related fields
        self.state = FrameState.INITIAL
        self.stack_ref_states: List[Tuple] = []
        self.branch: Optional[Frame] = None
        self.last_op = None

        # Import/export fields - Only set for the root frame of a file
        self.exports: List[LocatedRef] = []
        self.imported_triggers: Dict[str, Tuple[Trigger, Scope]] = dict()

        # Ensure trigger variables exist
        for child in trigger.children:
            self.get_ref_var(DanglingRef(child.name), True)

    def execute(self):
        """
        The main execution function in Dijon.  This will read all symbols in the Trigger and will break when
        a trigger is called, continuing execution when the trigger is finished.
        """

        if self.state == FrameState.FINISHED:
            raise DijonException("Finished frame being ran again")

        code = self.scope.trigger.code

        # Handle trigger calls
        if self.state == FrameState.CONTINUE or self.state == FrameState.BREAK:
            self._deref_pending()

            # Handle one trigger call
            if len(self.stack_ref_states) > 0:
                tup = self.stack_ref_states.pop()
                self.branch = Frame(tup[2][1], tup[2][0])
                self.stack_ref_states.append((tup[0], STATE_DEREF))
                self.state = FrameState.BREAK
                # TODO: Implement tail call optimization
                # 1. This is the last symbol  2. The symbol is either $ or ~  3. There is only one value on the stack
                return
            else:  # Finished with triggers and dereferences
                self.last_op.run_final(self)
                self.last_op = None
                self.branch = None
                self.pos += 1

        # Execute code symbols
        self.state = FrameState.RUNNING
        while self.pos < len(code) and self.state == FrameState.RUNNING:
            sym = code[self.pos]
            if isinstance(sym, Name):  # Push names onto stack
                self.stack.append(DanglingRef(sym.name))
                self.pos += 1
            elif isinstance(sym, Operation):  # Handle operations
                # Run the initial check, then dereference any requested stack values
                sym.run_initial(self)
                self.last_op = sym
                self.state = FrameState.CONTINUE

        # If finished normally
        if self.state == FrameState.RUNNING:
            self.state = FrameState.FINISHED

    def queue_deref(self, index):
        """
        Called from Operation subclasses to tell the interpreter to dereference/run triggers for the object at
        the -index stack position
        :param index: the stack position pointing at the object to be de-referenced
        """

        sv = self.stack[-index]

        if isinstance(sv, ReferenceValue):
            # Find the trigger depending on the reference type
            trigger = None
            if isinstance(sv, LocatedRef) and sv.callee_frame:
                trigger = sv.callee_frame.find_trigger(sv.name)
            elif isinstance(sv, DanglingRef):
                trigger = self.find_trigger(sv.name)

            if trigger:  # A trigger exists, so we need to break to run it
                self.stack_ref_states.append((-index, STATE_TRIGGER, trigger))
            else:
                self.stack_ref_states.append((-index, STATE_DEREF))

    def get_ref_var(self, ref: ReferenceValue, create=False) -> Optional[Variable]:
        """
        Gets the variable that the given reference is referencing.  If ref is an unresolved reference, then it
        will look for a variable matching that name starting at this scope, then looking through parent scopes.  If
        the reference is resolved, then it will start at the resolved variable and find its child pointed by the
        reference's extension if present
        :param ref: the reference possibly pointing to a variable
        :param create: whether to create the variable if it doesn't exist
        :return: the Variable referenced, if it exists, None otherwise
        """

        # Locate base variable and variable tree to get to desired variable
        var = None
        name = None

        if isinstance(ref, LocatedRef):
            var = ref.ref
            name = ref.extension
        elif isinstance(ref, DanglingRef):
            var = self._find_variable_base(ref.name)
            name = ref.name

            # The base variable does not exist, create it here
            if not var and create:
                var = self.scope

        if not var:
            return None

        # Successively get or create each part
        if name:
            parts = name.split('.')
            for i in range(len(parts)):
                if parts[i] not in var.indices:
                    if create:
                        var.indices[parts[i]] = Variable()
                    else:
                        return None
                var = var.indices[parts[i]]

        return var

    def find_trigger(self, name: str) -> Optional[Tuple[Trigger, Scope]]:
        """
        Finds the trigger matching the given name
        :param name: the name of the trigger
        :return: A tuple containing the Trigger object and its associated Scope, if it exists, None otherwise
        """

        scope = self.scope

        # Iterate through triggers, starting with this scope's children
        # Look for triggers matching this name, except not the currently executing trigger
        while scope is not None:
            for child in scope.trigger.children:
                if child.name == name and child != self.scope.trigger:
                    return child, scope

            scope = scope.parent

        # Look through the imported triggers
        if name in self.imported_triggers:
            return self.imported_triggers[name]

        return None

    # Private methods
    def _deref_pending(self):
        srs = self.stack_ref_states
        if not srs:
            return

        # Find all deref states and dereference them
        states = filter(lambda t: t[1] == STATE_DEREF, srs)
        for s in states:
            stack_val = self.stack[s[0]]
            if isinstance(stack_val, ReferenceValue):
                var = self.get_ref_var(stack_val)
                self.stack[s[0]] = DEFAULT_VALUE if not var else var.value

        # Remove deref states from state list
        self.stack_ref_states = list(filter(lambda t: t[1] != STATE_DEREF, srs))

    def _find_variable_base(self, name: str) -> Optional[Variable]:
        # Find the variable that contains the entire name variable path
        parts = name.split('.')
        s = self.scope

        # Check each scope to see if it contains the base variable
        while s is not None and parts[0] not in s.indices:
            s = s.parent

        return s


class Interpreter:
    """The main Dijon interpreter.  Code is imported and ran here."""

    def __init__(self):
        self.frames: List[Frame] = []
        self.imported_frames: Dict[Path, Frame] = dict()

        # Import path resolution
        self.path_list = [Path(getcwd())]
        file_path = Path(__file__)
        if file_path.is_file():
            self.path_list.append(file_path.parent)

    def run_file(self, path: str):
        """
        Run a Dijon file
        :param path: a string path to the dijon file to run
        :return: the Frame object that ran the code
        """

        p = Path(path)
        self.path_list.append(p.parent)
        return self._run(p)

    def run_code(self, code: str):
        """
        Runs valid dijon code from a string
        :param code: Dijon code
        :return: the Frame object that ran the code
        """

        return self._run(code=code)

    def _resolve_import(self, name: str, calling_path: Path) -> Frame:
        # Create list of base paths to start with
        possible_paths = list(self.path_list)
        possible_paths.append(calling_path)
        resolved_path = None

        for base_path in possible_paths:
            # Go up directories if two dots are in front of path
            curr_name = name
            while curr_name.startswith('..'):
                base_path = base_path.parent
                curr_name = curr_name[2:]
            if curr_name.startswith('.'):
                curr_name = curr_name[1:]

            # Add .dij to the end and split the path at the periods
            name_parts = curr_name.split('.')
            name_parts[-1] += '.dij'

            for part in name_parts:
                base_path = base_path.joinpath(part)

            if base_path.is_file():
                resolved_path = base_path
                break

        if not resolved_path:
            raise DijonException(f"The import '{name}' can't be found")

        # If the file was already imported, just return it
        if resolved_path in self.imported_frames:
            if self.imported_frames[resolved_path].state != FrameState.FINISHED:
                raise DijonException("Circular import detected")
            else:
                return self.imported_frames[resolved_path]

        return self._run(resolved_path)

    def _run(self, path: Optional[Path] = None, code: Optional[str] = None) -> Frame:
        # Set up source container
        if path:
            path = path.resolve()
            name = path.stem

            # Read all the data from the given file
            with path.open('rt') as file:
                try:
                    code = file.read() + '\n'
                except IOError as e:
                    raise Exception("There was an error opening the file for " + str(path), e)
        elif code:
            name = "<code>"
        else:
            raise ValueError("Either path or code must be set and valid")

        sf = SourceFile(name, code, path)

        # Set up the root frame and add special reserved variables
        frame = Frame(None, sf.root_trigger)
        frame.scope.indices["out"] = OutVariable()
        frame.scope.indices["nout"] = NumberOutVariable()
        frame.scope.indices["export"] = ExportVariable(frame)

        self.frames.append(frame)
        self.imported_frames[path] = frame

        # Import all prerequisite files and their exported variables/trigger
        for imp in sf.imports:
            imp_frame = self._resolve_import(imp, path)
            global_flag = imp in sf.global_imports
            imp_var = Variable()

            for exp in imp_frame.exports:
                # Add imported variable and a global alias if applicable
                imp_var.indices[exp.name] = exp.ref
                if global_flag:
                    frame.scope.indices[exp.name] = exp.ref

                # Add a link to the trigger from the imported file
                if exp.callee_frame:
                    trigger = exp.callee_frame.find_trigger(exp.name)
                    if trigger:
                        # Add trigger to the imported list, and an alias if the import is global
                        frame.imported_triggers[imp + '.' + exp.name] = trigger
                        if global_flag:
                            frame.imported_triggers[exp.name] = trigger

            # Add base imported variable to the outermost scope
            frame.scope.indices[imp] = imp_var

        # Execute the frame
        self._execute_until_return()
        return frame

    def _execute_until_return(self):
        # Run code until the starting top frame is popped
        start_len = len(self.frames)
        while len(self.frames) >= start_len:
            f = self.frames[-1]
            self._execute_safe()

            # Handle breaking or finished state
            if f.state == FrameState.BREAK:
                self.frames.append(f.branch)
            elif f.state == FrameState.FINISHED:
                self.frames.pop()

    def _execute_safe(self):
        # Catch any exceptions so a proper stack trace can be shown to the user
        try:
            self.frames[-1].execute()
        except DijonException as e:
            # Print stack trace and exit
            sys.stderr.write(f"Error: {e}\nStack trace:\n")
            for f in self.frames[::-1]:
                trigger = f.scope.trigger
                symbol = trigger.code[f.pos]
                sys.stderr.write(f"\t{trigger.source.name}.dij@{trigger.name} {symbol.line+1}:{symbol.pos+1}\n")
            sys.exit(1)
