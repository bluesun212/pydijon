"""
Dijon is an imperative, stack-oriented esoteric language with highly-nontraditional control flow constructs.
The heart of Dijon is the way it handles memory using both a stack and variables. However, variables also influence
the control flow using a concept called triggers, which are sections of code that execute when the trigger's
variable's value is requested. Variables can be declared in these triggers and are scoped inside them. Programs may
import and use triggers and variables from other files as well, most importantly the standard library.
"""

import interpreter

__all__ = ['interpreter']
__author__ = "Jared Jonas (bluesun212)"
__license__ = "GPL"
__version__ = "1.0"
__website__ = "https://github.com/bluesun212/pydijon"
