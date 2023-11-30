from __future__ import annotations

class My:
    pass

class His(My):
    pass

class Her:
    pass

your: My = My()
my: type[My] = His