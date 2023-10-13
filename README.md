# thiNK framework for ABM

This is an agent-based simulation codebase for an organization with multiple interacting agents based on an _NK_ framework by Kauffman (1989).

### Summary
The model heavily focuses on the social norms emerging from the interactions and sharing of knowledge among agents, and thus features networks. In that regard, this model combines _NK framework_ with the _Network Theory_.

### Platform
The code is written in _Python_ using an OOP paradigm that features aggregation relationship of type `has_many` between `Organization` and `Agent` classes, and a hidden `Nature` class.

### Codebase and directory structure
The big part of the codebase is the _NKPackage_ that contains utility commands for the comfortable _NK_ development. The documentation for the code is not available at the moment, so it is advised to look at the comments in the code for now.

The file `setup.py` contains the system's architecture written in an OOP paradigm. The files `main.py` and `test.py` contain the particular implementation, given the parameter set. The directory `refmaterial/` contains the utilities and useful functions.

### Performance
The `set_landscapes` method is the slowest part of the code, as it runs through every possible bitstring and maps it to a performance. At the moment, `jit` is used for its underlying `xcontrib_full` command under _NKPackage_. Also the `main.py` runs a `multiprocessing.Pool` for faster simulations. _CUDA_ is being considered at the moment, but without clear idea how to implement it.

### Credits
The code is created by Ravshan S.K. I'm on Twitter [@ravshansk](https://twitter.com/ravshansk). The research is funded by the [D!ARC](https://www.aau.at/digital-age-research-center/decide/) research center. 
