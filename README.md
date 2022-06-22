# PyDijon version 1.0
PyDijon is the reference interpreter for [Dijon](https://github.com/bluesun212/dijon), an imperative, stack-oriented esoteric language with highly non-traditional control flow constructs.  It is written in Python 3.8 and doesn't require any extra packages.  For more information about the language itself, please follow the above link. 

# Quickstart
### Prerequisites
Please install Git and Python 3.8+.  

### Installation
The following shell code will set up the project.  It will download this project from GitHub, download the language reference project into the `lang` folder, then copy `std.dij` into the `dijon` folder.
```commandline
git clone https://github.com/bluesun212/pydijon
mkdir lang
cd lang
git clone https://github.com/bluesun212/dijon
cd ..
cp lang/std.dij dijon/std.dij
```

### Usage
The following command will run the file `hello_world.dij` in the folder `lang/examples/`:
```commandline
python3 dijon -f lang/examples/hello_world.dij
```