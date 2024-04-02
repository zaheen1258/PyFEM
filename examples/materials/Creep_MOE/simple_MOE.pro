############################################################################
#  This Python file is part of PyFEM, the code that accompanies the book:  #
#                                                                          #
#    'Non-Linear Finite Element Analysis of Solids and Structures'         #
#    R. de Borst, M.A. Crisfield, J.J.C. Remmers and C.V. Verhoosel        #
#    John Wiley and Sons, 2012, ISBN 978-0470666449                        #
#                                                                          #
#  The code is written by J.J.C. Remmers, C.V. Verhoosel and R. de Borst.  #
#                                                                          #
#  The latest stable version can be downloaded from the web-site:          #
#     http://www.wiley.com/go/deborst                                      #
#                                                                          #
#  A github repository, with the most up to date version of the code,      #
#  can be found here:                                                      #
#     https://github.com/jjcremmers/PyFEM                                  #
#                                                                          #
#  The code is open source and intended for educational and scientific     #
#  purposes only. If you use PyFEM in your research, the developers would  #
#  be grateful if you could cite the book.                                 #
#                                                                          #
#  Disclaimer:                                                             #
#  The authors reserve all rights but do not guarantee that the code is    #
#  free from errors. Furthermore, the authors shall not be liable in any   #
#  event caused by the use of the program.                                 #
############################################################################

input = "simple_MOE.dat";

Continuum =
{
  type = "SmallStrainContinuum";

  material =
  {
    type   = "PowerLawCreepMOE";
    E      = 160e3;
    nu     = 0.33;
    rhoc = 4669.9511535360325;
    rhow = 4406641771830.477;
    Temp = 1050.0690046472204;
    flux = 9.998075715064236e-08;
  };
};

solver =
{
  type = "NonlinearSolver";
  maxCycle = 8;
  iterMax = 1
  dtime    = 1;
  time = 1;
  tol = 1e-12;
};

outputModules = ["vtk", "GraphWriter"];

vtk =
{
  type = "MeshWriter";
};

GraphWriter =
{
  onScreen = true;

  columns = [ "disp" , "load" ];

  disp =
  {
    type = "state";
    node = 5;
    dof  = 'w';
  };
 
  load =
  {
    type = "fint";
    node = load_nodes1;
    dof  = 'w';
  };
};


