# this file defines the Nix package
{ lib, buildPythonPackage
, bokeh
, cython
, click
, dill
, future
, h5py
, joblib
, matplotlib
, mlflow
, moviepy
, nose
, numpy
, opencv3
, pandas
, pims
, pytest
, pytorch
, pytorch-lightning
, requests
, scipy
, seaborn
, tables
, torchvision
, tqdm
, tifffile
}:

buildPythonPackage rec {
  pname = "babelfish-models";
  version = "0.1.0";
  src = ./.;
  doCheck = false;

  propagatedBuildInputs = [
    bokeh
    cython
    click
    dill
    future
    h5py
    joblib
    matplotlib
    mlflow
    moviepy
    nose
    numpy
    opencv3
    pandas
    pims
    pytest
    pytorch
    pytorch-lightning
    requests
    scipy
    seaborn
    tables
    torchvision
    tqdm
    tifffile
 ];

   meta = with lib; {
    description = "Zebrafish analysis";
    homepage = "https://github.com/tbenst/babelfish-models";
    maintainers = [ maintainers.tbenst ];
    license = licenses.gpl3;
  };
}
