```
ml gcc/7.3.1
ml cuda/11.1.1
module load cmake
module load spectrum-mpi/rolling-release
rm -fr CMakeFiles/ 
rm -fr CMakeCache.txt
rm -fr spack.lock
source ~/spack/share/spack/setup-env.sh
despacktivate
spack compiler find
spack env create -d . spack.yaml
spack env activate .
spack install
spacktivate .
cmake .
make clean
make
mkdir files
lalloc 9  lrun  -M -gpu -N 9 -T 4 -g 1 ping_pong >files/mixed
python python.py
```
