# change &track use_cuda=1 to run with gpu
mpirun -np 4 ../build/genesis4 fel.in -o fel_gpu > fel_gpu.log 2>&1

# it's better to run use_cuda=1 without mpi
../build/genesis4 fel.in -o fel_gpu > fel_gpu.log 2>&1

# change &track use_cuda=0 to run with cpu
mpirun -np 4 ../build/genesis4 fel.in -o fel_cpu > fel_cpu.log 2>&1