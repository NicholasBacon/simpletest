#include <vector>
#include <iostream>
#include <iomanip>

#include <chrono>
#include "mpi.h"
#include <math.h>


#include <list>
#include <cstring>    /* memset & co. */
#include <ctime>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <iostream>
#include <numeric>

#include<unistd.h>
#include <iostream>

//#include "input.hpp"
#include <iostream>
#include <fstream>
#include "Kokkos_Core.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <list>

#define FS_LAYOUT Kokkos::LayoutRight
//#define FS_LAYOUT Kokkos::LayoutLeft
//#define FS_LAYOUT Kokkos::DefaultExecutionSpace::array_layout

// double view types
typedef typename Kokkos::View<float ******, FS_LAYOUT>
FS6D;
typedef typename Kokkos::View<float *****, FS_LAYOUT> FS5D;
typedef typename Kokkos::View<float ****, FS_LAYOUT> FS4D;
typedef typename Kokkos::View<float ***, FS_LAYOUT> FS3D;
typedef typename Kokkos::View<float **, FS_LAYOUT> FS2D;
typedef typename Kokkos::View<float *, FS_LAYOUT> FS1D;

typedef typename Kokkos::View<float ******, FS_LAYOUT>::HostMirror FS6DH;
typedef typename Kokkos::View<float *****, FS_LAYOUT>::HostMirror FS5DH;
typedef typename Kokkos::View<float ****, FS_LAYOUT>::HostMirror FS4DH;
typedef typename Kokkos::View<float ***, FS_LAYOUT>::HostMirror FS3DH;
typedef typename Kokkos::View<float **, FS_LAYOUT>::HostMirror FS2DH;
typedef typename Kokkos::View<float *, FS_LAYOUT>::HostMirror FS1DH;

// int view types
typedef typename Kokkos::View<int ******, FS_LAYOUT> FS6D_I;
typedef typename Kokkos::View<int *****, FS_LAYOUT> FS5D_I;
typedef typename Kokkos::View<int ****, FS_LAYOUT> FS4D_I;
typedef typename Kokkos::View<int ***, FS_LAYOUT> FS3D_I;
typedef typename Kokkos::View<int **, FS_LAYOUT> FS2D_I;
typedef typename Kokkos::View<int *, FS_LAYOUT> FS1D_I;

typedef typename Kokkos::View<int ******, FS_LAYOUT>::HostMirror FS6DH_I;
typedef typename Kokkos::View<int *****, FS_LAYOUT>::HostMirror FS5DH_I;
typedef typename Kokkos::View<int ****, FS_LAYOUT>::HostMirror FS4DH_I;
typedef typename Kokkos::View<int ***, FS_LAYOUT>::HostMirror FS3DH_I;
typedef typename Kokkos::View<int **, FS_LAYOUT>::HostMirror FS2DH_I;
typedef typename Kokkos::View<int *, FS_LAYOUT>::HostMirror FS1DH_I;

typedef typename Kokkos::MDRangePolicy<Kokkos::Rank < 2>>
policy_f;
typedef typename Kokkos::MDRangePolicy<Kokkos::Rank < 3>>
policy_f3;
typedef typename Kokkos::MDRangePolicy<Kokkos::Rank < 4>>
policy_f4;
typedef typename Kokkos::MDRangePolicy<Kokkos::Rank < 5>>
policy_f5;


int n = 10;
int startbytes = 1;
int endbytes = 150000;

bool server = false;
int testcount = 10;
int testamount = 10;
//int NG_START_PACKET_SIZE = 1028;
//int max_datasize = 10000000 * 2;
int maxbuffercount = 100000000; // 100M floats


bool selfPack;
int total_size;


void pingandpong(FS1D a, int amount, int count, int blocklength, int stride, FS1D temp, int rank, bool copy) {


    int MPI_TAG2 = 1;
    auto kok = Kokkos::MDRangePolicy < Kokkos::Rank < 3 >> ( { 0, 0, 0 },
    { amount, count, blocklength } );

    std::list<double> times0;
    for (int k = 0; k < testcount; ++k) {

        double t0 = MPI_Wtime();
        for (int j = 0; j < testamount; ++j) {
            if (rank % 2 == 0) {

                Kokkos::parallel_for(
                        kok, KOKKOS_LAMBDA(
                const int am,
                const int c,
                const int b) {

                    temp(am * count * blocklength + c * blocklength + b) = a(
                            am * count * stride + c * stride + b);

                });

                Kokkos::fence();


                int temp_rank = rank + 1;
                MPI_Send(temp.data(), amount * count * blocklength, MPI_FLOAT, temp_rank, MPI_TAG2,
                         MPI_COMM_WORLD);
                MPI_Recv(temp.data(), amount * count * blocklength, MPI_FLOAT, temp_rank, MPI_TAG2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//        if (copy) {
//            Kokkos::deep_copy(recv, recv_H);
//        }

                Kokkos::parallel_for(
                        kok, KOKKOS_LAMBDA(
                const int am,
                const int c,
                const int b) {
                    a(am * count * stride + c * stride + b) = temp(
                            am * count * blocklength + c * blocklength + b);
                });
                Kokkos::fence();
            } else {
                int temp_rank = rank - 1;


                MPI_Recv(temp.data(), amount * count * blocklength, MPI_FLOAT, temp_rank, MPI_TAG2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Kokkos::parallel_for(
                        kok, KOKKOS_LAMBDA(
                const int am,
                const int c,
                const int b) {
                    a(am * count * blocklength + c * blocklength + b) = temp(
                            am * count * blocklength + c * blocklength + b);
                });
                Kokkos::fence();
                Kokkos::parallel_for(
                        kok, KOKKOS_LAMBDA(
                const int am,
                const int c,
                const int b) {

                    temp(am * count * blocklength + c * blocklength + b) = a(
                            am * count * stride + c * stride + b);

                });
                Kokkos::fence();
                MPI_Send(temp.data(), amount * count * blocklength, MPI_FLOAT, temp_rank, MPI_TAG2,
                         MPI_COMM_WORLD);
            }

        }
        double tfinal = (MPI_Wtime() - t0) / (testamount);


        times0.push_back(tfinal);


    }

    times0.sort();
    int count1 = 0;
    double low = 0;
    double med = 0;
    double high = 0;
    double total = 0;
    for (const auto &item: times0) {
        total += item;
        if (count1 == 0) {
            low = item;
        }
        if (count1 == (testcount) / 2) {
            med = item;
        }
        if (count1 == testcount - 1) {
            high = item;
        }

        count1++;

    }

    double mean0 = total / times0.size();


    printf("h-%i,%i,%i,%i,%15.9f,%15.9f,%15.9f,%15.9f\n", amount * count * blocklength * 4, count, blocklength, stride,
           mean0, low, med, high);
    fflush(stdout);
}


void pongandping(void *buffer, int i, int rank) {

    MPI_Status status;
    int size_s;


    std::list<double> times0;
    for (int k = 0; k < testcount; ++k) {

        double t0 = MPI_Wtime();
        for (int j = 0; j < testamount; ++j) {
            if (rank == 0) {
                MPI_Send(buffer, i, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, i, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
            }
            if (rank == 1) {
                MPI_Recv(buffer, i, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send(buffer, i, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }
        }
        double tfinal = (MPI_Wtime() - t0) / (testamount);


        times0.push_back(tfinal);


    }

    times0.sort();
    int count = 0;
    double low = 0;
    double med = 0;
    double high = 0;
    double total = 0;
    for (const auto &item: times0) {
        total += item;
        if (count == 0) {
            low = item;
        }
        if (count == (testcount) / 2) {
            med = item;
        }
        if (count == testcount - 1) {
            high = item;
        }

        count++;

    }

    double mean0 = total / times0.size();


    printf("e-%i,%15.9f,%15.9f,%15.9f,%15.9f\n", i * 4, mean0, low, med, high);
    fflush(stdout);


}


void pingpong(void *buffer, int i, int rank, MPI_Datatype datatype, int x, int y, int z) {

    MPI_Status status;
    int size_s;


    MPI_Type_size(datatype, &size_s);
    std::list<double> times0;
    for (int k = 0; k < testcount; ++k) {

        double t0 = MPI_Wtime();
        for (int j = 0; j < testamount; ++j) {
            if (rank == 0) {
                MPI_Send(buffer, i, datatype, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, i, datatype, 1, 0, MPI_COMM_WORLD, &status);
            }
            if (rank == 1) {
                MPI_Recv(buffer, i, datatype, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send(buffer, i, datatype, 0, 0, MPI_COMM_WORLD);
            }
        }
        double tfinal = (MPI_Wtime() - t0) / (testamount);


        times0.push_back(tfinal);


    }

    times0.sort();
    int count = 0;
    double low = 0;
    double med = 0;
    double high = 0;
    double total = 0;
    for (const auto &item: times0) {
        total += item;
        if (count == 0) {
            low = item;
        }
        if (count == (testcount) / 2) {
            med = item;
        }
        if (count == testcount - 1) {
            high = item;
        }

        count++;

    }

    double mean0 = total / times0.size();


    printf("d-%i,%i,%i,%i,%15.9f,%15.9f,%15.9f,%15.9f\n", i * size_s, x, y, z, mean0, low, med, high);
    fflush(stdout);


}


int main(int argc, char *argv[]) {


    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    Kokkos::initialize(argc, argv);



    FS1D a = Kokkos::View<float *, FS_LAYOUT>("data", maxbuffercount);
    FS1D holder = Kokkos::View<float *, FS_LAYOUT>("holder", maxbuffercount);

    char *buffer = (char *) a.data();

    server = rank == 0;

    for (int i = 1; i <= 10; ++i) {
        for (int j = 1; j <= i; ++j) {
            for (int k = 1; k <= j; ++k) {


                MPI_Datatype oldType = MPI_FLOAT;
                MPI_Datatype type;
                MPI_Type_vector(k, j, i, oldType, &type);

                MPI_Type_commit(&type);
//                printf("datatype- %i %i %i\n ",  k, k, j);

                for (int amount = 1; amount < 200000; amount *= 2) {

                    pingpong(buffer, amount, rank, type, k, j, i);
                    MPI_Barrier(MPI_COMM_WORLD );
                    pingandpong(a, amount, k, j, i, holder, rank, false);
                    MPI_Barrier(MPI_COMM_WORLD );
                    pongandping(buffer, amount * k * j, rank);
                    MPI_Barrier(MPI_COMM_WORLD );

                }


                MPI_Type_free(&type);

            }
        }
    }


    MPI_Finalize();
    Kokkos::finalize();


    return 0;
}
