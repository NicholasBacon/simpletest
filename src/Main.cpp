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
typedef typename Kokkos::View<float *, FS_LAYOUT> FS1D;

int _size;
void *topBuff;
void *bottomBuff;
void *leftBuff;
void *rightBuff;
long long int maxamout = 51200;
long long int maxbuffercount = maxamout * maxamout;

void doWork(FS1D a) {


    int size = _size;

    Kokkos::parallel_for(size * size - size * 4, KOKKOS_LAMBDA(
    const int spot) {
        int y = spot / size + 1;
        int x = spot % size + 1;
        float temp = a(y * size + (x + 1)) + a(y * size + (x - 1)) + a((y + 1) * size + x) + a((y - 1) * size + x) +
                     a((y - 1) * size + (x + 1)) + a((y - 1) * size + (x - 1)) + a((y + 1) * size + (x - 1)) +
                     a((y + 1) * size + (x + 1));


        if (temp == 2.0) {
            if (a(y * size + x) == 1.0) {
                a(y * size + x) = 1.0;
            }
        } else if (temp == 3.0) {

            if (a(y * size + x) == 1.0) {
                a(y * size + x) = 1.0;
            } else {
                a(y * size + x) = 1.0;


            }
        } else {

            a(y * size + x) = 0.0;
        }

        a(y * size + x) = 1.0;

    });
    Kokkos::fence();

}

void pack(FS1D l, FS1D r, FS1D buff) {

    int size = _size;
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(
    const int spot) {
        l[spot] = buff[spot * size];
        r[spot] = buff[spot * size + size - 1];
    });
    Kokkos::fence();

}


void unpack(FS1D l, FS1D r, FS1D buff) {

    int size = _size;
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(
    const int spot) {
        buff[spot * size] = l[spot];
        buff[spot * size + size - 1] = r[spot];
    });
    Kokkos::fence();

}


void setAddress(float *buff) {
    topBuff = buff;
    bottomBuff = buff + (_size * (_size - 1));
    leftBuff = buff;
    rightBuff = buff + _size - 1;

}

int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}

int main(int argc, char *argv[]) {


    fflush(stdout);
    int rank, num_procs;
    int maxsq = 1;
    int kokkos = atoi(argv[1]);


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    Kokkos::initialize(argc, argv);
    while (maxsq*maxsq<num_procs){
        maxsq++;
    }


    fflush(stdout);
    FS1D a = Kokkos::View<float *, FS_LAYOUT>("data", maxbuffercount);
    FS1D sendL = Kokkos::View<float *, FS_LAYOUT>("data", maxamout);
    FS1D resvL = Kokkos::View<float *, FS_LAYOUT>("data", maxamout);
    FS1D sendR = Kokkos::View<float *, FS_LAYOUT>("data", maxamout);
    FS1D resvR = Kokkos::View<float *, FS_LAYOUT>("data", maxamout);
    auto *buffer = (float *) a.data();

    int testamont = 1000;


    setAddress(buffer);
    for (int workx = 1; workx < 101; workx = 10 * workx) {
        _size = 51200;
        MPI_Datatype oldType = MPI_FLOAT;
        MPI_Datatype type;
        MPI_Type_contiguous(_size, oldType, &type);
        MPI_Type_commit(&type);
        int size_0;
        MPI_Type_size(type, &size_0);


        double singalwork = MPI_Wtime();
        for (int zz = 0; zz < testamont; ++zz) {

            for (int j = 0; j < workx; ++j) {
                doWork(a);
            }

        }
        double work = (MPI_Wtime() - singalwork) / testamont;
        printf("p-%i-%i-%i-%i,%15.9f,%15.9f,%15.9f\n", 0, 1, workx, size_0, work, 0, work);
        printf("p-%i-%i-%i-%i,%15.9f,%15.9f,%15.9f\n", 1, 1, workx, size_0, work, 0, work);
        fflush(stdout);
        MPI_Type_free(&type);
    }

    for (int sq = 2; sq < maxsq+1; ++sq) {


        for (int workx = 1; workx < 101; workx = 10 * workx) {


            MPI_Barrier(MPI_COMM_WORLD);

            //warmup
            if (sq * sq > rank) {
                for (int zz = 0; zz < 100; ++zz) {
                    MPI_Request request[8];

                    MPI_Irecv(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[0]);
                    MPI_Irecv(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[1]);
                    MPI_Irecv(resvR.data(), 1, type, right, 123, MPI_COMM_WORLD, &request[2]);
                    MPI_Irecv(resvL.data(), 1, type, left, 123, MPI_COMM_WORLD, &request[3]);
                    MPI_Isend(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[4]);
                    MPI_Isend(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[5]);
                    MPI_Isend(sendL.data(), 1, type, left, 123, MPI_COMM_WORLD, &request[6]);
                    MPI_Isend(sendR.data(), 1, type, right, 123, MPI_COMM_WORLD, &request[7]);
                    MPI_Waitall(8, request, MPI_STATUSES_IGNORE);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (sq * sq > rank) {
                int i = 51200 / (sq * sq);

                _size = i;
                MPI_Datatype oldType = MPI_FLOAT;
                MPI_Datatype type;
                MPI_Type_contiguous(_size, oldType, &type);
                MPI_Datatype oldType1 = MPI_FLOAT;
                MPI_Datatype type1;
                MPI_Type_contiguous(_size, oldType1, &type1);

                MPI_Type_commit(&type);
                MPI_Type_commit(&type1);
                int size_0;
                int size_1;

                MPI_Type_size(type, &size_0);
                MPI_Type_size(type, &size_1);


                std::list<double> times0;
                std::list<double> times1;
                std::list<double> times2;


                for (int zz = 0; zz < testamont; ++zz) {
                    double t0 = MPI_Wtime();
                    MPI_Request request[8];

                    if (kokkos) {
                        MPI_Irecv(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[0]);
                        MPI_Irecv(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[1]);
                        MPI_Irecv(resvR.data(), 1, type, right, 123, MPI_COMM_WORLD, &request[2]);
                        MPI_Irecv(resvL.data(), 1, type, left, 123, MPI_COMM_WORLD, &request[3]);
                        MPI_Isend(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[4]);
                        MPI_Isend(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[5]);
                        pack(sendL, sendR, a);
                        MPI_Isend(sendL.data(), 1, type, left, 123, MPI_COMM_WORLD, &request[6]);
                        MPI_Isend(sendR.data(), 1, type, right, 123, MPI_COMM_WORLD, &request[7]);
                        MPI_Waitall(8, request, MPI_STATUSES_IGNORE);
                        unpack(resvL, resvR, a);
                    } else {
                        MPI_Irecv(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[0]);
                        MPI_Irecv(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[1]);
                        MPI_Irecv(resvR.data(), 1, type, right, 123, MPI_COMM_WORLD, &request[2]);
                        MPI_Irecv(resvL.data(), 1, type, left, 123, MPI_COMM_WORLD, &request[3]);
                        MPI_Isend(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[4]);
                        MPI_Isend(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[5]);
                        MPI_Isend(sendL.data(), 1, type, left, 123, MPI_COMM_WORLD, &request[6]);
                        MPI_Isend(sendR.data(), 1, type, right, 123, MPI_COMM_WORLD, &request[7]);
                        MPI_Waitall(8, request, MPI_STATUSES_IGNORE);
                    }


                    double sendingData = MPI_Wtime();
                    for (int j = 0; j < workx; ++j) {
                        doWork(a);
                    }

                    double work = MPI_Wtime();

                    times0.push_back(work - sendingData);
                    times1.push_back(sendingData - t0);
                    times2.push_back(work - t0);
                }


                double totalsend = 0;
                double totalwork = 0;
                double total = 0;
                for (const auto &item: times0) {
                    totalwork += item;

                }

                for (const auto &item: times1) {
                    totalsend += item;

                }
                for (const auto &item: times2) {
                    total += item;

                }
                double work = totalwork / times0.size();
                double send = totalsend / times0.size();
                double totaltime = total / times0.size();
                MPI_Type_free(&type);
                MPI_Type_free(&type1);
                printf("p-%i-%i-%i-%i,%15.9f,%15.9f,%15.9f\n", kokkos, sq, workx, size_0, work, send, totaltime);
                fflush(stdout);
            }
        }
    }
    MPI_Finalize();
    Kokkos::finalize();

    return 0;
}


