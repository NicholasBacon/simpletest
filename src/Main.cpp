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
typedef typename Kokkos::View<float *, FS_LAYOUT> FS1D;

int _size;
void *topBuff;
void *bottomBuff;
void *leftBuff;
void *rightBuff;
long long int maxamout = 40000;
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
//    int sq = atoi(argv[1]);


    int maxsq = 1;


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    Kokkos::initialize(argc, argv);
    while (maxsq * maxsq < num_procs) {
        maxsq++;
    }
    maxsq++;
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


        {


            int i = maxamout;

            _size = i;
            MPI_Datatype oldType = MPI_FLOAT;
            MPI_Datatype type;
            MPI_Type_contiguous(_size, oldType, &type);
            MPI_Type_commit(&type);

            int size_0;


            MPI_Type_size(type, &size_0);


            std::list<double> times0;
            std::list<double> times1;
            std::list<double> times2;


            for (int zz = 0; zz < testamont; ++zz) {
                double t0 = MPI_Wtime();


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
            printf("np-%i-%i-%i,%15.9f,%15.9f,%15.9f\n", 1, workx, size_0, work, send, totaltime);
            printf("p-%i-%i-%i,%15.9f,%15.9f,%15.9f\n", 1, workx, size_0, work, send, totaltime);

            fflush(stdout);


        }


        for (int sq = 2; sq < maxsq; ++sq) {


            int x = rank % sq;
            int y = rank / sq;


            int left = y * sq + positive_modulo((x - 1), sq);
            int right = y * sq + positive_modulo((x + 1), sq);
            int top = positive_modulo((y + 1), sq) * sq + x;
            int bottom = positive_modulo((y - 1), sq) * sq + x;

            MPI_Barrier(MPI_COMM_WORLD);


            if (sq * sq > rank) {


//warmup
                for (int zz = 0; zz < testamont; ++zz) {


                    MPI_Request request[8];
                    MPI_Irecv(topBuff, maxamout, MPI_FLOAT, top, 123, MPI_COMM_WORLD, &request[0]);
                    MPI_Irecv(bottomBuff, maxamout, MPI_FLOAT, bottom, 123, MPI_COMM_WORLD, &request[1]);
                    MPI_Irecv(resvR.data(), maxamout, MPI_FLOAT, right, 123, MPI_COMM_WORLD, &request[2]);
                    MPI_Irecv(resvL.data(), maxamout, MPI_FLOAT, left, 123, MPI_COMM_WORLD, &request[3]);
                    MPI_Isend(topBuff, maxamout, MPI_FLOAT, top, 123, MPI_COMM_WORLD, &request[4]);
                    MPI_Isend(bottomBuff, maxamout, MPI_FLOAT, bottom, 123, MPI_COMM_WORLD, &request[5]);
                    MPI_Isend(sendL.data(), maxamout, MPI_FLOAT, left, 123, MPI_COMM_WORLD, &request[6]);
                    MPI_Isend(sendR.data(), maxamout, MPI_FLOAT, right, 123, MPI_COMM_WORLD, &request[7]);
                    MPI_Waitall(8, request, MPI_STATUSES_IGNORE);


                }


            }


            MPI_Barrier(MPI_COMM_WORLD);


            if (sq * sq > rank) {


                int i = maxamout / (sq * sq);

                _size = i;
                MPI_Datatype oldType = MPI_FLOAT;
                MPI_Datatype type;
                MPI_Type_contiguous(_size, oldType, &type);
                MPI_Type_commit(&type);
                int size_0;


                MPI_Type_size(type, &size_0);


                std::list<double> times0;
                std::list<double> times1;
                std::list<double> times2;


                for (int zz = 0; zz < testamont; ++zz) {
                    double t0 = MPI_Wtime();


                    if (false) {
                        MPI_Request request[4];
                        MPI_Request unpacked[4];

                        MPI_Irecv(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[0]);
                        MPI_Irecv(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[1]);
                        MPI_Irecv(resvR.data(), 1, type, right, 123, MPI_COMM_WORLD, &unpacked[0]);
                        MPI_Irecv(resvL.data(), 1, type, left, 123, MPI_COMM_WORLD, &unpacked[1]);

                        MPI_Isend(topBuff, 1, type, top, 123, MPI_COMM_WORLD, &request[2]);
                        MPI_Isend(bottomBuff, 1, type, bottom, 123, MPI_COMM_WORLD, &request[3]);
                        pack(sendL, sendR, a);
                        MPI_Isend(sendR.data(), 1, type, right, 123, MPI_COMM_WORLD, &unpacked[2]);
                        MPI_Isend(sendL.data(), 1, type, left, 123, MPI_COMM_WORLD, &unpacked[4]);
                        MPI_Waitall(4, request, MPI_STATUSES_IGNORE);
                        unpack(resvL, resvR, a);
                        MPI_Waitall(4, request, MPI_STATUSES_IGNORE);
                    } else {
                        MPI_Request request[8];
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
                printf("p-%i-%i-%i,%15.9f,%15.9f,%15.9f\n", sq, workx, size_0, work, send, totaltime);
                fflush(stdout);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if (sq * sq > rank) {

//    for (int i = 200; i <maxamout ; i=i*2) {
//        printf("%i\n",i);
//    }
                int i = maxamout / (sq * sq);

                _size = i;
                MPI_Datatype oldType = MPI_FLOAT;
                MPI_Datatype type;
                MPI_Type_contiguous(_size, oldType, &type);
                MPI_Type_commit(&type);

                int size_0;


                MPI_Type_size(type, &size_0);


                std::list<double> times0;
                std::list<double> times1;
                std::list<double> times2;


                for (int zz = 0; zz < testamont; ++zz) {
                    double t0 = MPI_Wtime();
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
                printf("np-%i-%i-%i,%15.9f,%15.9f,%15.9f\n", sq, workx, size_0, work, send, totaltime);
                fflush(stdout);


            }


        }


    }
    MPI_Finalize();
    Kokkos::finalize();

    return 0;
}


