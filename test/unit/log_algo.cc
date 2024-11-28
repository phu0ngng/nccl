/******************************************************************************************/
/*  Usage: log_algo {rs,ag} minranks maxranks rank size stepSize nChannels nSteps verbose */
/*  Default values:      rs        0 minranks    0    8   524288         2      8 depends */
/*  All arguments are optional and have default values                                    */
/******************************************************************************************/
#include "collectives.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int log2Up(int n) {
  int l = 1;
  while (l<n) l*= 2;
  return l;
}

struct fifo {
  uint64_t head;
  uint64_t tail;
  struct fifoElement* send;
  struct fifoElement* recv;
  int postSend;
  int postRecv;
};

struct fifoElement {
  int offset;    // Offset in step
  int rank;      // Data for a given rank
  uint64_t step; // Step in the FIFO
  struct fifoElement* next; // Chaining
};

void fifoInit(struct fifo* fifo) {
  fifo->head = 0;
  fifo->tail = 0;
  fifo->send = NULL;
  fifo->recv = NULL;
}

void fifoSend(struct fifo* fifo, int shiftRecv, int nranks, int offset, int stepOffset, int rank, int postSend, int* error) {
  *error = 0;
  struct fifoElement* e = fifo->send;
  while (e) {
    if (e->step == fifo->tail+stepOffset && e->offset == offset) {
      if (e->rank != rank) {
        *error = 1;
      }
      break;
    }
    e = e->next;
  }

  if (e == NULL) {
    //printf("Add send queue offset %d step %ld rank %d\n", offset, fifo->tail+stepOffset, rank);
    e = (struct fifoElement*)malloc(sizeof(struct fifoElement));
    e->offset = offset;
    e->rank = rank;
    e->step = fifo->tail + stepOffset;
    e->next = NULL;
    struct fifoElement* last = fifo->send;
    if (last == NULL) fifo->send = e;
    else {
      while (last->next) last = last->next;
      last->next = e;
    }
  }

  if (postSend) {
    if (fifo->postSend) {
      *error = 3;
    }
    fifo->postSend = 1;
    e = fifo->send;
    struct fifoElement* prev = NULL;
    struct fifoElement* recvEnd = fifo->recv;
    if (recvEnd) while (recvEnd->next) recvEnd = recvEnd->next;
    // Move all elements from this step from send queue to recv queue
    while (e) {
     struct fifoElement* next = e->next;
     if (e->step == fifo->tail) {
       // Dequeue from send list
       if (prev) prev->next = e->next;
       else fifo->send = e->next;
       // Enqueue on recv list, shift rank
       e->rank = (e->rank + nranks + shiftRecv) % nranks;
       //printf("Move from send to recv rank %d offset %d step %ld\n", e->rank, e->offset, e->step);
       e->next = NULL;
       if (recvEnd) recvEnd->next = e;
       else fifo->recv = e;
       recvEnd = e;
     } else {
       prev = e;
     }
     e = next;
    }
    fifo->tail++;
  }
}

void fifoRecv(struct fifo* fifo, int offset, int stepOffset, int rank, int postRecv, int* error) {
  *error = 0;
  struct fifoElement* e = fifo->recv;
  while (e) {
    if (e->step == fifo->head+stepOffset && e->offset == offset) {
      //printf("Found recv queue offset %d step %ld rank %d\n", offset, fifo->head+stepOffset, rank);
      if (e->rank != rank) {
        //printf("Wrong rank %d / %d\n", e->rank, rank);
        *error = 1;
      }
      break;
    }
    e = e->next;
  }
  if (e == NULL) {
    //printf("Nothing found at step %ld offset %d\n", fifo->head+stepOffset, offset);
    *error = 2;
  }
  if (postRecv) {
    if (fifo->postRecv) {
      *error = 3;
    }
    fifo->postRecv = 1;
    struct fifoElement* e = fifo->recv;
    struct fifoElement* prev = NULL;
    while (e) {
      struct fifoElement* next = e->next;
      if (e->step == fifo->head) {
        if (prev) prev->next = e->next;
        else fifo->recv = e->next;
        free(e);
      } else {
        prev = e;
      }
      e = next;
    }
    fifo->head++;
  }
}

int runRSAlgo(int nranks, int rank, size_t size, int nChannels, int nsteps, int stepSize, int verbose) {
  int errors = 0;
  const ssize_t channelSize = size / nChannels;
  struct fifo* fifos = (struct fifo*)malloc(sizeof(struct fifo)*log2Up(nranks)*nranks);
  int nsend = 0, nrecv = 0;
  int last = -1;
  for (int i=0; i<log2Up(nranks); i++) fifoInit(fifos+i);
  PatRSAlgorithm<char> algo(stepSize, nsteps, NCCL_PAT_NWORKERS/WARP_SIZE, 0, channelSize, size, stepSize, rank, nranks);
  int parFactor = algo.getParallelFactor();
  if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
  if (verbose) printf("| Reduce Scatter nranks %6d size %8ld nChannels %2d nsteps %1d stepSize %7d parFactor %2d |\n", nranks, size, nChannels, nsteps, stepSize, parFactor);
  if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
  if (verbose) printf("|Rank|RecvDim|SendDim|InputOffset|OutputOffset|RecvOffset|SendOffset|Nelems|PostRecv|PostSend|USL|\n");
  struct ncclPatStep ps;
  int step = 0;
  do {
    if (step++ % parFactor == 0) {
      if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
      for (int i=0; i<log2Up(nranks); i++) fifos[i].postSend = fifos[i].postRecv = 0;
      if (last > 0) {
        errors++;
        if (verbose) printf("%c[0;31mNew step after last steps.%c[00m\n", 0x1b, 0x1b);
      }
      last = -1;
    }
    algo.getNextOp(&ps);
    if (ps.flags & PatSkipped) {
      if (verbose) printf("|%4d|       |       |           |            |          |          |      |        |        |%1d%1d%1d|\n", rank, ps.flags&PatUsed ? 1 : 0, ps.flags&PatSkipped ? 1 : 0, ps.last);
      continue;
    }
    nsend += ps.postSend;
    nrecv += ps.postRecv;
    int peer = ps.inpIx/size;
    int rcolor = 0, scolor = 0, lcolor = 0;
    int sendError = 0, recvError = 0;
    int errorColor[] = { 0, 33, 31, 35 };
    int newLast = ps.last ? 1 : 0;
    if (last != -1 && newLast != last) {
      lcolor = errorColor[2];
    }
    last = newLast;
    if (ps.sendDim != -1) {
      fifoSend(fifos+ps.sendDim, -(1<<ps.sendDim), nranks, ps.sendOffset, ps.stepOffset, peer, ps.postSend, &sendError);
      scolor = errorColor[sendError];
    }
    if (ps.recvDim != -1) {
      fifoRecv(fifos+ps.recvDim, ps.recvOffset, 0, peer, ps.postRecv, &recvError);
      rcolor = errorColor[recvError];
    }
    char rcolorstr[] = " [0;XXm";
    char scolorstr[] = " [0;XXm";
    char lcolorstr[] = " [0;XXm";
    char rcolorend[] = " [00m";
    char scolorend[] = " [00m";
    char lcolorend[] = " [00m";
    rcolorstr[0] = scolorstr[0] = lcolorstr[0] = 0;
    rcolorend[0] = scolorend[0] = lcolorend[0] = 0;
    if (rcolor) { sprintf(rcolorstr, "%c[0;%dm", 0x1b, rcolor); rcolorend[0] = 0x1b; errors++; }
    if (scolor) { sprintf(scolorstr, "%c[0;%dm", 0x1b, scolor); scolorend[0] = 0x1b; errors++; }
    if (lcolor) { sprintf(lcolorstr, "%c[0;%dm", 0x1b, lcolor); lcolorend[0] = 0x1b; errors++; }
    if (verbose) printf("|%4d|%3d/%3d|%3d/%3d|%11ld|%12ld|%s%10d%s|%s%1d/%8d%s|%6d|%8d|%8d|%1d%1d%s%1d%s|\n",
        rank, ps.recvDim, peer, ps.sendDim, peer, ps.inpIx, ps.outIx,
        rcolorstr, ps.recvOffset, rcolorend, scolorstr, ps.stepOffset, ps.sendOffset, scolorend,
        ps.nelem, ps.postRecv, ps.postSend,
        ps.flags&PatUsed ? 1 : 0, ps.flags&PatSkipped ? 1 : 0, lcolorstr, ps.last, lcolorend);
  } while (ps.last != 2);
  if (nsend != nrecv) { if (verbose) printf("%c[0;31mError : sends don't match receives%c[00m\n", 0x1b, 0x1b); errors++; }
  for (int i=0; i<log2Up(nranks); i++) {
    for (struct fifoElement* e = fifos[i].send; e; e = e->next) {
      if (verbose) printf("%c[0;31mRemaining element in send fifo dimension %d offset %d rank %d step %ld%c[00m\n", 0x1b, i, e->offset, e->rank, e->step, 0x1b);
      errors++;
    }
    for (struct fifoElement* e = fifos[i].recv; e; e = e->next) {
      if (verbose) printf("%c[0;31mRemaining element in recv fifo dimension %d offset %d rank %d step %ld%c[00m\n", 0x1b, i, e->offset, e->rank, e->step, 0x1b);
      errors++;
    }
  }
  if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
  free(fifos);
  return errors;
}

int runAGAlgo(int nranks, int rank, size_t size, int nChannels, int nsteps, int stepSize, int verbose) {
  int errors = 0;
  const ssize_t channelSize = size / nChannels;
  struct fifo* fifos = (struct fifo*)malloc(sizeof(struct fifo)*log2Up(nranks)*nranks);
  int nsend = 0, nrecv = 0;
  int last = -1;
  for (int i=0; i<log2Up(nranks); i++) fifoInit(fifos+i);
  PatAGAlgorithm<char> algo(stepSize, nsteps, NCCL_PAT_NWORKERS/WARP_SIZE, 0, channelSize, size, stepSize, rank, nranks);
  int parFactor = algo.getParallelFactor();
  if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
  if (verbose) printf("| All Gather     nranks %6d size %8ld nChannels %2d nsteps %1d stepSize %7d parFactor %2d |\n", nranks, size, nChannels, nsteps, stepSize, parFactor);
  if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
  if (verbose) printf("|Rank|RecvDim|SendDim|InputOffset|OutputOffset|RecvOffset|SendOffset|Nelems|PostRecv|PostSend|USL|\n");
  struct ncclPatStep ps;
  int step = 0;
  do {
    if (step++ % parFactor == 0) {
      if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
      for (int i=0; i<log2Up(nranks); i++) fifos[i].postSend = fifos[i].postRecv = 0;
      if (last > 0) {
        errors++;
        if (verbose) printf("%c[0;31mNew step after last steps.%c[00m\n", 0x1b, 0x1b);
      }
      last = -1;
    }
    algo.getNextOp(&ps);
    if (ps.flags & PatSkipped) {
      if (verbose) printf("|%4d|       |       |           |            |          |          |      |        |        |%1d%1d%1d|\n", rank, ps.flags&PatUsed ? 1 : 0, ps.flags&PatSkipped ? 1 : 0, ps.last);
      continue;
    }
    nsend += ps.postSend;
    nrecv += ps.postRecv;
    int peer = ps.outIx/size;
    int rcolor = 0, scolor = 0, lcolor = 0;
    int sendError = 0, recvError = 0;
    int errorColor[] = { 0, 33, 31, 35 };
    int newLast = ps.last ? 1 : 0;
    if (last != -1 && newLast != last) {
      lcolor = errorColor[2];
    }
    last = newLast;
    if (ps.sendDim != -1) {
      fifoSend(fifos+ps.sendDim, 1<<ps.sendDim, nranks, ps.sendOffset, 0, peer, ps.postSend, &sendError);
      scolor = errorColor[sendError];
    }
    if (ps.recvDim != -1) {
      fifoRecv(fifos+ps.recvDim, ps.recvOffset, ps.stepOffset, peer, ps.postRecv, &recvError);
      rcolor = errorColor[recvError];
    }
    char rcolorstr[] = " [0;XXm";
    char scolorstr[] = " [0;XXm";
    char lcolorstr[] = " [0;XXm";
    char rcolorend[] = " [00m";
    char scolorend[] = " [00m";
    char lcolorend[] = " [00m";
    rcolorstr[0] = scolorstr[0] = lcolorstr[0] = 0;
    rcolorend[0] = scolorend[0] = lcolorend[0] = 0;
    if (rcolor) { sprintf(rcolorstr, "%c[0;%dm", 0x1b, rcolor); rcolorend[0] = 0x1b; errors++; }
    if (scolor) { sprintf(scolorstr, "%c[0;%dm", 0x1b, scolor); scolorend[0] = 0x1b; errors++; }
    if (lcolor) { sprintf(lcolorstr, "%c[0;%dm", 0x1b, lcolor); lcolorend[0] = 0x1b; errors++; }
    if (verbose) printf("|%4d|%3d/%3d|%3d/%3d|%11ld|%12ld|%s%1d/%8d%s|%s%10d%s|%6d|%8d|%8d|%1d%1d%s%1d%s|\n",
        rank, ps.recvDim, peer, ps.sendDim, peer, ps.inpIx, ps.outIx,
        rcolorstr, ps.stepOffset, ps.recvOffset, rcolorend, scolorstr, ps.sendOffset, scolorend,
        ps.nelem, ps.postRecv, ps.postSend,
        ps.flags&PatUsed ? 1 : 0, ps.flags&PatSkipped ? 1 : 0, lcolorstr, ps.last, lcolorend);
  } while (ps.last != 2);
  if (nsend != nrecv) { if (verbose) printf("%c[0;31mError : sends don't match receives%c[00m\n", 0x1b, 0x1b); errors++; }
  for (int i=0; i<log2Up(nranks); i++) {
    for (struct fifoElement* e = fifos[i].send; e; e = e->next) {
      if (verbose) printf("%c[0;31mRemaining element in send fifo dimension %d offset %d rank %d step %ld%c[00m\n", 0x1b, i, e->offset, e->rank, e->step, 0x1b);
      errors++;
    }
    for (struct fifoElement* e = fifos[i].recv; e; e = e->next) {
      if (verbose) printf("%c[0;31mRemaining element in recv fifo dimension %d offset %d rank %d step %ld%c[00m\n", 0x1b, i, e->offset, e->rank, e->step, 0x1b);
      errors++;
    }
  }
  if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+------+--------+--------+---+\n");
  free(fifos);
  return errors;
}

int main(int argc, char *argv[]) {
  setlinebuf(stdout);
  int totalErrors = 0;
  const char* op         = argc > 1 ? argv[1] : "rs";
  int nranksStart  = argc > 2 ? strtol(argv[2], NULL, 0) : 128;
  int nranksEnd    = argc > 3 ? strtol(argv[3], NULL, 0) : nranksStart;
  int rank         = argc > 4 ? strtol(argv[4], NULL, 0) : 0;
  ssize_t size     = argc > 5 ? strtoll(argv[5], NULL, 0) : 8;
  ssize_t stepSize = argc > 6 ? strtoll(argv[6], NULL, 0) : 512*1024;
  int nChannels    = argc > 7 ? strtol(argv[7], NULL, 0) : 2;
  int nsteps       = argc > 8 ? strtol(argv[8], NULL, 0) : 8;
  int verbose      = argc > 9 ? strtol(argv[9], NULL, 0) : argc > 4 ? 1 : 0;
  if (argc > 4) {
    for (int nr=nranksStart; nr<=nranksEnd; nr++) {
      if (strcmp(op, "rs") == 0) {
        totalErrors = runRSAlgo(nr, rank, size, nChannels, nsteps, stepSize, verbose);
      } else {
        totalErrors = runAGAlgo(nr, rank, size, nChannels, nsteps, stepSize, verbose);
      }
    }
  } else {
    printf("+-----------+-----------------------------+\n");
    printf("| nchannels |      1       |     2        |\n");
    printf("|      size |1K 16K 256K 8M|1K 16K 256K 8M|\n");
    printf("|    nranks |v   v   v    v|v   v   v    v|\n");
    printf("+-----------+--------------+--------------+\n");
    stepSize = 512*1024;
    for (int nr=nranksStart; nr<=nranksEnd; nr++) {
      printf("| %9d |", nr);
      for (nChannels=1; nChannels<3; nChannels++) {
        for (size = 1024; size <= 8*1024*1024; size *= 2) {
          int errors;
          if (strcmp(op, "rs") == 0) {
            totalErrors += errors = runRSAlgo(nr, rank, size, nChannels, nsteps, stepSize, verbose);
          } else {
            totalErrors += errors = runAGAlgo(nr, rank, size, nChannels, nsteps, stepSize, verbose);
          }
          printf("%c", errors?'X':'.');
          fflush(stdout);
        }
        printf("|");
      }
      printf("\n");
    }
    printf("+-----------+-----------------------------+\n");
    printf("| Total     | %10d errors           |\n", totalErrors);
    printf("+-----------+-----------------------------+\n");
  }
  return totalErrors ? 1 : 0;
}
