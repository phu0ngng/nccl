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

int runRSAlgo(int nranks, size_t size, int nChannels, int nsteps, int stepSize, int verbose) {
  int errors = 0;
  const ssize_t channelSize = size / nChannels;
  struct fifo* fifos = (struct fifo*)malloc(sizeof(struct fifo)*log2Up(nranks)*nranks);
  for (int rank=0; rank<nranks; rank+=verbose ? 1 : nranks-1) {
    int nsend = 0, nrecv = 0;
    for (int i=0; i<log2Up(nranks); i++) fifoInit(fifos+i);
    PatRSAlgorithm<char> algo(stepSize, nsteps, 0, channelSize, size, stepSize, rank, nranks);
    if (verbose) printf("|Rank|RecvDim|SendDim|InputOffset|OutputOffset|RecvOffset|SendOffset|Nelem|PostRecv|PostSend|\n");
    if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+-----+--------+--------+\n");
    int last = 0;
    while (last == 0) {
      int recvDim, sendDim, recvOffset, sendOffset, sendStepOffset, postRecv, postSend, nelem;
      size_t inpIx, outIx;
      algo.getNextOp(recvDim, sendDim, inpIx, outIx, recvOffset, sendOffset, sendStepOffset, nelem, postRecv, postSend, last);
      nsend += postSend;
      nrecv += postRecv;
      int peer = inpIx/size;
      int rcolor = 0, scolor = 0;
      int sendError = 0, recvError = 0;
      int errorColor[] = { 0, 33, 31 };
      if (sendDim != -1) {
        fifoSend(fifos+sendDim, -(1<<sendDim), nranks, sendOffset, sendStepOffset, peer, postSend, &sendError);
        scolor = errorColor[sendError];
      }
      if (recvDim != -1) {
        fifoRecv(fifos+recvDim, recvOffset, 0, peer, postRecv, &recvError);
        rcolor = errorColor[recvError];
      }
      char rcolorstr[] = " [0;XXm";
      char scolorstr[] = " [0;XXm";
      char rcolorend[] = " [00m";
      char scolorend[] = " [00m";
      rcolorstr[0] = scolorstr[0] = 0;
      rcolorend[0] = scolorend[0] = 0;
      if (rcolor) { sprintf(rcolorstr, "%c[0;%dm", 0x1b, rcolor); rcolorend[0] = 0x1b; errors++; }
      if (scolor) { sprintf(scolorstr, "%c[0;%dm", 0x1b, scolor); scolorend[0] = 0x1b; errors++; }
      if (verbose) printf("|%4d|%3d/%3d|%3d/%3d|%11ld|%12ld|%s%10d%s|%s%1d/%8d%s|%5d|%8d|%8d|\n",
          rank, recvDim, peer, sendDim, peer, inpIx, outIx,
          rcolorstr, recvOffset, rcolorend, scolorstr, sendStepOffset, sendOffset, scolorend,
          nelem, postRecv, postSend);
      if (postSend || postRecv) if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+-----+--------+--------+\n");
    }
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
  }
  free(fifos);
  return errors;
}
int runAGAlgo(int nranks, size_t size, int nChannels, int nsteps, int stepSize, int verbose) {
  int errors = 0;
  const ssize_t channelSize = size / nChannels;
  struct fifo* fifos = (struct fifo*)malloc(sizeof(struct fifo)*log2Up(nranks)*nranks);
  for (int rank=0; rank<nranks; rank+=verbose ? 1 : nranks-1) {
    int nsend = 0, nrecv = 0;
    for (int i=0; i<log2Up(nranks); i++) fifoInit(fifos+i);
    PatAGAlgorithm<char> algo(stepSize, nsteps, 0, channelSize, size, stepSize, rank, nranks);
    if (verbose) printf("|Rank|RecvDim|SendDim|InputOffset|OutputOffset|RecvOffset|SendOffset|Nelem|PostRecv|PostSend|\n");
    if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+-----+--------+--------+\n");
    int last = 0;
    while (last == 0) {
      int recvDim, sendDim, recvOffset, sendOffset, recvStepOffset, postRecv, postSend, nelem;
      size_t inpIx, outIx;
      algo.getNextOp(recvDim, sendDim, inpIx, outIx, recvOffset, sendOffset, recvStepOffset, nelem, postRecv, postSend, last);
      nsend += postSend;
      nrecv += postRecv;
      int peer = outIx/size;
      int rcolor = 0, scolor = 0;
      int sendError = 0, recvError = 0;
      int errorColor[] = { 0, 33, 31 };
      if (sendDim != -1) {
        fifoSend(fifos+sendDim, 1<<sendDim, nranks, sendOffset, 0, peer, postSend, &sendError);
        scolor = errorColor[sendError];
      }
      if (recvDim != -1) {
        fifoRecv(fifos+recvDim, recvOffset, recvStepOffset, peer, postRecv, &recvError);
        rcolor = errorColor[recvError];
      }
      char rcolorstr[] = " [0;XXm";
      char scolorstr[] = " [0;XXm";
      char rcolorend[] = " [00m";
      char scolorend[] = " [00m";
      rcolorstr[0] = scolorstr[0] = 0;
      rcolorend[0] = scolorend[0] = 0;
      if (rcolor) { sprintf(rcolorstr, "%c[0;%dm", 0x1b, rcolor); rcolorend[0] = 0x1b; errors++; }
      if (scolor) { sprintf(scolorstr, "%c[0;%dm", 0x1b, scolor); scolorend[0] = 0x1b; errors++; }
      if (verbose) printf("|%4d|%3d/%3d|%3d/%3d|%11ld|%12ld|%s%1d/%8d%s|%s%10d%s|%5d|%8d|%8d|\n",
          rank, recvDim, peer, sendDim, peer, inpIx, outIx,
          rcolorstr, recvStepOffset, recvOffset, rcolorend, scolorstr, sendOffset, scolorend,
          nelem, postRecv, postSend);
      if (postSend || postRecv) if (verbose) printf("+----+-------+-------+-----------+------------+----------+----------+-----+--------+--------+\n");
    }
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
  }
  free(fifos);
  return errors;
}

/*
int runAGAlgo(int nranks, size_t size, int nChannels, int nsteps, int stepSize) {
  int errors = 0;
  const ssize_t channelSize = size / nChannels;
  int *offsets = (int*)malloc(sizeof(int)*log2Up(nranks)*nranks);
  for (int rank=0; rank<nranks; rank++) {
    int nsend = 0, nrecv = 0;
    for (int i=0; i<log2Up(nranks)*nranks; i++) offsets[i] = -1;
    PatAGAlgorithm<char> algo(stepSize, nsteps, 0, channelSize, size, stepSize, rank, nranks);
    printf("|Rank|RecvDim|SendDim|InputOffset|OutputOffset|RecvOffset|SendOffset|Nelem|PostRecv|PostSend|\n");
    int last = 0;
    int lastSendDim = -2;
    while (last == 0) {
      int recvDim, sendDim, recvOffset, sendOffset, recvStepOffset, postRecv, postSend, nelem;
      size_t inpIx, outIx;
      algo.getNextOp(recvDim, sendDim, inpIx, outIx, recvOffset, sendOffset, recvStepOffset, nelem, postRecv, postSend, last);
      if (sendDim != lastSendDim) {
        printf("+----+-------+-------+-----------+------------+----------+----------+-----+--------+--------+\n");
        lastSendDim = sendDim;
      }
      nsend += postSend;
      nrecv += postRecv;
      int peer = outIx/nelem;
      int sendPeer = sendDim >= 0 ? peer : -1;
      int recvPeer = recvDim >= 0 ? peer : -1;
      int rocolor = 0, socolor = 0, rpcolor = 0;
      if (recvDim != -1) {
        if (offsets[recvDim*nranks+sendPeer] == -1) offsets[recvDim*nranks+sendPeer] = recvOffset;
        else if (offsets[recvDim*nranks+sendPeer] != recvOffset) { socolor = 33; errors++; }
      }
      int found = -1;
      if (sendDim != -1) {
        for (int i=0; i<nranks; i++) {
          if (offsets[sendDim*nranks+i] == sendOffset) {
            found = i;
            if (((found-(1<<sendDim)+nranks)%nranks) != sendPeer) { rpcolor = 31; errors++; }
            offsets[sendDim*nranks+i] = -1;
            break;
          }
        }
        if (found == -1) { rocolor = 31; errors++; }
      }
      char rocolorstr[] = " [0;XXm";
      char socolorstr[] = " [0;XXm";
      char rpcolorstr[] = " [0;XXm";
      char rocolorend[] = " [00m";
      char socolorend[] = " [00m";
      char rpcolorend[] = " [00m";
      rocolorstr[0] = socolorstr[0] = rpcolorstr[0] = 0;
      rocolorend[0] = socolorend[0] = rpcolorend[0] = 0;
      if (rocolor) { sprintf(rocolorstr, "%c[0;%dm", 0x1b, rocolor); rocolorend[0] = 0x1b; }
      if (socolor) { sprintf(socolorstr, "%c[0;%dm", 0x1b, socolor); socolorend[0] = 0x1b; }
      if (rpcolor) { sprintf(rpcolorstr, "%c[0;%dm", 0x1b, rpcolor); rpcolorend[0] = 0x1b; }
      printf("|%4d|%3d/%s%3d%s|%3d/%3d|%11ld|%12ld|%s%1d/%8d%s|%s%10d%s|%5d|%8d|%8d|\n",
          rank, recvDim, rpcolorstr, recvPeer, rpcolorend, sendDim, sendPeer, inpIx, outIx,
          rocolorstr, recvStepOffset, recvOffset, rocolorend, socolorstr, sendOffset, socolorend,
          nelem, postRecv, postSend);
    }
    if (nsend != nrecv) { printf("%c[0;31mError : sends don't match receives%c[00m\n", 0x1b, 0x1b); errors++; }
  }
  free(offsets);
  return errors;
}*/

int main(int argc, char *argv[]) {
  setlinebuf(stdout);
  int errors = 0;
  if (argc < 3) { printf("Usage: log_algo <rs|ag> <nranks> [<size> <stepsize> <nchannels> <nsteps>]\n"); return 1; }
  char* op = argv[1];
  int nranks = strtol(argv[2], NULL, 0);
  ssize_t size = -1;
  ssize_t stepSize = -1;
  int nChannels = -1;
  int nsteps = -1;
  if (argc > 3) {
    size = strtoll(argv[3], NULL, 0);
    stepSize = strtoll(argv[4], NULL, 0);
    nChannels = strtoll(argv[5], NULL, 0);
    nsteps = strtoll(argv[6], NULL, 0);
    if (strcmp(op, "rs") == 0) {
      errors += runRSAlgo(nranks, size, nChannels, nsteps, stepSize, 1);
    } else {
      errors += runAGAlgo(nranks, size, nChannels, nsteps, stepSize, 1);
    }
  } else {
    printf("+-----------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+\n");
    printf("|    nsteps |                 1                 |                 2                 |                 4                 |                 8                 |\n");
    printf("| nchannels |        1        |       2         |         1       |        2        |         1       |        2        |         1       |        2        |\n");
    printf("|      size |1K 16K  512K  64M|1K 16K  512K  64M|1K 16K  512K  64M|1K 16K  512K  64M|1K 16K  512K  64M|1K 16K  512K  64M|1K 16K  512K  64M|1K 16K  512K  64M|\n");
    printf("|    nranks |v   v    v      v|v   v    v      v|v   v    v      v|v   v    v      v|v   v    v      v|v   v    v      v|v   v    v      v|v   v    v      v|\n");
    printf("+-----------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+\n");
    stepSize = 512*1024;
    for (int nr=2; nr<=nranks; nr++) {
      printf("| %9d |", nr);
      for (nsteps = 1; nsteps <=8; nsteps *= 2) {
        for (nChannels=1; nChannels<3; nChannels++) {
          for (size = 1024; size < 128*1024*1024; size *= 2) {
            if (strcmp(op, "rs") == 0) {
              errors = runRSAlgo(nr, size, nChannels, nsteps, stepSize, 0);
            } else {
              errors = runAGAlgo(nr, size, nChannels, nsteps, stepSize, 0);
            }
            printf("%c", errors?'X':'.');
          }
          printf("|");
        }
      }
      printf("\n");
    }
  }
  return errors ? 1 : 0;
}
