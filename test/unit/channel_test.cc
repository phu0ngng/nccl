#include "channel.h"
#include "comm.h"

#include <unistd.h>
static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
	return;
    }
  }
}

#undef NCCLCHECK
#define NCCLCHECK(cmd) do {                              \
  ncclResult_t result = cmd;                             \
  if (result!= ncclSuccess) {                            \
    char hostname[1024];                                 \
    getHostName(hostname, 1024);                         \
    printf("%s: NCCL failure %s:%d '%s'\n",              \
         hostname,                                       \
        __FILE__,__LINE__,ncclGetErrorString(result));   \
    exit(EXIT_FAILURE);                                  \
  }                                                      \
} while(0)

void usage() {
  fprintf(stderr, "Usage: channel_test <nranks>\n");
}

int main(int argc, char* argv[]) {
  int nranks;
  if (argc < 2) {
    usage();
    return 1;
  } else {
    nranks = atoi(argv[1]);
    if (nranks <= 0) {
      usage();
      return 1;
    }
  }
  struct ncclComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  comm->p2pnChannels = 16;
  comm->nNodes = nranks / 8;
  comm->maxLocalRanks = 8;
  comm->rank = 0;
  comm->node = 0;
  NCCLCHECK(ncclCalloc(&comm->rankToNode, nranks));
  NCCLCHECK(ncclCalloc(&comm->rankToLocalRank, nranks));
  for (int r=0; r<nranks; r++) {
    comm->rankToNode[r] = r/8;
    comm->rankToLocalRank[r] = r%8;
  }

  for (int r=0; r<nranks; r++) {
    printf("[%3d]", r);
    for (int i=0; i<4; i++) {
      int base = ncclP2pChannelBaseForRound(comm, r);
      int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, i);
      printf(" %2d", channelId);
    }
    printf("\n");
  }
  free(comm->rankToLocalRank);
  free(comm->rankToNode);
  free(comm);
  return 0;
}
