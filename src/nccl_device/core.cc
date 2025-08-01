#include "core.h"
#include "comm.h"
#include "nccl_device/impl/core__funcs.h"

NCCL_API_CXX(ncclSymTeam, ncclSymTeamWorld, struct ncclComm* comm);
ncclSymTeam ncclSymTeamWorld(struct ncclComm* comm) {
  ncclSymTeam ans;
  ans.nRanks = comm->nRanks;
  ans.rank = comm->rank;
  ans.stride = 1;
  return ans;
}

NCCL_API_CXX(ncclSymTeam, ncclSymTeamNear, struct ncclComm* comm);
ncclSymTeam ncclSymTeamNear(struct ncclComm* comm) {
  ncclSymTeam ans;
  ans.nRanks = comm->symrState.nearSize;
  ans.rank = comm->symrState.nearSelf;
  ans.stride = 1;
  return ans;
}

NCCL_API_CXX(ncclSymTeam, ncclSymTeamRail, struct ncclComm* comm);
ncclSymTeam ncclSymTeamRail(struct ncclComm* comm) {
  ncclSymTeam ans;
  ans.nRanks = comm->nRanks/comm->symrState.nearSize;
  ans.rank = comm->rank/comm->symrState.nearSize;
  ans.stride = comm->symrState.nearSize;
  return ans;
}

NCCL_API_CXX(int, ncclSymTeamRankToWorld, struct ncclComm* comm, ncclSymTeam team, int rank);
int ncclSymTeamRankToWorld(struct ncclComm* comm, ncclSymTeam team, int rank) {
  return comm->rank + (rank - team.rank)*team.stride;
}

NCCL_API_CXX(int, ncclSymTeamRankToNear, struct ncclComm* comm, ncclSymTeam team, int rank);
int ncclSymTeamRankToNear(struct ncclComm* comm, ncclSymTeam team, int rank) {
  return comm->symrState.nearSelf + (rank - team.rank)*team.stride;
}
