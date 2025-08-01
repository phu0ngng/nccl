#include "core.h"
#include "comm.h"
#include "nccl_device/impl/core__funcs.h"

NCCL_API_CXX(ncclTeam, ncclTeamWorld, struct ncclComm* comm);
ncclTeam ncclTeamWorld(struct ncclComm* comm) {
  ncclTeam ans;
  ans.nRanks = comm->nRanks;
  ans.rank = comm->rank;
  ans.stride = 1;
  return ans;
}

NCCL_API_CXX(ncclTeam, ncclTeamNear, struct ncclComm* comm);
ncclTeam ncclTeamNear(struct ncclComm* comm) {
  ncclTeam ans;
  ans.nRanks = comm->symrState.nearSize;
  ans.rank = comm->symrState.nearSelf;
  ans.stride = 1;
  return ans;
}

NCCL_API_CXX(ncclTeam, ncclTeamRail, struct ncclComm* comm);
ncclTeam ncclTeamRail(struct ncclComm* comm) {
  ncclTeam ans;
  ans.nRanks = comm->nRanks/comm->symrState.nearSize;
  ans.rank = comm->rank/comm->symrState.nearSize;
  ans.stride = comm->symrState.nearSize;
  return ans;
}

NCCL_API_CXX(int, ncclTeamRankToWorld, struct ncclComm* comm, ncclTeam team, int rank);
int ncclTeamRankToWorld(struct ncclComm* comm, ncclTeam team, int rank) {
  return comm->rank + (rank - team.rank)*team.stride;
}

NCCL_API_CXX(int, ncclTeamRankToNear, struct ncclComm* comm, ncclTeam team, int rank);
int ncclTeamRankToNear(struct ncclComm* comm, ncclTeam team, int rank) {
  return comm->symrState.nearSelf + (rank - team.rank)*team.stride;
}
